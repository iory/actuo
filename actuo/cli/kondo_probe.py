#!/usr/bin/env python3
"""Read-only first contact with a Kondo ARM-H7 board.

Commands nothing. It asks the board what firmware it runs, and -- given a symbol
map -- which servos it can see and where they are. Nothing here can move a
joint: the only packets it sends are the version request and memory *reads*.

This is deliberately the first thing to run against new hardware, the same way
``actuo-can-probe`` is for a CAN arm. It is also how you obtain the firmware
version string that ``actuo-kondo-symbols`` needs.

Usage
-----
::

    actuo-kondo-probe --channel /dev/ttyUSB1 --version-only
    actuo-kondo-probe --channel /dev/ttyUSB1 \\
        --symbols kondoh7_symbols.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional, Sequence, Set

import yaml


from actuo.drivers.actuators.kondo_ics import (  # noqa: E402
    CENTER_PULSE, PULSE_PER_DEG, SERVO_OFF_PULSE, SERVO_ON_PULSE)
from actuo.drivers.buses.kondo_h7_bus import (  # noqa: E402
    KondoH7Bus, compare_discovery)
from actuo.drivers.kondo_h7_protocol import (  # noqa: E402
    ProtocolError, SERVO_VECTOR_LENGTH, SYSTEM_SLOTS, build_version,
    decode_servo_slot, load_symbol_map, servo_slot_read, system_slot_read)
from actuo.drivers.transports.kondo_h7 import (  # noqa: E402
    DEFAULT_BITRATE, KondoH7Transport)

from actuo.core.contracts import Frame  # noqa: E402


def ask(transport, body, timeout):
    """Send one request and return its reply payload, or ``None``."""
    reply = transport.transact([Frame(payload=body)], timeout)[0]
    return None if reply is None else reply.payload


def read_version(transport, timeout):
    payload = ask(transport, build_version(), timeout)
    if payload is None:
        return None
    return payload.decode("utf-8", errors="replace").strip("\x00").strip()


def read_slot(transport, base, slot, timeout):
    payload = ask(transport, servo_slot_read(base, slot), timeout)
    if payload is None:
        return None
    return decode_servo_slot(slot, payload)


def load_expected_ids(path: str) -> List[int]:
    """Return sorted unique Kondo servo ids declared in a robot config."""
    with open(path, "r") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise ValueError("{}: expected a YAML mapping".format(path))
    joints = document.get("joints")
    if not isinstance(joints, list):
        raise ValueError("{}: expected 'joints' to be a list".format(path))

    expected: Set[int] = set()
    for index, joint in enumerate(joints):
        if not isinstance(joint, dict):
            raise ValueError("{}: joints[{}] is not a mapping".format(path, index))
        actuator = joint.get("actuator")
        if not isinstance(actuator, dict):
            continue
        if actuator.get("plugin") != "kondo_ics":
            continue
        if "id" not in actuator:
            raise ValueError(
                "{}: joints[{}].actuator.plugin is 'kondo_ics' but no 'id' is set"
                .format(path, index))
        expected.add(int(actuator["id"]))
    return sorted(expected)


def _ids_text(ids: Optional[Sequence[int]]) -> str:
    if ids is None:
        return "unavailable"
    return "none" if not ids else ", ".join(str(value) for value in ids)


def _read_version_retrying(transport, args):
    """Ask for the version up to ``--attempts`` times.

    Not stubbornness: rcb4 retries ten times right after a power-on because
    "the ACK value becomes unstable for some reason", and a single request
    reports a healthy board as dead. It also makes the silent verdict below
    mean something -- ten unanswered requests is evidence, one is a guess.
    """
    for attempt in range(max(1, args.attempts)):
        version = read_version(transport, args.timeout)
        if version is not None:
            if attempt:
                print("(answered on attempt {} of {})".format(
                    attempt + 1, args.attempts))
            return version
        time.sleep(0.1)
    return None


def _after_usb_reset(transport, args):
    """Reset the USB device, wait for it to come back, and ask again."""
    from actuo.drivers.usb_reset import reset_usb_device

    try:
        described = reset_usb_device(args.channel)
    except Exception as error:                                 # noqa: BLE001
        print("could not reset the USB device: {}".format(error))
        return None
    print("USB reset {}; waiting for it to re-enumerate".format(described))
    transport.close()
    # Two seconds, as rcb4 waits: opening before udev has recreated the node
    # fails in a way that reads as the board still being dead.
    time.sleep(2.0)
    try:
        transport.open()
    except Exception as error:                                 # noqa: BLE001
        print("the port did not come back after the reset: {}".format(error))
        return None
    return _read_version_retrying(transport, args)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Read-only probe of a Kondo ARM-H7 board.")
    parser.add_argument("--channel", required=True,
                        help="serial device, e.g. /dev/ttyUSB1")
    parser.add_argument("--bitrate", type=int, default=DEFAULT_BITRATE)
    parser.add_argument("--timeout", type=float, default=0.2,
                        help="seconds to wait for one reply")
    parser.add_argument("--symbols",
                        help="firmware symbol map (from actuo-kondo-symbols). "
                             "Without it, only the version can be read.")
    parser.add_argument("--version-only", action="store_true",
                        help="read the firmware version and stop")
    parser.add_argument("--scan", action="store_true",
                        help="run startup discovery scans (read-only): "
                             "per-port ids, per-servo flags/health counters, "
                             "and optional config comparison.")
    parser.add_argument("--config",
                        help="robot config YAML for expected-vs-found "
                             "comparison (used with --scan)")
    parser.add_argument("--attempts", type=int, default=10,
                        help="version requests before calling the board "
                             "silent (default 10). rcb4 retries ten times too, "
                             "with the note that the first reply after a "
                             "power-on is unstable; one request throws away a "
                             "board that answers on the third.")
    parser.add_argument("--reset-usb", action="store_true",
                        help="if the board stays silent, reset its USB device "
                             "and try again. This is a real remedy, not a "
                             "shot in the dark: the link can wedge with the "
                             "device still enumerated and the port still "
                             "opening, and neither reopening nor a physical "
                             "replug clears it. Needs write access to "
                             "/dev/bus/usb.")
    args = parser.parse_args(argv)

    transport = KondoH7Transport(args.channel, args.bitrate, args.timeout)
    transport.open()
    try:
        version = _read_version_retrying(transport, args)
        if version is None and args.reset_usb:
            version = _after_usb_reset(transport, args)
        if version is None:
            print("the board did not answer {} version requests.".format(
                args.attempts))
            print("  Nothing was commanded. The port opened and the write "
                  "succeeded, so this says the BOARD is not answering -- it "
                  "does not say why.")
            print("  Three causes look identical from here:")
            print("    * the board is unpowered, or its firmware has stopped;")
            print("    * the cable or the adapter;")
            print("    * a WEDGED usb-serial link -- the device stays "
                  "enumerated and the port still opens, and neither reopening "
                  "it nor unplugging and replugging clears it. This has been "
                  "observed on real hardware.")
            if not args.reset_usb:
                print("  For the third, re-run with --reset-usb, which is what "
                      "rcb4 does for the same fault.")
            return 1
        print("firmware version: {!r}".format(version))

        if args.version_only:
            return 0

        if args.scan:
            if not args.symbols:
                print("\n--scan needs --symbols; discovery reads board structs "
                      "by firmware address and will not guess those addresses.")
                return 2
            symbols = load_symbol_map(args.symbols)
            try:
                symbols.require_version(version)
            except ProtocolError as error:
                print("\n{}".format(error))
                return 2

            bus = KondoH7Bus(transport, [], symbols)
            try:
                discovery = bus.discover()
            except ProtocolError as error:
                print("\nscan failed: {}".format(error))
                return 2

            timeout_count = read_slot(
                transport, symbols.address("servo_vector"), "timeout_count",
                args.timeout)
            rxerror_count = read_slot(
                transport, symbols.address("servo_vector"), "rxerror_count",
                args.timeout)
            if timeout_count is None or rxerror_count is None:
                print("\nscan failed: board did not answer comm-health slot reads")
                return 1

            print("symbol map matches the board.")
            print("\nstartup discovery:")
            print("  firmware: {!r}".format(discovery.firmware_version))
            print("  present ids: {}".format(_ids_text(discovery.present_ids)))
            if discovery.timed_out_ids is None:
                print("  timed-out ids: unavailable ({})"
                      .format(discovery.ics_channel_scan_error))
            else:
                print("  timed-out ids: {}".format(_ids_text(discovery.timed_out_ids)))

            print("\nper-port ICS ids (from Control.device_port, second source):")
            print("  caveat: the index->servo-id packing is measured on one board")
            print("          only; ServoStruct.flag/port stay authoritative, and")
            print("          the port cross-check below is what confirms it here.")
            if discovery.per_port is None:
                print("  unavailable: {}".format(discovery.ics_channel_scan_error))
            else:
                for port in sorted(discovery.per_port):
                    ids = discovery.per_port[port]
                    print("  J{}: {}".format(port, _ids_text(ids)))

            flags = bus.scan_servo_flags()
            ports = bus.scan_servo_ports()
            if discovery.per_port is None:
                print("\nport cross-check (ServoStruct.port vs inferred device_port):")
                print("  unavailable: {}".format(discovery.ics_channel_scan_error))
            else:
                inferred_ports = {}
                for port, ids in discovery.per_port.items():
                    for servo_id in ids:
                        inferred_ports[int(servo_id)] = int(port)
                disagreements = bus.compare_servo_ports_with_inferred_ics_ports(
                    ports, inferred_ports)
                print("\nport cross-check (ServoStruct.port vs inferred device_port):")
                if disagreements:
                    details = ", ".join(
                        "id {} (ServoStruct.port={}, device_port={})".format(
                            servo_id, values[0], values[1])
                        for servo_id, values in sorted(disagreements.items()))
                    print("  DISAGREE: inferred mapping is wrong on this robot; "
                          "do not trust Control.device_port ({})".format(details))
                else:
                    print("  AGREE: inferred mapping is confirmed on this robot "
                          "for ids both sources can represent.")

            print("\nservo flags and comm health:")
            print("{:>4}  {:>5}  {:>5}  {:>13}  {:>13}".format(
                "id", "flag", "port", "timeout_count", "rxerror_count"))
            for servo_id in range(SERVO_VECTOR_LENGTH):
                print("{:>4}  {:>5}  {:>5}  {:>13}  {:>13}".format(
                    servo_id, int(flags[servo_id]), ports[servo_id],
                    timeout_count[servo_id], rxerror_count[servo_id]))

            if args.config:
                try:
                    expected_ids = load_expected_ids(args.config)
                except ValueError as error:
                    print("\nconfig parse error: {}".format(error))
                    return 2
                missing, unexpected, timed_out = compare_discovery(
                    discovery, expected_ids)
                print("\nconfig comparison ({})".format(args.config))
                print("  expected ids: {}".format(_ids_text(expected_ids)))
                print("  missing ids: {}".format(_ids_text(sorted(missing))))
                print("  unexpected ids: {}".format(_ids_text(sorted(unexpected))))
                if timed_out is None:
                    print("  timed-out ids: unavailable "
                          "({})".format(discovery.ics_channel_scan_error))
                else:
                    print("  timed-out ids: {}".format(_ids_text(sorted(timed_out))))

                timeout_scan_unavailable = timed_out is None
                timeout_hits = set() if timed_out is None else set(timed_out)
                suspicious = bool(missing or timeout_hits)
                # A confirmed fault outranks a missing check. `missing` comes
                # from the VERIFIED flag scan, so when it is non-empty we know a
                # servo is absent -- reporting that as INCOMPLETE ("we could not
                # tell") would understate something we did determine. The
                # inferred device_port scan is routinely unavailable (any symbol
                # map without `Control`), so this pair is common, not exotic.
                if suspicious:
                    verdict = "SUSPICIOUS"
                elif timeout_scan_unavailable:
                    verdict = "INCOMPLETE"
                else:
                    verdict = "OK"
                reason = []
                if missing:
                    reason.append(
                        "verified flag scan is missing id(s): {}"
                        .format(_ids_text(sorted(missing))))
                else:
                    reason.append("verified flag scan found no missing ids")
                if unexpected:
                    reason.append(
                        "unexpected discovered id(s): {}"
                        .format(_ids_text(sorted(unexpected))))
                else:
                    reason.append("no unexpected discovered ids")
                if timeout_scan_unavailable:
                    reason.append(
                        "device_port timeout scan unavailable ({})"
                        .format(discovery.ics_channel_scan_error))
                elif timeout_hits:
                    reason.append(
                        "device_port timeout bit set for id(s): {}"
                        .format(_ids_text(sorted(timeout_hits))))
                else:
                    reason.append("device_port timeout scan found no timed-out ids")
                print("VERDICT: {} - {}".format(verdict, "; ".join(reason)))

            print("\ntransport counters: checksum_errors={} framing_errors={}"
                  .format(transport.checksum_errors, transport.framing_errors))
            print("nothing was commanded.")
            return 0

        if not args.symbols:
            print("\nno --symbols given, so no address is known for "
                  "servo_vector; stopping here rather than guessing one.")
            return 0

        symbols = load_symbol_map(args.symbols)
        try:
            symbols.require_version(version)
        except ProtocolError as error:
            print("\n{}".format(error))
            return 2
        print("symbol map matches the board.")

        servo_base = symbols.address("servo_vector")
        flags = read_slot(transport, servo_base, "flag", args.timeout)
        angles = read_slot(transport, servo_base, "current_angle", args.timeout)
        references = read_slot(transport, servo_base, "ref_angle", args.timeout)
        ports = read_slot(transport, servo_base, "port", args.timeout)
        if flags is None or angles is None:
            print("the board answered the version but not a memory read.")
            return 1

        present = [i for i in range(SERVO_VECTOR_LENGTH) if flags[i] > 0]
        print("\nservos present: {}".format(present if present else "none"))
        if present:
            print("{:>4}  {:>5}  {:>9}  {:>8}  {:>6}".format(
                "id", "uart", "pulse", "deg", "servo"))
            for servo_id in present:
                pulse = angles[servo_id]
                reference = references[servo_id] if references else None
                if reference == SERVO_OFF_PULSE:
                    servo = "off"
                elif reference == SERVO_ON_PULSE:
                    servo = "on"
                elif reference is None:
                    servo = "?"
                else:
                    servo = "on"
                degrees = ("{:8.2f}".format((pulse - CENTER_PULSE) / PULSE_PER_DEG)
                           if pulse else "       -")
                print("{:>4}  {:>5}  {:>9}  {}  {:>6}".format(
                    servo_id, ports[servo_id] if ports else "?", pulse,
                    degrees, servo))
            print("\nA pulse of 0 means the board has never heard from that "
                  "servo -- it is not an angle.")

        battery = ask(transport,
                      system_slot_read(symbols.address("SysB"),
                                       "battery_voltage"),
                      args.timeout)
        if battery is not None:
            volts = SYSTEM_SLOTS["battery_voltage"].decode(battery)[0]
            print("\nbattery: {:.2f} V".format(volts))

        print("\ntransport counters: checksum_errors={} framing_errors={}"
              .format(transport.checksum_errors, transport.framing_errors))
        print("nothing was commanded.")
        return 0
    finally:
        transport.close()


if __name__ == "__main__":
    sys.exit(main())
