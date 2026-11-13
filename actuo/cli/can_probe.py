#!/usr/bin/env python3
"""Read-only CAN bring-up probe: prove the stack before anything moves.

Sends NOTHING to the motors by default. It only listens to the periodic status
upload and decodes it through the same Transport/Actuator/Bus objects the live
driver uses, so a green run means node ids, bitrate, wiring, frame layout and
unit conversion are all correct -- verified without energising a single joint.

    actuo-can-probe --config <robot>.yaml
    actuo-can-probe --config <robot>.yaml --seconds 5

Run this before the ros2_control pipeline on any new bus, adapter or robot.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time


import can  # noqa: E402

from actuo.drivers import can_errors  # noqa: E402
from actuo.drivers.actuators.cubemars import ERROR_NAMES  # noqa: E402
from actuo.drivers.buses.can_bus import StaleFeedbackError  # noqa: E402
from actuo.drivers.config import build_bus, build_transport, load_document  # noqa: E402


def describe_error(state) -> str:
    if state.error is None:
        return "?"
    return ERROR_NAMES.get(state.error, "unknown code {}".format(state.error))


#: Standard CAN bitrates worth trying, most likely first for CubeMars.
CANDIDATE_BITRATES = (1000000, 500000, 250000, 125000, 800000, 100000)


def survey(transport, seconds):
    """Listen and classify raw traffic. Sends nothing.

    Returns a dict of counters. This is the evidence that separates "wrong
    bitrate" (error frames, garbled ids) from "wrong node id" (clean 8-byte
    frames on ids we simply do not expect).
    """
    result = {
        "total": 0,
        "errors": 0,
        "ids": {},          # arbitration id -> count
        "extended": 0,
        "standard": 0,
        "dlc8": 0,
        "sample": {},       # arbitration id -> first payload seen
    }
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        frame = transport.recv(timeout=max(0.0, deadline - time.monotonic()))
        if frame is None:
            break
        result["total"] += 1
        if frame.meta.get("error_frame"):
            result["errors"] += 1
            continue
        address = int(frame.address) if frame.address is not None else -1
        result["ids"][address] = result["ids"].get(address, 0) + 1
        result["sample"].setdefault(address, frame.payload)
        if frame.meta.get("extended"):
            result["extended"] += 1
        else:
            result["standard"] += 1
        if frame.meta.get("dlc") == 8:
            result["dlc8"] += 1
    return result


def looks_like_cubemars(result):
    """CubeMars servo upload is an 8-byte frame; a healthy bus is error-free."""
    if result["total"] == 0:
        return False
    if result["errors"] > result["total"] * 0.05:
        return False
    return result["dlc8"] > 0


def scan_bitrates(build, seconds):
    """Open the adapter at each candidate bitrate and report what appears.

    Only listens. A bus read at the wrong bitrate yields error frames and
    nonsense ids; the right one yields clean 8-byte status frames.
    """
    print("[probe] scanning bitrates (listening only, {:.1f}s each)".format(seconds))
    verdicts = []
    for bitrate in CANDIDATE_BITRATES:
        transport = build(bitrate)
        try:
            transport.open()
        except Exception as err:
            print("[probe]   {:>8} : could not open ({})".format(bitrate, err))
            continue
        try:
            result = survey(transport, seconds)
        finally:
            transport.close()

        node_ids = sorted({a & 0xFF for a in result["ids"]})
        healthy = looks_like_cubemars(result)
        print("[probe]   {:>8} : {:5d} frames, {:5d} error, {:2d} distinct id(s), "
              "{:5d} 8-byte  {}".format(
                  bitrate, result["total"], result["errors"], len(result["ids"]),
                  result["dlc8"], "<-- looks like real traffic" if healthy else ""))
        if result["ids"]:
            print("[probe]              node id(s) seen: {}".format(node_ids))
        verdicts.append((bitrate, result, healthy))

    good = [v for v in verdicts if v[2]]
    print()
    if not good:
        print("[probe] no bitrate produced clean 8-byte frames.", file=sys.stderr)
        print("[probe] That points at wiring/termination/power rather than "
              "bitrate or node ids.", file=sys.stderr)
        return 1

    for bitrate, result, _ in good:
        node_ids = sorted({a & 0xFF for a in result["ids"]})
        print("[probe] BITRATE {} looks correct. Motors on node id(s): {}".format(
            bitrate, node_ids))
        for address, count in sorted(result["ids"].items(), key=lambda kv: -kv[1]):
            payload = result["sample"][address]
            decoded = ""
            if len(payload) == 8:
                pos, spd, cur, temp, err = struct.unpack(">hhhbB", payload)
                decoded = ("pos={:.1f}deg spd={}erpm I={:.2f}A T={}C err={}".format(
                    pos * 0.1, spd * 10, cur * 0.01, temp,
                    ERROR_NAMES.get(err, err)))
            print("[probe]   id=0x{:X} node={} x{}  {}".format(
                address, address & 0xFF, count, decoded))
        print("[probe] => set 'bitrate: {}' and the 'id:' fields in the config "
              "accordingly.".format(bitrate))
    return 0


def diagnose(build, bitrate, seconds):
    """Decode the bus's error frames instead of guessing from their raw ids.

    Listen-only. Prints the adapter's clock and computed bit timing (a wrong
    sample point at 1 Mbps errors every frame), then classifies the errors.
    """
    print("[probe] diagnosing at {} bit/s (listening only)".format(bitrate))
    transport = build(bitrate)
    transport.open()
    try:
        gs = getattr(getattr(transport, "_bus", None), "gs_usb", None)
        if gs is not None:
            try:
                cap = gs.device_capability
                print("[probe] adapter clock: {} Hz, bitrate range {}..{}".format(
                    cap.fclk_can, getattr(cap, "brp_min", "?"),
                    getattr(cap, "brp_max", "?")))
                timing = can.BitTiming.from_sample_point(
                    f_clock=cap.fclk_can, bitrate=bitrate, sample_point=87.5)
                print("[probe] computed timing: {}".format(timing))
            except Exception as err:
                print("[probe] (could not read adapter timing: {})".format(err))

        counts = {}
        samples = {}
        total = 0
        data_frames = 0
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            frame = transport.recv(timeout=max(0.0, deadline - time.monotonic()))
            if frame is None:
                break
            total += 1
            if not frame.meta.get("error_frame"):
                data_frames += 1
                continue
            decoded = can_errors.decode(int(frame.address or 0), frame.payload)
            key = (
                tuple(decoded["classes"]),
                tuple(decoded["protocol"]),
                decoded["location"],
            )
            counts[key] = counts.get(key, 0) + 1
            samples[key] = decoded
    finally:
        transport.close()

    print("[probe] {} frames in {:.1f}s ({} were real data frames)".format(
        total, seconds, data_frames))
    if not counts:
        if data_frames:
            print("[probe] no errors -- the bus is healthy at this bitrate.")
        else:
            print("[probe] silence: nothing is transmitting at this bitrate.")
        return 0

    print("[probe] error breakdown:")
    for key, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        decoded = samples[key]
        print("[probe]   x{:<6} class={} protocol={} at={} tx_err={} rx_err={}".format(
            count,
            ", ".join(decoded["classes"]) or "-",
            ", ".join(decoded["protocol"]) or "-",
            decoded["location"],
            decoded["tx_errors"],
            decoded["rx_errors"],
        ))

    print()
    print("[probe] what this means:")
    for key, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:1]:
        for hint in can_errors.explain(samples[key]):
            print("[probe]   - {}".format(hint))
    return 1


def report_unmatched(bus) -> None:
    """Say what WAS on the wire, not just that nothing matched.

    A silent bus and a bus full of motors using other node ids are completely
    different faults with the same symptom, so print the evidence that tells
    them apart.
    """
    # Errors first: they mean the wire is failing, which makes every other
    # observation meaningless. Reporting an error mask as a node id is how this
    # probe twice claimed "motors are talking on node id 12".
    if bus.error_frames:
        print("[probe] {} BUS ERROR frames -- the bus is not carrying data at "
              "all. This is an electrical/timing fault, NOT a node-id "
              "mismatch.".format(bus.error_frames), file=sys.stderr)
        print("[probe] run with --diagnose to decode them.", file=sys.stderr)
        if not bus.unmatched_addresses:
            return

    seen = bus.unmatched_addresses
    print("[probe] data frames that matched no configured id: {} "
          "across {} distinct arbitration id(s)".format(
              bus.unmatched_frames, len(seen)), file=sys.stderr)

    if not seen:
        if not bus.error_frames:
            print("[probe] NOTHING was received at all -- this points at the "
                  "bitrate, the wiring/termination, or motor power, not at "
                  "node ids.", file=sys.stderr)
        return

    print("[probe] observed traffic (a CubeMars status id is "
          "node_id | (packet << 8), so the low byte is the node id):",
          file=sys.stderr)
    for address, count in sorted(seen.items(), key=lambda kv: -kv[1]):
        print("[probe]   id=0x{:08X}  packet={:<3}  low byte (node id)={:<4} "
              "x{}".format(address, (address >> 8) & 0x1FFFFF, address & 0xFF, count),
              file=sys.stderr)

    node_ids = sorted({a & 0xFF for a in seen})
    print("[probe] => motors are talking on node id(s): {}".format(node_ids),
          file=sys.stderr)
    print("[probe] => the config expects: {}".format(
        sorted(int(a.id) for a in bus.actuators)), file=sys.stderr)
    print("[probe] Fix the 'id:' fields in the config to the observed ids "
          "(or renumber the drivers).", file=sys.stderr)


def format_state(name, actuator, state) -> str:
    parts = [
        "{:<10s} id={:<4} pos={:8.3f} rad ({:7.2f} deg)".format(
            name, str(actuator.id), state.position, state.position * 180.0 / 3.141592653589793
        )
    ]
    # Only print what is actually known -- absent stays absent.
    if state.velocity is not None:
        parts.append("vel={:7.3f} rad/s".format(state.velocity))
    else:
        parts.append("erpm={:8.1f}".format(state.aux.get("erpm", float("nan"))))
    parts.append("I={:6.2f}A".format(state.aux.get("current_a", float("nan"))))
    parts.append("T={:3.0f}C".format(state.temperature))
    parts.append("err={}".format(describe_error(state)))
    return "  ".join(parts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Listen-only CubeMars/CAN bring-up probe (sends nothing)"
    )
    parser.add_argument("--config", required=True, help="robot bus YAML")
    parser.add_argument("--seconds", type=float, default=3.0,
                        help="how long to listen (default: 3)")
    parser.add_argument("--rate", type=float, default=2.0,
                        help="print rate in Hz (default: 2)")
    # Transport-only overrides so one joint map serves every machine. The same
    # USB2CAN adapter is socketcan/can0 on a Linux robot PC and gs_usb/index 0
    # on a mac; the motors' calibration is identical and stays in the file.
    parser.add_argument("--interface", default=None,
                        help="override bus.transport.interface (e.g. gs_usb, socketcan)")
    parser.add_argument("--channel", default=None,
                        help="override bus.transport.channel (gs_usb wants an "
                             "integer device index, e.g. 0)")
    parser.add_argument("--bitrate", type=int, default=None,
                        help="override bus.transport.bitrate (ignored by socketcan)")
    parser.add_argument("--scan-bitrates", action="store_true",
                        help="listen at each common bitrate and report which one "
                             "yields clean frames, plus the node ids present")
    parser.add_argument("--diagnose", action="store_true",
                        help="decode the bus's error frames and say which layer "
                             "is broken (listen-only)")
    args = parser.parse_args(argv)

    overrides = {}
    if args.interface is not None:
        overrides["interface"] = args.interface
    if args.channel is not None:
        # gs_usb selects the device by integer index, socketcan by name.
        try:
            overrides["channel"] = int(args.channel)
        except ValueError:
            overrides["channel"] = args.channel
    if args.bitrate is not None:
        overrides["bitrate"] = args.bitrate

    if args.scan_bitrates or args.diagnose:
        # Build a fresh transport per bitrate from the same config block.
        document = load_document(args.config)
        base = dict(document["bus"]["transport"])
        base.update(overrides)

        def build(bitrate):
            spec = dict(base)
            spec["bitrate"] = bitrate
            return build_transport(spec)

        if args.diagnose:
            return diagnose(
                build, bitrate=overrides.get("bitrate", base.get("bitrate", 1000000)),
                seconds=args.seconds,
            )
        return scan_bitrates(build, seconds=max(0.5, args.seconds / 4.0))

    bus, names = build_bus(args.config, transport_overrides=overrides or None)
    print("[probe] config: {}".format(args.config))
    print("[probe] transport: {}".format(type(bus.transport).__name__))
    for name, actuator in zip(names, bus.actuators):
        caps = actuator.capabilities
        print("[probe]   {:<10s} id={:<4} modes={}".format(
            name, str(actuator.id), [m.value for m in caps.modes]))

    try:
        bus.transport.open()
    except Exception as err:
        print("[probe] could not open the CAN bus: {}".format(err), file=sys.stderr)
        text = str(err).lower()
        if "permission" in text or "access denied" in text:
            print("[probe] libusb could not claim the adapter. On macOS a gs_usb/"
                  "candleLight device needs root:", file=sys.stderr)
            print("[probe]   sudo -E PYTHONPATH=$PYTHONPATH python3 {} ...".format(
                os.path.basename(__file__)), file=sys.stderr)
        elif "no such device" in text or "network is down" in text:
            print("[probe] Linux: bring the link up first:", file=sys.stderr)
            print("[probe]   sudo ip link set can0 type can bitrate 1000000 && "
                  "sudo ip link set up can0", file=sys.stderr)
        elif "socketcan" in text or "not supported" in text:
            print("[probe] socketcan is Linux-only. On macOS pass "
                  "--interface gs_usb --channel 0 (candleLight/InnoMaker USB2CAN) "
                  "or --interface slcan --channel /dev/tty.usbmodemXXXX",
                  file=sys.stderr)
        return 2

    print("[probe] listening for {:.1f}s -- NOT sending any command".format(args.seconds))
    try:
        try:
            bus.read_batch(timeout=max(2.0, args.seconds))
        except StaleFeedbackError as err:
            print("[probe] FAIL: {}".format(err), file=sys.stderr)
            report_unmatched(bus)
            return 1

        deadline = time.monotonic() + args.seconds
        period = 1.0 / args.rate
        next_print = 0.0
        faults = set()
        while time.monotonic() < deadline:
            bus.poll(timeout=0.01)
            now = time.monotonic()
            if now >= next_print:
                print("--")
                for name, actuator in zip(names, bus.actuators):
                    state = bus.last_state(actuator.id)
                    if state is None:
                        continue
                    print(format_state(name, actuator, state))
                    if state.error:
                        faults.add((name, describe_error(state)))
                next_print = now + period

        bus.check_fresh()
    except StaleFeedbackError as err:
        print("[probe] FAIL: feedback went stale: {}".format(err), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[probe] interrupted")
    finally:
        bus.transport.close()

    if faults:
        print("[probe] FAIL: motors reporting faults:", file=sys.stderr)
        for name, reason in sorted(faults):
            print("  {}: {}".format(name, reason), file=sys.stderr)
        return 1

    print("[probe] OK: all {} motors reported healthy, decoded cleanly, "
          "nothing was commanded".format(len(names)))
    if bus.unmatched_frames:
        # Not necessarily a fault -- other nodes may legitimately share the bus.
        print("[probe] note: {} frame(s) from other ids also on this bus: {}".format(
            bus.unmatched_frames,
            sorted({a & 0xFF for a in bus.unmatched_addresses})))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
