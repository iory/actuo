#!/usr/bin/env python3
"""Read and drive a KJS-03 / AIRRELAY vacuum board on a DYNAMIXEL bus.

The board sits on the same bus as the servos and speaks DYNAMIXEL 2.0, but it
is not a motor: it carries a pressure sensor, an IMU, and (on the AIRRELAY
variant) a pump + relay driving a vacuum gripper. This tool exercises it through
the same ``Kjs03Sensor`` plugin the driver uses, so what works here works in the
stack -- there is no second, tool-only code path.

Commands
--------
    read     monitor pressure + IMU (safe, read-only)
    grip     pump ON + relay ON   -> draws vacuum   (~-45 kPa measured with one cup)
    hold     pump OFF + relay ON  -> holds the vacuum with the pump stopped
    release  everything OFF        -> vents to atmosphere (0 kPa)
    pump     pump ON only          -> pump runs, cup unconnected
    off      alias of release

Why ``release`` is its own thing, and not the firmware's VALVE_STATE
-------------------------------------------------------------------
Measured on hardware: the firmware's VALVE_STATE "release" leaves the relay
energised and *holds* the vacuum (~-42 kPa) rather than venting. The gripper is
only actually released by dropping the relay, so every command here writes
GPIO_STATE directly (``Kjs03Sensor.encode_air``). A release that does not
release would be silent breakage: a command reporting success while doing
something else.

Examples
--------
    actuo-air --channel /dev/ttyUSB0 read
    actuo-air --channel /dev/ttyUSB0 grip --watch 4
    actuo-air --channel /dev/ttyUSB0 release
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

from actuo.core.contracts import CapabilityError
from actuo.drivers import dynamixel_protocol as dxl
from actuo.drivers.sensors.kjs03 import AIR_OPS, Kjs03Sensor
from actuo.drivers.transports.serial_transport import SerialTransport


def _read_identity(transport: SerialTransport, sensor: Kjs03Sensor,
                   timeout: float) -> Optional[dict]:
    """Read the board's identity block; return its fields or None if silent.

    Doubles as the "is this really a KJS-03" check -- ``decode_identity``
    refuses a model/magic mismatch rather than reading a stranger's table.
    """
    reply = transport.transact(sensor.encode_identity_query(), timeout)[0]
    if reply is None:
        return None
    return sensor.decode_identity(reply)


def _read_state(transport: SerialTransport, sensor: Kjs03Sensor,
                timeout: float) -> Optional[dict]:
    reply = transport.transact(sensor.encode_query(), timeout)[0]
    if reply is None:
        return None
    return sensor.decode_state(reply)


def _fmt_state(state: dict) -> str:
    """One compact line: pressure, gyro, accel, and any hardware error."""
    parts = []
    if "air/pressure" in state:
        parts.append("P={:6.2f}kPa".format(state["air/pressure"]))
    gyro = tuple(state.get("imu/angular_velocity.{}".format(a)) for a in "xyz")
    acc = tuple(state.get("imu/linear_acceleration.{}".format(a)) for a in "xyz")
    if all(v is not None for v in gyro):
        parts.append("gyro=[{:+.2f} {:+.2f} {:+.2f}]rad/s".format(*gyro))
    if all(v is not None for v in acc):
        mag = (acc[0] ** 2 + acc[1] ** 2 + acc[2] ** 2) ** 0.5
        parts.append("acc=[{:+.2f} {:+.2f} {:+.2f}]|{:.2f}|m/s^2".format(
            acc[0], acc[1], acc[2], mag))
    err = int(state.get("hardware_error", 0))
    flagged = [n for bit, n in ((0x01, "imu"), (0x08, "adc")) if err & bit]
    if flagged:  # proximity/magenc absent is normal on AIRRELAY; only flag real ones
        parts.append("HW_ERR={}".format(",".join(flagged)))
    return "  ".join(parts)


def _apply(transport: SerialTransport, sensor: Kjs03Sensor, op: str,
           timeout: float) -> bool:
    """Send an air command and confirm the write was acknowledged."""
    frame = sensor.encode_air(op)
    ack = transport.transact([frame], timeout)[0]
    return ack is not None


def _monitor(transport: SerialTransport, sensor: Kjs03Sensor, seconds: float,
             timeout: float, period: float = 0.3) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        state = _read_state(transport, sensor, timeout)
        if state is None:
            print("  (no reply)")
        else:
            print("  " + _fmt_state(state))
        time.sleep(period)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Read/drive a KJS-03 AIRRELAY vacuum board over DYNAMIXEL.")
    parser.add_argument("command",
                        choices=["read"] + sorted(AIR_OPS),
                        help="read (monitor) or an air op: %s" % ", ".join(
                            sorted(AIR_OPS)))
    parser.add_argument("--config", default=None,
                        help="robot config YAML; reads channel + the kjs03 "
                             "sensor's id from it (its sensors: section)")
    parser.add_argument("--sensor", default=None,
                        help="with --config, name of the sensor to use "
                             "(default: the only/first kjs03 board)")
    parser.add_argument("--channel", default=None,
                        help="serial device, e.g. /dev/ttyUSB0 "
                             "(overrides the config's channel)")
    parser.add_argument("--id", type=int, default=None,
                        help="DYNAMIXEL id of the board (default 19, or from "
                             "--config)")
    parser.add_argument("--baud", type=int, default=1000000)
    parser.add_argument("--timeout", type=float, default=0.05)
    parser.add_argument("--watch", type=float, default=0.0, metavar="SEC",
                        help="after a command, monitor pressure for SEC seconds")
    parser.add_argument("--seconds", type=float, default=3.0,
                        help="how long 'read' monitors (default 3 s)")
    args = parser.parse_args(argv)

    channel, device_id = args.channel, args.id
    if args.config:
        from actuo.drivers.config import build_sensors, load_document
        document = load_document(args.config)
        try:
            cfg_channel = document["bus"]["transport"]["channel"]
        except (KeyError, TypeError):
            cfg_channel = None
        channel = channel or cfg_channel
        boards = [(n, s) for n, s in build_sensors(document.get("sensors"))
                  if isinstance(s, Kjs03Sensor)
                  and (args.sensor is None or n == args.sensor)]
        if not boards:
            print("no kjs03 sensor{} in {}".format(
                " named %r" % args.sensor if args.sensor else "", args.config),
                file=sys.stderr)
            return 2
        if device_id is None:
            device_id = boards[0][1].id
    if channel is None:
        print("need --channel, or --config with a bus.transport.channel",
              file=sys.stderr)
        return 2
    if device_id is None:
        device_id = 19

    transport = SerialTransport(channel, args.baud, timeout=args.timeout,
                                protocol=dxl.PROTOCOL_2)
    try:
        transport.open()
    except Exception as error:              # noqa: BLE001 - report and exit
        print("cannot open {}: {}".format(channel, error), file=sys.stderr)
        return 2

    try:
        sensor = Kjs03Sensor(device_id)
        identity = _read_identity(transport, sensor, args.timeout)
        if identity is None:
            print("id {} did not answer on {}. Is the board powered and on "
                  "this bus? (try: actuo-dxl-probe --channel {})".format(
                      device_id, channel, channel), file=sys.stderr)
            return 1
        # One state read teaches the plugin the board revision (reg 99), which
        # decides has_valves -- do it before any air command so the gripper
        # check is answered by the hardware, not by an assumption.
        first = _read_state(transport, sensor, args.timeout)
        print("KJS-03 id {}: model {:#06x}  fw {}  IMU +/-{:.0f} dps / +/-{:.0f} g"
              "  {}".format(
                  device_id, identity["model_number"], identity["firmware_version"],
                  identity["gyro_range_dps"], identity["accel_range_g"],
                  "AIRRELAY (has gripper)" if sensor.has_valves
                  else "KJS-03 (no gripper)"))

        if args.command == "read":
            _monitor(transport, sensor, args.seconds, args.timeout)
            return 0

        # An air op.
        try:
            ok = _apply(transport, sensor, args.command, args.timeout)
        except CapabilityError as error:
            print(str(error), file=sys.stderr)
            return 1
        value = AIR_OPS[args.command]
        print(">>> {}  (GPIO_STATE=0x{:02X}: pump={}, relay={})  ack={}".format(
            args.command, value,
            "ON" if value & 0x02 else "off",
            "ON" if value & 0x01 else "off",
            "ok" if ok else "NO ACK"))
        if args.watch > 0:
            _monitor(transport, sensor, args.watch, args.timeout)
        return 0
    finally:
        transport.close()


if __name__ == "__main__":
    sys.exit(main())
