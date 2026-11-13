#!/usr/bin/env python3
"""Read the Feetech serial hand from its config. Sends only READ requests.

The serial counterpart of ``actuo-can-probe``: it builds the bus from a robot
config, then polls each servo's present state and prints it. It commands no
motion -- a SerialBus read is a READ instruction and its reply, nothing that
moves a finger.

    actuo-hand-probe --config <hand>.yaml
    # e.g. a config for one Amazing Hand (Feetech SCS0009 servos)

Use it to confirm the wiring, the servo ids, and that every servo answers before
trusting anything that writes.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time


from actuo.drivers.config import build_bus  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="read a Feetech serial hand (sends only READ requests)")
    parser.add_argument("--config", required=True, help="robot bus YAML")
    parser.add_argument("--seconds", type=float, default=3.0,
                        help="how long to read (default 3)")
    parser.add_argument("--rate", type=float, default=5.0,
                        help="reads per second (default 5)")
    parser.add_argument("--channel", default=None,
                        help="override the serial device path")
    parser.add_argument("--bitrate", type=int, default=None)
    args = parser.parse_args(argv)

    overrides = {}
    if args.channel is not None:
        overrides["channel"] = args.channel
    if args.bitrate is not None:
        overrides["bitrate"] = args.bitrate

    bus, names = build_bus(args.config, transport_overrides=overrides or None)
    print("[hand-probe] transport: {}".format(type(bus.transport).__name__))
    for name, actuator in zip(names, bus.actuators):
        print("[hand-probe]   {:<8s} id={}".format(name, actuator.id))
    print("[hand-probe] reading for {:.0f}s -- sending only READ requests".format(
        args.seconds))

    bus.transport.open()
    deadline = time.monotonic() + args.seconds
    period = 1.0 / args.rate
    healthy = False
    try:
        while time.monotonic() < deadline:
            states = bus.read_batch()
            print("--", flush=True)
            all_ok = True
            for name, actuator, state in zip(names, bus.actuators, states):
                if state is None:
                    print("{:<8s} id={:<3d} NO REPLY".format(name, actuator.id))
                    all_ok = False
                    continue
                err = "0x{:02x}".format(state.error) if state.error else "none"
                print("{:<8s} id={:<3d} pos={:8.2f} deg  vel={:6.2f} rad/s  "
                      "load={:4.0f}  T={:3.0f}C  err={}".format(
                          name, actuator.id, math.degrees(state.position),
                          state.velocity or 0.0,
                          (state.aux or {}).get("load", 0.0),
                          state.temperature or 0.0, err))
            healthy = healthy or all_ok
            time.sleep(period)
    except KeyboardInterrupt:
        print("\n[hand-probe] interrupted")
    finally:
        bus.transport.close()

    if bus.missed:
        print("[hand-probe] {} missed reply(ies) over the run".format(bus.missed),
              file=sys.stderr)
    if healthy:
        print("[hand-probe] OK: all {} servos reported at least once, decoded "
              "cleanly, nothing was commanded".format(len(names)))
        return 0
    print("[hand-probe] some servos never answered -- check ids/wiring/power",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
