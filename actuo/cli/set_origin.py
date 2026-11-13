#!/usr/bin/env python3
"""Zero one CAN actuator's encoder at the pose it is standing in, right now.

Why this exists
---------------
A CubeMars AK80-6 arm joint was seen to lose its encoder origin across power
cycles. Correcting it by hand-editing ``zero_offset_deg`` in the robot config is
not a fix: a hand-written offset tracks whichever power-up it was measured on, so
it is *unreliable by construction*. Observed errors: 106.1 deg (2026-07-28),
11.2 deg (2026-07-29, the origin having returned), 172.3 deg (2026-07-30).

``comm_can_set_origin`` with the permanent flag writes the origin to the driver's
own non-volatile memory, which is the only place it can survive a power cycle.
This tool is the deliberate, auditable way to do that.

The one rule this tool is built around
--------------------------------------
**A write is confirmed by its EFFECT, never by its success.** This is not
caution for its own sake -- it is the lesson from a DYNAMIXEL servo whose
Homing Offset register accepted a write, read back exactly what was written,
returned error byte 0x00, and had *zero effect*. Every check available said it
had worked, and it was completely broken.

So this tool refuses to claim success until it has re-read the actuator's
reported position and seen it change. CAN servo mode makes that easy: the driver
uploads its position unsolicited, so the proof arrives on its own.

Safety
------
* Torque is never enabled. ``SET_ORIGIN_HERE`` is an encoder operation and no
  motion frame is emitted, so the joint cannot move because of this tool.
* Nothing is written without ``--yes``. Without it the tool reads, prints what
  it would do, and exits.
* The origin becomes wherever the joint is standing **at this instant**, so the
  pose must already be the datum. The tool prints every joint's reading first
  precisely so that can be judged before committing: on a correctly-posed arm
  the OTHER joints read near zero, and one that does not is the tell that the
  arm is not where you think it is.

Usage
-----
    # read-only: what is where, and what would happen
    actuo-set-origin --config <robot>.yaml --joint joint3

    # temporary origin (lost on power cycle) -- useful to rehearse
    actuo-set-origin --config <robot>.yaml --joint joint3 --yes

    # the real thing: writes the driver's non-volatile memory
    actuo-set-origin --config <robot>.yaml --joint joint3 --permanent --yes

After a permanent write, set that joint's ``zero_offset_deg`` to 0.0 in the
config: the offset existed only to cancel a raw origin that is now actually
zero. ``position_limit_deg`` must NOT be shifted -- the physical datum has not
moved, only the encoder's opinion of it.

Reference [1]: CubeMars AK series CAN protocol manual, section 5.2
(``comm_can_set_origin``); see actuo/drivers/references.md.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time


from actuo.drivers.config import build_bus  # noqa: E402


def _read_all(bus, names, seconds):
    """Drain status frames for ``seconds`` and return {name: state}."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        bus.poll(timeout=0.05)
    return {
        name: bus.last_state(actuator.id)
        for name, actuator in zip(names, bus.actuators)
    }


def _print_states(states, label):
    print("[origin] {}".format(label))
    for name, state in states.items():
        if state is None:
            print("[origin]   {:<10s} NO DATA".format(name))
            continue
        print("[origin]   {:<10s} {:>8.2f} deg   T={:.0f}C  err={}".format(
            name, math.degrees(state.position), state.temperature or 0.0,
            state.error))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Zero one CAN actuator's encoder at its present position")
    parser.add_argument("--config", required=True)
    parser.add_argument("--joint", required=True,
                        help="joint name from the config")
    parser.add_argument("--permanent", action="store_true",
                        help="write the origin to the driver's NON-VOLATILE "
                             "memory so it survives a power cycle. Without "
                             "this the origin is lost on the next power-off.")
    parser.add_argument("--yes", action="store_true",
                        help="actually write. Without it this tool only reads.")
    parser.add_argument("--settle", type=float, default=1.5,
                        help="seconds of status frames to collect before and "
                             "after the write (default 1.5)")
    parser.add_argument("--tolerance-deg", type=float, default=2.0,
                        help="how close to 0 the joint must read afterwards "
                             "for the write to count as confirmed (default 2)")
    parser.add_argument("--interface", default=None)
    parser.add_argument("--channel", default=None)
    parser.add_argument("--bitrate", type=int, default=None)
    args = parser.parse_args(argv)

    overrides = {}
    if args.interface:
        overrides["interface"] = args.interface
    if args.channel:
        overrides["channel"] = args.channel
    if args.bitrate:
        overrides["bitrate"] = args.bitrate

    bus, names = build_bus(args.config,
                           transport_overrides=overrides or None)
    if args.joint not in names:
        raise SystemExit(
            "[origin] no such joint: {!r}. This bus has: {}".format(
                args.joint, ", ".join(names)))
    index = names.index(args.joint)
    actuator = bus.actuators[index]

    if not hasattr(actuator, "encode_set_origin"):
        raise SystemExit(
            "[origin] {} is a {} and does not support setting an origin. This "
            "is a capability, not an oversight -- it is not faked.".format(
                args.joint, type(actuator).__name__))

    bus.transport.open()
    try:
        before = _read_all(bus, names, args.settle)
        _print_states(before, "before (every joint, so the POSE can be judged)")

        state = before[args.joint]
        if state is None:
            raise SystemExit(
                "[origin] {} (id {}) never reported. Refusing to write an "
                "origin to a motor whose position is unknown.".format(
                    args.joint, actuator.id))

        current_deg = math.degrees(state.position)
        print("")
        print("[origin] joint      : {} (id {})".format(args.joint, actuator.id))
        print("[origin] reads now  : {:.2f} deg".format(current_deg))
        print("[origin] after write: 0.00 deg (this pose becomes the origin)")
        print("[origin] storage    : {}".format(
            "NON-VOLATILE -- survives power cycles"
            if args.permanent else "temporary -- lost on the next power-off"))
        print("[origin] torque     : never enabled by this tool; no motion "
              "frame is sent")

        if not args.yes:
            print("")
            print("[origin] read-only: nothing written. Re-run with --yes{} "
                  "to write.".format(" --permanent" if args.permanent else ""))
            return 0

        print("")
        print("[origin] writing...")
        bus.transport.write([actuator.encode_set_origin(
            permanent=args.permanent)])

        # THE POINT OF THIS TOOL. A write that "succeeded" proves nothing: on
        # a Dynamixel arm a Homing Offset write was accepted, read back
        # correctly, returned error 0x00 and had no effect whatsoever. Only the
        # reported position can confirm it, so that is what is checked.
        after = _read_all(bus, names, args.settle)
        _print_states(after, "after")

        confirmed = after[args.joint]
        if confirmed is None:
            raise SystemExit(
                "[origin] FAILED: {} stopped reporting after the write. The "
                "origin may or may not have been set -- power-cycle and read "
                "it before trusting anything.".format(args.joint))

        new_deg = math.degrees(confirmed.position)
        print("")
        if abs(new_deg) > args.tolerance_deg:
            raise SystemExit(
                "[origin] FAILED, and this is exactly the failure mode this "
                "check exists for: the frame was sent with no error, but {} "
                "still reads {:.2f} deg instead of ~0. The write had NO EFFECT. "
                "Do not edit zero_offset_deg -- the origin was not moved.".format(
                    args.joint, new_deg))

        print("[origin] CONFIRMED: {} moved {:.2f} deg -> {:.2f} deg by its own "
              "report.".format(args.joint, current_deg, new_deg))
        if args.permanent:
            print("[origin] The origin is in the driver's non-volatile memory.")
            print("[origin] NOW EDIT THE CONFIG: set this joint's "
                  "zero_offset_deg to 0.0.")
            print("[origin]   Its only job was to cancel a raw origin that is "
                  "now really zero.")
            print("[origin] Do NOT shift position_limit_deg: the physical "
                  "datum has not moved,")
            print("[origin]   only the encoder's opinion of it, so the "
                  "envelope still fences the")
            print("[origin]   same part of the mechanism.")
            print("[origin] VERIFY BY POWER-CYCLING and reading it again -- "
                  "that is the only")
            print("[origin]   test of 'permanent' that means anything.")
        else:
            print("[origin] This origin is TEMPORARY and will be lost on the "
                  "next power-off.")
        return 0
    finally:
        bus.transport.close()
        print("[origin] bus closed")


if __name__ == "__main__":
    raise SystemExit(main())
