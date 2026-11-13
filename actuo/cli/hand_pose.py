#!/usr/bin/env python3
"""Drive the Amazing Hand between two poses: OPEN and GRASP.

The simplest useful demo -- open the hand (ready to take a bottle) and grasp
(close onto it). Poses are the upstream demo's proven values (src/AmazingHand
PythonExample: OpenHand a=-35 b=+35, CloseHand a=+90 b=-90), which in this
config's coordinates are commanded directly in degrees.

Both servos of every finger move together in ONE sync-write, which is exactly
why it does not jam: a finger's two servos are opposed, and moving one while the
other is limp binds the linkage (that is what defeated per-servo limit
detection). Coordinated motion is the finger's real range.

    actuo-hand-pose --config <hand>.yaml --pose cycle
    # --pose open | grasp | cycle   (cycle = open, grasp, open)

Every command is clamped to each servo's position_limit_deg, so a pose outside
the taught envelope is refused, not driven.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time


from actuo.core.contracts import ActuatorCommand, OperatingMode  # noqa: E402
from actuo.drivers.actuators import feetech_scs as f  # noqa: E402
from actuo.core.contracts import Frame  # noqa: E402
from actuo.drivers.config import build_bus  # noqa: E402

# Upstream demo values, in this config's degrees (MiddlePos folded into the
# calibration, so MY position deg == the demo's Angle). 'a'/'b' = the two servos
# of each finger, keyed off the joint-name suffix.
POSES = {
    "open":  {"a": -35.0, "b": 35.0},
    "grasp": {"a": 90.0, "b": -90.0},
}


def pose_commands(names, pose):
    """One POSITION command per joint for the named pose (radians)."""
    commands = []
    for name in names:
        side = name.rsplit("_", 1)[-1]      # 'a' or 'b'
        if side not in POSES[pose]:
            raise ValueError("joint {!r} does not end in _a/_b; cannot map to a "
                             "finger side".format(name))
        commands.append(ActuatorCommand(mode=OperatingMode.POSITION,
                                        position=math.radians(POSES[pose][side])))
    return commands


def _set_speed(bus, speed):
    """Write goal_speed to every servo, so the move is gentle, not a snap."""
    frames = [Frame(payload=f.build_write(a.id, f.REG_GOAL_SPEED, f.u16_be(speed)),
                    address=a.id) for a in bus.actuators]
    bus.transport.write(frames)


def _report(bus, names, label):
    bus.read_batch(0.1)
    bus.read_batch(0.1)
    print("[hand-pose] {}:".format(label))
    for name, a in zip(names, bus.actuators):
        st = bus.last_state(a.id)
        deg = math.degrees(st.position) if st else float("nan")
        load = (st.aux or {}).get("load", 0.0) if st else 0.0
        print("    {:<8s} id={:<3d} {:7.2f} deg  load {:4.0f}".format(
            name, a.id, deg, load))


def main(argv=None):
    p = argparse.ArgumentParser(description="open/grasp the Amazing Hand")
    p.add_argument("--config", required=True)
    p.add_argument("--pose", choices=("open", "grasp", "cycle"), default="cycle")
    p.add_argument("--speed", type=int, default=400,
                   help="servo goal_speed raw value; lower is gentler (default "
                        "400). 0 lets the servo use its own default (fast).")
    p.add_argument("--hold", type=float, default=1.6,
                   help="seconds to hold each pose in a cycle")
    p.add_argument("--release", action="store_true",
                   help="torque off at the end (hand goes limp)")
    p.add_argument("--channel", default=None)
    p.add_argument("--bitrate", type=int, default=None)
    p.add_argument("--yes", action="store_true", help="skip the confirmation")
    args = p.parse_args(argv)

    overrides = {}
    if args.channel is not None:
        overrides["channel"] = args.channel
    if args.bitrate is not None:
        overrides["bitrate"] = args.bitrate

    bus, names = build_bus(args.config, transport_overrides=overrides or None)
    sequence = ["open", "grasp", "open"] if args.pose == "cycle" else [args.pose]

    print("[hand-pose] will move the hand: {}".format(" -> ".join(sequence)))
    print("[hand-pose] THIS MOVES THE FINGERS. grasp closes them onto whatever "
          "is in the hand. Watch it; keep a hand on the power.")
    if not args.yes:
        try:
            if input("[hand-pose] type MOVE to proceed: ").strip() != "MOVE":
                print("[hand-pose] aborted; nothing was sent")
                return 1
        except EOFError:
            print("[hand-pose] no tty; pass --yes if you mean it", file=sys.stderr)
            return 1

    bus.transport.open()
    try:
        _report(bus, names, "start (as found)")
        if args.speed > 0:
            _set_speed(bus, args.speed)
        bus.enable_all()
        # enable_all writes 8 torque_enable regs, each of which the servo ACKs.
        # Those ACKs collide with the next sync-write on the half-duplex bus and
        # eat the first pose command, so drain them (a read resets the input and
        # re-syncs) before commanding a pose.
        time.sleep(0.1)
        bus.read_batch(0.1)
        for pose in sequence:
            print("\n[hand-pose] -> {}".format(pose.upper()), flush=True)
            bus.write_batch(pose_commands(names, pose))
            time.sleep(args.hold)
            _report(bus, names, "reached {}".format(pose))
    except ValueError as err:
        # A pose outside a servo's envelope is refused here, not driven.
        print("\n[hand-pose] REFUSED: {}".format(err), file=sys.stderr)
        bus.release_all()
        bus.transport.close()
        return 1
    except KeyboardInterrupt:
        print("\n[hand-pose] interrupted")
    finally:
        if args.release:
            bus.release_all()
            print("[hand-pose] released; the hand is limp")
        bus.transport.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
