#!/usr/bin/env python3
"""Check the configured pole_pairs against the hardware. Commands nothing, ever.

`pole_pairs` is a datasheet number, and a wrong one silently rescales how fast
the hardware moves -- exactly the class of error this stack refuses to guess at.
But it does not have to be taken on faith either: a CubeMars driver uploads BOTH
its speed and its position, so back-driving a joint by hand makes its own two
streams check each other.

    ERPM = output_deg/s * pole_pairs * gear_ratio * 60/360

so the slope of reported-erpm against the erpm predicted from the position
derivative is 1.0 exactly when the configured pole_pairs is right, and is the
ratio of wrong-to-right when it is not.

    actuo-verify-poles --config <robot>.yaml

Then move each joint by hand, briskly -- a slow push produces no speed to
measure. Verified on a 3-joint CubeMars arm, 2026-07-17: joint1 slope 1.005
(implied 14.1 vs a configured 14), joint2 slope 0.970 (implied 20.4 vs 21).

Worth knowing: the drivers report speed while completely unpowered. That is not
obvious -- a controller that only estimated speed from its own commands would
report zero here -- and it is what makes this check possible at all.
"""

from __future__ import annotations

import argparse
import collections
import math
import os
import sys
import time


from actuo.drivers.config import build_bus  # noqa: E402

# Below this the position derivative is mostly quantisation noise (the encoder
# reports 0.1 deg steps), so samples there would drag the slope toward nothing.
MIN_ERPM = 40.0


def main(argv=None):
    p = argparse.ArgumentParser(description="verify pole_pairs by back-driving")
    p.add_argument("--config", required=True, help="robot bus YAML")
    p.add_argument("--seconds", type=float, default=120.0)
    p.add_argument("--samples", type=int, default=40,
                   help="samples per joint before a verdict is offered")
    p.add_argument("--tolerance", type=float, default=0.15,
                   help="fractional slope error still called a match")
    p.add_argument("--interface", default=None)
    p.add_argument("--channel", default=None)
    p.add_argument("--bitrate", type=int, default=None)
    args = p.parse_args(argv)

    overrides = {}
    if args.interface:
        overrides["interface"] = args.interface
    if args.channel:
        try:
            overrides["channel"] = int(args.channel)
        except ValueError:
            overrides["channel"] = args.channel
    if args.bitrate:
        overrides["bitrate"] = args.bitrate

    bus, names = build_bus(args.config, transport_overrides=overrides or None)
    by_id = {a.id: (n, a) for n, a in zip(names, bus.actuators)}

    ungeared = [n for n, a in zip(names, bus.actuators)
                if a.calibration.pole_pairs <= 0]
    if ungeared:
        print("[verify-poles] no pole_pairs configured for: {} -- nothing to "
              "check there".format(", ".join(ungeared)), file=sys.stderr)

    bus.transport.open()
    print("[verify-poles] NOTHING will be commanded. Move each joint by hand, "
          "briskly.", flush=True)
    prev = {}
    acc = collections.defaultdict(lambda: [0.0, 0.0, 0])
    deadline = time.monotonic() + args.seconds
    try:
        while time.monotonic() < deadline:
            bus.poll(timeout=0.05)
            now = time.monotonic()
            for device_id, (name, actuator) in by_id.items():
                if actuator.calibration.pole_pairs <= 0:
                    continue
                state = bus.last_state(device_id)
                if state is None or state.aux is None:
                    continue
                erpm = state.aux.get("erpm")
                if erpm is None:
                    continue
                pos = state.position
                if device_id in prev:
                    p0, t0 = prev[device_id]
                    dt = now - t0
                    if dt < 0.06:
                        continue
                    rad_s = (pos - p0) / dt
                    predicted = actuator.calibration.rad_s_to_erpm(rad_s)
                    if abs(predicted) > MIN_ERPM:
                        a = acc[device_id]
                        a[0] += erpm * predicted     # least-squares slope,
                        a[1] += predicted * predicted  # through the origin
                        a[2] += 1
                prev[device_id] = (pos, now)
            if acc and all(acc[d][2] >= args.samples for d in by_id
                           if by_id[d][1].calibration.pole_pairs > 0):
                break
    except KeyboardInterrupt:
        pass
    finally:
        bus.transport.close()

    print("\n{:<10s} {:>7s} {:>8s} {:>9s} {:>8s}  {}".format(
        "joint", "samples", "slope", "implied", "config", "verdict"))
    bad = 0
    for device_id, (name, actuator) in sorted(by_id.items()):
        configured = actuator.calibration.pole_pairs
        if configured <= 0:
            continue
        num, den, n = acc[device_id]
        if n < 10 or den <= 0:
            print("{:<10s} {:>7d} {:>8s} {:>9s} {:>8d}  not moved enough to "
                  "judge".format(name, n, "-", "-", configured))
            bad += 1
            continue
        slope = num / den
        implied = configured * slope
        ok = abs(slope - 1.0) <= args.tolerance
        if not ok:
            bad += 1
        print("{:<10s} {:>7d} {:>8.3f} {:>9.1f} {:>8d}  {}".format(
            name, n, slope, implied, configured,
            "ok" if ok else "*** MISMATCH: the hardware says ~{:.0f} ***".format(
                implied)))

    if bad:
        print("\n[verify-poles] {} joint(s) unconfirmed. An unverified "
              "pole_pairs only affects rad/s and N*m conversions -- POSITION "
              "mode never uses it -- but VELOCITY and CURRENT are scaled by "
              "it.".format(bad), file=sys.stderr)
        return 1
    print("\n[verify-poles] every configured pole_pairs matches the hardware.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
