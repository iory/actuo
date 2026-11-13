#!/usr/bin/env python3
"""Find one joint's REAL top speed, by letting the driver do the move itself.

    actuo-maxspeed --config <robot>.yaml --shm-name <name> --joint joint2 --to 60 --speed 60      # start here
    actuo-maxspeed --config <robot>.yaml --shm-name <name> --joint joint2 --to -30 --speed 250

Why this exists, and what it gives up
-------------------------------------
Through ros2_control a CubeMars AK arm tops out around 150 deg/s, and the limit
is neither the motors nor ros2_control: it is a **position loop inside a position
loop**. JTC emits position setpoints and the CubeMars driver runs its own
position controller to chase them, and that cascade lags by ~130 ms. The
tracking error is ``speed x tau``, so it hits JointTrajectoryController's
0.35 rad state tolerance at about 150 deg/s and the goal aborts. An AK70-9 at
24 V is good for ~960 deg/s, so that is under a fifth of the hardware.

``SET_POS_SPD`` (servo packet 6) removes the cascade instead of fighting it: one
frame says "go to X, at up to V", and the driver runs the whole trapezoid on its
own. There is no streamed reference, so there is no tracking error to exceed, and
the driver moves at whatever it can.

What that costs, stated plainly, because it is not a free upgrade:

* **No trajectory control.** You get a point-to-point move, not a path. Nothing
  interpolates, nothing coordinates two joints, nothing monitors following.
* **No ros2_control at all** -- this talks to the bus directly, so the servicer and
  ros2_control must be DOWN. Its guards (the jump guard, the supervisor, the
  overspeed check) are not in the loop either.
* **The only way to stop it is to cut torque.** The driver owns the move once the
  frame is sent.
* The taught envelope IS still enforced, by the actuator model itself
  (``CubeMarsActuator._check_position_limit``), because that lives below all of the
  above. That is the one guard you keep.

The real fix for full performance *inside* ros2_control is a **velocity** command
interface (the driver's ``SET_RPM``), which puts only one position loop in the path
and keeps every controller-side guard. That is a project-sized change; this tool is
for measuring what the hardware can actually do.

A CONSTANT target, streamed
---------------------------
``SET_POS_SPD`` is a trapezoidal *move* and every frame restarts the driver's
planner, so the first version of this tool sent the target exactly once. That does
not work: the driver disables its output when commands stop arriving, so it moved
**1.4 deg at 2 A and gave up**. One frame is not a move, it is a twitch.

The distinction that matters is not "once vs streamed", it is whether the target
**changes**. Re-planning every cycle toward a *fixed, distant* target still
accelerates toward it, which is why ``actuo-jog-joint`` reaches its target exactly.
Re-planning toward a target that advances 0.14 deg per cycle is what leaves the arm
standing still, and that is why a robot config driven by ros2_control should set
``use_pos_spd: false``. So this streams ONE unchanging target until the joint
arrives.

**The joint goes limp when this tool exits.** The driver disables its output a
second or two after commands stop, so the arm falls to wherever gravity takes it.
The driver does not keep holding -- measured, an AK70-9 joint finished a move at
-38.70 deg and sat at -1.40 deg with 0.00 A eight seconds later. Two consecutive
runs disagreed about the start position by 98 deg for exactly that reason. If
you need a pose held, something must keep commanding it: the servicer plus
ros2_control, or a joint-move tool that keeps streaming the target.

Reachable speed is set by ACCELERATION, not by the cap
------------------------------------------------------
``--accel`` defaults to the same number as ``--speed`` (the vendor's own examples
do that) and it is far too low to reach a high ``--speed``. A trapezoid that cannot
finish ramping is a triangle, and then the cap is irrelevant: the top speed is
``sqrt(accel * travel)``. Measured: ``--speed 400`` over 20.8 deg reached 94 deg/s,
and ``sqrt(400 * 20.8) = 91``. The tool now says so before it sends.

Measured on an AK70-9 arm joint (143 deg envelope), 2026-07-30:

=========  =========  =========  ==========  =======
--speed    --accel    travel     PEAK        peak I
=========  =========  =========  ==========  =======
60         60          41 deg     51 deg/s   3.05 A
300        800         99 deg    275 deg/s   5.06 A
580        2600       102 deg    **416**     5.25 A
=========  =========  =========  ==========  =======

**416 deg/s is 2.75x what the same joint manages through ros2_control** (151), at
5.25 A against a rated 6.25 -- so torque is not the limit either. Sanity check on
the peak: a triangular profile should peak at about twice its average, and 416/229
= 1.81.
"""

from __future__ import annotations

import argparse
import math
import sys
import time


from actuo.core.contracts import ActuatorCommand, OperatingMode      # noqa: E402
from actuo.drivers.actuators.cubemars import ERROR_NAMES             # noqa: E402
from actuo.drivers.config import build_bus                           # noqa: E402
from actuo.drivers.shm import lock_holder, lock_path                 # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Point-to-point move at a commanded speed, driver-side")
    parser.add_argument("--config", required=True)
    parser.add_argument("--joint", required=True)
    parser.add_argument("--to", type=float, required=True,
                        help="target angle, DEGREES (taught envelope enforced)")
    parser.add_argument("--speed", type=float, required=True,
                        help="speed cap for the driver's trapezoid, deg/s at the "
                             "OUTPUT shaft")
    parser.add_argument("--accel", type=float, default=None,
                        help="deg/s^2 cap (default: same number as --speed, "
                             "which the vendor's own examples use)")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--rate", type=float, default=50.0,
                        help="Hz to re-send the SAME target. Not a control rate: "
                             "the driver plans the move, this only keeps its "
                             "output enabled (20 Hz is enough in practice).")
    parser.add_argument("--tolerance-deg", type=float, default=2.0)
    parser.add_argument("--release", action="store_true",
                        help="send an explicit zero-torque frame when the move "
                             "ends. Largely cosmetic: the driver disables its "
                             "output within a second or two of commands stopping "
                             "anyway, so the joint goes limp either way. Use it "
                             "to make that intent explicit in a script.")
    parser.add_argument("--shm-name", required=True,
                        help="shm name the bus servicer for this bus runs "
                             "under; its lock is checked so this tool refuses "
                             "to run beside a live servicer")
    parser.add_argument("--yes", action="store_true",
                        help="skip the confirmation prompt")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--interface", default=None)
    parser.add_argument("--channel", default=None)
    parser.add_argument("--bitrate", type=int, default=None)
    args = parser.parse_args(argv)

    # Refuse to share the bus. A running servicer is commanding these motors every
    # cycle; adding a second commander means two planners fighting over one joint,
    # and whichever frame lands last wins. Check its lock (shm.lock_path of the
    # servicer's shm name) rather than a process list -- the lock is what
    # actually decides, and the kernel drops it when the holder dies.
    who = lock_holder(args.shm_name)
    if who is not None:
        who = who or "unknown pid"
        raise SystemExit(
            "[maxspeed] a bus servicer is RUNNING on shm {!r} (pid {}, lock {}). "
            "This tool drives the motors directly, so two things would be "
            "commanding them at once. Stop the servicer (pid {}) and its "
            "ros2_control stack first.".format(
                args.shm_name, who, lock_path(args.shm_name), who))

    overrides = {}
    for key in ("interface", "channel", "bitrate"):
        if getattr(args, key):
            overrides[key] = getattr(args, key)
    bus, names = build_bus(args.config, transport_overrides=overrides or None)
    if args.joint not in names:
        raise SystemExit("[maxspeed] no such joint: {!r}. This bus has: {}".format(
            args.joint, ", ".join(names)))
    actuator = bus.actuators[names.index(args.joint)]
    calibration = actuator.calibration

    if not hasattr(actuator, "use_pos_spd"):
        raise SystemExit(
            "[maxspeed] {} is a {}, which has no SET_POS_SPD equivalent.".format(
                args.joint, type(actuator).__name__))

    speed_rad = math.radians(args.speed)
    accel_dps2 = args.speed if args.accel is None else args.accel
    erpm = calibration.rad_s_to_erpm(speed_rad)
    erpm_accel = calibration.rad_s_to_erpm(math.radians(accel_dps2))
    if not 1 <= int(round(erpm / 10.0)) <= 32767:
        raise SystemExit(
            "[maxspeed] {:.0f} deg/s is {:.0f} ERPM, which does not fit the "
            "wire's int16 of ERPM/10.".format(args.speed, erpm))

    # Override in memory only. A config's max_velocity_erpm is typically fitted
    # to demonstrations (e.g. a p95 of recorded moves) -- a statement about the
    # task, not the hardware -- and this
    # tool exists to go past it. Writing it to the file would silently raise the
    # fence for every other consumer.
    # dataclasses.replace, not assignment: CubeMarsLimits is FROZEN, which is the
    # right choice for a calibration value and means an override has to be
    # deliberate rather than a stray attribute write.
    import dataclasses
    actuator.use_pos_spd = True
    actuator.limits = dataclasses.replace(
        actuator.limits,
        max_velocity_erpm=erpm,
        max_acceleration_erpm_s=erpm_accel)

    bus.transport.open()
    try:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            bus.poll(timeout=0.05)
        state = bus.last_state(actuator.id)
        if state is None or state.position is None:
            raise SystemExit(
                "[maxspeed] {} (id {}) never reported; refusing to command a "
                "joint whose position is unknown.".format(
                    args.joint, actuator.id))
        here = math.degrees(state.position)
        low, high = [math.degrees(x) for x in calibration.position_limit]

        print("[maxspeed] joint     : {} (id {})".format(args.joint, actuator.id))
        print("[maxspeed] envelope  : {:+.2f} .. {:+.2f} deg".format(low, high))
        print("[maxspeed] now       : {:+.2f} deg, T={:.0f}C, err={}".format(
            here, state.temperature or 0.0,
            ERROR_NAMES.get(state.error or 0, state.error)))
        print("[maxspeed] target    : {:+.2f} deg  ({:.2f} deg away)".format(
            args.to, abs(args.to - here)))
        print("[maxspeed] speed cap : {:.0f} deg/s = {:.0f} ERPM  (accel {:.0f} "
              "deg/s^2)".format(args.speed, erpm, accel_dps2))
        print("[maxspeed] mode      : SET_POS_SPD. The driver runs the trapezoid "
              "itself; nothing here")
        print("[maxspeed]             can stop it once sent except cutting "
              "torque.")

        # The sqrt(accel * distance) ceiling, for the third time in this project.
        # A trapezoid that cannot finish ramping is a triangle, and then the speed
        # CAP is irrelevant -- only accel and travel matter. Measured: --speed 400
        # over 20.8 deg at the default accel reached 94 deg/s, and
        # sqrt(400 * 20.8) = 91. The default --accel equals --speed, which is the
        # vendor's own example and is far too low to reach a high --speed.
        travel = abs(args.to - here)
        reachable = math.sqrt(accel_dps2 * travel) if travel > 0 else 0.0
        if reachable < args.speed * 0.95:
            print("[maxspeed] ** the {:.0f} deg/s cap is NOT reachable here **"
                  .format(args.speed))
            print("[maxspeed]   over {:.1f} deg at {:.0f} deg/s^2 the profile is "
                  "a TRIANGLE and tops out at".format(travel, accel_dps2))
            print("[maxspeed]   sqrt(accel*travel) = {:.0f} deg/s. To reach "
                  "{:.0f}, use --accel {:.0f} or move further."
                  .format(reachable, args.speed,
                          args.speed ** 2 / travel if travel > 0 else 0.0))
        else:
            print("[maxspeed] reachable : {:.0f} deg/s cap is within "
                  "sqrt(accel*travel) = {:.0f}".format(args.speed, reachable))
        if not (low <= args.to <= high):
            raise SystemExit(
                "[maxspeed] REFUSED: {:+.2f} is outside the taught envelope "
                "[{:+.2f}, {:+.2f}]. Teach the range or pick a target inside "
                "it.".format(args.to, low, high))

        if args.dry_run:
            print("[maxspeed] dry run: no frame sent.")
            return 0
        if not args.yes:
            reply = input("[maxspeed] type MOVE to send: ")
            if reply.strip() != "MOVE":
                print("[maxspeed] not sent.")
                return 1

        command = ActuatorCommand(mode=OperatingMode.POSITION,
                                  position=math.radians(args.to))
        batch = [command if a is actuator else None for a in bus.actuators]
        sent = time.monotonic()
        next_send = sent
        period = 1.0 / args.rate

        peak = 0.0
        peak_current = 0.0
        peak_temp = state.temperature or 0.0
        arrived = None
        fault = 0
        while time.monotonic() - sent < args.timeout:
            now = time.monotonic()
            if now >= next_send:
                # The SAME target every time. See the module docstring: a fixed
                # distant target survives the planner restart, a moving one does
                # not, and stopping altogether makes the driver disable output.
                bus.write_batch(batch)
                next_send = now + period
            bus.poll(timeout=0.005)
            st = bus.last_state(actuator.id)
            if st is None:
                continue
            if st.velocity is not None:
                peak = max(peak, abs(st.velocity))
            if st.aux and st.aux.get("current_a") is not None:
                peak_current = max(peak_current, abs(st.aux["current_a"]))
            if st.temperature is not None:
                peak_temp = max(peak_temp, st.temperature)
            if st.error:
                fault = st.error
                break
            if abs(math.degrees(st.position) - args.to) <= args.tolerance_deg:
                arrived = time.monotonic() - sent
                break

        final = bus.last_state(actuator.id)
        moved = abs(math.degrees(final.position) - here)
        print("")
        print("[maxspeed] travelled : {:.2f} deg  (now {:+.2f})".format(
            moved, math.degrees(final.position)))
        elapsed = (arrived if arrived is not None
                   else time.monotonic() - sent)
        if peak > 0.0:
            print("[maxspeed] PEAK      : {:.1f} deg/s   ({:.0f}% of the {:.0f} "
                  "deg/s asked)".format(
                      math.degrees(peak), 100.0 * math.degrees(peak) / args.speed,
                      args.speed))
            # Cross-check. A peak far above the average over a short move is a
            # single noisy sample, not a speed the joint held.
            if elapsed > 0.05:
                print("[maxspeed] average   : {:.1f} deg/s over {:.2f} s  "
                      "(a peak far above this is one noisy sample, not a speed "
                      "the joint held)".format(moved / elapsed, elapsed))
        else:
            print("[maxspeed] PEAK      : no velocity reported (pole_pairs "
                  "unset?)")
        print("[maxspeed] peak I    : {:.2f} A     peak T: {:.0f}C".format(
            peak_current, peak_temp))
        if fault:
            print("[maxspeed] FAULT     : {} (code {})".format(
                ERROR_NAMES.get(fault, "unknown"), fault))
        elif arrived is not None:
            print("[maxspeed] arrived in {:.2f} s".format(arrived))
        else:
            print("[maxspeed] did NOT arrive within {:.1f} s -- it may still be "
                  "moving, or it stalled.".format(args.timeout))

        if args.release:
            bus.release_all()
        # Whether we release or not, the joint is about to go limp: the driver
        # disables its output when commands stop arriving, and this process is
        # about to stop sending. Measured: an AK70-9 joint was commanded to -40,
        # ended the move at -38.70, and 8 s after the tool exited sat at -1.40 deg with
        # 0.00 A -- it had fallen to its gravity rest. Two consecutive runs also
        # disagreed about the start position by 98 deg for exactly this reason.
        #
        # Saying "the driver is still holding this target" was therefore a false
        # reassurance, and on an arm with a payload that is the worst kind.
        print("[maxspeed] ** THE JOINT WILL GO LIMP within a second or two of "
              "this tool exiting **")
        print("[maxspeed]   The driver disables its output when commands stop, "
              "so the arm falls to")
        print("[maxspeed]   wherever gravity takes it. To hold a pose you need "
              "something that keeps")
        print("[maxspeed]   commanding: the servicer plus ros2_control, or "
              "a joint-move tool that keeps streaming the target.")
        return 0
    finally:
        bus.transport.close()
        print("[maxspeed] bus closed")


if __name__ == "__main__":
    raise SystemExit(main())
