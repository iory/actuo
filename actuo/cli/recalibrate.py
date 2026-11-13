#!/usr/bin/env python3
"""Re-zero a joint at its present pose and teach its limits -- in ONE command.

    actuo-recalibrate --joint joint3 --config <robot>.yaml --shm-name <name>

What it does, in order:

1. **Re-zero.** Whatever pose the joint is standing in right now becomes 0. The
   new ``zero_offset_deg`` is written to the config.
2. **Teach.** You hand-move the joint to both ends; the extremes reached are
   recorded. Listen-only -- nothing is ever commanded and torque is never
   enabled, so the joint cannot move because of this tool.
3. **Write both files.** ``position_limit_deg`` into the YAML *and* the URDF's
   ``<limit>`` plus its ros2_control ``command_interface`` bounds.
4. **Invalidate the remembered gear-window reference**, which was recorded in
   the OLD frame and is now a lie.

Why one command
---------------
These four steps are not independent, and doing three of them is worse than
doing none:

* Re-zeroing without re-teaching leaves an envelope fencing a *different part of
  the mechanism* than it was taught on. On a CubeMars AK80-6 arm joint that once
  left the upper bound 119.80 deg above the rest pose while the mechanism ends
  9.4 deg above it -- a fence permitting a 110 deg drive past a hard stop.
* Writing the YAML without the URDF leaves ``skrobot`` clamping to stale limits
  **silently**, so the model reports a plausible wrong angle.
* Leaving the remembered window reference behind points the next start-up at a
  pose in a frame that no longer exists.

Each of those has already happened once. Hence one command, and hence it refuses
half-completion: nothing is written until the teach succeeds.

Safety
------
The whole procedure is listen-only. It emits no motion frame and never enables
torque -- the joint moves only because a hand moves it.

The taught envelope is inside a safe range **by construction**: a hand put the
joint at every point of it. The inward margin exists so a command never targets
a hard stop, and is deliberately NOT applied where doing so would exclude the
pose the joint rests at under gravity -- an envelope that cannot contain the
rest pose makes torque impossible to enable after any release.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import time


from actuo.drivers import config_edit, window_state          # noqa: E402
from actuo.drivers.config import build_bus                   # noqa: E402
from actuo.cli.teach_limits import teach_one                # noqa: E402


def _read_pose(bus, names, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        bus.poll(timeout=0.05)
    return {n: bus.last_state(a.id) for n, a in zip(names, bus.actuators)}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Re-zero one joint here, teach its limits, write both files")
    parser.add_argument("--config", required=True, help="robot bus YAML")
    parser.add_argument("--joint", required=True)
    parser.add_argument("--urdf", default="",
                        help="xacro/URDF whose <limit> and ros2_control bounds "
                             "must move with the YAML. Strongly recommended: "
                             "skrobot clamps to it silently.")
    # window_state keys its reference files by the servicer's shm name, so it
    # must be the one that robot's servicer runs under; there is no default.
    parser.add_argument("--shm-name", required=True,
                        help="which robot's remembered gear-window reference to "
                             "invalidate (it is recorded in the old frame); the "
                             "shm name that robot's servicer runs under")
    parser.add_argument("--no-rezero", action="store_true",
                        help="teach the limits ONLY, leaving zero_offset_deg "
                             "alone. Use when the datum is right and the "
                             "envelope is merely too tight -- which is the "
                             "common case, because where a limp arm comes to "
                             "rest changes with what it was holding and how it "
                             "was released, while the datum does not. Re-zeroing "
                             "to fix an envelope would invalidate every angle "
                             "recorded against the old datum for no reason.")
    parser.add_argument("--shift-limits", action="store_true",
                        help="re-zero at the present pose and SHIFT the existing "
                             "envelope by the same amount, instead of teaching a "
                             "new one by hand. The fence then still describes the "
                             "same PHYSICAL range -- which is the only thing that "
                             "makes a re-zero without a re-teach safe. Use it when "
                             "the mechanism has NOT changed and only the datum "
                             "should move; teach properly (the default) whenever "
                             "the hardware itself was touched, because a shifted "
                             "envelope inherits every error the old one had.")
    parser.add_argument("--window-reference", type=float, default=None,
                        metavar="DEG",
                        help="roughly where this joint is, in the DATUM frame, "
                             "used only to establish its gear window before "
                             "teaching (--no-rezero). Needs to be right to a "
                             "quarter of a window, not exactly. Overrides the "
                             "remembered position and the config's "
                             "startup_reference_deg.")
    parser.add_argument("--margin-deg", type=float, default=2.0,
                        help="shrink each taught extreme inward by this much, "
                             "so a command never targets a hard stop")
    parser.add_argument("--keep-rest-inside", action="store_true", default=True,
                        help="never apply the margin on a side that would "
                             "exclude the pose the joint rests at (default on)")
    parser.add_argument("--min-travel-deg", type=float, default=5.0,
                        help="refuse to write a suspiciously small range")
    parser.add_argument("--settle", type=float, default=1.5)
    parser.add_argument("--dry-run", action="store_true",
                        help="do everything except write the files")
    parser.add_argument("--interface", default=None)
    parser.add_argument("--channel", default=None)
    parser.add_argument("--bitrate", type=int, default=None)
    args = parser.parse_args(argv)
    if args.shift_limits and args.no_rezero:
        raise SystemExit(
            "[recal] --shift-limits and --no-rezero contradict each other: the "
            "first shifts the envelope BECAUSE the datum moved, the second does "
            "not move the datum. Pick one.")

    overrides = {}
    for key in ("interface", "channel", "bitrate"):
        if getattr(args, key):
            overrides[key] = getattr(args, key)

    bus, names = build_bus(args.config, transport_overrides=overrides or None)
    if args.joint not in names:
        raise SystemExit("[recal] no such joint: {!r}. This bus has: {}".format(
            args.joint, ", ".join(names)))
    index = names.index(args.joint)
    actuator = bus.actuators[index]
    calibration = actuator.calibration

    bus.transport.open()
    try:
        states = _read_pose(bus, names, args.settle)
        print("[recal] pose now (every joint, so you can judge it):")
        for name, state in states.items():
            print("[recal]   {:<10s} {:>9.2f} deg".format(
                name, math.degrees(state.position) if state else float("nan")))
        state = states[args.joint]
        if state is None or state.position is None:
            raise SystemExit(
                "[recal] {} never reported. Refusing to re-zero a joint whose "
                "position is unknown.".format(args.joint))

        # ESTABLISH THE GEAR WINDOW FIRST, but only when keeping the datum.
        #
        # This tool runs standalone, so its actuator starts at window_offset 0 --
        # i.e. in the RAW frame the driver powered up in, which for a CubeMars
        # AK80-6 can be a whole number of 60 deg windows away from the datum.
        # With --rezero that is harmless: the re-zero absorbs it and
        # deliberately sets the window to 0. With --no-rezero the datum frame is
        # exactly what the taught numbers must be expressed in, so skipping this
        # writes an envelope a whole window out.
        #
        # It did, on this tool's first run: an AK80-6 joint was taught
        # [93.60, 215.10] when the truth was [-86.40, 35.10] -- 180 deg, three
        # windows, and every number individually plausible. Caught only by
        # cross-checking against a reading the running servicer had taken with
        # the window established.
        # Only for a joint that USES window recovery. A joint with no
        # startup_reference_deg has opted out of it (see config.py), which means
        # no window correction is ever applied to it -- its raw frame IS its
        # datum frame, and teaching in it is correct. Demanding a reference from
        # such a joint would refuse a perfectly ordinary re-teach for a problem
        # it does not have. An explicit --window-reference still opts in.
        uses_window = (args.window_reference is not None
                       or getattr(actuator, "startup_reference_rad", None)
                       is not None)
        # --shift-limits needs it too, and for a sharper reason than
        # --no-rezero does: the shift SUBTRACTS the present reading from
        # bounds written in the datum frame, so a reading carrying an
        # unresolved k*60 moves the fence by k*60 as well -- onto a part
        # of the mechanism nobody taught, with every number still looking
        # reasonable. A full re-teach does not care, because a hand
        # visits both ends in whatever frame the driver powered up in.
        if ((args.no_rezero or args.shift_limits) and uses_window
                and hasattr(actuator, "snap_to_window")):
            if args.window_reference is not None:
                reference = math.radians(args.window_reference)
            else:
                # Prefer the remembered position over the config constant, for
                # the reason window_state.py exists: a constant predicts where a
                # limp arm settles, and it is bad at it.
                remembered = window_state.load_positions(args.shm_name)
                reference = remembered.get(
                    args.joint, actuator.startup_reference_rad)
            try:
                offset = actuator.snap_to_window(state.position, reference)
            except ValueError as err:
                raise SystemExit(
                    "[recal] {}: cannot establish the gear window, so the "
                    "datum frame is unknown and nothing can be {} in it. "
                    "Pass --window-reference DEG with roughly where this joint "
                    "is (a quarter of a window is close enough), or teach the "
                    "envelope by hand instead -- a hand visits both ends in "
                    "whatever frame the driver powered up in, so it needs no "
                    "window at all. {}".format(
                        args.joint,
                        "shifted" if args.shift_limits else "taught", err))
            if offset:
                print("[recal] gear window {:+d} ({:+.2f} deg) established from "
                      "the {} reference; teaching in the DATUM frame".format(
                          offset, offset * calibration.gear_window_deg,
                          "--window-reference" if args.window_reference is not None
                          else "remembered/config"))
            # Re-read so `here_deg` below is in the datum frame.
            states = _read_pose(bus, names, 0.5)
            state = states[args.joint]

        # ---- 1. re-zero -------------------------------------------------
        # counts_to_rad is (counts - zero_offset_counts) * 0.1 (+ window), so
        # making THIS pose read zero means the offset becomes the raw reading.
        old_offset_deg = calibration.zero_offset_counts / 10.0
        here_deg = math.degrees(state.position)
        # state.position has `direction` applied; the raw wire frame does not.
        # direction is +-1, so multiplying undoes it. Getting this wrong would
        # only show on a reversed joint -- i.e. not on the one being fixed
        # today, and then catastrophically on the next robot.
        frame_deg = here_deg * calibration.direction
        raw_deg = frame_deg + old_offset_deg - (
            calibration.window_offset * calibration.gear_window_deg)
        new_offset_deg = raw_deg
        print("")
        if args.no_rezero:
            new_offset_deg = old_offset_deg
            print("[recal] === KEEPING THE DATUM ({}) ===".format(args.joint))
            print("[recal]   zero_offset_deg stays {:+.2f}; this joint reads "
                  "{:+.2f} deg right now".format(old_offset_deg, here_deg))
            print("[recal]   teaching the envelope only. The rest pose is "
                  "wherever the arm settled,")
            print("[recal]   which is what the envelope has to contain -- not "
                  "necessarily 0.")
        else:
            print("[recal] === RE-ZERO {} ===".format(args.joint))
            print("[recal]   reads now      : {:+.2f} deg".format(here_deg))
            print("[recal]   raw on the wire: {:+.2f} deg".format(raw_deg))
            print("[recal]   zero_offset_deg: {:+.2f} -> {:+.2f}".format(
                old_offset_deg, new_offset_deg))
            print("[recal]   this pose becomes 0.00 deg")

            # Apply it in memory NOW, so the teach below records extremes in the
            # NEW frame. Teaching in the old frame and writing the numbers into a
            # re-zeroed config is precisely the mismatch this command exists to
            # prevent.
            calibration.zero_offset_counts = new_offset_deg * 10.0
            calibration.window_offset = 0

        # ---- 2. teach, or shift the envelope that already exists ---------
        if args.shift_limits:
            # A physical angle that read `x` in the old frame reads
            # `x - here_deg` in the new one, so subtracting the same number from
            # both bounds leaves the fence on the same PART OF THE MECHANISM.
            # That is the entire justification for skipping the teach: nothing
            # new is being learned about the joint, the numbers are only being
            # restated in the new frame.
            #
            # What it does NOT do is make a wrong envelope right. Every error in
            # the old bounds survives the subtraction unchanged. If a hand has
            # not been at both ends since the mechanism was last touched, teach.
            old_low_deg, old_high_deg = [
                math.degrees(x) for x in calibration.position_limit]
            fenced_low = old_low_deg - here_deg
            fenced_high = old_high_deg - here_deg
            print("")
            print("[recal] === SHIFT THE ENVELOPE ({}) === no hand needed"
                  .format(args.joint))
            print("[recal]   [{:+.2f}, {:+.2f}] - {:+.2f} = [{:+.2f}, {:+.2f}]"
                  .format(old_low_deg, old_high_deg, here_deg,
                          fenced_low, fenced_high))
            print("[recal]   same physical range, restated about the new datum.")
            if not fenced_low <= 0.0 <= fenced_high:
                raise SystemExit(
                    "[recal] REFUSED: the new datum (this pose) would sit "
                    "OUTSIDE the shifted envelope [{:+.2f}, {:+.2f}]. That can "
                    "only happen if the joint is currently outside its old "
                    "envelope, and an envelope that excludes the pose the robot "
                    "is in can never be entered -- servo_on refuses from there "
                    "forever.".format(fenced_low, fenced_high))
            rest_deg = 0.0
            low_deg = fenced_low
            high_deg = fenced_high
        else:
            print("")
            print("[recal] === TEACH {} === listen-only; nothing is commanded, "
                  "torque stays off.".format(args.joint))
            low_rad, high_rad = teach_one(bus, args.joint, actuator, args)
            low_deg, high_deg = math.degrees(low_rad), math.degrees(high_rad)
            print("")
            print("[recal]   reached by hand: {:+.2f} .. {:+.2f} deg  "
                  "(span {:.2f})".format(low_deg, high_deg, high_deg - low_deg))

            if high_deg - low_deg < args.min_travel_deg:
                raise SystemExit(
                    "[recal] REFUSED: only {:.2f} deg of travel was recorded, "
                    "under the {:.2f} deg minimum. A joint that was not really "
                    "moved would get an envelope locking it where it happened "
                    "to sit.".format(high_deg - low_deg, args.min_travel_deg))

        # Shrink inward, EXCEPT where that would exclude the rest pose. The
        # margin exists to stop us driving INTO a hard stop, not to fence off a
        # pose the robot already occupies unpowered -- an envelope that excludes
        # the rest pose makes torque impossible to enable from a cold start.
        # Where the joint rests, which is what the envelope must contain. 0 after
        # a re-zero by construction; otherwise wherever the arm actually settled.
            rest_deg = 0.0 if not args.no_rezero else here_deg
            fenced_low = low_deg + args.margin_deg
            fenced_high = high_deg - args.margin_deg
            notes = []
            if args.keep_rest_inside and fenced_low > rest_deg:
                fenced_low = low_deg
                notes.append("lower margin dropped: it would exclude the rest pose")
            if args.keep_rest_inside and fenced_high < rest_deg:
                fenced_high = high_deg
                notes.append("upper margin dropped: it would exclude the rest pose")
            print("[recal]   position_limit_deg: [{:.2f}, {:.2f}]  "
                  "(margin {:.2f} deg)".format(
                      fenced_low, fenced_high, args.margin_deg))
            for note in notes:
                print("[recal]   NOTE {}".format(note))

        if args.dry_run:
            print("")
            print("[recal] dry run: nothing written.")
            return 0

        # ---- 3. write both files ---------------------------------------
        stamp = time.strftime("%Y-%m-%d %H:%M")
        wrote = []
        with open(args.config) as handle:
            text = handle.read()
        if not args.no_rezero:
            text = config_edit.set_joint_value(
                text, args.joint, "zero_offset_deg",
                "{:.2f}".format(new_offset_deg))
        text = config_edit.set_joint_value(
            text, args.joint, "position_limit_deg",
            "[{:.2f}, {:.2f}]".format(fenced_low, fenced_high))
        # Its gear-window reference is wherever it rests, which after a re-zero
        # is 0 by construction. Only a first-run fallback either way -- the
        # servicer's remembered position supersedes it (window_state).
        #
        # Written ONLY for a joint that already uses window recovery. Writing it
        # for one that does not OPTS IT IN, which is not this command's business
        # and is not free: the joint then has to satisfy a +-quarter-window
        # reference at every start-up or refuse to start. joint0 showed why that
        # is the wrong default -- it has never lost an origin, and as a yaw axis
        # with 235 deg of travel and no rest pose it is precisely the joint whose
        # position a hand is most likely to have changed by more than the +-10 deg
        # its 40 deg window allows.
        if config_edit.find_joint_value(
                text, args.joint, "startup_reference_deg") is not None:
            text = config_edit.set_joint_value(
                text, args.joint, "startup_reference_deg",
                "{:.1f}".format(rest_deg))
        shutil.copyfile(args.config, args.config + ".bak")
        with open(args.config, "w") as handle:
            handle.write(text)
        wrote.append(args.config)

        if args.urdf:
            with open(args.urdf) as handle:
                urdf = handle.read()
            urdf = config_edit.set_urdf_joint_limits(
                urdf, args.joint, math.radians(fenced_low),
                math.radians(fenced_high))
            shutil.copyfile(args.urdf, args.urdf + ".bak")
            with open(args.urdf, "w") as handle:
                handle.write(urdf)
            wrote.append(args.urdf)
        else:
            print("[recal] WARNING: no --urdf given. The URDF still carries the "
                  "OLD limits, and skrobot clamps to them SILENTLY -- the model "
                  "will report plausible wrong angles until you update it.")

        # ---- 4. the window reference -------------------------------------
        # Only a RE-ZERO invalidates it: the remembered positions are angles in
        # the datum's frame, so moving the datum makes them lies. Teaching the
        # envelope alone leaves the frame untouched, and the memory is then a
        # perfectly good reference that must NOT be thrown away -- deleting it
        # would force the next start-up back onto the config constant, which is
        # exactly the worse reference this whole mechanism replaced.
        state_file = window_state.state_path(args.shm_name)
        if args.no_rezero:
            if os.path.exists(state_file):
                print("[recal] kept {} -- the datum did not move, so the "
                      "remembered positions are still valid".format(state_file))
        elif os.path.exists(state_file):
            os.unlink(state_file)
            print("[recal] removed {} -- it recorded positions in the OLD "
                  "frame".format(state_file))

        # Verify by reading back, not by the write having succeeded.
        with open(args.config) as handle:
            check = handle.read()
        back = config_edit.find_joint_value(
            check, args.joint, "position_limit_deg")
        if back is None or "{:.2f}".format(fenced_low) not in back:
            raise SystemExit(
                "[recal] FAILED: wrote position_limit_deg but it reads back as "
                "{!r}. The config was NOT updated correctly; restore "
                "{}.bak".format(back, args.config))
        if not args.no_rezero:
            back = config_edit.find_joint_value(
                check, args.joint, "zero_offset_deg")
            if back is None or abs(float(back) - new_offset_deg) > 0.005:
                raise SystemExit(
                    "[recal] FAILED: wrote zero_offset_deg but it reads back as "
                    "{!r}. The config was NOT updated correctly; restore "
                    "{}.bak".format(back, args.config))

        print("")
        print("[recal] written ({}):".format(stamp))
        for path in wrote:
            print("[recal]   {}  (backup at {}.bak)".format(path, path))
        print("[recal] Next launch reads these -- no copy/paste step.")
        print("")
        print("[recal] NOW READ THE COMMENTS AROUND {}'s NEW VALUES.".format(
            args.joint))
        print("[recal]   This tool deliberately preserves the prose in the "
              "config -- that reasoning")
        print("[recal]   is the calibration's provenance and a YAML round-trip "
              "would delete it. But")
        print("[recal]   it CANNOT know which of it your new numbers just made "
              "false. A fresh value")
        print("[recal]   next to stale reasoning misleads whoever reads it "
              "next into re-deriving the")
        print("[recal]   wrong thing. Say what you measured, and mark what it "
              "supersedes.")
        return 0
    finally:
        bus.transport.close()
        print("[recal] bus closed")


if __name__ == "__main__":
    raise SystemExit(main())
