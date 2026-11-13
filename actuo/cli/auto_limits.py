#!/usr/bin/env python3
"""Find a joint's travel by pushing it gently until it stops. DRIVES THE ROBOT.

Hand teaching (``actuo-teach-limits``) stops where the operator's patience does:
on a CubeMars AK80-6 arm joint the hand-taught span turned out to be a slice of
a much larger travel. This pushes instead, in CURRENT mode, at a torque you choose, and
calls "the position stopped changing while I was still pushing" a stop.

    actuo-auto-limits --config <robot>.yaml \
        --joint joint2 --torque 0.5

Read all of this before running it.

WHAT IT FINDS IS NOT WHAT YOU WANT
----------------------------------
It finds the **mechanical** envelope. You want the **safe** one. They differ
wherever the robot can hit itself, and this tool does NO self-collision
checking (that belongs in a separate safety layer above this package, and none
is wired in here). "Where it stopped" may be a
hard stop, one link meeting another, or the robot meeting the table -- and this
routine cannot tell those apart. Hand teaching had a human making that call;
automating it deletes the human, not the judgement. **Look at the robot while
this runs, and sanity-check every number it prints against the geometry.**

GRAVITY IS THE HARD PART, AND ONE PUSH CANNOT SETTLE IT
-------------------------------------------------------
A joint pushed against gravity stops when the torques balance, and from here
that is indistinguishable from a hard stop: the position holds, and the
commanded current flows either way. Measured on a gravity-loaded AK80-6 arm
joint, low direction:

    0.6 N*m -> stopped at -117.1 deg
    1.0 N*m -> stopped at -172.0 deg     (+54.9 -- so -117 was never a stop)
    1.2 N*m -> stopped at -172.1 deg     (+0.1  -- THIS is the stop)

A single push at 0.6 would have written -117.1 into the config as a limit and
been wrong by 55 degrees, confidently. So this ramps the torque and only calls
an end a stop once **raising the torque stops buying travel**. That is the whole
discrimination: gravity yields to more torque, a stop does not.

"It never moved at all" is still its own answer:

* ``--torque`` is REQUIRED and per-joint. There is no sensible default: the
  right value depends on the robot's mass, this joint's pose, and which way
  gravity pulls. A number that barely lifts one joint will slam another.
* A joint that never moves at all is reported as UNDETERMINED and gets no limit.
  It is not silently written down as a stop.
* Each direction is only believed if the joint travelled ``--min-travel-deg``
  first. Anything less means "it did not really go anywhere", not "the stop is
  right here".

WHY CURRENT AND NOT PWM
-----------------------
PWM is the one mode that needs no datasheet number, and it is the wrong tool:
at stall the back-EMF is zero, so duty maps straight to current through the
winding resistance (~0.17 ohm phase-to-phase on the AK80-6) and a "small" duty
is several amps into a stalled motor. CURRENT bounds the thing that actually
matters. It needs ``torque_constant``; supply it.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time


from actuo.core.contracts import ActuatorCommand, OperatingMode  # noqa: E402
from actuo.drivers.actuators.cubemars import ERROR_NAMES  # noqa: E402
from actuo.drivers.buses.can_bus import (  # noqa: E402
    OverspeedError, StaleFeedbackError)
from actuo.drivers.config import build_bus, load_document  # noqa: E402


class Abort(RuntimeError):
    """Something the operator must know about; the joint is released first."""


def _push(bus, actuator, index, torque, args, direction, log):
    """Drive one direction until the position stops changing. Returns (deg, why).

    Only this function puts a motion frame on the wire, and it releases every
    other actuator in the same batch on every cycle -- a limit search must move
    exactly one joint.
    """
    # Where the joint sits BEFORE anything is pushed. Sampling this after the
    # first command would drop that cycle's motion out of the travel, and travel
    # is what --min-travel-deg judges "did it actually go anywhere" on.
    bus.poll(timeout=0.0)
    first = bus.last_state(actuator.id)
    if first is None:
        raise Abort("{} has never reported".format(actuator.id))
    start = math.degrees(first.position)
    peak = start
    last_move = time.monotonic()
    deadline = time.monotonic() + args.timeout
    settle = args.settle
    moved_total = 0.0

    while True:
        now = time.monotonic()
        if now > deadline:
            raise Abort("{} never settled within {:.0f}s -- it is still moving, "
                        "so this is not a stop".format(actuator.id, args.timeout))

        batch = [None] * len(bus.actuators)
        batch[index] = ActuatorCommand(
            mode=OperatingMode.CURRENT, current=direction * torque)
        bus.write_batch(batch)

        bus.poll(timeout=0.0)
        bus.check_fresh()
        bus.check_speed(tolerance=args.overspeed_tolerance)

        state = bus.last_state(actuator.id)
        if state is None:
            raise Abort("{} stopped reporting".format(actuator.id))
        if state.error:
            raise Abort("{} faulted: {}".format(
                actuator.id, ERROR_NAMES.get(state.error, state.error)))
        if state.temperature is not None and state.temperature > args.max_temp:
            raise Abort(
                "{} reached {:.0f}C (limit {:.0f}). Pushing into a stop heats a "
                "motor fast; let it cool.".format(
                    actuator.id, state.temperature, args.max_temp))

        deg = math.degrees(state.position)
        travel = abs(deg - start)
        moved_total = max(moved_total, travel)

        # "Still moving" is judged against the extreme reached, not the last
        # sample: a joint creeping back under gravity would otherwise look like
        # continuous motion and never settle.
        if (direction > 0 and deg > peak + args.noise_deg) or \
           (direction < 0 and deg < peak - args.noise_deg):
            peak = deg
            last_move = now

        if abs(deg - start) > args.sanity_deg:
            raise Abort(
                "{} has travelled {:.0f} deg, past --sanity-deg {:.0f}. "
                "Stopping rather than letting it keep going.".format(
                    actuator.id, travel, args.sanity_deg))

        if now - last_move > settle:
            return peak, moved_total

        log(deg, travel, state.temperature,
            (state.aux or {}).get('current_a'))
        time.sleep(1.0 / args.rate)


def _ramp_direction(bus, actuator, index, direction, label, start_torque,
                    max_torque, args):
    """Ramp torque one way until raising it stops buying travel. Moves the joint.

    Returns ``(stop_deg, note)`` where ``stop_deg`` is None if the end was not
    determined (never moved, or still moving at the ceiling). A single push
    cannot tell a hard stop from a gravity balance, so this is the routine that
    ramps: gravity yields to more torque, a stop does not.
    """
    next_log = [0.0]
    amps_cmd = [0.0]

    def log(deg, travel, temp, amps):
        now = time.monotonic()
        if now >= next_log[0]:
            # Commanded amps beside reported amps: if they disagree the joint is
            # not stuck, the command is not landing, and more torque teaches you
            # nothing.
            print("      {:8.2f} deg  (moved {:6.2f})  T={:.0f}C  "
                  "I={:.2f}A (asked {:.2f}A)".format(
                      deg, travel, temp or 0.0, amps or 0.0, amps_cmd[0]),
                  flush=True)
            next_log[0] = now + 0.5

    torque = start_torque
    stop_deg = None
    moved = 0.0
    while True:
        amps_cmd[0] = direction * actuator.calibration.torque_to_amps(torque)
        print("    {} at {:.3f} N*m ...".format(label, torque), flush=True)
        here, travelled = _push(bus, actuator, index, torque, args, direction, log)
        moved = max(moved, travelled)

        # Convergence only MEANS something once the joint has actually gone
        # somewhere. Two pushes that both fail to move it also "converge" -- to
        # the start pose -- and treating that as a stop made the ramp give up on
        # the very joints that needed more torque (joint1 quit at 1.8 N*m of a
        # 4.8 ceiling because it had not budged). So: below --min-travel-deg,
        # a matching position is "still stuck", never "found the stop", and the
        # ramp keeps climbing.
        moved_enough = moved >= args.min_travel_deg
        if moved_enough and stop_deg is not None \
                and abs(here - stop_deg) < args.converged_deg:
            # Real travel, and more torque bought almost none: a real stop.
            return here, "stop at {:.2f} deg (travelled {:.2f})".format(here, moved)
        stop_deg = here

        if torque * args.torque_step > max_torque:
            if not moved_enough:
                return None, ("UNDETERMINED: never moved past {:.2f} deg even at "
                              "{:.3f} N*m (the ceiling). Either it started "
                              "against this stop, or the torque never beat "
                              "gravity -- raise max_torque_nm if the joint can "
                              "take it.".format(moved, torque))
            return None, ("UNDETERMINED: still moving at {:.3f} N*m (the "
                          "ceiling). Each torque bought more travel, which is "
                          "gravity, not a stop -- raise max_torque_nm."
                          .format(torque))
        torque *= args.torque_step


def _hand_direction(bus, actuator, index, label, args):
    """Listen-only: the operator hand-moves the joint to one stop; record it.

    For the end a push cannot reach -- the direction that lifts a joint against
    gravity, where a constant current just balances the weight and never finds
    the stop (measured on a CubeMars arm joint: 5 N*m still climbing). Here the operator
    supplies the muscle and this only watches, so nothing is commanded and there
    is no torque and no heat.

    Returns the extreme reached in ``label``'s direction (max for high, min for
    low) over a timed window, or None if the joint was not moved.
    """
    direction = +1 if label == "high" else -1
    bus.poll(timeout=0.0)
    first = bus.last_state(actuator.id)
    if first is None:
        raise Abort("{} has never reported".format(actuator.id))
    start = math.degrees(first.position)
    extreme = start
    print("[auto-limits]   HAND-MOVE {} to its {} stop now -- {:.0f}s window, "
          "nothing is commanded.".format(actuator.id, label, args.hand_seconds),
          flush=True)

    deadline = time.monotonic() + args.hand_seconds
    next_log = 0.0
    while time.monotonic() < deadline:
        bus.poll(timeout=0.0)
        bus.check_fresh()
        state = bus.last_state(actuator.id)
        if state is None:
            raise Abort("{} stopped reporting".format(actuator.id))
        if state.error:
            raise Abort("{} faulted: {}".format(
                actuator.id, ERROR_NAMES.get(state.error, state.error)))
        deg = math.degrees(state.position)
        if (direction > 0 and deg > extreme) or (direction < 0 and deg < extreme):
            extreme = deg
        now = time.monotonic()
        if now >= next_log:
            print("      {:8.2f} deg   ({} extreme {:8.2f})".format(
                deg, label, extreme), flush=True)
            next_log = now + 0.5
        time.sleep(1.0 / args.rate)

    if abs(extreme - start) < args.min_travel_deg:
        return None, ("UNDETERMINED: hand-moved only {:.2f} deg. Move it further "
                      "into the {} stop.".format(abs(extreme - start), label))
    return extreme, "hand stop at {:.2f} deg (moved {:.2f})".format(
        extreme, abs(extreme - start))


def calibrate_joint(bus, index, name, start_torque, max_torque, modes, args):
    """Search the ends of ONE joint. Returns {'low': deg, 'high': deg}, partial ok.

    ``modes`` maps each direction to 'push' (drive into it in CURRENT mode) or
    'hand' (operator moves it, listen-only). Mixing them is the point: a joint's
    gravity-helped end is found by pushing, its gravity-fought end by hand, and
    the two combine into one limit.

    Raises on a fault/overspeed/stale so the caller can stop the whole run --
    those mean something is wrong, not that this joint is simply undetermined.
    """
    actuator = bus.actuators[index]
    found = {}
    for token in ("low", "high"):
        mode = modes.get(token)
        if mode is None or mode == "skip":
            continue
        if mode == "hand":
            print("\n[auto-limits] {} {} by HAND:".format(name, token), flush=True)
            stop_deg, note = _hand_direction(bus, actuator, index, token, args)
        elif mode == "push":
            actuator.capabilities.require(OperatingMode.CURRENT)
            direction = +1 if token == "high" else -1
            print("\n[auto-limits] {} {} by PUSH (start {:.3f}, ceiling {:.3f} "
                  "N*m):".format(name, token, start_torque, max_torque),
                  flush=True)
            stop_deg, note = _ramp_direction(
                bus, actuator, index, direction, token, start_torque,
                max_torque, args)
        else:
            raise Abort("{} {}: mode must be push/hand/skip, got {!r}".format(
                name, token, mode))
        print("[auto-limits]   {} {}".format(token, note), flush=True)
        if stop_deg is not None:
            found[token] = stop_deg
    return found


def _limit_line(name, found, margin):
    """The position_limit_deg line for a joint, or a reason there is none."""
    if set(found) != {"low", "high"}:
        return None, "{}: only {} end(s) found; both are needed".format(
            name, sorted(found) or "no")
    low = found["low"] + margin
    high = found["high"] - margin
    if low >= high:
        return None, "{}: the two stops are < {:.1f} deg apart".format(
            name, 2 * margin)
    return "      position_limit_deg: [{:.2f}, {:.2f}]".format(low, high), None


def main(argv=None):
    p = argparse.ArgumentParser(description="find joint travel by pushing gently")
    p.add_argument("--config", required=True)
    p.add_argument("--joint", required=True,
                   help="ONE joint, or 'all' to sweep every joint that has an "
                        "auto_limit block in the config. This MOVES the robot.")
    p.add_argument("--torque", type=float, default=None,
                   help="N*m at the output shaft. Overrides the config's "
                        "auto_limit.torque_nm for a single joint; ignored for "
                        "'all', where each joint uses its own configured value. "
                        "No default: too-low reports UNDETERMINED (safe), and "
                        "the right value is per-joint (a number that lifts "
                        "joint1 slams joint3).")
    p.add_argument("--torque-step", type=float, default=1.5,
                   help="multiply the torque by this and push again, until the "
                        "stop stops moving (default 1.5x). This is what tells a "
                        "hard stop from a gravity balance: gravity yields to "
                        "more torque, a stop does not.")
    p.add_argument("--max-torque", type=float, default=None,
                   help="stop ramping here. Overrides config auto_limit."
                        "max_torque_nm for a single joint. Reaching it without "
                        "converging is UNDETERMINED, not written as a limit.")
    p.add_argument("--converged-deg", type=float, default=1.0,
                   help="a torque step that buys less travel than this has "
                        "found the stop")
    p.add_argument("--low", choices=("push", "hand", "skip"), default=None,
                   help="how to find the LOW end of a single joint, overriding "
                        "its config auto_limit.low. 'push' drives into it in "
                        "CURRENT mode; 'hand' is listen-only while the operator "
                        "moves it (for the gravity-fought end a push only "
                        "balances); 'skip' leaves it. Ignored for 'all'.")
    p.add_argument("--high", choices=("push", "hand", "skip"), default=None,
                   help="how to find the HIGH end of a single joint, overriding "
                        "its config auto_limit.high (see --low).")
    p.add_argument("--hand-seconds", type=float, default=25.0,
                   help="listen-only window for a 'hand' direction: move the "
                        "joint into that stop within this many seconds "
                        "(default 25). Nothing is commanded.")
    p.add_argument("--margin-deg", type=float, default=2.0,
                   help="shrink each end inward by this, so a command never "
                        "targets the stop itself")
    p.add_argument("--min-travel-deg", type=float, default=5.0,
                   help="a direction that moves less than this is UNDETERMINED, "
                        "not a stop")
    p.add_argument("--settle", type=float, default=1.5,
                   help="seconds of no further progress that counts as stopped")
    p.add_argument("--noise-deg", type=float, default=0.3,
                   help="movement below this is encoder quantisation (0.1 deg "
                        "steps), not travel")
    p.add_argument("--sanity-deg", type=float, default=200.0,
                   help="abort if one direction travels further than this")
    p.add_argument("--max-temp", type=float, default=55.0)
    p.add_argument("--timeout", type=float, default=40.0, help="per direction")
    p.add_argument("--rate", type=float, default=50.0)
    p.add_argument("--overspeed-tolerance", type=float, default=4.0)
    p.add_argument("--interface", default=None)
    p.add_argument("--channel", default=None)
    p.add_argument("--bitrate", type=int, default=None)
    p.add_argument("--yes", action="store_true",
                   help="skip the confirmation prompt")
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

    # Per-joint torque AND per-direction mode live in the config's auto_limit
    # blocks -- the per-robot place for per-robot numbers -- so 'all' needs
    # nothing on the command line and cannot be given one wrong value for every
    # joint. Each block: torque_nm/max_torque_nm (for push directions) and
    # low/high = push|hand|skip (default push).
    document = load_document(args.config)
    torques = {}          # name -> (start, ceiling); absent if never pushed
    modes = {}            # name -> {'low': mode, 'high': mode}
    for joint in document.get("joints", []):
        al = joint.get("auto_limit")
        if not al:
            continue
        name = joint["name"]
        modes[name] = {"low": al.get("low", "push"), "high": al.get("high", "push")}
        if "torque_nm" in al:
            torques[name] = (
                float(al["torque_nm"]),
                float(al.get("max_torque_nm", al["torque_nm"] * 4.0)))

    if args.joint == "all":
        targets = [n for n in names if n in modes]
        skipped = [n for n in names if n not in modes]
        if skipped:
            print("[auto-limits] no auto_limit block for {}; skipping".format(
                ", ".join(skipped)), file=sys.stderr)
        if not targets:
            print("[auto-limits] no joint has an auto_limit block; nothing to "
                  "do", file=sys.stderr)
            return 2
    else:
        if args.joint not in names:
            print("no such joint {!r}; this bus has: {}".format(
                args.joint, ", ".join(names)), file=sys.stderr)
            return 2
        modes.setdefault(args.joint, {"low": "push", "high": "push"})
        if args.low is not None:
            modes[args.joint]["low"] = args.low
        if args.high is not None:
            modes[args.joint]["high"] = args.high
        if args.torque is not None:
            torques[args.joint] = (
                args.torque, args.max_torque or args.torque * 4.0)
        elif args.max_torque is not None and args.joint in torques:
            torques[args.joint] = (torques[args.joint][0], args.max_torque)
        targets = [args.joint]

    # A joint pushed in some direction needs a torque; a hand-only joint does
    # not. Fail loudly rather than reaching a push with no torque to give it.
    for name in targets:
        pushes = [d for d, m in modes[name].items() if m == "push"]
        if pushes and name not in torques:
            print("[auto-limits] {} is set to push {} but has no torque; give "
                  "--torque or add torque_nm to its auto_limit block".format(
                      name, "/".join(pushes)), file=sys.stderr)
            return 2

    print("[auto-limits] will calibrate: {}".format(", ".join(targets)))
    for name in targets:
        m = modes[name]
        detail = "low={} high={}".format(m["low"], m["high"])
        if name in torques:
            detail += "  ({:.3f}..{:.3f} N*m for push)".format(*torques[name])
        print("[auto-limits]   {}: {}".format(name, detail))
    print("[auto-limits] THIS MOVES THE ROBOT, into its stops, on purpose.")
    print("[auto-limits] It finds the MECHANICAL travel. This tool does no "
          "self-collision checking, so a 'stop' may be the robot hitting itself "
          "or the table -- look at the robot and sanity-check every number.")
    print("[auto-limits] Keep a hand on the power. A CAN link drop was seen "
          "once while pushing hard; if it recurs the run aborts.")
    if not args.yes:
        try:
            if input("[auto-limits] type MOVE to proceed: ").strip() != "MOVE":
                print("[auto-limits] aborted; nothing was sent")
                return 1
        except EOFError:
            print("[auto-limits] no tty for the prompt; pass --yes if you mean "
                  "it", file=sys.stderr)
            return 1

    bus.transport.open()
    results = {}
    try:
        bus.read_batch(timeout=5.0)
        for name in targets:
            index = names.index(name)
            start, ceiling = torques.get(name, (None, None))
            results[name] = calibrate_joint(
                bus, index, name, start, ceiling, modes[name], args)
    except (Abort, StaleFeedbackError, OverspeedError) as err:
        # A fault/overspeed/stale is about the hardware, not this joint's range:
        # stop the whole sweep rather than moving on to push the next joint.
        print("\n[auto-limits] ABORT: {}".format(err), file=sys.stderr)
        try:
            bus.release_all()
        except Exception:                                     # noqa: BLE001
            pass
        bus.transport.close()
        return 1
    except KeyboardInterrupt:
        print("\n[auto-limits] interrupted")
    finally:
        # Always let go. This routine's whole job is to end up pressed against
        # something, and leaving the driver holding that is how a motor cooks.
        try:
            bus.release_all()
        except Exception:                                     # noqa: BLE001
            pass
        bus.transport.close()
        print("[auto-limits] released; all joints are limp")

    print("\n" + "=" * 68)
    print("Paste each line into that joint's `actuator:` block -- AFTER checking")
    print("against the robot that neither end is the robot hitting itself:")
    print("=" * 68)
    any_ok = False
    for name in targets:
        line, why = _limit_line(name, results.get(name, {}), args.margin_deg)
        if line:
            any_ok = True
            print("    # {}".format(name))
            print(line)
        else:
            print("    # {} -- NO LIMIT: {}".format(name, why))
    return 0 if any_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
