#!/usr/bin/env python3
"""Find each finger servo's travel by GENTLY driving it into its stops.

The Amazing Hand's SCS0009 servos do not back-drive (two opposed servos per
finger bind the linkage), so the "limp it and hand-move" teach used for
back-drivable arm joints (``actuo-teach-limits``) cannot work here. But these are
POSITION servos, and that gives an automatic method those joints lack: command
the goal outward and watch ``present_position`` stop following. When the finger
jams, the servo keeps being told to go further while its measured
position pins -- a growing *following error*. That, not the sound, is the stop.

It is gentler to get right than an arm joint, too: fingers are light, so there is no
gravity balance to mistake for a stop, and the push force is set directly by
lowering ``max_torque_limit`` for the duration -- a jam presses softly, not hard.

One servo at a time, low torque, backing off the instant a stop is found. Still
motion on a hand with no self-collision model: a detected "stop" may be a finger
against its neighbour or the palm, which the routine cannot tell from a hard
stop. Watch it, and keep a hand on the power.

    actuo-hand-find-limits --config <hand>.yaml --joint l_f1_a
    actuo-hand-find-limits --config <hand>.yaml --joint all

Emits a ``position_limit_deg`` per joint, shrunk inward by a margin.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time


from actuo.drivers.actuators import feetech_scs as f  # noqa: E402
from actuo.core.contracts import Frame  # noqa: E402
from actuo.drivers.config import build_bus  # noqa: E402

SOFT_MARGIN_COUNTS = 2   # stay this far inside the servo's own angle limits


class Abort(RuntimeError):
    """A fault/overheat -- stop the whole run, not just this joint."""


def find_stop(step_fn, start, direction, step, bound,
              follow_error, settle, noise, max_steps):
    """Advance the goal outward until ``present`` stops following it.

    Parameters
    ----------
    step_fn : callable(goal_raw) -> present_raw
        Commands ``goal_raw`` and returns the resulting present position. Owns
        the real timing, reads, and safety guards; this function is pure control
        logic so it can be unit-tested with a fake.
    start, bound : int
        Start count and the count not to command past (the servo's soft limit).
    direction : int
        +1 toward higher counts, -1 toward lower.
    step, noise, follow_error : int
        Counts per increment; movement below ``noise`` is "not advancing"; a
        goal-minus-present gap above ``follow_error`` means the servo is pinned.
    settle : int
        Consecutive non-advancing steps (while pinned) that confirm a stop.

    Returns
    -------
    (stop_raw, note) : (Optional[int], str)
        ``stop_raw`` is the furthest present reached, or None if the soft bound
        was hit while still tracking (no mechanical stop inside the range).
    """
    peak = start
    goal = start
    stuck = 0
    for _ in range(max_steps):
        goal += direction * step
        goal = min(goal, bound) if direction > 0 else max(goal, bound)
        present = step_fn(goal)

        advanced = (present - peak) if direction > 0 else (peak - present)
        if advanced > noise:
            peak = present
            stuck = 0
        else:
            stuck += 1

        pinned = abs(goal - present) > follow_error
        if stuck >= settle and pinned:
            return peak, "stop at raw {} (follow err {})".format(
                peak, abs(goal - present))

        if goal == bound:
            if abs(goal - present) <= follow_error:
                # Tracked cleanly to the servo's own angle limit with no jam:
                # the servo will not go past this anyway, so ITS soft limit is
                # the usable limit for this direction -- a real bound, not None.
                return present, ("reached the servo's soft limit at raw {} with "
                                 "no mechanical stop -- using it".format(present))
            return peak, "stop at raw {} (at soft bound)".format(peak)
    return peak, "gave up after {} steps at raw {}".format(max_steps, peak)


def _read_reg(transport, sid, addr, length, retries=4):
    """Read ``length`` bytes, validating the reply is this servo's and the right
    size. Retries because a write's status ACK can land in the read's slot: the
    servo answers WRITE with a 0-data status packet, and one of those grabbed
    instead of the READ reply is a short packet, not data -- reject and re-ask.
    """
    for _ in range(retries):
        reply = transport.transact(
            [Frame(payload=f.build_read(sid, addr, length), address=sid)], 0.05)[0]
        if reply is None:
            continue
        parsed = f.parse_status(reply.payload)
        if parsed and parsed[0] == sid and len(parsed[2]) == length:
            return parsed[2]
    return None


def _write_reg(transport, sid, addr, data):
    transport.write([Frame(payload=f.build_write(sid, addr, data), address=sid)])


def _read_present(transport, sid):
    """One block read -> (position_raw, load_raw, temperature) or None.

    Position, load and temperature in a single transaction (present block at 56),
    so each sweep step is one round trip and the load is watched for free.
    """
    data = _read_reg(transport, sid, f.PRESENT_BLOCK_ADDR, f.PRESENT_BLOCK_LEN)
    if data is None:
        return None
    pos = f.from_u16_be(data[0:2]) & 0x3FF
    load = f.from_u16_be(data[4:6]) & 0x3FF     # magnitude, 0..1023
    temp = data[7]
    return pos, load, temp


def calibrate_joint(transport, name, actuator, args):
    """Drive one servo to both stops gently; return (low_raw, high_raw)."""
    sid = actuator.id
    limits = _read_reg(transport, sid, f.REG_MIN_ANGLE_LIMIT, 4)
    if limits is None:
        raise Abort("{} (id {}) did not answer".format(name, sid))
    soft_lo = f.from_u16_be(limits[0:2]) + SOFT_MARGIN_COUNTS
    soft_hi = f.from_u16_be(limits[2:4]) - SOFT_MARGIN_COUNTS

    original_torque = _read_reg(transport, sid, f.REG_MAX_TORQUE_LIMIT, 2)
    block = _read_present(transport, sid)
    if block is None:
        raise Abort("{} (id {}) did not report present state".format(name, sid))
    start = block[0]
    print("[find-limits] {} (id {}): start raw {}, soft range [{}, {}], "
          "pushing at torque {} of 1000 (load abort > {})".format(
              name, sid, start, soft_lo, soft_hi, args.torque_limit,
              args.max_load))

    def step_fn(goal_raw):
        _write_reg(transport, sid, f.REG_GOAL_POSITION, f.u16_be(int(goal_raw)))
        time.sleep(args.step_delay)
        block = _read_present(transport, sid)
        if block is None:
            raise Abort("{} stopped reporting".format(name))
        present, load, temp = block
        if temp > args.max_temp:
            raise Abort("{} reached {} C (limit {})".format(name, temp, args.max_temp))
        if load > args.max_load:
            raise Abort("{} load {} over limit {} -- backing off".format(
                name, load, args.max_load))
        if args.verbose:
            print("      goal {:4d}  present {:4d}  (follow {:+4d})  "
                  "load {:4d}  T {}C".format(
                      int(goal_raw), present, int(goal_raw) - present, load, temp),
                  flush=True)
        return present

    # Gentle: lower the servo's torque ceiling so a jam presses softly.
    _write_reg(transport, sid, f.REG_MAX_TORQUE_LIMIT, f.u16_be(args.torque_limit))
    _write_reg(transport, sid, f.REG_TORQUE_ENABLE, bytes([1]))
    try:
        found = {}
        for direction, label, bound in ((-1, "low", soft_lo), (+1, "high", soft_hi)):
            print("  {} ...".format(label), flush=True)
            here_block = _read_present(transport, sid)
            here = here_block[0] if here_block else start
            stop, note = find_stop(
                step_fn, here, direction, args.step, bound,
                args.follow_error, args.settle, args.noise, args.max_steps)
            print("  {}: {}".format(label, note), flush=True)
            if stop is not None:
                found[label] = stop
            # Back off toward the middle so we never hold a jam.
            _write_reg(transport, sid, f.REG_GOAL_POSITION,
                       f.u16_be(int(start)))
            time.sleep(0.3)
        return found
    finally:
        _write_reg(transport, sid, f.REG_TORQUE_ENABLE, bytes([0]))    # limp
        if original_torque is not None:
            _write_reg(transport, sid, f.REG_MAX_TORQUE_LIMIT, original_torque)


def _limit_line(name, found, actuator, margin_deg):
    if set(found) != {"low", "high"}:
        return None, "{}: only {} end(s) found".format(name, sorted(found) or "no")
    low_deg = math.degrees(actuator.calibration.counts_to_rad(found["low"]))
    high_deg = math.degrees(actuator.calibration.counts_to_rad(found["high"]))
    low, high = sorted((low_deg, high_deg))
    low += margin_deg
    high -= margin_deg
    if low >= high:
        return None, "{}: ends < {:.1f} deg apart".format(name, 2 * margin_deg)
    return "      position_limit_deg: [{:.2f}, {:.2f}]".format(low, high), None


def main(argv=None):
    p = argparse.ArgumentParser(
        description="find finger-servo travel by gently driving into the stops")
    p.add_argument("--config", required=True)
    p.add_argument("--joint", required=True, help="ONE joint or 'all'; MOVES it")
    p.add_argument("--torque-limit", type=int, default=300,
                   help="servo max_torque_limit during the push, of 1000 "
                        "(default 300 = gentle). Restored afterwards.")
    p.add_argument("--step", type=int, default=4,
                   help="goal advance per cycle, raw counts (~0.3 deg each)")
    p.add_argument("--step-delay", type=float, default=0.06)
    p.add_argument("--follow-error", type=int, default=15,
                   help="goal-minus-present gap that means the servo is pinned")
    p.add_argument("--settle", type=int, default=5,
                   help="consecutive pinned, non-advancing steps = a stop")
    p.add_argument("--noise", type=int, default=2,
                   help="present change below this is jitter, not travel")
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--max-temp", type=float, default=60.0)
    p.add_argument("--max-load", type=int, default=800,
                   help="abort if present_load magnitude exceeds this, of 1023 "
                        "(default 800). max_torque_limit already caps the force; "
                        "this is the backstop.")
    p.add_argument("--margin-deg", type=float, default=3.0)
    p.add_argument("--channel", default=None)
    p.add_argument("--bitrate", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--yes", action="store_true", help="skip the confirmation")
    args = p.parse_args(argv)

    overrides = {}
    if args.channel is not None:
        overrides["channel"] = args.channel
    if args.bitrate is not None:
        overrides["bitrate"] = args.bitrate

    bus, names = build_bus(args.config, transport_overrides=overrides or None)
    if args.joint == "all":
        targets = list(names)
    elif args.joint in names:
        targets = [args.joint]
    else:
        print("no such joint {!r}; this hand has: {}".format(
            args.joint, ", ".join(names)), file=sys.stderr)
        return 2

    print("[find-limits] will drive: {}".format(", ".join(targets)))
    print("[find-limits] THIS MOVES THE FINGERS into their stops, gently.")
    print("[find-limits] No self-collision model -- a 'stop' may be one finger "
          "on another. Watch it; keep a hand on the power.")
    if not args.yes:
        try:
            if input("[find-limits] type MOVE to proceed: ").strip() != "MOVE":
                print("[find-limits] aborted; nothing was sent")
                return 1
        except EOFError:
            print("[find-limits] no tty; pass --yes if you mean it", file=sys.stderr)
            return 1

    bus.transport.open()
    results = {}
    index = {n: i for i, n in enumerate(names)}
    try:
        for name in targets:
            actuator = bus.actuators[index[name]]
            results[name] = calibrate_joint(bus.transport, name, actuator, args)
    except Abort as err:
        print("\n[find-limits] ABORT: {}".format(err), file=sys.stderr)
        bus.release_all()
        bus.transport.close()
        return 1
    except KeyboardInterrupt:
        print("\n[find-limits] interrupted")
    finally:
        try:
            bus.release_all()
        except Exception:                                     # noqa: BLE001
            pass
        bus.transport.close()
        print("[find-limits] released; the hand is limp")

    print("\n" + "=" * 68)
    print("Paste each line into that joint's config (check it against the hand):")
    print("=" * 68)
    any_ok = False
    for name in targets:
        actuator = bus.actuators[index[name]]
        line, why = _limit_line(name, results.get(name, {}), actuator,
                                args.margin_deg)
        if line:
            any_ok = True
            print("    # {}".format(name))
            print(line)
        else:
            print("    # {} -- NO LIMIT: {}".format(name, why))
    return 0 if any_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
