#!/usr/bin/env python3
"""Measure what a continuous-rotation Kondo servo actually does per command.

    # Kondo ARM-H7 board
    actuo-wheel-calibrate --channel /dev/... \
        --symbols .../kondoh7_v0.6.5_symbols.yaml --ids 20
    # Kondo RCB-4mini board (ids are ICS numbers)
    actuo-wheel-calibrate --board rcb4mini \
        --channel /dev/... --ids 20 21 22 23 \
        --rotation-checked "<date>: rcb4 ics.py read_rotation on the adapter"

*** THE WHEEL WILL SPIN. Lift the robot so its wheels are off the ground. ***

Why this exists
---------------
A Kondo servo in continuous-rotation mode (`ServoStruct.rotation == 1`) reads
the SAME pulse field as a position servo, but treats it as a speed: 7500 is
stop, and either side of it turns. rcb4 drives the wheels through the identical
`angle_vector` path as the arms -- an "angle" in degrees, converted by the same
7500 + deg * (4000/135) -- so no new wire format is needed, only a meaning.

What is NOT known is the meaning's scale. The relationship is proportional, not
calibrated: nothing on the wire says how fast a given offset turns the wheel,
and these servos have a DEADBAND, so small offsets turn it not at all. Both
numbers are per-servo and have to be measured before a mecanum controller's
`rad/s` can be honoured rather than approximated.

How the speed is measured, and why it is the wrap PERIOD
-------------------------------------------------------
The servo keeps reporting `current_angle` while it spins, and the reading is a
sawtooth: it sweeps, wraps, sweeps again. **One wrap is one revolution**, so the
time between two wraps is the revolution period, and 1/period is the wheel's
speed in rev/s. That needs no knowledge of the pot's scale at all.

The obvious alternative -- unwrap the sawtooth, fit a slope in pulses/s, divide
by the pot's span -- was what this did first, and it is wrong here twice over.
It assumed a span (8000 pulses; the measured range is 3000..12000), and worse,
dividing by a span assumes the pot reports the WHOLE revolution. Measured, a
wrap jumps about 7400 pulses across a 9000-pulse range, so roughly a sixth of
each turn is not reported -- a blind sector. A rate divided by the full span
would be a confident number about a sensor rather than about a wheel.

So: at least two wraps, or no speed is reported. A wheel turning too slowly to
complete a revolution inside the window gets `n/a` and a longer window, not an
extrapolation.

Every step ends with a stop, and so does every exit path.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


import struct  # noqa: E402

from actuo.core.contracts import Frame  # noqa: E402
from actuo.drivers.kondo_h7_protocol import (  # noqa: E402
    build_multi_servo, decode_servo_slot, load_symbol_map, servo_slot_read)
from actuo.drivers.transports.kondo_h7 import KondoH7Transport  # noqa: E402

CENTER_PULSE = 7500
PULSE_PER_DEG = 4000.0 / 135.0
#: A step this large cannot be real motion between two samples; it is the
#: reading wrapping. Expressed as a fraction of the range the run actually
#: covered, so nothing here assumes what that range is -- an earlier version
#: assumed 8000 pulses and the measured range turned out to be 3000..12000.
WRAP_FRACTION = 0.5


def parse_args(argv):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--board", choices=("h7", "rcb4mini"), default="h7",
                   help="h7 = Kondo ARM-H7 (reads each servo's rotation mode "
                        "from the board); rcb4mini = Kondo RCB-4mini, where "
                        "--ids are ICS numbers and the rotation mode cannot be "
                        "read, so --rotation-checked is required")
    p.add_argument("--channel", required=True)
    p.add_argument("--symbols",
                   help="ARM-H7 firmware symbol map (required with --board h7)")
    p.add_argument("--rotation-checked", default=None, metavar="TEXT",
                   help="RCB-4mini only: how and when these servos were seen to "
                        "be in rotation mode, e.g. read with rcb4 ics.py on the "
                        "Dual USB Adapter. The board cannot tell, and a position "
                        "servo commanded these offsets swings by up to 100 deg.")
    p.add_argument("--ics-bitrate", type=int, default=1250000,
                   help="RCB-4mini only: the rate the servos are set to")
    p.add_argument("--ids", type=int, nargs="+", required=True,
                   help="servo ids to calibrate. They are done ONE AT A TIME "
                        "and stopped in between: a robot on a stand with four "
                        "wheels spinning is harder to read and harder to stop.")
    p.add_argument("-o", "--output",
                   help="write the measured numbers as a YAML fragment for the "
                        "robot's bus config. Without it they are printed only, "
                        "which on the second robot means someone retypes them.")
    p.add_argument("--offsets", type=float, nargs="+",
                   default=[2, 4, 6, 8, 10, 12, 15, 20, 30, 60, 100],

                   help="command offsets in 'degrees', the unit rcb4 sends. "
                        "Each is tried both directions. The default is dense "
                        "between 2 and 20 because that is where a measured "
                        "wheel turned out to live: below is deadband, above is "
                        "saturation, and a coarse sweep sees a straight line "
                        "through neither.")
    p.add_argument("--seconds", type=float, default=5.0,
                   help="MINIMUM time to hold each offset. A slow wheel is then "
                        "held longer -- see --wraps and --max-dwell.")
    p.add_argument("--wraps", type=int, default=3,
                   help="whole revolutions to see before an offset is done "
                        "(default 3). A FIXED dwell is what made the first run "
                        "of this tool wrong: at 5 s a wheel turning 0.15 rev/s "
                        "produces under two wraps, gets discarded as 'no clean "
                        "rate', and the fit then runs a straight line through "
                        "the fast points only. That is how a servo which is "
                        "either stopped or at full speed came out as a tidy "
                        "proportional map.")
    p.add_argument("--max-dwell", type=float, default=25.0,
                   help="give up waiting for --wraps after this long. A wheel "
                        "that manages one revolution in 25 s is reported as "
                        "the slow speed it is, not as n/a.")
    p.add_argument("--rate", type=float, default=0.0,
                   help="feedback sampling rate; 0 = as fast as the bus "
                        "answers (about 500 Hz), which is what makes a wrap "
                        "unmistakable")
    p.add_argument("--move-byte", type=int, default=1,
                   help="board move duration byte. rcb4 uses 1 for wheels "
                        "(wheel_frame_count), i.e. apply the speed now rather "
                        "than ramp to it.")
    p.add_argument("--max-spread", type=float, default=0.25,
                   help="how much the revolution periods may vary before the "
                        "speed is discarded. Consistent revolutions are the "
                        "evidence that the wraps were revolutions.")
    p.add_argument("--identify", type=float, default=None, metavar="OFFSET_DEG",
                   help="do not calibrate: just turn each --ids servo, one at "
                        "a time, at this command offset for --seconds, and ask "
                        "which corner of the robot moved. The measured speed "
                        "map says nothing about WHERE a servo is, and the "
                        "mecanum kinematics are entirely about where -- a "
                        "front-left/front-right swap does not slow the base, "
                        "it makes it strafe when told to drive forward. Lift "
                        "the robot first.")
    p.add_argument("--yes", action="store_true",
                   help="skip the are-the-wheels-off-the-ground prompt")
    return p.parse_args(argv)


class H7Board:
    """A Kondo ARM-H7: servos by flat id, rotation mode read from the board."""

    def __init__(self, channel, symbols_path):
        self.transport = KondoH7Transport(channel, 1000000, 0.3)
        self.transport.open()
        self.base = load_symbol_map(symbols_path).address("servo_vector")

    def not_in_rotation_mode(self, ids):
        """The ids the board says are position servos."""
        reply = self.transport.transact(
            [Frame(payload=servo_slot_read(self.base, "rotation"))], 0.3)[0]
        rotation = decode_servo_slot("rotation", reply.payload)
        return [i for i in ids if not rotation[i]]

    def read_angle(self, servo_id):
        reply = self.transport.transact(
            [Frame(payload=servo_slot_read(self.base, "current_angle"))], 0.3)[0]
        return None if reply is None else decode_servo_slot(
            "current_angle", reply.payload)[servo_id]

    def command(self, servo_id, pulse, move_byte):
        self.transport.transact([Frame(payload=build_multi_servo(
            [servo_id], [pulse], move_byte))], 0.3)

    def close(self, ids):
        self.transport.close()


class Rcb4MiniBoard:
    """A Kondo RCB-4mini: servos by ICS number, rotation mode not readable.

    Connecting is what ``kondo_rcb4_bus`` does at startup (switch the COM
    rate, stop the ROM program, assign the ICS numbers, all in RAM). Each
    wheel is then switched on at STOP by writing its record's reference: the
    usual servo-on copies the pot into the reference, which a wheel reads as a
    speed. On exit the wheels are released.
    """

    def __init__(self, channel, ids, ics_bitrate):
        from actuo.drivers.actuators.kondo_ics import (
            KondoIcsActuator, KondoIcsCalibration)
        from actuo.drivers.buses.kondo_rcb4_bus import KondoRcb4Bus
        from actuo.drivers.transports.kondo_rcb4 import KondoRcb4Transport
        transport = KondoRcb4Transport(channel)
        transport.open()
        # Position actuators only as handles for the bus's connect/assign; no
        # position command is ever sent through them.
        self.bus = KondoRcb4Bus(
            transport,
            [KondoIcsActuator(i, KondoIcsCalibration(), velocity=1) for i in ids],
            joint_names=["ics{}".format(i) for i in ids],
            ics_bitrate=ics_bitrate)
        for item in self.bus.verify_servos():
            print("[board] {}: {}".format(item.name, item.message), flush=True)
        for servo_id in ids:
            self.bus._stop_wheel(servo_id)

    def not_in_rotation_mode(self, ids):
        """Nothing: this board cannot tell. The operator vouched instead."""
        return []

    def read_angle(self, servo_id):
        from rcb4.protocol.rcb4mini import servo_record_address
        raw = self.bus._read_ram(servo_record_address(servo_id) + 4, 2)
        pulse = struct.unpack("<H", raw)[0]
        return None if pulse == 0 else pulse

    def command(self, servo_id, pulse, move_byte):
        from rcb4.protocol.rcb4mini import build_multi_servo as move
        self.bus._command(move([servo_id], [pulse], move_byte))

    def close(self, ids):
        try:
            self.bus.release_all()
        finally:
            self.bus.transport.close()


def calibrate_one(board, servo_id, args, log):
    """Sweep one wheel and return what was measured, or None if it never moved."""
    def read_angle():
        return board.read_angle(servo_id)

    def command(offset_deg):
        pulse = int(round(CENTER_PULSE + offset_deg * PULSE_PER_DEG))
        board.command(servo_id, pulse, args.move_byte)
        return pulse

    rows = []
    log("\n{:>8} {:>7} {:>11} {:>8} {:>7} {:>7}".format(
        "offset", "pulse", "rev/s", "moving%", "wraps", "spread"))
    log("-" * 54)
    try:
        for magnitude in args.offsets:
            for offset in (magnitude, -magnitude):
                pulse = command(offset)
                times, angles = [], []
                started = time.time()
                # Hold until enough whole revolutions have gone by, not for a
                # fixed time: resolution at the slow end is the whole question
                # here, and a fixed dwell has none.
                seen_wraps = 0
                last = None
                while True:
                    elapsed = time.time() - started
                    if elapsed >= args.max_dwell:
                        break
                    if elapsed >= args.seconds and seen_wraps >= args.wraps:
                        break
                    value = read_angle()
                    if value is not None:
                        times.append(elapsed)
                        angles.append(float(value))
                        if last is not None and abs(float(value) - last) > 2000:
                            seen_wraps += 1
                        last = float(value)
                    if args.rate > 0:
                        time.sleep(max(0.0, 1.0 / args.rate - 0.002))
                command(0.0)
                time.sleep(0.3)

                if len(angles) < 20:
                    log("{:>8.1f} {:>7} {:>11}".format(offset, pulse, "no data"))
                    continue
                raw = np.array(angles)
                t = np.array(times)
                steps = np.diff(raw)
                span = float(raw.max() - raw.min())
                moving = float(np.mean(np.abs(steps) > 0.5))
                wrap_at = (np.where(np.abs(steps) > WRAP_FRACTION * span)[0]
                           if span > 100 else np.array([], dtype=int))
                periods = np.diff(t[wrap_at]) if len(wrap_at) > 1 else np.array([])
                if len(periods) == 0 and len(wrap_at) == 1:
                    # Exactly one wrap in the whole dwell. Not a clean period,
                    # but it is not nothing either: it says the wheel turned at
                    # roughly one revolution per dwell, which at the slow end is
                    # the difference between a number and a shrug. Reported with
                    # its spread left as NaN so it cannot pass the cleanliness
                    # filter and be fitted as if it were solid.
                    elapsed_total = float(t[-1] - t[0])
                    if elapsed_total > 0:
                        log("{:>8.1f} {:>7} {:>11} {:>7.0f}% {:>7} {:>7}".format(
                            offset, pulse, "~{:.2f}".format(1.0 / elapsed_total),
                            100 * moving, 1, "1 wrap"))
                if len(periods) >= 1:
                    period = float(np.median(periods))
                    rev_s = 1.0 / period if period > 1e-6 else 0.0
                    # Direction from the sign of the ordinary (non-wrap) steps
                    # THAT ARE NON-ZERO. Sampling runs faster than the board
                    # updates the angle, so most consecutive samples are equal
                    # and the median step is 0.0 in BOTH directions -- which
                    # silently made every reverse run read as forward, and then
                    # fitted a gain through +1.1 and +0.4 rev/s and got 0.0017.
                    # Measured: 97 % of the non-zero steps rise going forward,
                    # 0 % going back. Unambiguous once the zeros are dropped.
                    ordinary = steps[np.abs(steps) <= WRAP_FRACTION * span]
                    ordinary = ordinary[ordinary != 0]
                    if len(ordinary) and np.mean(ordinary > 0) < 0.5:
                        rev_s = -rev_s
                    spread = (float(np.std(periods) / period)
                              if len(periods) > 1 and period > 0 else 0.0)
                else:
                    rev_s, spread = float("nan"), float("nan")
                rows.append(dict(offset=offset, pulse=pulse, rev_s=rev_s,
                                 wraps=len(wrap_at), spread=spread,
                                 moving=moving, span=span))
                log("{:>8.1f} {:>7} {:>11} {:>7.0f}% {:>7} {:>7}".format(
                    offset, pulse,
                    "n/a" if rev_s != rev_s else "{:.2f}".format(rev_s),
                    100 * moving, len(wrap_at),
                    "n/a" if spread != spread else "{:.0%}".format(spread)))
    finally:
        command(0.0)

    # A speed counts only if whole revolutions were seen and they were of
    # consistent length. An erratic spread means the wrap detector caught
    # something that was not a revolution.
    clean = [r for r in rows if r["rev_s"] == r["rev_s"] and r["wraps"] >= 2
             and r["spread"] <= args.max_spread]
    if not clean:
        return None

    # The deadband is the largest offset that did NOT turn the wheel -- so it
    # only exists if the sweep actually bracketed it. When the smallest offset
    # tested already produced a clean speed, the deadband is somewhere below it
    # and was not measured; reporting the lowest tested value would be quoting
    # the sweep's own lower bound back as a property of the servo. It happened:
    # a sweep starting at 4 deg found the wheel already turning at 0.29 rev/s
    # there and duly emitted `deadband_deg: 4.0`.
    slowest_tested = min(abs(float(o)) for o in args.offsets)

    # PER DIRECTION -- but NOT because these servos are directionally
    # asymmetric. They are not: measured on a Kondo humanoid's wheel servo 21 at 6 deg, forward
    # ran 0.17 rev/s when tried first and 0.33 when tried second, and reverse
    # ran 0.17 or 0.50 depending on the same thing. Near the deadband the speed
    # is not repeatable at all, and since this sweep always tries +offset
    # before -offset, that irreproducibility reads as a clean "dead forward,
    # alive in reverse" story. It is an artefact of the order.
    #
    # What the split is really for is to notice when the two disagree, which is
    # the signal that the offset is in the irreproducible region and the number
    # should not be trusted -- see --repeats, which is the actual test.
    def _deadband(rows_in_direction):
        if not rows_in_direction:
            return None
        smallest = min(abs(r["offset"]) for r in rows_in_direction)
        return None if smallest == slowest_tested else smallest

    forward_dead = _deadband([r for r in clean if r["offset"] > 0])
    reverse_dead = _deadband([r for r in clean if r["offset"] < 0])
    # The larger of the two is what the model must use. Not because the smaller
    # direction is wrong, but because a command down there produces a speed
    # nobody can predict -- and on a mecanum base an unpredictable wheel is an
    # unusable one, whichever way it happens to turn.
    if forward_dead is None or reverse_dead is None:
        deadband = None
    else:
        deadband = max(forward_dead, reverse_dead)
    top = max(abs(r["rev_s"]) for r in clean)
    # Saturation begins at the smallest offset already within 5 % of the top
    # speed: past it, more command buys nothing and a linear map would lie.
    saturate = min(abs(r["offset"]) for r in clean if abs(r["rev_s"]) >= 0.95 * top)
    low = slowest_tested if deadband is None else deadband
    band = [r for r in clean if low <= abs(r["offset"]) <= saturate]
    slope = (np.polyfit([r["offset"] for r in band], [r["rev_s"] for r in band], 1)[0]
             if len(band) >= 2 else float("nan"))
    # How well a straight line actually describes the band, as the worst
    # deviation in rev/s. A slope quoted without this is a claim nobody can
    # check, and "is this servo proportional at all" is exactly the question
    # this tool exists to answer.
    if len(band) >= 2:
        fit = np.polyfit([r["offset"] for r in band],
                         [r["rev_s"] for r in band], 1)
        residual = float(max(
            abs(r["rev_s"] - (fit[0] * r["offset"] + fit[1])) for r in band))
    else:
        residual = float("nan")
    forward = [r["rev_s"] for r in clean if r["offset"] > 0]
    reverse = [r["rev_s"] for r in clean if r["offset"] < 0]
    # Both tops are reported because they are not the same servo behaviour: a
    # wheel that runs faster one way is a real asymmetry a mecanum base will
    # steer with.
    return dict(servo_id=servo_id, deadband_deg=deadband,
                forward_deadband=forward_dead, reverse_deadband=reverse_dead,
                slowest_tested=slowest_tested, residual=residual,
                saturate_deg=saturate, top_rev_s=top, slope=float(slope),
                forward_top=max(forward) if forward else float("nan"),
                reverse_top=min(reverse) if reverse else float("nan"),
                rows=rows)


def emit_yaml(results, args):
    """The measured numbers, as something a robot config can include."""
    lines = [
        "# Measured by actuo-wheel-calibrate. These are THIS robot's servos:",
        "# repeat the run per robot rather than copying the block, which is the",
        "# whole reason it is emitted instead of printed.",
        "#",
        "#   deadband_deg  below this the wheel does not turn at all",
        "#   saturate_deg  above this more command buys no more speed",
        "#   rev_s_per_command_deg   wheel revolutions/s per command deg,",
        "#                           fitted BETWEEN those two",
        "#   Speeds are the WHEEL's, from the time between sawtooth wraps: one",
        "#   wrap is one revolution, so no pot scale is assumed. Roughly a",
        "#   sixth of each turn is not reported at all (a blind sector), which",
        "#   is why a slope through the raw reading would not be a wheel speed.",
        "#",
        "# Only the band between them is proportional. Mapping a controller's",
        "# rad/s across the whole range with one gain would promise speeds the",
        "# servo cannot reach and creep where it cannot move.",
        "wheels:",
    ]
    for r in results:
        lines.append("  - servo_id: {}".format(r["servo_id"]))
        if r["deadband_deg"] is None:
            # NOT a number, because the sweep never saw this wheel stationary.
            # Emitting the lowest offset tested would quote the sweep's own
            # lower bound back as a property of the servo, and the config that
            # read it would refuse every command below a threshold nobody
            # measured.
            lines += [
                "    # NO deadband_deg: the wheel already turned at the",
                "    # smallest offset swept ({:.1f} deg), so the deadband is".format(
                    r["slowest_tested"]),
                "    # below it and was never measured. Re-run with smaller",
                "    # --offsets to bracket it. The config REQUIRES this key,",
                "    # so this block is deliberately incomplete rather than",
                "    # quietly filled in with the sweep's lower bound.",
            ]
        else:
            if r["forward_deadband"] != r["reverse_deadband"]:
                lines += [
                    "    # The two directions disagreed: {:.1f} deg forward vs",
                    "    # {:.1f} deg reverse. That is NOT a directional",
                    "    # property of the servo -- near the deadband its speed",
                    "    # is not repeatable, and this sweep always tries",
                    "    # forward first, so the irreproducibility reads as an",
                    "    # asymmetry. The larger is used: below it the speed is",
                    "    # unpredictable, and an unpredictable wheel is an",
                    "    # unusable one on a mecanum base.",
                ]
                lines[-8] = lines[-8].format(r["forward_deadband"])
                lines[-7] = lines[-7].format(r["reverse_deadband"])
            lines.append(
                "    deadband_deg: {:.1f}".format(r["deadband_deg"]))
        lines += [
            "    saturate_deg: {:.1f}".format(r["saturate_deg"]),
            "    rev_s_per_command_deg: {:.4f}".format(r["slope"]),
            "    top_rev_s: {:.3f}".format(r["top_rev_s"]),
            "    forward_top_rev_s: {:.3f}".format(r["forward_top"]),
            "    reverse_top_rev_s: {:.3f}".format(r["reverse_top"]),
            "    # worst deviation from the fitted line: {:.3f} rev/s".format(
                r["residual"]),
        ]
    lines.append("")
    return "\n".join(lines)


def identify(board, args, log):
    """Turn one wheel at a time and let the operator say which corner it is.

    Nothing on the bus knows where a servo is bolted, and the mecanum kinematics
    are entirely about where: swap front-left with front-right and the base
    strafes when told to drive forward, at full speed and with every wheel doing
    exactly what it was asked. So the mapping is *observed*, once, and written
    into the robot's config -- it is not derivable and it must not be guessed.
    """
    offset = float(args.identify)
    log("\nEach wheel will turn for {:.1f}s at {:+.1f} command deg. Watch which "
        "corner moves, and which WAY.".format(args.seconds, offset))
    mapping = {}
    for servo_id in args.ids:
        log("\n=== servo {} -- turning now ===".format(servo_id))
        pulse = int(round(CENTER_PULSE + offset * PULSE_PER_DEG))
        board.command(servo_id, pulse, args.move_byte)
        time.sleep(args.seconds)
        board.command(servo_id, CENTER_PULSE, args.move_byte)
        answer = input("  which corner turned? "
                       "[fl / fr / rl / rr, or blank if nothing moved] ").strip().lower()
        if answer in ("fl", "fr", "rl", "rr"):
            mapping[answer] = servo_id
        else:
            log("  servo {} not placed.".format(servo_id))

    duplicates = len(set(mapping.values())) != len(mapping)
    log("\n--- observed mapping ---")
    for corner in ("fl", "fr", "rl", "rr"):
        log("  {}: {}".format(
            corner, mapping.get(corner, "NOT IDENTIFIED")))
    if len(mapping) < 4 or duplicates:
        log("\nIncomplete. Do NOT write a partial mapping into the config: a "
            "corner nobody watched is a corner that will be wrong, and it will "
            "be wrong as a confident, full-speed motion in the wrong direction.")
        return 1
    log("\nPut these ids on the matching mecanum joints in the robot's config, "
        "and check the SIGN too: a wheel that turned backwards needs "
        "direction: -1 there.")
    return 0


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.board == "h7":
        if not args.symbols:
            print("--board h7 needs --symbols (the firmware symbol map).")
            return 2
    elif not args.rotation_checked:
        print("--board rcb4mini needs --rotation-checked: the RCB-4mini cannot "
              "read whether a servo is in rotation mode, and these offsets "
              "would swing a position servo. Read the flag with the servo on "
              "the Dual USB Adapter in ICS mode (rcb4 ics.py, read_rotation) "
              "first, and say how and when here.")
        return 2
    printed = []

    def log(text):
        print(text, flush=True)
        printed.append(text)

    if not args.yes:
        answer = input("{} wheel(s) WILL SPIN, one at a time. Are they off "
                       "the ground? [y/N] ".format(len(args.ids)))
        if answer.strip().lower() not in ("y", "yes"):
            print("stopping; nothing was commanded.")
            return 1
    if args.board == "h7":
        board = H7Board(args.channel, args.symbols)
    else:
        board = Rcb4MiniBoard(args.channel, args.ids, args.ics_bitrate)
        log("rotation mode vouched for, not read: {}".format(
            args.rotation_checked))

    try:
        wrong = board.not_in_rotation_mode(args.ids)
        if wrong:
            print("servo(s) {} report rotation=0: they are POSITION servos, and "
                  "these offsets would be angles, not speeds. Refusing."
                  .format(", ".join(str(i) for i in wrong)))
            return 2

        if args.identify is not None:
            return identify(board, args, log)

        results = []
        for servo_id in args.ids:
            log("\n=== servo {} ===".format(servo_id))
            measured = calibrate_one(board, servo_id, args, log)
            if measured is None:
                log("  nothing produced a clean, linear rate. Either the wheel "
                    "did not turn or the reported angle does not track it -- "
                    "read the wraps and R^2 columns before trusting anything.")
                continue
            results.append(measured)
            if measured["deadband_deg"] is None:
                log("  DEADBAND NOT MEASURED: the wheel already turned at "
                    "{:.1f} deg, the smallest offset swept. Re-run with "
                    "smaller --offsets.".format(measured["slowest_tested"]))
            elif measured["forward_deadband"] != measured["reverse_deadband"]:
                log("  the directions disagreed ({:.1f} deg forward, {:.1f} deg "
                    "reverse), which means the speed down there is NOT "
                    "repeatable rather than that the servo is asymmetric. "
                    "Using {:.1f}.".format(measured["forward_deadband"],
                                           measured["reverse_deadband"],
                                           measured["deadband_deg"]))
            else:
                log("  deadband below {:.1f} deg,".format(
                    measured["deadband_deg"]))
            log("  saturates at {:.1f} deg, {:.4f} rev/s per command deg in "
                "between (worst deviation from that line {:.3f} rev/s), top "
                "{:.2f} rev/s ({:.0f} deg/s at the wheel)".format(
                    measured["saturate_deg"], measured["slope"],
                    measured["residual"], measured["top_rev_s"],
                    360 * measured["top_rev_s"]))

        if results:
            document = emit_yaml(results, args)
            print("\n" + document)
            if args.output:
                with open(args.output, "w") as handle:
                    handle.write(document)
                print("wrote {}".format(args.output))
        return 0
    finally:
        try:
            for servo_id in args.ids:
                board.command(servo_id, CENTER_PULSE, args.move_byte)
            print("\nall commanded wheels stopped.")
        except Exception as error:                          # noqa: BLE001
            print("\nCOULD NOT SEND THE STOP: {}. Kill power if anything is "
                  "still turning.".format(error))
        board.close(args.ids)


if __name__ == "__main__":
    raise SystemExit(main())
