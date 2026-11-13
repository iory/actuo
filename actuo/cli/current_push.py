#!/usr/bin/env python3
"""Hand-driven current push: YOU raise the torque, YOUR eyes find the stop.

The auto-ramp (``actuo-auto-limits``) tries to tell a hard stop from a gravity
balance by watching whether more torque still buys travel. On a joint whose
gravity torque climbs steeply that discrimination is fragile: a step that fails
to move the joint reads as "converged on a stop" when it only means "this much
torque balanced gravity here" -- observed on joint2, which the ramp called a
stop at 70 deg when the real stop is ~30 deg.

A human is better at this, but NOT because of the sound. Both cases stall the
motor -- rotor held, current flowing -- so both can buzz; the buzz alone does
not tell a hard stop from losing to gravity. What a human has that the auto-ramp
does not is two things:

* eyes -- a mechanical stop is the joint pressed against something solid (the
  frame, the table, itself); a gravity balance is the joint floating in mid-air.
  Look at which it is. This is the reliable signal, and it is visual.
* judgement about torque -- the auto-ramp climbs by a fixed factor and gives up
  when one step buys no travel, which a steep gravity curve defeats (it read
  joint2's 70 deg balance as a stop because 0.9 N*m could not beat gravity
  there). By hand you keep raising the torque -- 1.35, 2.0, 3.0 N*m -- until it
  is clear the joint will not move no matter what, and you can SEE it is against
  a hard object while it refuses.

So the loop is: raise the torque, and either the joint moves again (it was only
a balance -- keep going) or it stays pinned against something visibly solid even
as you add more (a real stop -- record it). The buzz is just "stalled"; your
eyes say whether it is stalled on a wall or on gravity.

* raise / lower the commanded torque a step at a time,
* record THIS end only when more torque will not move it AND it is visibly
  against something -- not merely because it stopped,
* the other end likewise, and it prints the ``position_limit_deg`` line.

It commands exactly ONE joint in CURRENT mode; the others are never sent a
frame. Zero torque is limp -- the joint hangs. This is a bare current source
with no position hold, on an arm with no self-collision checking: keep a hand on
the power, and do not hold a stall any longer than it takes to hear it (a
pinned motor heats fast).

    actuo-current-push --config <robot>.yaml --joint joint2
    # (run it in a REAL terminal -- it reads single keypresses)

Keys
----
    w / up        torque up one step        s / down    torque down one step
    space         torque to zero (limp)     d           flip push direction
    [             record LOW  end here       ]          record HIGH end here
    + / -         bigger / smaller step      q / Ctrl-C  quit (releases)

Guards (each forces the joint limp and warns, keeping you in control):
    temperature, feedback staleness, a speed cap (CURRENT mode has no speed
    envelope of its own), and a travel backstop.
"""

from __future__ import annotations

import argparse
import math
import os
import select
import sys
import termios
import time
import tty


from actuo.core.contracts import ActuatorCommand, OperatingMode  # noqa: E402
from actuo.drivers.actuators.cubemars import ERROR_NAMES  # noqa: E402
from actuo.drivers.buses.can_bus import (  # noqa: E402
    StaleFeedbackError)
from actuo.drivers.config import build_bus  # noqa: E402


# Arrow keys arrive as the three bytes ESC '[' 'A'/'B'; map the final byte.
_ARROWS = {"A": "up", "B": "down", "C": "right", "D": "left"}


def _read_keys(timeout=0.0):
    """Return the list of keypresses waiting on stdin, non-blocking.

    Single characters come back verbatim; the up/down arrows are folded to
    ``'up'``/``'down'``. Reads whatever is buffered so a burst of presses is not
    lost between cycles.

    Parameters
    ----------
    timeout : float
        Seconds to wait for the first byte. ``0`` polls and returns at once.

    Returns
    -------
    list of str
        Zero or more key tokens.
    """
    if not select.select([sys.stdin], [], [], timeout)[0]:
        return []
    data = os.read(sys.stdin.fileno(), 64).decode("latin-1")
    keys = []
    i = 0
    while i < len(data):
        ch = data[i]
        if ch == "\x1b" and data[i + 1:i + 2] == "[" and i + 2 < len(data):
            keys.append(_ARROWS.get(data[i + 2], "esc"))
            i += 3
            continue
        keys.append(ch)
        i += 1
    return keys


def _fmt_recorded(recorded):
    """One-line summary of the ends captured so far."""
    parts = []
    for end in ("low", "high"):
        if end in recorded:
            parts.append("{}={:.2f}".format(end, recorded[end]))
        else:
            parts.append("{}=--".format(end))
    return "  ".join(parts)


def run(bus, index, name, actuator, args):
    """Interactive loop for one joint. Returns the ``{'low', 'high'}`` recorded.

    Streams the operator's commanded current at ``args.rate`` Hz, reads feedback
    every cycle, and applies the safety guards. Never raises for an operator
    quit; re-raises a stale-feedback fault, which means a motor went silent and
    is not something to push through.
    """
    period = 1.0 / args.rate
    torque = 0.0
    direction = -1                      # -1 drives toward LOW (smaller angle)
    step = args.step
    recorded = {}

    bus.poll(timeout=0.0)
    first = bus.last_state(actuator.id)
    if first is None:
        raise RuntimeError("{} has never reported".format(actuator.id))
    start_deg = math.degrees(first.position)
    prev_deg = start_deg
    prev_t = time.monotonic()
    speed = 0.0
    next_send = 0.0
    next_print = 0.0
    tripped = ""

    sys.stdout.write(
        "\n[current-push] {} (id {}). w/s torque, d flip, space limp, "
        "[ ] record low/high, q quit.\n".format(name, actuator.id))
    sys.stdout.flush()

    while True:
        now = time.monotonic()

        for key in _read_keys():
            if key in ("q", "\x03", "\x04"):      # q, Ctrl-C, Ctrl-D
                return recorded
            elif key in ("w", "k", "up"):
                torque = min(args.max_torque, round(torque + step, 4))
                tripped = ""
            elif key in ("s", "j", "down"):
                torque = max(0.0, round(torque - step, 4))
            elif key == " ":
                torque = 0.0
            elif key == "d":
                direction = -direction
            elif key in ("+", "="):
                step = min(1.0, round(step * 2.0, 4))
            elif key in ("-", "_"):
                step = max(0.01, round(step / 2.0, 4))
            elif key == "[":
                recorded["low"] = math.degrees(
                    bus.last_state(actuator.id).position)
                sys.stdout.write("\n[current-push] LOW  recorded at "
                                 "{:.2f} deg\n".format(recorded["low"]))
            elif key == "]":
                recorded["high"] = math.degrees(
                    bus.last_state(actuator.id).position)
                sys.stdout.write("\n[current-push] HIGH recorded at "
                                 "{:.2f} deg\n".format(recorded["high"]))

        if now >= next_send:
            if torque > 0.0:
                batch = [None] * len(bus.actuators)
                batch[index] = ActuatorCommand(
                    mode=OperatingMode.CURRENT, current=direction * torque)
                bus.write_batch(batch)
            else:
                bus.release_all()          # zero torque == limp, honestly
            next_send = now + period

        bus.poll(timeout=0.005)
        try:
            bus.check_fresh()
        except StaleFeedbackError:
            bus.release_all()
            raise

        state = bus.last_state(actuator.id)
        if state is None:
            bus.release_all()
            raise RuntimeError("{} stopped reporting".format(actuator.id))

        deg = math.degrees(state.position)
        dt = now - prev_t
        if dt > 0.0:
            # Speed from the position stream, so it needs no pole_pairs and
            # cannot inherit a bad one. Light smoothing keeps quantisation
            # (0.1 deg steps) from reading as spikes.
            inst = (deg - prev_deg) / dt
            speed = 0.7 * speed + 0.3 * inst
        prev_deg, prev_t = deg, now

        # --- guards: each drops the joint limp and stays in the loop ----------
        if state.error:
            torque = 0.0
            tripped = "FAULT {}".format(
                ERROR_NAMES.get(state.error, state.error))
        elif state.temperature is not None and state.temperature > args.max_temp:
            torque = 0.0
            tripped = "HOT {:.0f}C".format(state.temperature)
        elif abs(speed) > args.max_speed:
            torque = 0.0
            tripped = "OVERSPEED {:.0f} deg/s -- cut".format(speed)
        elif abs(deg - start_deg) > args.sanity_deg:
            torque = 0.0
            tripped = "TRAVEL {:.0f} deg past start -- cut".format(
                deg - start_deg)

        if now >= next_print:
            amps = (state.aux or {}).get("current_a")
            arrow = "LOW <-" if direction < 0 else "-> HIGH"
            sys.stdout.write(
                "\r{:5.2f} N*m {:>7}  pos {:8.2f} deg  {:6.1f} deg/s  "
                "I={:5.2f}A  T={:2.0f}C  step {:.2f}  [{}]  {:<28}".format(
                    torque, arrow, deg, speed, amps or 0.0,
                    state.temperature or 0.0, step, _fmt_recorded(recorded),
                    tripped))
            sys.stdout.flush()
            next_print = now + 0.1


def main(argv=None):
    p = argparse.ArgumentParser(
        description="hand-driven current push; you raise torque and judge the "
                    "stop by eye (pressed vs floating), not by sound")
    p.add_argument("--config", required=True)
    p.add_argument("--joint", required=True, help="ONE joint; this moves it")
    p.add_argument("--step", type=float, default=0.1,
                   help="torque increment per keypress, N*m (default 0.1)")
    p.add_argument("--max-torque", type=float, default=5.0,
                   help="the torque keys will not go above, N*m (default 5.0)")
    p.add_argument("--max-speed", type=float, default=60.0,
                   help="cut torque if the joint exceeds this, deg/s "
                        "(default 60). CURRENT mode has no speed envelope, so "
                        "this is the runaway backstop.")
    p.add_argument("--sanity-deg", type=float, default=300.0,
                   help="cut torque after this much travel from the start pose")
    p.add_argument("--max-temp", type=float, default=70.0)
    p.add_argument("--rate", type=float, default=50.0)
    p.add_argument("--margin-deg", type=float, default=2.0,
                   help="shrink each recorded end inward by this for the "
                        "suggested limit line")
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
    if args.joint not in names:
        print("no such joint {!r}; this bus has: {}".format(
            args.joint, ", ".join(names)), file=sys.stderr)
        return 2
    index = names.index(args.joint)
    actuator = bus.actuators[index]
    try:
        actuator.capabilities.require(OperatingMode.CURRENT)
    except Exception as err:                                   # noqa: BLE001
        print("{}: {}".format(args.joint, err), file=sys.stderr)
        return 2

    if not sys.stdin.isatty():
        print("current-push reads single keypresses and needs a real terminal; "
              "run it directly, not through a pipe or a `!` wrapper.",
              file=sys.stderr)
        return 2

    print("[current-push] {} (id {}) in CURRENT mode. THIS MOVES THE ROBOT."
          .format(args.joint, actuator.id))
    print("[current-push] Zero torque is LIMP -- the joint hangs. No self-"
          "collision checking here; keep a hand on the power.")
    print("[current-push] Both a hard stop and losing to gravity STALL the "
          "motor (it may buzz either way). Tell them apart by EYE: a stop is "
          "pressed against something solid; a balance floats. Raise torque -- "
          "if it moves again it was a balance; record only a pinned end.")

    bus.transport.open()
    old = termios.tcgetattr(sys.stdin.fileno())
    recorded = {}
    try:
        bus.read_batch(timeout=5.0)
        tty.setcbreak(sys.stdin.fileno())     # single keys, Ctrl-C still works
        recorded = run(bus, index, name=args.joint, actuator=actuator,
                       args=args)
    except (StaleFeedbackError, RuntimeError) as err:
        print("\n[current-push] STOPPED: {}".format(err), file=sys.stderr)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old)
        try:
            bus.release_all()
        except Exception:                                     # noqa: BLE001
            pass
        bus.transport.close()
        print("\n[current-push] released; the joint is limp")

    print("\n" + "=" * 68)
    if set(recorded) == {"low", "high"}:
        low = recorded["low"] + args.margin_deg
        high = recorded["high"] - args.margin_deg
        if low < high:
            print("Paste into {}'s `actuator:` block (already shrunk {:.1f} deg "
                  "inward):".format(args.joint, args.margin_deg))
            print("      position_limit_deg: [{:.2f}, {:.2f}]".format(low, high))
        else:
            print("recorded low {:.2f} >= high {:.2f} after margin -- check "
                  "which end is which".format(low, high))
    else:
        print("recorded {} -- both ends are needed for a limit line".format(
            _fmt_recorded(recorded)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
