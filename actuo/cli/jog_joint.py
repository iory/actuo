#!/usr/bin/env python3
"""Move ONE joint a small amount inside its taught limits, then put it back.

The first-motion tool. It commands exactly one joint; the others are never sent
anything, so they stay as they are. It keeps the safety properties that
first-motion testing on CubeMars servos showed to be necessary: an operator
confirmation, a feedback refresh after that prompt, and per-cycle checks on
fault code, temperature and feedback age.

    actuo-jog-joint --config <robot>.yaml --joint joint1
    # macOS: --interface gs_usb --channel 0

Nothing moves until you type MOVE. Use --dry-run to see the plan only.

NOTE: CubeMars servo mode has no "release" command. After a move the driver
keeps holding its last target. Cut power when you
want the joint free again.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time


from actuo.core.contracts import ActuatorCommand, OperatingMode  # noqa: E402
from actuo.drivers.actuators.cubemars import ERROR_NAMES  # noqa: E402
from actuo.drivers.buses.can_bus import CanBus, StaleFeedbackError  # noqa: E402
from actuo.drivers.config import build_bus  # noqa: E402

MAX_TEMPERATURE_C = 80.0   # driver-board abort threshold, deg C


def check(bus, name, actuator):
    """Per-cycle safety gate. Raises on anything unhealthy."""
    bus.check_fresh()                      # feedback age
    state = bus.last_state(actuator.id)
    if state.error:
        raise RuntimeError("{} faulted: {}".format(
            name, ERROR_NAMES.get(state.error, "code {}".format(state.error))))
    if state.temperature >= MAX_TEMPERATURE_C:
        raise RuntimeError("{} is at {:.0f} C".format(name, state.temperature))
    return state


def refresh_after_prompt(bus, timeout=3.0):
    """input() blocks CAN reception, so the staleness watchdog must restart from
    fresh data rather than instantly tripping on frames buffered before the
    operator answered."""
    deadline = time.monotonic() + timeout
    seen = 0
    while time.monotonic() < deadline and seen < 3:
        seen += bus.poll(timeout=0.1)
    bus.check_fresh()


def hold(bus, actuator, target, seconds, rate, name, tolerance_deg, settle=0.3):
    """Stream one target at ``rate`` Hz until it is reached and settled."""
    command = [ActuatorCommand(mode=OperatingMode.POSITION, position=target)]
    period = 1.0 / rate
    deadline = time.monotonic() + seconds
    next_send = 0.0
    in_tolerance_since = None
    next_print = 0.0

    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_send:
            bus.write_batch(command)       # single-actuator bus: only this joint
            next_send = now + period
        bus.poll(timeout=0.005)
        state = check(bus, name, actuator)

        error_deg = abs(math.degrees(state.position - target))
        if error_deg <= tolerance_deg:
            if in_tolerance_since is None:
                in_tolerance_since = now
            elif now - in_tolerance_since >= settle:
                print("    reached {:.2f} deg (err {:.2f} deg)".format(
                    math.degrees(state.position), error_deg))
                return True
        else:
            in_tolerance_since = None

        if now >= next_print:
            print("    at {:7.2f} deg  target {:7.2f}  err {:5.2f}  I={:5.2f}A  "
                  "T={:.0f}C".format(math.degrees(state.position),
                                     math.degrees(target), error_deg,
                                     state.aux.get("current_a", float("nan")),
                                     state.temperature), end="\r", flush=True)
            next_print = now + 0.2
    print()
    raise TimeoutError("{} did not reach {:.2f} deg in {:.0f}s".format(
        name, math.degrees(target), seconds))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Jog one joint inside its taught limits")
    parser.add_argument("--config", required=True)
    parser.add_argument("--joint", required=True, help="joint name from the config")
    parser.add_argument("--delta-deg", type=float, default=5.0,
                        help="how far to move from the current pose (default 5)")
    parser.add_argument("--tolerance-deg", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--rate", type=float, default=20.0,
                        help="command rate, Hz (default 20)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and exit without commanding")
    parser.add_argument("--yes", action="store_true", help="skip the MOVE prompt")
    parser.add_argument("--interface", default=None)
    parser.add_argument("--channel", default=None)
    parser.add_argument("--bitrate", type=int, default=None)
    args = parser.parse_args(argv)

    overrides = {}
    if args.interface is not None:
        overrides["interface"] = args.interface
    if args.channel is not None:
        try:
            overrides["channel"] = int(args.channel)
        except ValueError:
            overrides["channel"] = args.channel
    if args.bitrate is not None:
        overrides["bitrate"] = args.bitrate

    full_bus, names = build_bus(args.config, transport_overrides=overrides or None)
    if args.joint not in names:
        print("unknown joint {!r}; config has {}".format(args.joint, names),
              file=sys.stderr)
        return 2
    actuator = full_bus.actuators[names.index(args.joint)]

    limit = actuator.calibration.position_limit
    if limit is None:
        print("{} has no position_limit_deg in the config. Teach it first with "
              "actuo-teach-limits -- refusing to move a joint with no taught "
              "envelope.".format(args.joint), file=sys.stderr)
        return 2
    low, high = limit

    # A bus holding ONLY this actuator: write_batch then physically cannot emit a
    # frame for any other motor. The others are left exactly as they are rather
    # than being energised to hold a pose nobody asked for.
    bus = CanBus(full_bus.transport, [actuator], stale_after=full_bus.stale_after)
    bus.transport.open()
    try:
        bus.read_batch(timeout=5.0)
        state = check(bus, args.joint, actuator)
        start = state.position

        print("[jog] joint      : {} (id {})".format(args.joint, actuator.id))
        print("[jog] taught limit: {:.2f} .. {:.2f} deg".format(
            math.degrees(low), math.degrees(high)))
        print("[jog] current     : {:.2f} deg, T={:.0f}C, err={}".format(
            math.degrees(start), state.temperature,
            ERROR_NAMES.get(state.error, state.error)))

        # The taught extremes were shrunk inward by a margin, so the joint can
        # legitimately be sitting just outside its own limit right now. Say so
        # instead of failing obscurely inside the encoder later.
        if not low <= start <= high:
            nearest = low if start < low else high
            print("\n[jog] REFUSING: {} is at {:.2f} deg, outside its taught "
                  "limit [{:.2f}, {:.2f}].".format(
                      args.joint, math.degrees(start), math.degrees(low),
                      math.degrees(high)), file=sys.stderr)
            print("[jog] Hand-move it to about {:.2f} deg (inside the range) and "
                  "retry. Driving it in from outside is a move this tool will "
                  "not make for you.".format(math.degrees(nearest)),
                  file=sys.stderr)
            return 1

        # Prefer the direction with room; clamp so the target is always legal.
        delta = math.radians(abs(args.delta_deg))
        if start + delta <= high:
            target = start + delta
        elif start - delta >= low:
            target = start - delta
        else:
            target = high if (high - start) > (start - low) else low
        if not low <= target <= high:
            target = min(max(target, low), high)

        moved = math.degrees(abs(target - start))
        print("[jog] plan        : {:.2f} -> {:.2f} deg ({:.2f} deg move), then "
              "back to {:.2f}".format(math.degrees(start), math.degrees(target),
                                      moved, math.degrees(start)))
        print("[jog] envelope    : {:.0f} ERPM / {:.0f} ERPM/s (from the config)"
              .format(actuator.limits.max_velocity_erpm,
                      actuator.limits.max_acceleration_erpm_s))
        if moved < 0.5:
            print("[jog] no room to move inside the limits; nothing to do.")
            return 0

        if args.dry_run:
            print("\n[jog] dry run: no CAN command transmitted.")
            return 0

        if not args.yes:
            answer = input("\n[jog] Clear the mechanism and support it, then "
                           "type MOVE: ").strip()
            if answer != "MOVE":
                print("[jog] cancelled. Nothing was sent.")
                return 0
            refresh_after_prompt(bus)

        print("\n[jog] moving to {:.2f} deg".format(math.degrees(target)))
        hold(bus, actuator, target, args.timeout, args.rate, args.joint,
             args.tolerance_deg)

        print("[jog] returning to {:.2f} deg".format(math.degrees(start)))
        hold(bus, actuator, start, args.timeout, args.rate, args.joint,
             args.tolerance_deg)

        print("\n[jog] done. NOTE: the driver keeps holding this target -- cut "
              "power to free the joint.")
        return 0

    except KeyboardInterrupt:
        print("\n[jog] interrupted; stopped commanding.", file=sys.stderr)
        return 130
    except (RuntimeError, TimeoutError, ValueError, StaleFeedbackError) as err:
        print("\n[jog] ABORT: {}".format(err), file=sys.stderr)
        # Say WHY the bus went quiet: silence and an error storm are different
        # faults, and reporting neither is what made this hard to diagnose.
        if bus.error_frames:
            print("[jog] saw {} bus ERROR frames -- the adapter's bit timing or "
                  "the wiring is wrong, not the node id.".format(bus.error_frames),
                  file=sys.stderr)
        elif bus.unmatched_frames:
            print("[jog] saw {} data frames from other ids: {}".format(
                bus.unmatched_frames,
                sorted({a & 0xFF for a in bus.unmatched_addresses})),
                file=sys.stderr)
        else:
            print("[jog] the bus was completely silent.", file=sys.stderr)
        print("[jog] stopped commanding. Use the power cutoff if motion is unsafe.",
              file=sys.stderr)
        return 1
    finally:
        bus.transport.close()


if __name__ == "__main__":
    raise SystemExit(main())
