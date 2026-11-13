#!/usr/bin/env python3
"""Drive a joint into its mechanical stop, on purpose, and MEASURE the result.

    actuo-impact-test --config <robot>.yaml --shm-name <name> --joint joint3 --overtravel-deg 15 --speed 300 --repeats 5

A structural test: hit the hard stop at a known speed, repeatedly, and watch
whether the structure moves. **This tool exists to do the one thing every other
tool in this package refuses to do**, so read the three sections below before
running it.

What it measures, and why that and not "did it break"
-----------------------------------------------------
The headline number is the **contact angle**: where the joint comes to rest
while being pushed into the stop at a repeatable torque. That is the right
metric, and "it did not break" is not a metric:

* It is gravity-independent. The joint is held against the stop by the motor,
  not by where a limp arm happens to hang, so it is repeatable between runs and
  between poses.
* A fastener backing out, a bracket yielding, a stop insert crushing, a spline
  fretting -- all of them move the contact angle, and they move it long before
  anything visibly fails. **Drift across repeats is the signal.**
* It is measured under a *known* load, because the driver saturates its own
  current. Peak current is logged next to it, so a contact angle that drifted
  while the current also changed can be told apart from one that drifted on its
  own.

Every impact also records approach speed, peak current, peak temperature, and
the settled rest angle before and after, into a CSV.

The three hazards, all specific to CubeMars AK actuators on CAN
---------------------------------------------------------------
1. **Losing the bus during the push is the dangerous failure, not the impact.**
   ``SET_POS_SPD`` is a target the driver holds: if CAN goes down mid-push, the
   motor keeps driving into the stop at whatever current it takes, indefinitely,
   and nothing on the host can command it to stop. **The motor power switch
   is the only remedy -- be in reach of it.** This is not hypothetical: on a
   CubeMars AK arm the bus has dropped under high motor current, and an impact
   draws more current than anything else the arm does. The tool watches for staleness
   and error frames and releases, but a release frame cannot be delivered over a
   bus that is already gone.

2. **A stall is the worst thermal case there is** -- maximum current with no
   rotation, hence no fan effect and no back-EMF relief -- and **the temperature
   this tool can read is the DRIVER BOARD's, not the winding's**. A CubeMars AK
   uploads only the board sensor over CAN; the winding sensor exists (it can
   raise fault code 1) but is UART-only. So the number in the log stays
   comfortable while the part that actually cooks is unobservable. That is why
   ``--rest-seconds`` defaults to 20 and why ``--hold`` is bounded: the tool
   cannot see the thing it is protecting.

3. **The taught envelope is widened IN MEMORY ONLY, never in a file.** Editing
   ``position_limit_deg`` in the robot config for a test is a trap: an envelope
   that outlives the test fences a part of the mechanism nobody taught, and the
   next process to read the file inherits it silently. The override here dies
   with the process.

What it will not do
-------------------
* It will not run while the bus servicer holds the lock -- two writers on one
  bus means whichever frame lands last wins, which during an impact test is not
  something to leave to chance.
* It will not start from outside the taught envelope, or with a joint that has
  never reported, or with a live driver fault.
* It will not push for longer than ``--hold`` after contact, and it releases in
  a ``finally``: an exception, a Ctrl-C and a normal exit all end with zero
  current.
"""

import argparse
import csv
import dataclasses
import math
import os
import sys
import time


from actuo.core.contracts import ActuatorCommand, OperatingMode      # noqa: E402
from actuo.drivers.config import build_bus                           # noqa: E402
from actuo.drivers.actuators.cubemars import ERROR_NAMES             # noqa: E402
from actuo.drivers import window_state                              # noqa: E402
from actuo.drivers.shm import lock_holder, lock_path                 # noqa: E402


def _drive_home(bus, actuator, calibration, home_deg, approach_speed, rate,
                log, phase, temp_limit, here_deg, contact_speed=6.0,
                contact_samples=4):
    """Drive gently back to the home angle, then release.

    Watches for a stall exactly as the impact does, and this is not symmetry for
    its own sake: THE RETURN CAN HIT SOMETHING TOO. In one test an AK80-6
    joint driven home from -84 deg met an obstruction at -10 deg and -- with no
    detector on the return move -- stalled there at 40 A until the CAN bus
    dropped. The impact itself had been the safe part.

    Returns (reached, stall_deg, stall_current, abort). `reached` False with a
    stall_deg means the joint met something; False with an `abort` means the bus
    stopped reporting and the move was cut, which is recoverable and NOT the
    same thing.
    """
    if abs(here_deg - home_deg) < 1.0:
        return True, None, None, None
    fast = actuator.limits
    slow = calibration.rad_s_to_erpm(math.radians(approach_speed))
    actuator.limits = dataclasses.replace(
        fast, max_velocity_erpm=slow, max_acceleration_erpm_s=slow)
    # The home move gets a GENEROUS arrival tolerance and knows its target.
    # It is a repositioning move, not a measurement, and a position loop that
    # settles 2.7 deg out has arrived for every purpose this move has. With a
    # 0.5 deg tolerance, settling anywhere outside that read as a contact -- at
    # 0.1 A -- and ended a ten-cycle endurance run on its first cycle.
    # CURRENT ONLY on the way home. What this move needs to catch is the joint
    # meeting something solid, which shows as saturation current. The
    # speed-based rules cannot help here and actively hurt: a position loop
    # settling onto its target decelerates over more than the arrival tolerance,
    # so "it slowed down near home" fires every single time. Measured: a climb
    # stopped 8.8 deg from home drawing 9.2 A -- a third of what that joint
    # pulls just climbing -- and was called a contact, ending an eight-cycle run
    # on cycle one.
    detector = _ContactDetector(contact_speed, contact_samples,
                                min_peak=max(10.0, contact_speed * 2.0),
                                target_deg=home_deg, arrive_tol=5.0,
                                use_collapse=False)
    arrived = [False]

    def watch(row):
        if abs(row["position_deg"] - home_deg) < 5.0:
            arrived[0] = True
            return True
        return detector(row)

    try:
        samples, abort = _stream(
            bus, actuator, home_deg, rate,
            max(2.0, abs(here_deg - home_deg) / approach_speed * 3.0 + 1.0),
            log, phase, temp_limit, on_sample=watch)
    finally:
        actuator.limits = fast
        _release(bus)
    if arrived[0]:
        return True, None, None, None
    if detector.contact is not None:
        current = samples[-1].get("current_a") if samples else None
        return False, detector.contact, current, None
    return False, None, None, abort


class _ContactDetector:
    """Decides, sample by sample, that the joint has stopped moving.

    Online, and as EARLY as the evidence allows. The whole value of an impact
    test is the moment of impact; every millisecond of pushing after it is pure
    cost -- heat in a winding this tool cannot see, and current on a bus that
    has been seen to drop under it. So the rule is two-tier:

    * **One sample is enough when the current says so.** A joint drawing many
      times its approach current while not turning is stopped; there is nothing
      to confirm. The threshold is self-calibrating -- a multiple of what the
      SAME joint drew while moving freely a moment earlier -- so it needs no
      per-motor rating and works on a joint whose datasheet nobody has read.
    * `contact_samples` consecutive slow samples otherwise, for a driver that
      reports no current, or a gentle stop that never spikes.

    Measured on a CubeMars AK80-6 arm joint: approach 0.5-1.4 A, impact 39.4 A, and the
    four-sample rule could not complete because the bus was already saturating
    -- 0.42 s at 40 A followed by a dead bus. The current rule fires on the
    first stopped sample instead.

    Contact requires the joint to have MOVED first, so a joint already resting
    against its stop reads as "never moved" rather than as an instant contact
    at its starting angle.
    """

    def __init__(self, contact_speed, needed, min_peak, current_ratio=3.0,
                 current_floor=3.0, target_deg=None, arrive_tol=2.0,
                 use_collapse=True):
        self.contact_speed = float(contact_speed)
        self.needed = int(needed)
        self.min_peak = float(min_peak)
        self.current_ratio = float(current_ratio)
        self.current_floor = float(current_floor)
        self.moved = False
        self.still = []
        self.peak = 0.0
        self.contact = None
        self.reason = None
        self.free_current = 0.0
        # ARRIVING is not hitting anything, and the difference is invisible in
        # position and speed alone -- both go still either way. Measured on an
        # AK80-6 joint: a sweep to -116.00 reported "contact" at -115.75
        # drawing 0.54 A, i.e. it simply got there. Reporting that as a contact
        # angle would have put a number in the drift column that tracks the
        # COMMAND, not the mechanism, and it would have been beautifully
        # repeatable -- the worst kind of wrong.
        self.free_currents = []
        self.previous_speed = 0.0
        self.start_deg = None
        self.use_collapse = bool(use_collapse)
        self.target_deg = target_deg
        self.arrive_tol = float(arrive_tol)
        self.arrived = False

    def __call__(self, row):
        speed = abs(row["velocity_dps"])
        self.peak = max(self.peak, speed)
        current = row.get("current_a")

        # The current rule is tested FIRST, before the "is it still moving"
        # branch, and the order is not cosmetic. min_peak is an absolute speed
        # (12 deg/s); a joint decelerating out of 563 deg/s passes through 15
        # deg/s, which that branch calls "moving" and returns on -- so the
        # current rule was never reached on the one sample that had all the
        # evidence: 15 deg/s and 40 A. With the other order, a run with a
        # textbook impact in its log reported no contact.
        if self.start_deg is None:
            self.start_deg = row["position_deg"]
        travelled = abs(row["position_deg"] - self.start_deg)
        # ARRIVING draws current too. A joint decelerating into its commanded
        # target brakes, and braking is current: an AK70-9 joint read -8.2 A settling on
        # a target it reached untouched, against 0.6 A while cruising, and every
        # current rule below fired on it. So the arrival guard has to sit in
        # front of ALL of them, not just the collapse rule -- an arrival looks
        # like an impact in speed AND in current, and only the target tells them
        # apart.
        near_target = (self.target_deg is not None
                       and abs(row["position_deg"] - self.target_deg)
                       <= self.arrive_tol)

        # ALREADY AGAINST IT. A joint that has barely left where it started and
        # is already drawing saturation current is touching its stop right now,
        # whatever its speed reads -- and its speed can read anything, because
        # 2 deg of grinding off a 0.1 deg encoder produces whatever number the
        # sample interval happens to give (49 deg/s, measured, at 40.0 A). No
        # collapse ever comes, because there was no flight to collapse out of.
        # The contact angle is nonetheless real: it is where the joint is.
        if (self.moved and not near_target and travelled < 5.0
                and current is not None
                and self.free_current > 0.0
                and abs(current) >= max(self.free_current * self.current_ratio,
                                        self.current_floor)):
            self.contact = row["position_deg"]
            self.reason = ("already against this end: {:.1f} deg of travel at "
                           "{:.1f} A".format(travelled, abs(current)))
            self.previous_speed = speed
            return True

        collapsed = speed <= max(self.contact_speed, 0.15 * self.peak)
        at_target = (self.target_deg is not None
                     and abs(row["position_deg"] - self.target_deg)
                     <= self.arrive_tol)

        # A COLLAPSE FASTER THAN THE COMMANDED RAMP is an impact, and it needs no
        # current at all. Whatever stopped the joint was not this program: the
        # trapezoid can only shed `accel` per second, so a joint that sheds far
        # more than that in one sample was stopped by the world.
        #
        # The "was it moving" half of the test is an ABSOLUTE speed, not a
        # fraction of the peak, for the same reason the stop half is. A climb
        # against gravity on a bus delivering irregularly reads 270, 138, -155,
        # 278, 533, 105, 254, 610 deg/s while ascending steadily; the peak of
        # that is 781 and a half-of-peak test then refuses to believe a genuine
        # 137 -> 5 deg/s stop at 40 A. In that measured run the joint hit its
        # stop, the detector declined, and the run ended blind.
        #
        # This is the only rule that works going DOWNHILL. Descending, the joint
        # brakes against gravity the whole way and genuinely draws 35 A while
        # moving freely, so the current rule's reference is 35 A, its trigger is
        # 170 A, and it can never fire -- measured on a descent that
        # ended in a textbook 708 -> 5 deg/s stop at 40 A and was reported as no
        # contact. Speed collapse sees it immediately.
        # The collapse test uses an ABSOLUTE small speed, not a fraction of the
        # peak. Differenced velocity off a 0.1 deg encoder arriving at irregular
        # intervals is noisy -- one recorded climb reads 210, 213, 200, 267, 204,
        # 40, 105, 256 deg/s while accelerating smoothly -- and a
        # fraction-of-peak threshold turns that 40 into an impact 24 deg before
        # the real one. Every genuine contact measured on an AK80-6/AK70-9 arm
        # collapsed to under 16 deg/s in the sample it landed.
        # An absolute floor OR five percent of the peak, whichever is larger.
        # The floor alone cannot see a fast impact: bare SET_POS put an AK80-6 joint
        # into its stop at 2245 deg/s and the sample after landing still read
        # 86 deg/s -- grinding, at 40 A, and "not stopped" by any 18 deg/s test.
        # Five percent alone cannot be trusted either, because differenced
        # velocity is noisy enough at low speed to dip under it mid-flight. The
        # larger of the two is the one that is right in each regime.
        stopped_dead = speed <= max(self.contact_speed * 3.0, 0.05 * self.peak)
        if (self.use_collapse and self.moved and stopped_dead and not at_target
                and self.previous_speed >= 5.0 * self.contact_speed
                and self.peak >= 4.0 * self.min_peak):
            self.contact = row["position_deg"]
            self.reason = "{:.0f} -> {:.0f} deg/s in one sample".format(
                self.previous_speed, speed)
            self.previous_speed = speed
            return True

        if (self.moved and not near_target and current is not None
                and self.free_current > 0.0):
            if collapsed:
                threshold = max(self.free_current * self.current_ratio,
                                self.current_floor)
                if abs(current) >= threshold:
                    self.contact = row["position_deg"]
                    self.reason = ("current {:.1f} A against {:.1f} A while "
                                   "moving freely".format(abs(current),
                                                          self.free_current))
                    self.previous_speed = speed
                    return True

        self.previous_speed = speed
        if speed >= self.min_peak:
            self.moved = True
            self.still = []
            # What this joint draws while it is NOT hitting anything: the
            # LOWER QUARTILE, not the median and certainly not the maximum.
            # The samples always include the impact itself -- a joint
            # decelerating from 576 deg/s to 114 in one sample is still
            # "moving" by any speed test and is already drawing 39.6 A. Taking
            # the maximum lifted the reference to 39.6 A and the detector sat
            # through a 40 A stall without firing. The median survives that on
            # a long approach but not on a SHORT one: a climb that had already
            # reached its stop produced exactly two moving samples, 10.7 A and
            # 40.0 A, and their median is the impact defining its own baseline.
            if current is not None:
                self.free_currents.append(abs(current))
                ordered = sorted(self.free_currents)
                self.free_current = ordered[max(0, (len(ordered) - 1) // 4)]
            return False
        if not self.moved:
            return False
        # The CURRENT rule gets a looser speed test than the counting rule, and
        # the looser one is the honest one. A joint that went from 563 deg/s to
        # 15 in a single sample while its draw went from 1 A to 40 A has hit
        # something; insisting it first reach 6 deg/s only delays the cut by a
        # sample, and at stall current a sample is exactly what stops arriving
        # next (in one measured run the bus went blind before the joint was ever
        # "slow enough" and no contact was reported at all).
        if speed > self.contact_speed:
            self.still = []
            return False

        if (self.target_deg is not None
                and abs(row["position_deg"] - self.target_deg) <= self.arrive_tol):
            self.arrived = True
            return True

        if not self.use_collapse:
            return False
        self.still.append(row["position_deg"])
        if len(self.still) >= self.needed:
            self.contact = sum(self.still) / len(self.still)
            self.reason = "{} consecutive samples under {:.0f} deg/s".format(
                self.needed, self.contact_speed)
            return True
        return False


def _release(bus):
    """Send zero current. Never raises; returns True if it got out.

    A release is attempted at every step of this test, including the steps
    reached BECAUSE the bus just failed. Letting it raise turns an orderly
    abort -- which reports the contact angle, writes the CSV, and tells the
    operator whether the motor is still pushing -- into a traceback that does
    none of those things. Two test runs died that way, one of them inside the
    very release meant to make the joint safe.
    """
    try:
        bus.release_all()
        return True
    except Exception:
        return False


def _write_csv(path, log):
    """Write the samples, if there are any and somewhere to put them.

    Called from the success path AND from the failure path. A run that dies
    mid-push is exactly the run whose samples matter -- it holds the approach,
    the contact, and whatever the current was doing when the bus went -- and
    the first version of this tool threw all of it away with the traceback.
    """
    if not path or not log:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["phase", "t", "position_deg", "velocity_dps",
                                "current_a", "temperature_c", "error"])
        writer.writeheader()
        writer.writerows(log)
    print("[impact] {} samples -> {}".format(len(log), path))


def _sample(bus, actuator):
    """One reading, or None if nothing has arrived yet."""
    state = bus.last_state(actuator.id)
    if state is None or state.position is None:
        return None
    return state


def _wait_for_status(bus, actuator, seconds):
    """Poll, commanding nothing, until the joint reports a NEW position."""
    end = time.monotonic() + seconds
    last = None
    while time.monotonic() < end:
        bus.poll(timeout=0.05)
        state = _sample(bus, actuator)
        if state is not None and state.position != last:
            if last is not None:
                return True
            last = state.position
    return False


def _settle(bus, actuator, seconds):
    """Poll for `seconds` with nothing commanded, and return where it settled."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        bus.poll(timeout=0.02)
    state = _sample(bus, actuator)
    return None if state is None else math.degrees(state.position)


def _stream(bus, actuator, target_deg, rate, duration, log, phase,
            temp_limit, on_sample=None, blind_s=0.15):
    """Stream one held POSITION target and record every reading.

    GOING BLIND STOPS THE PUSH. If no new status has arrived for `blind_s`, the
    command is cut immediately -- not because the joint is known to be in
    trouble, but because it is no longer known to be anything. Everything that
    decides to stop here (contact, stall, over-temperature) is computed from
    status frames, so a silent bus disables every one of those checks at once
    while the target keeps being commanded.

    On a CubeMars AK arm this is not a hypothetical failure mode, it is THE
    failure mode: a 700 deg/s sweep delivered 10 samples in 4 seconds -- the
    stall detector had almost nothing to look at -- and the run ended with the
    bus dead. Current is what breaks the bus, and blindness arrives before the
    break, so blindness is the earliest signal available.

    Returns (samples, abort_reason). `samples` is a list of dicts. Streaming
    stops early on driver fault, on status staleness, on over-temperature, or
    when `on_sample` returns True.
    """
    command = ActuatorCommand(mode=OperatingMode.POSITION,
                              position=math.radians(target_deg))
    batch = [command if a is actuator else None for a in bus.actuators]
    period = 1.0 / rate
    started = time.monotonic()
    next_send = started
    samples = []
    previous = None
    abort = None
    last_fresh = started
    last_position = None

    while time.monotonic() - started < duration:
        now = time.monotonic()
        if now >= next_send:
            try:
                bus.write_batch(batch)
            except Exception as err:
                # Losing the bus mid-push is a FORESEEN outcome of this test,
                # not a programming error: the impact is the moment of highest
                # motor current, and that is when a marginal bus goes. So
                # it aborts through the normal path -- which releases, prints
                # the contact angle that WAS measured, and writes the CSV --
                # rather than through a traceback that discards all three.
                abort = "the transmit queue jammed: {}".format(
                    str(err).split("\n")[0])
                break
            next_send += period
        bus.poll(timeout=0.005)
        # Deliberately NOT bus.check_fresh(): that raises if ANY actuator on the
        # bus is silent, and this tool drives exactly one. A different joint
        # having gone quiet is worth telling the operator about -- it is not a
        # reason to abandon a measurement on this one, and it made the whole
        # test unrunnable the first time a neighbour dropped off. The blind
        # watchdog below is per-joint and is the check
        # that actually matters here: it asks whether THIS joint is still
        # visible while THIS joint is being driven.
        state = _sample(bus, actuator)
        if state is None:
            if time.monotonic() - last_fresh > blind_s:
                abort = ("no status for {:.0f} ms -- cutting the command rather "
                         "than pushing blind".format(blind_s * 1000.0))
                break
            continue
        if state.position != last_position:
            last_fresh = time.monotonic()
            last_position = state.position
        elif time.monotonic() - last_fresh > blind_s:
            abort = ("no status for {:.0f} ms -- cutting the command rather "
                     "than pushing blind".format(blind_s * 1000.0))
            break
        if state.error:
            abort = "driver fault {} ({})".format(
                state.error, ERROR_NAMES.get(state.error, "?"))
            break
        if state.temperature is not None and state.temperature >= temp_limit:
            abort = "driver board reached {:.0f}C (limit {:.0f})".format(
                state.temperature, temp_limit)
            break

        elapsed = time.monotonic() - started
        position = math.degrees(state.position)
        if previous is not None and elapsed > previous[0]:
            velocity = (position - previous[1]) / (elapsed - previous[0])
        else:
            velocity = 0.0
        if previous is not None and position == previous[1]:
            # Same frame re-read: the bus has not delivered a new sample, so a
            # differenced velocity of 0 here would be an artefact, not a stop.
            continue
        previous = (elapsed, position)

        row = {
            "phase": phase,
            "t": round(elapsed, 4),
            "position_deg": round(position, 3),
            "velocity_dps": round(velocity, 2),
            "current_a": state.aux.get("current_a"),
            "temperature_c": state.temperature,
            "error": state.error,
        }
        samples.append(row)
        log.append(row)
        if on_sample is not None and on_sample(row):
            break

    return samples, abort


def _find_contact(samples, contact_speed, needed, min_peak):
    """Where the joint stopped while still being pushed.

    Returns (contact_deg, peak_speed, index) or (None, peak_speed, None).
    Contact requires the joint to have MOVED first -- `min_peak` -- so that a
    joint already resting against the stop is reported as "never moved" rather
    than as an instant contact at its starting angle.
    """
    peak = max((abs(r["velocity_dps"]) for r in samples), default=0.0)
    if peak < min_peak:
        return None, peak, None
    moving = False
    still = 0
    for index, row in enumerate(samples):
        if abs(row["velocity_dps"]) >= min_peak * 0.5:
            moving = True
            still = 0
            continue
        if not moving:
            continue
        if abs(row["velocity_dps"]) <= contact_speed:
            still += 1
            if still >= needed:
                window = samples[index - needed + 1:index + 1]
                mean = sum(r["position_deg"] for r in window) / len(window)
                return mean, peak, index
        else:
            still = 0
    return None, peak, None


def main():
    parser = argparse.ArgumentParser(
        description="Drive a joint into its mechanical stop and measure the "
                    "contact angle across repeats.")
    parser.add_argument("--config", required=True, help="robot bus YAML "
                                                        "(READ ONLY -- never written)")
    parser.add_argument("--joint", required=True)
    parser.add_argument("--max-aborts", type=int, default=3,
                        help="stop after this many aborts IN A ROW. One abort is "
                             "survivable -- the release has already happened and "
                             "the bus usually recovers once the current stops -- but a "
                             "link that will not carry one cycle will not carry "
                             "the next.")
    parser.add_argument("--recover-s", type=float, default=10.0,
                        help="how long to wait for status to return after an "
                             "abort before giving up")
    parser.add_argument("--set-pos", action="store_true",
                        help="send the bare target (servo packet 4, SET_POS) "
                             "instead of a trapezoid with a speed envelope "
                             "(packet 6, SET_POS_SPD). The driver then runs its "
                             "own position loop flat out and --speed/--accel do "
                             "nothing: there is no envelope to put them in. For "
                             "an impact test that is the honest form -- the "
                             "target sits past the stop, so the loop never "
                             "reaches its deceleration phase and the joint "
                             "arrives at whatever speed the motor can produce. "
                             "It is also the maximum-current form, so it is the "
                             "hardest on the bus.")
    parser.add_argument("--up-speed", type=float, default=None,
                        help="speed cap for the UPPER end only (default: "
                             "--speed). Climbing is torque-limited on a "
                             "gravity-loaded joint: the motor saturates its "
                             "current the whole way up whatever you ask for, so "
                             "asking for more buys no motion and costs the bus "
                             "-- measured, a 15000 deg/s^2 climb request reached "
                             "55 deg/s at 40 A and killed the link before the "
                             "joint had moved a degree. Give the climb what it "
                             "can actually use and spend the speed on the "
                             "descent, where gravity is helping.")
    parser.add_argument("--up-accel", type=float, default=None,
                        help="acceleration for the UPPER end only (default: "
                             "--accel). See --up-speed.")
    parser.add_argument("--ignore-envelope", action="store_true",
                        help="do not use the taught envelope to pick targets or "
                             "to veto the start. Requires --travel-deg. Use it "
                             "when the envelope demonstrably no longer describes "
                             "the mechanism -- which on a CubeMars AK happens the "
                             "moment the encoder origin slips a gear window, "
                             "since the bounds then fence a part of the travel "
                             "that has moved out from under them. WHAT PROTECTS "
                             "THE JOINT THEN is the stall detector (one sample, "
                             "current cut immediately) and the person watching. "
                             "Nothing else.")
    parser.add_argument("--travel-deg", type=float, default=None,
                        help="with --ignore-envelope: how far to drive from the "
                             "present pose toward each end before giving up. "
                             "The joint is expected to meet something first.")
    parser.add_argument("--sweep", action="store_true",
                        help="STOP TO STOP, with no home position: drive into "
                             "one end until it stalls, then straight into the "
                             "other, and repeat. Three things fall out of that "
                             "which the home-and-return form does not give. The "
                             "run-up is the whole travel, so it is the fastest "
                             "impact the joint can produce. There is no home to "
                             "fail to reach -- the previous stop IS the next "
                             "start. And the SPAN between the two stalls is "
                             "frame-independent: an origin that slips by a gear "
                             "window moves both ends together and leaves the "
                             "span untouched, so a span that changes is the "
                             "MECHANISM changing and a span that holds while "
                             "both ends move is the ENCODER. Implies --both.")
    parser.add_argument("--both", action="store_true",
                        help="hit BOTH ends every repeat (upper, home, lower, "
                             "home), using |--overtravel-deg| at each. A "
                             "fastener that only slips under load in one "
                             "direction shows up as the two ends drifting by "
                             "different amounts, which a one-ended test cannot "
                             "see.")
    parser.add_argument("--overtravel-deg", type=float, required=True,
                        help="how far PAST the taught envelope bound to aim. "
                             "The sign picks the end: positive aims past the "
                             "upper bound, negative past the lower. The joint "
                             "is expected to hit the mechanism before it gets "
                             "there -- that is the point -- so this only has to "
                             "be enough to keep the driver pushing.")
    parser.add_argument("--speed", type=float, required=True,
                        help="approach speed cap, deg/s at the output shaft")
    parser.add_argument("--accel", type=float, default=None,
                        help="deg/s^2 (default: equal to --speed, the vendor's "
                             "example, which is usually too low to reach it -- "
                             "the reachable speed over a short approach is "
                             "sqrt(accel*travel))")
    parser.add_argument("--from", dest="start_deg", type=float, default=0.0,
                        help="the home angle: the joint is driven back here after "
                             "every impact and starts the next one from here, so "
                             "the approach distance is identical every time. "
                             "Default 0, the datum. Must be inside the taught "
                             "envelope. Pass the joint's current angle to skip "
                             "the first return.")
    parser.add_argument("--approach-speed", type=float, default=40.0,
                        help="deg/s for the gentle retreat move between impacts")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--hold", type=float, default=0.0,
                        help="seconds to keep pushing AFTER contact. DEFAULT 0: "
                             "stop the moment the joint stops. This is the stall "
                             "time -- maximum current into a joint that is not "
                             "turning -- and it is what heats the winding, which "
                             "the tool cannot see (see the module docstring). "
                             "Raise it only if the test is specifically about "
                             "sustained load rather than impact.")
    parser.add_argument("--rest-seconds", type=float, default=20.0,
                        help="limp cool-down between impacts")
    parser.add_argument("--temp-limit", type=float, default=70.0,
                        help="abort if the DRIVER BOARD reaches this. The board "
                             "protects itself at 100C; this is well under it "
                             "because the winding is hotter and unreadable.")
    parser.add_argument("--timeout", type=float, default=4.0,
                        help="give up on an impact that never makes contact")
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--tolerance-deg", type=float, default=2.0,
                        help="how close to the commanded target counts as "
                             "ARRIVED rather than as a contact. A joint that "
                             "simply got where it was told has hit nothing, and "
                             "recording that as a contact angle would put a "
                             "number in the drift column that tracks the "
                             "command instead of the mechanism.")
    parser.add_argument("--blind-ms", type=float, default=150.0,
                        help="cut the command if no new status has arrived for "
                             "this long. Every stop condition is computed from "
                             "status, so a silent bus disables all of them at "
                             "once while the target keeps being driven.")
    parser.add_argument("--contact-speed", type=float, default=6.0,
                        help="deg/s below which the joint counts as stopped")
    parser.add_argument("--contact-samples", type=int, default=4,
                        help="consecutive slow samples needed to call it contact")
    parser.add_argument("--window-reference", type=float, default=None,
                        metavar="DEG",
                        help="roughly where this joint is, in the DATUM frame, "
                             "used only to pick its gear window. Needs to be "
                             "right to a quarter of a window, never better. "
                             "Overrides the remembered position and the "
                             "config's startup_reference_deg.")
    parser.add_argument("--shm-name", required=True,
                        help="the shm name the robot's bus servicer runs under: "
                             "its lock is checked so the test refuses to run "
                             "beside a live servicer, and its remembered "
                             "window position is read (see window_state.py)")
    parser.add_argument("--csv", default=None,
                        help="write every sample here (default: no CSV)")
    parser.add_argument("--yes", action="store_true",
                        help="skip the typed confirmation")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and send nothing")
    parser.add_argument("--interface", default=None)
    parser.add_argument("--channel", default=None)
    parser.add_argument("--bitrate", type=int, default=None)
    args = parser.parse_args()

    if lock_holder(args.shm_name) is not None:
        raise SystemExit(
            "[impact] REFUSED: a servicer holds {}. Two writers on one bus "
            "means whichever frame lands last wins, and during an impact test "
            "that is not something to leave to chance. Stop the launch "
            "first.".format(lock_path(args.shm_name)))

    overrides = {}
    for key in ("interface", "channel", "bitrate"):
        if getattr(args, key):
            overrides[key] = getattr(args, key)
    bus, names = build_bus(args.config, transport_overrides=overrides or None)
    if args.joint not in names:
        raise SystemExit("[impact] no such joint: {!r}. This bus has: {}".format(
            args.joint, ", ".join(names)))
    actuator = bus.actuators[names.index(args.joint)]
    calibration = actuator.calibration

    if not hasattr(actuator, "use_pos_spd"):
        raise SystemExit(
            "[impact] {} is a {}, which has no SET_POS_SPD equivalent; this "
            "tool needs a driver-side speed envelope.".format(
                args.joint, type(actuator).__name__))

    accel = args.speed if args.accel is None else args.accel
    erpm = calibration.rad_s_to_erpm(math.radians(args.speed))
    erpm_accel = calibration.rad_s_to_erpm(math.radians(accel))
    if not 1 <= int(round(erpm / 10.0)) <= 32767:
        raise SystemExit(
            "[impact] {:.0f} deg/s is {:.0f} ERPM, which does not fit the "
            "wire's int16 of ERPM/10.".format(args.speed, erpm))

    # In memory only -- see the module docstring's hazard 3. CubeMarsLimits is
    # frozen, so this has to be a deliberate replace rather than a stray write.
    actuator.use_pos_spd = not args.set_pos
    actuator.limits = dataclasses.replace(
        actuator.limits,
        max_velocity_erpm=erpm,
        max_acceleration_erpm_s=erpm_accel)

    low, high = [math.degrees(x) for x in calibration.position_limit]
    if args.ignore_envelope and args.travel_deg is None:
        raise SystemExit(
            "[impact] --ignore-envelope needs --travel-deg: with the taught "
            "bounds out of the picture there is nothing left to derive a target "
            "from, and a target has to come from somewhere explicit.")
    over = abs(args.overtravel_deg)
    if over == 0:
        raise SystemExit("[impact] --overtravel-deg 0 aims at nothing.")
    if args.sweep:
        args.both = True
    if args.both:
        # One repeat = one impact at EACH end, with a return home between them.
        # Alternating is worth more than convenience: a fastener that only slips
        # under load in one direction shows up as the two ends drifting by
        # DIFFERENT amounts, which a one-ended test cannot see at all.
        ends = [("upper", high + over), ("lower", low - over)]
    elif args.overtravel_deg > 0:
        ends = [("upper", high + over)]
    else:
        ends = [("lower", low - over)]

    bus.transport.open()
    log = []
    wrote_csv = [False]
    try:
        # Release frames, not silence. A CubeMars that has just been powered on
        # uploads NOTHING until it receives a frame, so a listen-only wait reads
        # as "the motors are dead" when they are merely idle (it once nearly
        # sent an operator to check a healthy power supply).
        # SET_CURRENT 0 is zero torque in any calibration, so this cannot move
        # the joint -- it just makes it talk.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            _release(bus)
            bus.poll(timeout=0.05)
        state = _sample(bus, actuator)
        if state is None:
            raise SystemExit(
                "[impact] {} (id {}) never reported, even after release frames. "
                "Refusing to drive a joint whose position is unknown into a "
                "hard stop.".format(args.joint, actuator.id))
        if state.error:
            raise SystemExit(
                "[impact] {} reports fault {} ({}). Clear it first.".format(
                    args.joint, state.error,
                    ERROR_NAMES.get(state.error, "?")))
        # Establish the gear window BEFORE reading a single angle as truth.
        # A standalone tool starts at window_offset 0, i.e. in the frame the
        # driver powered up in, which for an AK80-6 can be a whole multiple of
        # 60 deg from the datum the envelope was taught in. That is not a
        # hypothetical: an envelope of [93.60, 215.10] was once written for
        # exactly this reason -- every number plausible, all of them 180 deg
        # out. Aiming
        # an impact test with a 180 deg error would drive at full speed through
        # the middle of the range instead of into the stop.
        # --ignore-envelope needs NO datum. Its targets are measured from the
        # present pose and its headline number is the span between the two
        # stops, both of which are unchanged by an origin sitting a whole gear
        # window away. Demanding a window there would block the one mode that is
        # immune to the window being wrong -- which is exactly the mode you reach
        # for when it IS wrong.
        uses_window = (not args.ignore_envelope
                       and (args.window_reference is not None
                            or getattr(actuator, "startup_reference_rad", None)
                            is not None))
        if uses_window and hasattr(actuator, "snap_to_window"):
            if args.window_reference is not None:
                reference = math.radians(args.window_reference)
                source = "--window-reference"
            else:
                remembered = window_state.load_positions(args.shm_name)
                reference = remembered.get(args.joint,
                                           actuator.startup_reference_rad)
                source = ("remembered" if args.joint in remembered
                          else "config startup_reference_deg")
            try:
                offset = actuator.snap_to_window(state.position, reference)
            except ValueError as err:
                raise SystemExit(
                    "[impact] {}: cannot establish the gear window, so every "
                    "angle below -- including the stop this would drive into -- "
                    "is in an unknown frame. Pass --window-reference DEG with "
                    "roughly where this joint is. {}".format(args.joint, err))
            if offset:
                print("[impact] gear window {:+d} ({:+.2f} deg) from the {} "
                      "reference; angles below are in the DATUM frame".format(
                          offset, offset * calibration.gear_window_deg, source))
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                _release(bus)
                bus.poll(timeout=0.05)
            state = _sample(bus, actuator)

        here = math.degrees(state.position)
        if args.ignore_envelope:
            # Without --sweep, the sign of --overtravel-deg picks ONE end, the
            # same way it does with a taught envelope. Always doing both has a
            # cost on a gravity-loaded joint: on an AK80-6 arm joint the upward
            # leg held ~40 A for its whole length and is what took the bus
            # down, while the downward leg is gravity-assisted, lands on a
            # genuinely hard stop, and is the half worth repeating.
            if args.both:
                ends = [("upper", here + args.travel_deg),
                        ("lower", here - args.travel_deg)]
            elif args.overtravel_deg > 0:
                ends = [("upper", here + args.travel_deg)]
            else:
                ends = [("lower", here - args.travel_deg)]
            print("[impact] ** ENVELOPE IGNORED. Targets are {:+.2f} deg either "
                  "side of the present".format(args.travel_deg))
            print("[impact]    pose, and the ONLY thing that stops this joint "
                  "is the stall detector")
            print("[impact]    plus you. Do not leave it. **")
        elif not low <= here <= high:
            raise SystemExit(
                "[impact] {} is at {:+.2f}, outside its taught envelope "
                "[{:+.2f}, {:+.2f}]. Get it inside before adding overtravel to "
                "the outside.".format(args.joint, here, low, high))

        start = here if args.start_deg is None else args.start_deg
        if not args.ignore_envelope and not low <= start <= high:
            raise SystemExit(
                "[impact] --from {:+.2f} is outside the taught envelope "
                "[{:+.2f}, {:+.2f}].".format(start, low, high))
        worst = max(abs(t - start) for _, t in ends)
        reachable = math.sqrt(accel * worst) if worst > 0 else 0.0

        print("[impact] joint      : {} (id {})".format(args.joint, actuator.id))
        print("[impact] envelope   : {:+.2f} .. {:+.2f} deg (taught)".format(low, high))
        print("[impact] now        : {:+.2f} deg, T={:.0f}C".format(
            here, state.temperature or 0.0))
        print("[impact] start      : {:+.2f} deg".format(start))
        for edge, tgt in ends:
            print("[impact] target     : {:+.2f} deg = {:.2f} PAST the {} "
                  "bound".format(tgt, over, edge))
        if args.set_pos:
            print("[impact] speed      : NO CAP -- bare SET_POS, the driver's "
                  "own loop decides.")
            print("[impact]              --speed/--accel are ignored; the "
                  "target is past the stop, so")
            print("[impact]              nothing ever tells it to slow down.")
        else:
            print("[impact] speed      : {:.0f} deg/s cap, accel {:.0f} deg/s^2"
                  .format(args.speed, accel))
        if not args.set_pos and reachable < args.speed * 0.95:
            print("[impact]   ** over {:.1f} deg at this accel the profile is a "
                  "TRIANGLE and tops out".format(worst))
            print("[impact]      at sqrt(accel*travel) = {:.0f} deg/s, NOT {:.0f}. "
                  "Raise --accel to {:.0f} or --from further back."
                  .format(reachable, args.speed,
                          args.speed ** 2 / worst if worst > 0 else 0.0))
        elif not args.set_pos:
            print("[impact]   {:.0f} deg/s is reachable over {:.1f} deg "
                  "(sqrt(accel*travel) = {:.0f})".format(
                      args.speed, worst, reachable))
        print("[impact] repeats    : {} x (impact, hold {:.2f}s, rest {:.0f}s)"
              .format(args.repeats, args.hold, args.rest_seconds))
        print("[impact] envelope override is IN MEMORY ONLY; {} is not written."
              .format(os.path.basename(args.config)))
        print("[impact] ** BE IN REACH OF THE MOTOR POWER SWITCH. If the bus "
              "drops mid-push the")
        print("[impact]    driver holds this target and keeps pushing; no frame "
              "can stop it. **")

        if args.dry_run:
            print("[impact] dry run: nothing sent.")
            return 0
        if not args.yes:
            reply = input("[impact] type IMPACT to run: ")
            if reply.strip() != "IMPACT":
                print("[impact] not run.")
                return 1

        # Widen in memory so _check_position_limit does not veto the target.
        widest = [t for _, t in ends] + [low, high]
        calibration.position_limit = (
            math.radians(min(widest)), math.radians(max(widest)))

        results = []
        stop = False
        consecutive_aborts = 0
        max_consecutive_aborts = args.max_aborts
        for run in range(1, args.repeats + 1):
          if stop:
              break
          hit_this_run = False
          for edge, target in ends:
            print("\n[impact] --- impact {}/{} ({} end) ---".format(
                run, args.repeats, edge))

            if args.sweep:
                pass
            elif abs(here - start) > 1.0:
                print("[impact] going home to {:+.2f} at {:.0f} deg/s"
                      .format(start, args.approach_speed))
                reached, stalled, stall_a, home_abort = _drive_home(
                    bus, actuator, calibration, start, args.approach_speed,
                    args.rate, log, "home_{}_{}".format(run, edge), args.temp_limit,
                    here, args.contact_speed, args.contact_samples)
                if not reached and home_abort is not None:
                    print("[impact] the move home was cut: {}".format(home_abort))
                    consecutive_aborts += 1
                    if consecutive_aborts >= max_consecutive_aborts:
                        stop = True
                        break
                    if not _wait_for_status(bus, actuator, args.recover_s):
                        stop = True
                        break
                    fresh = _sample(bus, actuator)
                    if fresh is not None:
                        here = math.degrees(fresh.position)
                    continue
                if not reached:
                    print("[impact] CANNOT REACH HOME {:+.2f}: the joint "
                          "stopped at {} drawing {} A.".format(
                              start,
                              "?" if stalled is None else
                              "{:+.2f} deg".format(stalled),
                              "?" if stall_a is None else
                              "{:.1f}".format(stall_a)))
                    print("[impact] Current is cut. Nothing further is "
                          "commanded: an impact test that cannot")
                    print("[impact] return to the same start every time is not "
                          "measuring the same thing twice.")
                    stop = True
                    break
                here = start

            end_speed = (args.up_speed if (edge == "upper"
                                           and args.up_speed is not None)
                         else args.speed)
            end_accel = (args.up_accel if (edge == "upper"
                                           and args.up_accel is not None)
                         else accel)
            if not args.set_pos:
                actuator.limits = dataclasses.replace(
                    actuator.limits,
                    max_velocity_erpm=calibration.rad_s_to_erpm(
                        math.radians(end_speed)),
                    max_acceleration_erpm_s=calibration.rad_s_to_erpm(
                        math.radians(end_accel)))
            if not args.set_pos and (end_speed != args.speed
                                     or end_accel != accel):
                print("[impact] {} end: {:.0f} deg/s, {:.0f} deg/s^2".format(
                    edge, end_speed, end_accel))

            rest_before = _settle(bus, actuator, 1.0)
            detector = _ContactDetector(
                args.contact_speed, args.contact_samples,
                min_peak=max(10.0, args.contact_speed * 2.0),
                target_deg=target, arrive_tol=args.tolerance_deg)
            samples, abort = _stream(
                bus, actuator, target, args.rate, args.timeout, log,
                "impact_{}_{}".format(run, edge), args.temp_limit, on_sample=detector,
                blind_s=args.blind_ms / 1000.0)
            if detector.contact is not None:
                # Zero current first, then decide what to print. The return move
                # would unload the joint anyway, but it takes a trapezoid to do
                # it; a release takes one frame, and the frames are the scarce
                # thing here.
                _release(bus)
            contact, peak_speed = detector.contact, detector.peak
            index = max(0, len(samples) - args.contact_samples)

            if abort is not None:
                _release(bus)
                print("[impact] ABORTED mid-push: {}".format(abort))
                consecutive_aborts += 1
                results.append({"run": run, "contact_deg": contact,
                                "peak_speed_dps": peak_speed,
                                "peak_current_a": None, "peak_temp_c": None,
                                "rest_before_deg": rest_before,
                                "end": edge, "rest_after_deg": None, "note": abort})
                # An abort is not the end of the test. The dangerous thing --
                # commanding a joint nobody can see -- was already undone by the
                # release above, and the bus usually comes back on its own
                # once the current stops. Ending here meant a marginal link turned a
                # 10-cycle endurance test into a 1-cycle one, which is exactly
                # the test that cannot answer the question it was asked.
                #
                # Consecutive aborts DO end it: a link that will not carry one
                # cycle will not carry the next, and retrying forever is just
                # hammering a joint nobody can see.
                if consecutive_aborts >= max_consecutive_aborts:
                    print("[impact] {} aborts in a row -- stopping. The link is "
                          "not carrying this test.".format(consecutive_aborts))
                    stop = True
                    break
                print("[impact] waiting for the bus to come back...")
                if not _wait_for_status(bus, actuator, args.recover_s):
                    print("[impact] no status after {:.0f}s -- stopping."
                          .format(args.recover_s))
                    stop = True
                    break
                fresh = _sample(bus, actuator)
                if fresh is not None:
                    here = math.degrees(fresh.position)
                print("[impact] bus is back; continuing.")
                continue

            if contact is None:
                _release(bus)
                if detector.arrived:
                    print("[impact] REACHED {:+.2f} WITHOUT HITTING ANYTHING "
                          "-- there is nothing to measure".format(target))
                    print("[impact]   here: this joint has more travel this way "
                          "than it was asked for.")
                    print("[impact]   Raise {} until it meets something.".format(
                        "--travel-deg" if args.ignore_envelope
                        else "--overtravel-deg"))
                else:
                    print("[impact] no contact detected in {:.1f}s (peak "
                          "{:.0f} deg/s, {} samples).".format(
                              args.timeout, peak_speed, len(samples)))
                    if peak_speed < 1.0:
                        print("[impact]   The joint never moved.")
                    elif len(samples) < args.timeout * args.rate * 0.5:
                        print("[impact]   It MOVED, and the detector was "
                              "starved: {} samples in {:.1f}s is a bus that "
                              "stopped".format(len(samples), args.timeout))
                        print("[impact]   reporting, not a joint that never "
                              "stopped. Lower --speed or fix the bus.")
                results.append({"run": run, "contact_deg": None,
                                "peak_speed_dps": peak_speed,
                                "peak_current_a": None, "peak_temp_c": None,
                                "rest_before_deg": rest_before,
                                "end": edge, "rest_after_deg": None,
                                "note": ("reached the target, hit nothing"
                                         if detector.arrived else
                                         "no contact -- {} samples".format(
                                             len(samples)))})
                # NOT a reason to end the test. The other end is a different
                # question with a different answer, and free travel one way says
                # nothing about travel the other -- on a gravity-loaded joint the
                # two directions are not even the same experiment. Ending here
                # answered one of the two and reported it as the run.
                fresh = _sample(bus, actuator)
                if fresh is not None:
                    here = math.degrees(fresh.position)
                continue

            # Keep pushing for --hold beyond contact, then always release.
            held_samples, hold_abort = _stream(
                bus, actuator, target, args.rate, args.hold, log,
                "hold_{}_{}".format(run, edge), args.temp_limit)
            _release(bus)

            window = samples[index:] + held_samples
            currents = [r["current_a"] for r in window if r["current_a"] is not None]
            temps = [r["temperature_c"] for r in window if r["temperature_c"] is not None]
            if args.sweep:
                print("[impact] contact ({}) -- current cut. Next impact "
                      "starts here.".format(detector.reason))
            else:
                print("[impact] contact ({}) -- current cut. Returning to "
                      "{:+.2f} deg.".format(detector.reason, start))
            reached, stalled, stall_a, home_abort = \
                (True, None, None, None) if args.sweep else _drive_home(
                    bus, actuator, calibration, start, args.approach_speed,
                    args.rate, log, "return_{}_{}".format(run, edge), args.temp_limit,
                    contact, args.contact_speed, args.contact_samples)
            if not reached and home_abort is not None:
                # The bus went quiet on the way back, which happens right after
                # a hard stall -- it is the same recoverable event as an abort
                # during the impact, not a reason to end the test. Saying "did
                # not get back, stopped at ? drawing ? A" for it was reporting a
                # stall that never happened.
                print("[impact] the return was cut: {}. Recovering."
                      .format(home_abort))
                consecutive_aborts += 1
            elif not reached:
                print("[impact] did not get back to {:+.2f}: stopped at {} "
                      "drawing {} A -- something is there. Current cut.".format(
                          start,
                          "?" if stalled is None else "{:+.2f}".format(stalled),
                          "?" if stall_a is None else "{:.1f}".format(stall_a)))
                stop = True
            rest_after = _settle(bus, actuator, min(3.0, args.rest_seconds))
            here = (contact if args.sweep
                    else (rest_after if rest_after is not None else start))

            row = {
                "run": run,
                "end": edge,
                "contact_deg": round(contact, 3),
                "peak_speed_dps": round(peak_speed, 1),
                # By MAGNITUDE. Current is signed by direction, so max() on a
                # descent -- where every sample is negative -- returns the
                # SMALLEST draw and calls it the peak. An AK70-9 joint reported
                # "-0.70 A" at an impact that was pulling 17 A the other way.
                "peak_current_a": (round(max(currents, key=abs), 2)
                                   if currents else None),
                "peak_temp_c": max(temps) if temps else None,
                "rest_before_deg": None if rest_before is None else round(rest_before, 3),
                "rest_after_deg": None if rest_after is None else round(rest_after, 3),
                "note": hold_abort or "",
            }
            # Drift is measured against the first contact AT THE SAME END.
            # Comparing an upper contact with a lower one would report the
            # width of the range as a drift of ~140 deg on impact 1.
            same = [r for r in results
                    if r.get("end") == edge and r["contact_deg"] is not None]
            hit_this_run = True
            consecutive_aborts = 0
            results.append(row)
            drift = contact - (same[0]["contact_deg"] if same else contact)
            print("[impact] {} contact  {:+.2f} deg  (drift {:+.3f})  "
                  "approach {:.0f} deg/s  peak {} A  T {} C".format(
                      edge, contact, drift, peak_speed,
                      row["peak_current_a"], row["peak_temp_c"]))
            # Where it stopped, relative to where a hand got to. Stated, not
            # ruled on: current up + position frozen IS a stop, whatever is
            # taking the load. But a stop well inside the taught range is more
            # likely the motor running out of torque than the mechanism running
            # out of travel -- and that distinction decides whether the contact
            # angle means anything as a STRUCTURAL metric, because a torque
            # equilibrium moves with payload, supply voltage and temperature
            # while the structure is perfectly healthy.
            # Only meaningful for a contact INSIDE the taught range. One PAST
            # a bound is not "short of" anything -- it is the hand-taught bound
            # having been conservative, which is the normal case.
            short = (min(abs(contact - low), abs(contact - high))
                     if low <= contact <= high else 0.0)
            if short > 5.0 and not args.ignore_envelope:
                print("[impact]   NOTE it stopped {:.1f} deg short of the "
                      "nearest taught bound ({:+.2f}/{:+.2f}), and a hand "
                      "reached those.".format(short, low, high))
                print("[impact]   So something other than the mechanism's end "
                      "is taking the load -- most likely")
                print("[impact]   the motor's own torque limit, or an "
                      "obstruction. Drift in this angle is then")
                print("[impact]   not purely a structural signal.")
            if hold_abort:
                print("[impact] hold aborted: {}".format(hold_abort))
                stop = True
                break

            if not (run == args.repeats and edge == ends[-1][0]):
                print("[impact] resting {:.0f}s (limp)".format(args.rest_seconds))
                _settle(bus, actuator, args.rest_seconds)
                here = math.degrees(_sample(bus, actuator).position)

        print("\n[impact] === contact angle is the measurement ===")
        print("[impact] {:>3} {:>6} {:>12} {:>10} {:>10} {:>9} {:>8}".format(
            "run", "end", "contact", "drift", "approach", "peak I", "T"))
        first = {}
        for row in results:
            if row["contact_deg"] is not None:
                first.setdefault(row.get("end"), row["contact_deg"])
        for row in results:
            contact = row["contact_deg"]
            base = first.get(row.get("end"))
            drift = "-" if (contact is None or base is None) else \
                "{:+.3f}".format(contact - base)
            print("[impact] {:>3} {:>6} {:>12} {:>10} {:>10} {:>9} {:>8}  {}".format(
                row["run"], row.get("end", "-"),
                "-" if contact is None else "{:+.3f}".format(contact),
                drift,
                "{:.0f}".format(row["peak_speed_dps"]),
                "-" if row["peak_current_a"] is None else "{:.2f}".format(row["peak_current_a"]),
                "-" if row["peak_temp_c"] is None else "{:.0f}".format(row["peak_temp_c"]),
                row["note"]))
        if not hit_this_run and not stop:
            print("[impact] neither end met anything in run {}. There is "
                  "nothing to measure at this travel;".format(run))
            print("[impact] raise {}.".format(
                "--travel-deg" if args.ignore_envelope else "--overtravel-deg"))
            stop = True

        pairs = {}
        for row in results:
            if row["contact_deg"] is not None:
                pairs.setdefault(row["run"], {})[row.get("end")] = \
                    row["contact_deg"]
        spans = [(run, ends_seen["upper"] - ends_seen["lower"])
                 for run, ends_seen in sorted(pairs.items())
                 if "upper" in ends_seen and "lower" in ends_seen]
        if spans:
            print("[impact]")
            print("[impact] stop-to-stop SPAN, the frame-independent number:")
            for run, span in spans:
                print("[impact]   run {}: {:.3f} deg  (drift {:+.3f})".format(
                    run, span, span - spans[0][1]))
            print("[impact] A span that CHANGES is the mechanism changing. A "
                  "span that HOLDS while both")
            print("[impact] ends move together is the encoder's origin "
                  "slipping -- on a CubeMars AK that")
            print("[impact] slip is a whole gear window, {:.2f} deg, and it "
                  "moves nothing physical.".format(
                      calibration.gear_window_deg))
        for edge in sorted(first):
            moved = [r["contact_deg"] - first[edge] for r in results
                     if r.get("end") == edge and r["contact_deg"] is not None]
            if len(moved) > 1:
                print("[impact] {} end: total drift over {} impacts {:+.3f} deg"
                      .format(edge, len(moved), moved[-1]))
        if len(first) > 1:
            print("[impact] The two ends drifting by DIFFERENT amounts is the "
                  "informative case: it says")
            print("[impact] the load is being taken asymmetrically, which a "
                  "one-ended test cannot see.")
        if first:
                print("[impact] A contact angle that walks is the structure "
                      "moving. Compare it against")
                print("[impact] peak current in the same row before concluding "
                      "the part yielded: a")
                print("[impact] contact angle that moved while the current also "
                      "changed may be the")
                print("[impact] approach that differed, not the mechanism.")

        _write_csv(args.csv, log)
        wrote_csv[0] = True
        return 0
    finally:
        # Every exit path -- exception, Ctrl-C, normal return -- tries to end
        # limp. It can FAIL, and saying so is the whole point: the release
        # travels over the same bus that may have just died, and this tool is
        # used at the one moment the motor is pushing hardest. An operator
        # reading "released; the joint is limp" while the driver holds a target
        # against a hard stop is worse off than one told nothing.
        released = False
        try:
            released = _release(bus)
            if not released:
                raise RuntimeError("release frames could not be transmitted")
        except Exception as err:
            print("")
            print("[impact] ############################################")
            print("[impact] ## THE RELEASE DID NOT GET OUT: {}".format(
                type(err).__name__))
            print("[impact] ## The driver still holds its last target and is")
            print("[impact] ## pushing. No frame can reach it.")
            print("[impact] ## >>> CUT MOTOR POWER NOW <<<")
            print("[impact] ############################################")
        try:
            bus.transport.close()
        except Exception:
            pass
        if released:
            print("[impact] released; the joint is limp.")
        if not wrote_csv[0]:
            try:
                _write_csv(args.csv, log)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
