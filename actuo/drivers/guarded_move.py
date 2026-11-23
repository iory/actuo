"""Move one joint slowly and stop as soon as something resists it.

Resistance shows first as tracking error: the goal keeps advancing while the
joint lags behind it. A hard stop, a cable going taut or the table stall the
joint and the lag grows quickly; the load the actuator reports (where it
reports one) rises with it. Either, held for ``trip_duration``, stops the move
where the joint is, and it can then back away from the obstruction.

Ported from attachable_arm's feetech_drivers (one joint, one cycle at a time,
in the servicer's control loop).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple


class GuardedMove:
    """State of one guarded move, advanced once per control cycle.

    Parameters
    ----------
    start : float
        Commanded angle when the move begins [rad].
    displacement : float
        Signed distance to travel [rad], not zero.
    velocity : float
        Speed the goal advances at [rad/s], positive.
    max_tracking_error : float
        Stop when the joint lags the goal by more than this [rad].
    max_load : float or None
        Stop when the load pushing in the direction of travel exceeds this,
        in the actuator's ``load`` units (fraction of full scale); None or 0
        disables the check.
    trip_duration : float
        How long a limit must stay exceeded before the move stops [s], so a
        single noisy sample does not.
    settle_time : float
        After the goal reaches the end, keep guarding this long for the joint
        to catch up [s].
    backoff : float
        On a stop, back away from the obstruction by this much [rad].
    """

    def __init__(self, start: float, displacement: float, velocity: float,
                 max_tracking_error: float, max_load: Optional[float] = None,
                 trip_duration: float = 0.06, settle_time: float = 0.5,
                 backoff: float = 0.0):
        if displacement == 0 or not velocity > 0 or not max_tracking_error > 0:
            raise ValueError("displacement must be non-zero, velocity and "
                             "max_tracking_error positive")
        if backoff < 0 or trip_duration < 0 or settle_time < 0:
            raise ValueError("backoff, trip_duration and settle_time must be >= 0")
        if max_load is not None and max_load < 0:
            raise ValueError("max_load must be >= 0")
        self.start = float(start)
        self.end = self.start + float(displacement)
        self.sign = math.copysign(1.0, displacement)
        self.velocity = float(velocity)
        self.max_tracking_error = float(max_tracking_error)
        self.max_load = float(max_load) if max_load else 0.0
        self.trip_duration = float(trip_duration)
        self.settle_time = float(settle_time)
        self.backoff = float(backoff)
        self.over_since = math.nan
        self.reached_at = math.nan
        self.peak_error = 0.0
        self.peak_load = 0.0
        self.done = False
        self.stopped = False
        self.reason = ""
        self.position = self.start
        #: Where the joint should end up: the backed-off pose after a stop.
        self.target = self.start

    def finish(self, position: float, stopped: bool, reason: str,
               target: Optional[float] = None) -> None:
        self.done = True
        self.stopped = stopped
        self.reason = reason
        self.position = float(position)
        self.target = float(position if target is None else target)

    def step(self, commanded: float, position: float, load: Optional[float],
             now: float, period: float) -> float:
        """Check the last cycle and give the goal for the next one.

        Parameters
        ----------
        commanded : float
            Goal sent in the previous cycle [rad].
        position : float
            Angle read this cycle [rad].
        load : float or None
            Load read this cycle, signed like the torque the actuator exerts;
            None when the actuator reports none.
        now : float
            Time [s].
        period : float
            Seconds since the previous cycle.

        Returns
        -------
        float
            Goal to send now [rad]. Once :attr:`done`, :attr:`target` is where
            the joint is to be held.
        """
        error = self.sign * (commanded - position)
        push = self.sign * load if load is not None else 0.0
        self.peak_error = max(self.peak_error, error)
        self.peak_load = max(self.peak_load, push)
        exceeded = []
        if error > self.max_tracking_error:
            exceeded.append("tracking error {:.3f} rad > {:.3f}".format(
                error, self.max_tracking_error))
        if self.max_load > 0 and push > self.max_load:
            exceeded.append("load {:.3f} > {:.3f}".format(push, self.max_load))
        if exceeded:
            if math.isnan(self.over_since):
                self.over_since = now
            if now - self.over_since >= self.trip_duration:
                # Stop pushing where the joint is, then ease off the obstruction.
                self.finish(position, True, "; ".join(exceeded),
                            position - self.sign * self.backoff)
                return position
        else:
            self.over_since = math.nan
        remaining = self.end - commanded
        if remaining == 0:
            if math.isnan(self.reached_at):
                self.reached_at = now
            if now - self.reached_at >= self.settle_time:
                self.finish(position, False, "reached", self.end)
            return self.end
        return commanded + math.copysign(
            min(self.velocity * period, abs(remaining)), remaining)

    def summary(self) -> Tuple[bool, bool, str, float, float, float, float]:
        """(done, stopped, reason, start, position, peak_error, peak_load)."""
        return (self.done, self.stopped, self.reason, self.start, self.position,
                self.peak_error, self.peak_load)
