"""Stop a joint that pushes hard and gets nowhere.

ROS-free, like servo.py, so the rule is testable without a bus.

The failure it exists for
-------------------------
On 2026-10-13 a homing trajectory on a biped's CubeMars legs took a torque-on
twitch as its starting speed and drove a knee pitch joint (AK80-64) away from
home until it met something and stopped following. The command ran on to -69 deg while the
joint sat between -36 and -51 deg, and the drive's torque climbed past its
120 N*m rating to the 144 N*m end of its wire range for 0.35 s. Then the joint
gave 25 deg in 0.12 s with the torque collapsing: the gearbox had failed.
Nothing watched for it: the overspeed guard needs a joint that is moving, the
stale guard one that is silent, and this one was neither.

The rule
--------
A held joint in POSITION mode is stalled when, at once,

* it is more than ``error_rad`` from its command, and
* its reported effort is above ``effort_nm`` in magnitude,

and it has been so continuously for ``seconds``. One joint is enough.

The defaults (8 deg, 80 N*m, 0.12 s) come from logs of the same robot: across
eight runs of walking, falls and a multi-drive communication dropout, the
longest such stretch on any joint was 70 ms. The failing knee held it for
322 ms; with these defaults the guard would have tripped about 0.2 s before
the gear went, before the drive reached its torque ceiling.

Only joints that report an effort are judged. A NaN effort is "not reported",
never zero, so a drive that gives amps but no torque is skipped rather than
treated as unloaded.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable

from actuo.core.contracts import OperatingMode


class StallError(RuntimeError):
    """A held joint pushed past the effort threshold without following."""


class StallGuard:
    """Watch command-vs-measured error together with effort, per joint.

    Parameters
    ----------
    error_rad : float
        Distance from the command, in rad, that counts as not following.
    effort_nm : float
        Effort magnitude, in N*m at the output, that counts as pushing hard.
    seconds : float
        How long both must hold, uninterrupted, before :meth:`check` raises.
        ``0`` or less disables the guard.
    """

    def __init__(self, error_rad: float, effort_nm: float, seconds: float):
        self.error_rad = float(error_rad)
        self.effort_nm = float(effort_nm)
        self.seconds = float(seconds)
        #: joint -> monotonic time the current stretch began
        self._since: Dict[str, float] = {}

    @property
    def enabled(self) -> bool:
        return self.seconds > 0.0

    def check(self, joints: Iterable, now: float) -> None:
        """Raise StallError if any joint has been stalled for ``seconds``.

        Parameters
        ----------
        joints : iterable of JointServo
            This cycle's picture, as ``ServoSupervisor.inspect`` returns it.
        now : float
            Monotonic time of this cycle, in seconds.
        """
        if not self.enabled:
            return
        stalled = []
        for joint in joints:
            if not self._pushing(joint):
                self._since.pop(joint.name, None)
                continue
            began = self._since.setdefault(joint.name, now)
            if now - began >= self.seconds:
                stalled.append((joint, now - began))
        if stalled:
            raise StallError("; ".join(
                "{} commanded {:+.1f} deg but at {:+.1f} deg, pushing "
                "{:+.1f} N*m, for {:.0f} ms".format(
                    joint.name, math.degrees(joint.commanded),
                    math.degrees(joint.measured), joint.effort,
                    held * 1000.0)
                for joint, held in stalled)
                + " -- it is driving into something it cannot move "
                  "(limits {:.1f} deg / {:.0f} N*m / {:.0f} ms)".format(
                      math.degrees(self.error_rad), self.effort_nm,
                      self.seconds * 1000.0))

    def _pushing(self, joint) -> bool:
        if not joint.enabled or joint.mode is not OperatingMode.POSITION:
            return False
        values = (joint.commanded, joint.measured, joint.effort)
        if not all(math.isfinite(v) for v in values):
            return False
        return (abs(joint.commanded - joint.measured) > self.error_rad
                and abs(joint.effort) > self.effort_nm)
