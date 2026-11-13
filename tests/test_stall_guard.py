"""StallGuard: a held joint pushing hard without following."""

from __future__ import annotations

import math
import os
import sys

import pytest

_HERE = os.path.dirname(__file__)

from actuo.core.contracts import OperatingMode  # noqa: E402
from actuo.drivers.servo import JointServo  # noqa: E402
from actuo.drivers.stall_guard import StallError, StallGuard  # noqa: E402


def knee(commanded_deg, measured_deg, effort, enabled=True,
         mode=OperatingMode.POSITION, name="left_knee_pitch"):
    return JointServo(name=name, enabled=enabled,
                      measured=math.radians(measured_deg),
                      commanded=math.radians(commanded_deg),
                      error=0, error_name="", mode=mode, effort=effort)


def guard():
    return StallGuard(math.radians(8.0), 80.0, 0.12)


def test_trips_after_the_hold_time():
    g = guard()
    g.check([knee(-55.3, -40.5, -114.5)], 0.00)
    g.check([knee(-58.7, -43.5, -119.5)], 0.10)
    with pytest.raises(StallError, match="left_knee_pitch"):
        g.check([knee(-61.9, -45.2, -133.4)], 0.13)


def test_a_break_restarts_the_clock():
    g = guard()
    g.check([knee(-55.3, -40.5, -114.5)], 0.00)
    g.check([knee(-50.0, -45.0, -114.5)], 0.06)    # back within 8 deg
    g.check([knee(-55.3, -40.5, -114.5)], 0.10)
    g.check([knee(-55.3, -40.5, -114.5)], 0.21)    # 0.11 s since the break


@pytest.mark.parametrize("joint", [
    knee(-55.0, -40.0, -50.0),                        # far, but not pushing
    knee(-45.0, -40.0, -140.0),                       # pushing, but following
    knee(-55.0, -40.0, -140.0, enabled=False),        # limp
    knee(-55.0, -40.0, math.nan),                     # no effort reported
    knee(-55.0, -40.0, -140.0, mode=OperatingMode.VELOCITY),
])
def test_does_not_judge(joint):
    g = guard()
    g.check([joint], 0.0)
    g.check([joint], 1.0)


def test_zero_seconds_disables():
    g = StallGuard(math.radians(8.0), 80.0, 0.0)
    g.check([knee(-60.0, -40.0, -140.0)], 0.0)
    g.check([knee(-60.0, -40.0, -140.0)], 5.0)
