"""_wait_future counts its timeout on the robot's clock.

A simulator running at a quarter of real time must get the same simulated
seconds as a fast one (a heavy simulation can spend several wall seconds per
simulated second), and a clock that has stopped must be reported rather than
waited on forever.
"""

import threading
import time

import pytest

pytest.importorskip('skrobot.interfaces.ros2.base')

from rclpy.task import Future  # noqa: E402

from actuo_ros import robot_interface as robot  # noqa: E402


class _Stamp(object):
    def __init__(self, ns):
        self.nanoseconds = ns


class FakeClock(object):
    """Advances ``rate`` simulated seconds per wall second (0: stopped)."""

    def __init__(self, rate, start=100.0):
        self.rate = rate
        self.start = start
        self.t0 = time.monotonic()

    def now(self):
        sim = self.start + self.rate * (time.monotonic() - self.t0)
        return _Stamp(int(sim * 1e9))


def _done_after(seconds):
    future = Future()
    threading.Timer(seconds, lambda: future.set_result('ok')).start()
    return future


def test_slow_clock_gets_its_simulated_seconds():
    # 0.6 s of wall time is 0.15 s on a 0.25x clock: inside a 0.3 s timeout,
    # which wall time would have exceeded.
    assert robot._wait_future(_done_after(0.6), 0.3, 'x',
                              FakeClock(0.25)) == 'ok'


def test_timeout_on_the_robot_clock():
    with pytest.raises(RuntimeError, match='robot clock'):
        robot._wait_future(Future(), 0.2, 'x', FakeClock(1.0))


def test_stopped_clock_is_reported(monkeypatch):
    monkeypatch.setattr(robot, 'CLOCK_STALL', 0.3)
    with pytest.raises(RuntimeError, match='clock has not moved'):
        robot._wait_future(Future(), 10.0, 'x', FakeClock(0.0))


def test_wall_time_without_a_clock():
    with pytest.raises(RuntimeError, match='within 0.2 s'):
        robot._wait_future(Future(), 0.2, 'x')
