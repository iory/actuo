"""Unit tests for CollisionGate -- no scikit-robot needed.

A FakeMonitor stands in for SelfCollisionMonitor (same duck-typed surface:
index_of + check_angle_vector), declaring a collision region so the gate's
hold/latch/fail-safe behaviour can be asserted deterministically.
"""

from __future__ import annotations

import numpy as np

from actuo.safety.gate import CollisionGate
class FakeMonitor:
    """Collides when the single controlled joint value exceeds `threshold`."""

    def __init__(self, joint_names, controlled, threshold):
        self.joint_names = list(joint_names)
        self._controlled = controlled
        self._thr = threshold
        self.raise_on_check = False

    def index_of(self, name):
        return self.joint_names.index(name)

    def check_angle_vector(self, av):
        if self.raise_on_check:
            raise RuntimeError("checker exploded")
        val = av[self.joint_names.index(self._controlled)]
        if val > self._thr:
            return True, [("linkA", "linkB")]
        return False, []


def _gate(threshold=0.5, latched=True):
    names = ["j0", "j1", "j2"]
    mon = FakeMonitor(names, "j1", threshold)
    gate = CollisionGate(mon, ["j1"], base_angle_vector=np.zeros(3),
                         initial_safe=[0.0], latched=latched)
    return mon, gate


def test_passes_safe_commands():
    _, gate = _gate(threshold=0.5)
    for v in (0.0, 0.2, 0.4):
        r = gate.filter([v])
        assert not r.stopped
        assert np.isclose(r.applied[0], v)


def test_stops_and_holds_last_safe():
    _, gate = _gate(threshold=0.5)
    gate.filter([0.3])                 # safe -> last_safe = 0.3
    r = gate.filter([0.8])             # would collide
    assert r.stopped
    assert np.isclose(r.applied[0], 0.3)   # held at last safe, not 0.8
    assert r.colliding_pairs == [("linkA", "linkB")]


def test_latched_stays_stopped_even_if_next_is_safe():
    _, gate = _gate(threshold=0.5, latched=True)
    gate.filter([0.3])
    gate.filter([0.8])                 # latch
    r = gate.filter([0.1])             # safe again, but must stay latched
    assert r.stopped
    assert np.isclose(r.applied[0], 0.3)
    gate.reset(safe=[0.1])
    r2 = gate.filter([0.1])
    assert not r2.stopped


def test_unlatched_resumes_when_safe():
    _, gate = _gate(threshold=0.5, latched=False)
    gate.filter([0.3])
    assert gate.filter([0.8]).stopped
    r = gate.filter([0.2])             # unlatched: resumes on a safe command
    assert not r.stopped
    assert np.isclose(r.applied[0], 0.2)


def test_fail_safe_on_checker_error():
    mon, gate = _gate(threshold=0.5)
    gate.filter([0.3])
    mon.raise_on_check = True
    r = gate.filter([0.4])             # would be safe, but checker throws
    assert r.stopped                   # fail to STOP, never fail-open
    assert np.isclose(r.applied[0], 0.3)


# --------------------------------------------------------------------------
# Which checker the gate uses: the named check costs orders of magnitude more
# than the fast boolean path, so the choice must be recorded, not silent.
# --------------------------------------------------------------------------
class FastMonitor(FakeMonitor):
    """A monitor that offers the fast path, and counts which one is used."""

    def __init__(self, *args, **kwargs):
        super(FastMonitor, self).__init__(*args, **kwargs)
        self.slow_calls = 0
        self.fast_calls = 0

    def check_angle_vector(self, av):
        self.slow_calls += 1
        return super(FastMonitor, self).check_angle_vector(av)

    def check_angle_vector_fast(self, av):
        self.fast_calls += 1
        return FakeMonitor.check_angle_vector(self, av)


def test_the_gate_takes_the_fast_path_when_the_monitor_has_one():
    monitor = FastMonitor(["j0", "j1", "j2"], "j1", 0.5)
    gate = CollisionGate(monitor, ["j0", "j1", "j2"],
                         base_angle_vector=[0.0, 0.0, 0.0],
                         initial_safe=[0.0, 0.0, 0.0])
    assert gate.fast_path is True

    gate.filter([0.0, 0.2, 0.0])

    assert monitor.fast_calls == 1
    assert monitor.slow_calls == 0, \
        "the named check is the entire cost; it must not run every cycle"


def test_a_monitor_without_the_fast_path_still_works_and_says_so():
    """An upstreamed or duck-typed monitor must not break -- but the slowdown
    has to be visible, not mysterious."""
    monitor = FakeMonitor(["j0", "j1", "j2"], "j1", 0.5)
    gate = CollisionGate(monitor, ["j0", "j1", "j2"],
                         base_angle_vector=[0.0, 0.0, 0.0],
                         initial_safe=[0.0, 0.0, 0.0])

    assert gate.fast_path is False
    result = gate.filter([0.0, 0.9, 0.0])
    assert result.stopped is True
