"""Hardware: actuo.modular switching configurations on a rig.

Runs only when ``ACTUO_RIG`` names a rig file with a ``modular`` catalog. For
every configuration of the catalog: plugging exactly its servos makes it
current (when the catalog's own detection says it should), servos it shares
with the configuration before keep their torque without moving, and a
declaration switches to it whatever answers.
"""

from __future__ import annotations

import os
import time

import pytest

RIG = os.environ.get("ACTUO_RIG")

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(not RIG, reason="set ACTUO_RIG to a rig file to run hardware tests"),
]

SETTLE = 8.0
#: Raw counts a held servo may drift across a switch (encoder noise).
HELD_COUNTS = 3


@pytest.fixture(scope="module")
def rig():
    from actuo.testing.rig import ServoRig

    r = ServoRig.from_file(RIG)
    if r.modular is None:
        r.close()
        pytest.skip("the rig file names no modular catalog")
    r.restore()
    yield r
    r.restore()
    r.close()


def _wait(cond, timeout=SETTLE):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.05)
    return cond()


def _raw(mr, servo_id):
    state = mr.ri.bus.last_state(servo_id)
    return None if state is None else state.aux.get("raw_position")


def test_each_configuration_is_detected_and_shared_servos_keep_their_torque(rig):
    from actuo.modular import Catalog, ModularRobot

    catalog = Catalog(rig.modular)
    order = [c for c in catalog.names if catalog.detect(catalog.configurations[c].ids)[0] == c]
    if len(order) < 2:
        pytest.skip("fewer than two configurations can be told apart by their ids")
    first = catalog.configurations[order[0]]
    rig.only_ids(first.ids)
    time.sleep(0.5)
    failures = []
    mr = ModularRobot(rig.modular, initial=first.name)
    try:
        for name in order[1:] + order[:1]:
            previous = catalog.configurations[mr.current]
            target = catalog.configurations[name]
            shared = sorted(set(previous.joint_ids.values()) & set(target.joint_ids.values()))
            held_before = [j for j, i in previous.joint_ids.items() if i in shared]
            assert _wait(lambda held=held_before: set(mr.ri.attached_joints()) >= set(held)), \
                "{}: shared joints never attached".format(previous.name)
            if held_before:
                mr.ri.servo_on(held_before)
                time.sleep(0.3)
            raw_before = {i: _raw(mr, i) for i in shared}
            rig.only_ids(target.ids)
            if not _wait(lambda want=name: mr.current == want):
                failures.append("{} plugged: still {} ({})".format(
                    name, mr.current, mr.state().problems))
                continue
            held_after = sorted(mr.ri.supervisor.enabled_joints)
            expected = sorted(j for j, i in target.joint_ids.items() if i in shared)
            if held_after != expected:
                failures.append("{}: held {} after the switch, wanted {}".format(
                    name, held_after, expected))
            time.sleep(0.3)
            moved = {i: (raw_before[i], _raw(mr, i)) for i in shared
                     if raw_before[i] is None or _raw(mr, i) is None
                     or abs(_raw(mr, i) - raw_before[i]) > HELD_COUNTS}
            if moved:
                failures.append("{}: held servos moved {}".format(name, moved))
            mr.ri.servo_off()
            time.sleep(0.2)
    finally:
        mr.ri.servo_off()
        time.sleep(0.2)
        mr.close()
        rig.restore()
    assert not failures, "\n".join(failures)


def test_a_declaration_switches_whatever_answers_until_detection_resumes(rig):
    from actuo.modular import Catalog, ModularRobot

    catalog = Catalog(rig.modular)
    rig.restore()                       # everything plugged: detection may be ambiguous
    time.sleep(0.5)
    mr = ModularRobot(rig.modular)
    try:
        for name in catalog.names:
            state = mr.declare(name)
            assert mr.current == name and state.select == "declare"
            time.sleep(1.0)
            assert mr.current == name, "detection overrode the declaration"
        mr.resume_detection()
        assert mr.state().select == "detect"
    finally:
        mr.close()
        rig.restore()
