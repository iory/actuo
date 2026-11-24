"""Hardware: modules plugged and unplugged by a fault injector, checked end to end.

Runs only when ``ACTUO_RIG`` names a rig file (see :mod:`actuo.testing.rig`);
skipped everywhere else, CI included. Example::

    ACTUO_RIG=examples/rigs/feetech_4port/rig.yaml python -m pytest tests/hardware -v

The robot interface runs on the rig's real bus (``hot_plug``) with torque off
throughout: nothing here moves a servo. What is checked:

* every joint attaches with everything plugged;
* for **every combination** of plugged joints, ``attached_joints()`` becomes
  exactly that combination, and each ``robot_modes`` description of the rig
  selects from the live bus what it selects from the combination itself --
  the same robot, the same problems;
* a joint whose replies are cut (commands still reaching it) is detached, and
  comes back when the line is restored.
"""

from __future__ import annotations

import os

import pytest

RIG = os.environ.get("ACTUO_RIG")

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(not RIG, reason="set ACTUO_RIG to a rig file to run hardware tests"),
]

#: Seconds for the attached set to follow a change (the FEETECH bus took up to
#: 1.1 s to take a powered-up servo back on 2026-11-24).
SETTLE = 5.0


@pytest.fixture(scope="module")
def rig():
    from actuo.testing.rig import ServoRig

    r = ServoRig.from_file(RIG)
    r.restore()
    yield r
    r.restore()
    r.close()


@pytest.fixture(scope="module")
def ri(rig):
    from skrobot.model import RobotModel

    from actuo.robot_interface import ActuoRobotInterface

    if rig.robot_config is None or rig.urdf is None:
        pytest.skip("the rig file names no robot config or urdf")
    interface = ActuoRobotInterface(RobotModel.from_urdf(rig.urdf), rig.robot_config,
                                    required_joints=[])
    yield interface
    interface.close()


def _settle(rig, ri, joints):
    from actuo.testing.rig import wait_attached

    rig.only(joints)
    return wait_attached(ri, joints, SETTLE)


def test_every_joint_attaches_with_everything_plugged(rig, ri):
    reached, got, waited = _settle(rig, ri, rig.joints)
    assert reached, "attached {} after {:.1f}s, wanted {}".format(got, waited, rig.joints)


def test_attached_and_robot_modes_follow_every_combination(rig, ri):
    from actuo.drivers.robot_modes import RobotModes

    descriptions = [RobotModes(path) for path in rig.modes]
    failures = []
    for combo in rig.combinations():
        reached, got, waited = _settle(rig, ri, combo)
        if not reached:
            failures.append("plugged {}: attached {} after {:.1f}s".format(
                list(combo), got, waited))
            continue
        for modes in descriptions:
            live = modes.select(set(ri.attached_joints()))
            expected = modes.select(set(combo))
            if (live.attached, live.problems, live.missing_base) != (
                    expected.attached, expected.problems, expected.missing_base):
                failures.append("plugged {}, {}: live {} / {}, expected {} / {}".format(
                    list(combo), os.path.basename(modes.path), live.attached,
                    live.problems, expected.attached, expected.problems))
    rig.restore()
    assert not failures, "\n".join(failures)


def test_a_joint_whose_replies_are_cut_is_detached_and_comes_back(rig, ri):
    from actuo.testing.fault_injectors import DROP_REPLIES
    from actuo.testing.rig import wait_attached

    if DROP_REPLIES not in rig.injector.capabilities or not rig.drop_joints:
        pytest.skip("this rig cannot cut replies")
    failures = []
    for joint in rig.drop_joints:
        reached, got, _ = _settle(rig, ri, rig.joints)
        assert reached, "not everything attached before the test: {}".format(got)
        rig.drop_replies(joint)
        others = [j for j in rig.joints if j != joint]
        reached, got, waited = wait_attached(ri, others, SETTLE)
        if not reached:
            failures.append("replies of {} cut: attached {} after {:.1f}s".format(
                joint, got, waited))
        rig.restore()
        reached, got, waited = wait_attached(ri, rig.joints, SETTLE)
        if not reached:
            failures.append("after restoring {}: attached {} after {:.1f}s".format(
                joint, got, waited))
    assert not failures, "\n".join(failures)
