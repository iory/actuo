"""ActuoRobotInterface against the RCB-4mini emulator: a 2-joint arm, no ROS."""

import math
import os
import threading
import time

import pytest
import yaml

skrobot = pytest.importorskip("skrobot")
pytest.importorskip("rcb4.protocol")

from skrobot.model import RobotModel  # noqa: E402

from actuo.drivers.actuators.kondo_ics import CENTER_PULSE, PULSE_PER_DEG  # noqa: E402
from actuo.drivers.kondo_rcb4_emulator import Rcb4Emulator, serve  # noqa: E402
from actuo.robot_interface import ActuoRobotInterface  # noqa: E402

URDF = """<robot name="two">
  <link name="base"/><link name="l1"/><link name="l2"/>
  <joint name="j0" type="revolute"><parent link="base"/><child link="l1"/>
    <axis xyz="0 0 1"/><limit lower="-1.5" upper="1.5" effort="1" velocity="2"/></joint>
  <joint name="j1" type="revolute"><parent link="l1"/><child link="l2"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 1 0"/>
    <limit lower="-1.5" upper="1.5" effort="1" velocity="2"/></joint>
</robot>
"""


def start(tmp_path, servos=(2, 4), bus=None, **kwargs):
    """A 2-joint arm (j0 on ICS 2, j1 on ICS 4) on an emulated board.

    ``servos`` are the ICS numbers plugged in at start; ``bus`` adds to the
    bus block of the config. Returns (interface, board).
    """
    link = str(tmp_path / "rcb4")
    bus_block = {"plugin": "kondo_rcb4_bus", "state_slots": ["current_angle"],
                 "ics_bitrate": 1250000,
                 "transport": {"plugin": "kondo_rcb4", "channel": link,
                               "bitrate": 1250000}}
    bus_block.update(bus or {})
    config = {
        "bus": bus_block,
        "joints": [
            {"name": n, "actuator": {"plugin": "kondo_ics", "id": i, "velocity": 1,
                                     "direction": 1, "offset_deg": 0.0,
                                     "position_limit_deg": [-85.0, 85.0]}}
            for n, i in (("j0", 2), ("j1", 4))],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    urdf = tmp_path / "two.urdf"
    urdf.write_text(URDF)
    board = Rcb4Emulator(list(servos), servo_ics_bitrate=1250000, log=lambda m: None)
    threading.Thread(target=serve, args=(board, link, lambda m: None),
                     daemon=True).start()
    deadline = time.monotonic() + 5
    while not os.path.exists(link):
        assert time.monotonic() < deadline, "emulator did not start"
        time.sleep(0.01)
    interface = ActuoRobotInterface(RobotModel.from_urdf(str(urdf)), str(path),
                                    **kwargs)
    return interface, board


@pytest.fixture
def ri(tmp_path):
    interface, _ = start(tmp_path)
    yield interface
    interface.close()


def settle(ri, seconds=0.5):
    time.sleep(seconds)
    ri.update_robot_state()
    return ri.potentio_vector()


def test_a_limp_joint_does_not_follow_a_command(ri):
    before = ri.potentio_vector()
    ri.angle_vector([0.6, -0.4], time=0.3)
    assert not any(ri.wait_interpolation(timeout=2.0))
    after = settle(ri)
    assert after == pytest.approx(before, abs=1e-3)


def test_servo_on_moves_nothing_then_angle_vector_reaches_the_target(ri):
    before = ri.potentio_vector()
    ri.angle_vector([0.6, -0.4], time=0.3)   # planned while limp: dropped by servo_on
    ri.servo_on()
    assert settle(ri, 0.3) == pytest.approx(before, abs=1e-3)
    ri.angle_vector([0.6, -0.4], time=0.5)
    assert ri.is_interpolating()
    assert not any(ri.wait_interpolation(timeout=3.0))
    assert settle(ri) == pytest.approx([0.6, -0.4], abs=0.01)


def test_one_joint_by_name(ri):
    ri.servo_on(["j1"])
    ri.angle_vector({"j1": 0.3}, time=0.3)
    ri.wait_interpolation(timeout=3.0)
    measured = settle(ri)
    assert measured[1] == pytest.approx(0.3, abs=0.01)
    assert measured[0] == pytest.approx(0.0, abs=1e-3)   # j0 limp, untouched


def test_unknown_joint_is_refused(ri):
    with pytest.raises(KeyError):
        ri.servo_on(["elbow"])


def test_a_command_outside_the_envelope_is_fenced(ri):
    ri.servo_on()
    ri.angle_vector([1.4, 0.0], time=0.3)       # URDF allows 1.5, the config 85 deg
    ri.wait_interpolation(timeout=3.0)
    assert settle(ri)[0] <= math.radians(85.0) + 0.01


def test_required_joints_needs_a_hot_plug_bus(tmp_path):
    with pytest.raises(ValueError, match="hot-plug"):
        start(tmp_path, required_joints=["j0"])


# --------------------------------------------------------------------------
# Servos that come and go (hot_plug)
# --------------------------------------------------------------------------

HOT = {"hot_plug": True, "presence_period": 2, "absent_timeout_s": 0.1}


def wait_for(condition, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.02)


@pytest.fixture
def hot(tmp_path):
    """j1's servo (ICS 4) is not plugged in at start."""
    interface, board = start(tmp_path, servos=(2,), bus=HOT,
                             required_joints=["j0"])
    yield interface, board
    interface.close()


def test_a_missing_servo_does_not_stop_the_start(hot):
    ri, _ = hot
    assert ri.attached_joints() == ["j0"]
    assert ri.detached_joints() == ["j1"]
    assert ri.controllers() == {"default_controller": ["j0"]}
    measured = ri.potentio_vector()
    assert measured[0] == pytest.approx(0.0, abs=1e-3)
    assert math.isnan(measured[1])


def test_a_detached_joint_cannot_be_moved_or_switched_on(hot):
    ri, _ = hot
    with pytest.raises(ValueError, match="not attached"):
        ri.angle_vector({"j1": 0.3}, time=0.3)
    with pytest.raises(ValueError, match="not attached"):
        ri.servo_on(["j1"])
    ri.servo_on()                          # every attached joint: j0 only
    assert ri.supervisor.enabled_joints == ["j0"]
    ri.angle_vector([0.4, 0.9], time=0.3)  # a vector moves the attached joints
    assert not any(ri.wait_interpolation(timeout=3.0))
    assert settle(ri)[0] == pytest.approx(0.4, abs=0.01)


def test_a_required_joint_that_never_answers_refuses_the_start(tmp_path):
    with pytest.raises(RuntimeError, match="j1"):
        start(tmp_path, servos=(2,), bus=HOT, required_joints=["j1"],
              startup_timeout=1.0)


def test_a_servo_plugged_in_later_is_attached_limp_where_it_is(hot):
    ri, board = hot
    pulse = int(round(CENTER_PULSE + 20 * PULSE_PER_DEG))
    board.plug(4, pulse)
    wait_for(lambda: "j1" in ri.attached_joints())
    assert ri.supervisor.enabled_joints == []
    assert settle(ri, 0.2)[1] == pytest.approx(math.radians(20), abs=0.01)
    # Seeded from where it is: switching it on moves nothing.
    ri.servo_on(["j1"])
    assert settle(ri, 0.3)[1] == pytest.approx(math.radians(20), abs=0.01)
    ri.angle_vector({"j1": -0.2}, time=0.3)
    assert not any(ri.wait_interpolation(timeout=3.0))
    assert settle(ri)[1] == pytest.approx(-0.2, abs=0.01)


def test_a_servo_pulled_out_is_switched_off_and_comes_back_limp(tmp_path):
    ri, board = start(tmp_path, bus=HOT, required_joints=["j0", "j1"])
    try:
        ri.servo_on()
        ri.angle_vector([0.3, 0.6], time=2.0)
        time.sleep(0.3)
        board.unplug(4)
        wait_for(lambda: "j1" not in ri.attached_joints())
        assert ri.supervisor.enabled_joints == ["j0"]
        assert math.isnan(settle(ri, 0.0)[1])
        assert ri.controllers() == {"default_controller": ["j0"]}
        # j0 carries on to its goal.
        assert not any(ri.wait_interpolation(timeout=3.0))
        assert settle(ri)[0] == pytest.approx(0.3, abs=0.01)

        # Back in somewhere else: attached again, limp, from where it is now,
        # and nothing from before it was pulled is sent to it.
        pulse = int(round(CENTER_PULSE - 10 * PULSE_PER_DEG))
        board.plug(4, pulse)
        wait_for(lambda: "j1" in ri.attached_joints())
        assert ri.supervisor.enabled_joints == ["j0"]
        assert settle(ri, 0.3)[1] == pytest.approx(math.radians(-10), abs=0.01)
        assert board.enabled[4] is False
    finally:
        ri.close()
