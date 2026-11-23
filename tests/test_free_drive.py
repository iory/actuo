"""Free drive: the torque law, and the supervisor's way in and out of it.

No bus, no robot: a fake bus records the commands, and a gravity model is a
plain function.
"""

from __future__ import annotations

import numpy as np
import pytest

from actuo.core.contracts import ActuatorState, Capabilities, OperatingMode
from actuo.drivers.free_drive import (FreeDrive, FreeDriveConfig, FreeDriveJoint,
                                      free_drive_config)
from actuo.drivers.servo import ServoSupervisor

STRIDE = 3
NAMES = ["joint0", "joint1"]


def _config(**kwargs):
    joints = {n: FreeDriveJoint(friction=0.1, back_emf=0.5, hold=0.2,
                                stall_torque=5.0) for n in NAMES}
    return FreeDriveConfig(joints=joints, **kwargs)


# --- the torque law ---------------------------------------------------------


def test_at_rest_it_exerts_the_holding_torque():
    drive = FreeDrive(_config(), NAMES, [None, None])
    positions = [0.3, -0.2]
    drive.start(0, positions)
    torques, runaway = drive.torques([0], positions, [1.5, 9.0], 0.01)
    assert runaway == []
    assert torques == {0: pytest.approx(1.5)}


def test_soft_hold_pulls_back_and_is_dragged_past_the_band():
    # Friction help off (it starts above 10 rad/s): only the spring is left.
    drive = FreeDrive(_config(hold_band=0.02, friction_velocity=10.0,
                              max_velocity=100.0), NAMES, [None, None])
    drive.start(0, [0.0, 0.0])
    # 0.01 rad off the anchor: half the band, half the hold torque, back.
    torques, _ = drive.torques([0], [0.01, 0.0], [0.0, 0.0], 0.01)
    assert torques[0] == pytest.approx(-0.1, abs=0.02)
    # Pushed far: the anchor follows, so the pull stays at most `hold`.
    for _ in range(50):
        torques, _ = drive.torques([0], [0.5, 0.0], [0.0, 0.0], 0.01)
    assert abs(torques[0]) <= 0.2 + 1e-9
    assert drive.anchor[0] == pytest.approx(0.5 - 0.02)


def test_friction_and_back_emf_help_a_moving_joint():
    drive = FreeDrive(_config(hold_band=1.0), NAMES, [None, None])
    drive.start(0, [0.0, 0.0])
    angle = 0.0
    for _ in range(40):      # 0.5 rad/s, steady
        angle += 0.005
        torques, _ = drive.torques([0], [angle, 0.0], [0.0, 0.0], 0.01)
    velocity = drive.filter.velocity[0]
    assert velocity == pytest.approx(0.5, rel=0.05)
    spring = 0.2 * (drive.anchor[0] - angle) / 1.0
    assert torques[0] == pytest.approx(spring + 0.1 + 0.5 * velocity, rel=1e-6)


def test_no_help_past_the_envelope():
    drive = FreeDrive(_config(hold_band=1.0), NAMES, [(-1.0, 0.1), None])
    drive.start(0, [0.0, 0.0])
    angle = 0.0
    for _ in range(40):
        angle += 0.005
        torques, _ = drive.torques([0], [angle, 0.0], [0.0, 0.0], 0.01)
    # Out past 0.1 and moving further out: only the spring is left.
    spring = 0.2 * (drive.anchor[0] - angle) / 1.0
    assert torques[0] == pytest.approx(spring)


def test_a_runaway_is_reported_not_driven():
    drive = FreeDrive(_config(max_velocity=1.0), NAMES, [None, None])
    drive.start(0, [0.0, 0.0])
    torques, runaway = {}, []
    angle = 0.0
    for _ in range(20):
        angle += 0.05        # 5 rad/s
        torques, runaway = drive.torques([0], [angle, 0.0], [0.0, 0.0], 0.01)
        if runaway:
            break
    assert runaway and runaway[0][0] == 0
    assert 0 not in torques


def test_duty_is_torque_over_stall_and_capped():
    drive = FreeDrive(_config(duty_limit=0.3), NAMES, [None, None])
    assert drive.duty(0, 1.0) == pytest.approx(0.2)
    assert drive.duty(0, 10.0) == pytest.approx(0.3)
    assert drive.duty(0, -10.0) == pytest.approx(-0.3)


def test_config_from_a_robot_document():
    document = {
        "free_drive": {"dither": 0.1, "velocity_filter": [0.3, 0.05]},
        "joints": [
            {"name": "joint0", "free_drive": {"friction": 0.09, "stall_torque": 4.7}},
            {"name": "joint1"},
        ],
    }
    config = free_drive_config(document, "robot.yaml")
    assert set(config.joints) == {"joint0"}
    assert config.joints["joint0"].stall_torque == pytest.approx(4.7)
    assert config.dither == pytest.approx(0.1)
    assert config.velocity_filter == (0.3, 0.05)
    assert free_drive_config({"joints": [{"name": "joint0"}]}, "r.yaml") is None
    with pytest.raises(ValueError):
        free_drive_config({"joints": [{"name": "joint0", "free_drive": {}}]}, "r.yaml")
    with pytest.raises(ValueError):
        free_drive_config({"free_drive": {"gravity": [0, 0, -9.81]},
                           "joints": [{"name": "joint0",
                                       "free_drive": {"frction": 1}}]}, "r.yaml")


# --- the supervisor ---------------------------------------------------------


class Actuator:
    def __init__(self, node_id, modes):
        self.id = node_id
        self.capabilities = Capabilities(modes=tuple(modes))

    def position_refusal(self, position):
        return None


class Bus:
    def __init__(self, positions, modes):
        self.actuators = [Actuator(10 + i, modes) for i in range(len(positions))]
        self.states = {a.id: ActuatorState(position=p, error=0)
                       for a, p in zip(self.actuators, positions)}
        self.batches = []

    def last_state(self, device_id):
        return self.states.get(device_id)

    def move(self, index, position):
        self.states[self.actuators[index].id] = ActuatorState(position=position,
                                                              error=0)

    def write_batch(self, commands):
        self.batches.append(list(commands))

    def release_all(self):
        pass


class Shm:
    def __init__(self, commands):
        self.c = np.zeros(len(commands) * STRIDE)
        self.set(commands)

    def set(self, commands):
        for i, v in enumerate(commands):
            self.c[i * STRIDE] = v

    def read_command(self):
        return self.c


def _supervisor(modes=(OperatingMode.POSITION, OperatingMode.PWM), gravity=None):
    bus = Bus([0.3, -0.2], modes)
    shm = Shm([0.3, -0.2])
    drive = FreeDrive(_config(), NAMES, [None, None])
    sup = ServoSupervisor(bus, NAMES, 0.05, STRIDE, free_drive=drive,
                          gravity=gravity or (lambda angles: [1.0, 2.0]))
    assert sup.enable(shm)[0]
    return sup, bus, shm


def test_free_drive_sends_a_pwm_duty_and_leaves_the_rest_holding():
    sup, bus, shm = _supervisor()
    ok, reason = sup.set_free_drive(["joint0"])
    assert ok, reason
    sup.service(shm)
    first, second = bus.batches[-1]
    assert first.mode is OperatingMode.PWM
    assert first.pwm == pytest.approx(1.0 / 5.0)      # holding torque / stall
    assert second.mode is OperatingMode.POSITION and second.position == pytest.approx(-0.2)
    assert sup.inspect(shm)[0].free_drive


def test_free_drive_sends_a_torque_to_a_current_joint():
    sup, bus, shm = _supervisor(modes=(OperatingMode.POSITION, OperatingMode.CURRENT))
    assert sup.set_free_drive(["joint1"])[0]
    sup.service(shm)
    command = bus.batches[-1][1]
    assert command.mode is OperatingMode.CURRENT
    assert command.current == pytest.approx(2.0)


def test_refused_without_torque_or_a_torque_mode():
    sup, bus, shm = _supervisor()
    sup.disable(["joint1"])
    ok, reason = sup.set_free_drive()
    assert not ok and "joint1 has no torque" in reason
    assert sup.free_drive_joints == []          # all or nothing
    sup, _, _ = _supervisor(modes=(OperatingMode.POSITION,))
    ok, reason = sup.set_free_drive(["joint0"])
    assert not ok and "neither a torque nor a PWM" in reason


def test_leaving_holds_until_the_controller_commands_that_pose():
    sup, bus, shm = _supervisor()
    assert sup.set_free_drive(["joint0"])[0]
    # Pushed by hand to 1.0, slowly: a leap would be a runaway.
    for position in np.linspace(0.3, 1.0, 1400):
        bus.move(0, float(position))
        sup.service(shm)
    assert sup.free_drive_joints == ["joint0"]
    ok, reason = sup.set_free_drive(["joint0"], enable=False)
    assert ok and "holding where" in reason
    # The controller still commands 0.3: the joint holds 1.0, not thrown back.
    sup.service(shm)
    assert bus.batches[-1][0].mode is OperatingMode.POSITION
    assert bus.batches[-1][0].position == pytest.approx(1.0)
    assert sup.inspect(shm)[0].resyncing
    # Re-synced: the controller commands where the joint is; it follows again.
    shm.set([1.02, -0.2])
    sup.service(shm)
    assert bus.batches[-1][0].position == pytest.approx(1.02)
    assert not sup.inspect(shm)[0].resyncing
    shm.set([1.04, -0.2])
    sup.service(shm)
    assert bus.batches[-1][0].position == pytest.approx(1.04)


def test_torque_off_ends_free_drive():
    sup, bus, shm = _supervisor()
    assert sup.set_free_drive(["joint0"])[0]
    sup.disable(["joint0"])
    assert sup.free_drive_joints == []
    sup.service(shm)
    assert bus.batches[-1][0] is None


def test_no_free_drive_without_a_model():
    bus = Bus([0.0], (OperatingMode.POSITION, OperatingMode.PWM))
    sup = ServoSupervisor(bus, ["joint0"], 0.05, STRIDE)
    ok, reason = sup.set_free_drive()
    assert not ok and "no free_drive" in reason
    with pytest.raises(ValueError):
        ServoSupervisor(bus, ["joint0"], 0.05, STRIDE,
                        free_drive=FreeDrive(_config(), NAMES, [None, None]))


def test_a_runaway_leaves_free_drive_and_holds():
    sup, bus, shm = _supervisor()
    assert sup.set_free_drive(["joint0"])[0]
    sup.service(shm)
    bus.move(0, 0.9)                 # a 0.6 rad leap in one cycle
    sup.service(shm)
    assert sup.free_drive_joints == []
    assert "ran away" in sup.pop_filter_event()
    sup.service(shm)
    command = bus.batches[-1][0]
    assert command.mode is OperatingMode.POSITION
    assert command.position == pytest.approx(0.9)
    assert sup.inspect(shm)[0].resyncing


PENDULUM = """<robot name="pendulum">
  <link name="base"/>
  <link name="arm">
    <inertial><origin xyz="0.5 0 0"/><mass value="1.0"/>
      <inertia ixx="1e-4" iyy="1e-4" izz="1e-4" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="tool">
    <inertial><origin xyz="0 0 0"/><mass value="2.0"/>
      <inertia ixx="1e-4" iyy="1e-4" izz="1e-4" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="hinge" type="revolute"><parent link="base"/><child link="arm"/>
    <axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="10" velocity="1"/></joint>
  <joint name="mount" type="fixed"><parent link="arm"/><child link="tool"/>
    <origin xyz="1.0 0 0"/></joint>
</robot>"""


def test_gravity_model_on_a_pendulum():
    pytest.importorskip("skrobot")
    from actuo.dynamics.gravity import GravityModel

    # Arm along +x, axis +y: gravity (-z) pulls the arm down, i.e. turns it
    # by +m g L about +y, so the joint must exert -m g L.
    arm_only = GravityModel(PENDULUM, unmounted_links=["tool"])
    assert arm_only.torques({"hinge": 0.0}, ["hinge"])[0] == pytest.approx(
        -1.0 * 9.81 * 0.5, rel=1e-3)
    with_tool = GravityModel(PENDULUM)
    assert with_tool.torques({"hinge": 0.0}, ["hinge"])[0] == pytest.approx(
        -(1.0 * 0.5 + 2.0 * 1.0) * 9.81, rel=1e-3)
    # Hanging straight down: nothing to hold.
    assert with_tool.torques({"hinge": np.pi / 2}, ["hinge"])[0] == pytest.approx(
        0.0, abs=1e-6)
    with pytest.raises(ValueError):
        GravityModel(PENDULUM, unmounted_links=["no_such_link"])


def test_needs_service_while_free_and_after_each_transition():
    """A request/response bus writes only on a changed command; free drive
    changes none, so the supervisor has to say it needs the cycle."""
    sup, bus, shm = _supervisor()
    sup.service(shm)
    assert not sup.needs_service
    assert sup.set_free_drive(["joint0"])[0]
    assert sup.needs_service
    sup.service(shm)
    assert sup.needs_service              # every cycle while in free drive
    assert sup.set_free_drive(["joint0"], enable=False)[0]
    assert sup.needs_service              # the switch back must be written
    sup.service(shm)
    assert not sup.needs_service


def test_gravity_scale_scales_only_the_model():
    joints = {n: FreeDriveJoint(gravity_scale=0.5, stall_torque=5.0) for n in NAMES}
    drive = FreeDrive(FreeDriveConfig(joints=joints), NAMES, [None, None])
    drive.start(0, [0.0, 0.0])
    torques, _ = drive.torques([0], [0.0, 0.0], [1.5, 0.0], 0.01)
    assert torques[0] == pytest.approx(0.75)
