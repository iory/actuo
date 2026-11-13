"""Tests for the servo (torque) supervisor -- the safety rule, in isolation.

No ROS, no CAN, no robot: the point is that the rule is provable on its own.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)

from actuo.core.contracts import (  # noqa: E402
    ActuatorState, Capabilities, OperatingMode)
from actuo.drivers.servo import ServoSupervisor  # noqa: E402

STRIDE = 3


class FakeActuator:
    """The bits of the Actuator contract the supervisor actually asks about."""

    def __init__(self, node_id, refuse=None,
                 modes=(OperatingMode.POSITION, OperatingMode.VELOCITY,
                        OperatingMode.CURRENT), has_position_sense=True):
        self.id = node_id
        # None = "I can command anything". A string = the reason this actuator
        # would refuse, so a test can exercise the pre-enable check that exists
        # because a radian-space guard cannot see an unrepresentable wire value.
        self._refuse = refuse
        # The supervisor derives each joint's mode from what its actuator can
        # do, so this fake has to declare it. Default: all three, i.e. "whatever
        # the supervisor was constructed for" -- which is what every test here
        # assumed before wheels made the mode per joint.
        # has_position_sense is what decides whether a joint comes up driven:
        # no pose means no stale pose to be thrown to. A real wheel declares
        # False, so a fake that did not would test the wrong robot.
        self.capabilities = Capabilities(
            modes=tuple(modes), has_position_sense=has_position_sense)

    def position_refusal(self, position):
        return self._refuse


class FakeBus:
    """Records what the supervisor put on the wire, and nothing else."""

    def __init__(self, positions):
        self.actuators = [FakeActuator(104 + i) for i in range(len(positions))]
        self._states = {
            a.id: ActuatorState(position=p, temperature=30.0, error=0)
            for a, p in zip(self.actuators, positions)
        }
        self.commands = []     # batches; a None slot = that joint was released
        self.releases = 0

    def last_state(self, device_id):
        return self._states.get(device_id)

    def write_batch(self, commands):
        # Mirrors the real CanBus contract: None = release that actuator.
        self.commands.append([None if c is None else c.position
                              for c in commands])

    def release_all(self):
        self.releases += 1

    @property
    def positions_sent(self):
        """Batches that put a position on the wire for at least one joint.

        The distinction the safety tests turn on: an all-None batch is a
        release, not a command, however it was produced.
        """
        return [b for b in self.commands if any(v is not None for v in b)]

    def set_error(self, node_id, code):
        s = self._states[node_id]
        self._states[node_id] = ActuatorState(
            position=s.position, temperature=s.temperature, error=code)


class FakeShm:
    """Stands in for the command block. Tracks whether it was even looked at."""

    def __init__(self, commands):
        self._c = np.zeros(len(commands) * STRIDE)
        for i, v in enumerate(commands):
            self._c[i * STRIDE] = v
        self.reads = 0

    def read_command(self):
        self.reads += 1
        return self._c


def make(measured, commanded, threshold=0.05):
    bus = FakeBus(measured)
    shm = FakeShm(commanded)
    sup = ServoSupervisor(bus, ["joint1", "joint2", "joint3"], threshold, STRIDE,
                          error_names={5: "encoder fault"})
    return sup, bus, shm


AT = [1.70, 1.34, -0.24]


def test_starts_off():
    """A supervisor that came up enabled would apply torque nobody asked for."""
    sup, bus, _ = make(AT, AT)
    assert sup.all_off is True
    assert sup.any_enabled is False
    assert sup.all_enabled is False
    assert sup.enabled_joints == []


def test_off_never_reads_the_command_let_alone_sends_one():
    """Property 1: fully OFF has no reachable path from a setpoint to a frame."""
    sup, bus, shm = make(AT, [0.0, 0.0, 0.0])   # controller wants somewhere lethal
    for _ in range(10):
        sup.service(shm)
    assert bus.positions_sent == [], "OFF emitted a position command"
    assert bus.releases == 10
    assert shm.reads == 0, "OFF read the command block; it must not even look"


def test_on_sends_the_controllers_command():
    sup, bus, shm = make(AT, AT)
    ok, _ = sup.enable(shm)
    assert ok
    sup.service(shm)
    assert bus.commands == [pytest.approx(AT)]
    assert bus.releases == 0


def test_enable_is_refused_when_it_would_throw_the_robot():
    """The whole point: hand-moved robot + stale controller command."""
    moved = [1.70, 0.50, -0.24]              # operator pulled joint2 down
    sup, bus, shm = make(moved, AT)           # controller still says 1.34
    ok, reason = sup.enable(shm)

    assert ok is False
    assert sup.all_off is True
    assert "joint2" in reason
    assert "0.5000" in reason and "1.3400" in reason   # both numbers, not a vibe
    # and it points at the controller, not at moving the robot
    assert "trajectory controller" in reason
    sup.service(shm)
    assert bus.positions_sent == [], "a refused enable still armed the joint"
    assert bus.releases == 1


def test_threshold_separates_a_safe_mismatch_from_a_refused_one():
    """No claim about the exact boundary: at 0.05 rad, float arithmetic decides
    it, and a safety rule should not hinge on the last bit of a double."""
    inside, _, shm_a = make([1.70, 1.34, -0.24], [1.70, 1.38, -0.24], threshold=0.05)
    assert inside.enable(shm_a)[0] is True    # 0.04 rad: allowed

    outside, _, shm_b = make([1.70, 1.34, -0.24], [1.70, 1.40, -0.24], threshold=0.05)
    assert outside.enable(shm_b)[0] is False  # 0.06 rad: refused


def test_enable_refused_while_a_motor_reports_a_fault():
    sup, bus, shm = make(AT, AT)
    bus.set_error(105, 5)
    ok, reason = sup.enable(shm)
    assert ok is False
    assert "encoder fault" in reason          # named, not "error 5"


def test_force_is_explicit_and_says_so():
    """An override must be greppable, not a quietly widened threshold."""
    moved = [1.70, 0.50, -0.24]
    sup, bus, shm = make(moved, AT)
    assert sup.enable(shm)[0] is False

    ok, reason = sup.enable(shm, force=True)
    assert ok is True
    assert sup.all_enabled is True
    assert "FORCED" in reason
    assert "0.8400" in reason                 # states how far it will throw
    # and records what each joint was actually told, not just the verdict
    assert "joint2 at +0.5000 -> commanding +1.3400" in reason


def test_disable_is_always_allowed_and_warns_about_gravity():
    sup, bus, shm = make(AT, AT)
    sup.enable(shm)
    ok, reason = sup.disable()
    assert ok is True
    assert sup.all_off is True
    assert "limp" in reason and "gravity" in reason
    sup.service(shm)
    assert bus.positions_sent == []
    assert bus.releases == 1


def test_inspect_reports_the_jump_before_anyone_presses_a_button():
    moved = [1.70, 0.50, -0.24]
    sup, _, shm = make(moved, AT)
    joints = sup.inspect(shm)
    j2 = joints[1]
    assert j2.measured == pytest.approx(0.50)
    assert j2.commanded == pytest.approx(1.34)
    assert j2.mismatch == pytest.approx(0.84)
    assert sup.would_jump(j2) is True
    assert sup.would_jump(joints[0]) is False


def test_enable_logs_what_each_joint_was_commanded():
    """The forensic line. When a joint ran away, the logs said what was asked
    and what was measured, and nothing about what was COMMANDED -- which is the
    only number that explains a runaway."""
    sup, _, shm = make(AT, AT)
    ok, reason = sup.enable(shm)
    assert ok
    for name, value in zip(("joint1", "joint2", "joint3"), AT):
        assert "{} at {:+.4f} -> commanding {:+.4f}".format(name, value, value) in reason
    assert sup._last_enable == [
        ("joint1", 1.70, 1.70), ("joint2", 1.34, 1.34), ("joint3", -0.24, -0.24)]


def test_enabling_twice_is_idempotent_not_a_second_check():
    sup, _, shm = make(AT, AT)
    assert sup.enable(shm)[0] is True
    ok, reason = sup.enable(shm)
    assert ok is True and reason.startswith("already on")


# --- per-joint torque -----------------------------------------------------


def test_releasing_one_joint_leaves_the_others_holding():
    """The whole point of per-joint: freeing an elbow must not drop the arm."""
    sup, bus, shm = make(AT, AT)
    sup.enable(shm)
    ok, reason = sup.disable(joints=["joint2"])

    assert ok is True
    assert sup.enabled_joints == ["joint1", "joint3"]
    assert sup.all_enabled is False and sup.all_off is False
    assert "joint2" in reason
    assert "still holding: joint1, joint3" in reason   # says what did NOT go limp

    sup.service(shm)
    # ONE batch, holding two and releasing the middle -- not a command burst
    # followed by a separate release burst.
    assert bus.commands == [[pytest.approx(AT[0]), None, pytest.approx(AT[2])]]


def test_a_released_joint_cannot_be_commanded_even_while_others_are_on():
    """Property 1's per-joint form: limp means no position, every cycle."""
    sup, bus, shm = make(AT, [0.0, 0.0, 0.0])   # controller wants somewhere lethal
    sup.enable(shm, joints=["joint1"], force=True)
    for _ in range(10):
        sup.service(shm)
    for batch in bus.commands:
        assert batch[1] is None and batch[2] is None, \
            "a limp joint was handed a position"


def test_enable_of_one_joint_ignores_another_joints_mismatch():
    """Refusing joint1 because joint2 is limp-and-moved would teach `force`."""
    moved = [1.70, 0.50, -0.24]        # joint2 hand-moved, deliberately left off
    sup, _, shm = make(moved, AT)

    ok, reason = sup.enable(shm, joints=["joint1"])
    assert ok is True, reason
    assert sup.enabled_joints == ["joint1"]

    # ...but joint2 itself is still refused, on its own merits.
    ok, reason = sup.enable(shm, joints=["joint2"])
    assert ok is False
    assert "joint2" in reason


def test_an_unknown_joint_name_is_refused_not_silently_dropped():
    """A request naming a joint this bus lacks must fail loudly, never be
    silently dropped.

    Silently doing nothing while reporting success is how a panel's OFF button
    ends up lying about a limb.
    """
    sup, bus, shm = make(AT, AT)
    sup.enable(shm)

    ok, reason = sup.disable(joints=["elbow"])
    assert ok is False
    assert "elbow" in reason and "joint1, joint2, joint3" in reason
    assert sup.all_enabled is True, "a refused request still changed torque"

    ok, reason = sup.enable(shm, joints=["wrist"])
    assert ok is False
    assert "wrist" in reason


def test_empty_joint_list_means_every_joint_as_the_action_promises():
    sup, _, shm = make(AT, AT)
    sup.enable(shm, joints=[])
    assert sup.all_enabled is True
    sup.disable(joints=[])
    assert sup.all_off is True


def test_duplicate_names_are_collapsed_and_order_follows_the_robot():
    sup, _, shm = make(AT, AT)
    ok, reason = sup.enable(shm, joints=["joint3", "joint1", "joint3"])
    assert ok is True
    assert sup.enabled_joints == ["joint1", "joint3"]
    # Named once in the list of what was enabled and once in the commanded
    # detail -- but acted on once, which is what the duplicate could have broken.
    assert reason.count("joint3") == 2
    assert [n for n, _, _ in sup._last_enable] == ["joint1", "joint3"]
    assert "joint2" not in reason


def test_enable_refuses_a_pose_the_actuator_cannot_express():
    """The guard the 2026-07-26 DYNAMIXEL head fault needed and did not have
    (an XL430 goal outside 0..4095; see ``tests/test_dynamixel.py``).

    Every radian-space check passed -- the controller commanded exactly the
    measured angle, mismatch 0.0000 -- and enabling still slammed the joint,
    because that angle was not representable as a goal on that servo. So the
    actuator is asked before torque is applied, and its objection is a refusal
    like any other, not a crash three cycles later.
    """
    bus = FakeBus([0.0, 0.0])
    bus.actuators[0]._refuse = "id 5: 6.7058 rad converts to goal count 4392"
    supervisor = ServoSupervisor(bus, ["head", "elbow"], 0.05,
                                 command_stride=STRIDE)
    shm = FakeShm([0.0, 0.0])

    ok, reason = supervisor.enable(shm)
    assert not ok
    assert "cannot be commanded" in reason and "4392" in reason
    assert supervisor.all_off
    assert bus.positions_sent == [], "nothing may reach the wire on a refusal"


def _kondo_outside(limit_deg=30.0, at_deg=31.0):
    """A real KondoIcsActuator parked outside its taught envelope."""
    from actuo.drivers.actuators.kondo_ics import (
        KondoIcsActuator, KondoIcsCalibration)
    bus = FakeBus([math.radians(at_deg)])
    bus.actuators[0] = KondoIcsActuator(
        104, KondoIcsCalibration(
            position_limit=(math.radians(-limit_deg), math.radians(limit_deg))))
    return bus


def test_a_real_kondo_servo_outside_its_taught_limit_COMES_BACK():
    """It used to be refused, and that stranded it.

    A joint outside its envelope cannot be commanded to hold where it is --
    correctly -- so torque could not be enabled for it, so it could not be
    driven back, so the only way out was pushing it by hand. Velocity mode
    already resolved this deadlock by letting inward motion through and blocking
    outward; this is the same rule for position mode.
    """
    outside = math.radians(31.0)
    bus = _kondo_outside()
    supervisor = ServoSupervisor(bus, ["rarm_module1_joint1"], 0.05,
                                 command_stride=STRIDE)
    shm = FakeShm([outside])          # the controller commands where it IS

    ok, _ = supervisor.enable(shm)
    assert ok, "refusing here is what strands the joint"

    # ...and it is ANNOUNCED, because the jump guard cannot see this one: after
    # a re-sync the controller commands the measured pose, so `mismatch` is zero
    # while the joint is still about to move by however far outside it is.
    said = supervisor.pop_filter_event()
    assert "OUTSIDE its envelope" in said
    assert "1.0000" in said or "0.0175" in said or "will move" in said

    supervisor.service(shm)
    sent = bus.commands[-1][0]
    assert sent == pytest.approx(math.radians(30.0), abs=1e-6), \
        "the command should be the nearest edge, not the outside position"


def test_a_joint_outside_CANNOT_be_driven_further_out():
    """The narrowness is the argument: inward only, never outward."""
    bus = _kondo_outside()
    supervisor = ServoSupervisor(bus, ["rarm_module1_joint1"], 5.0,
                                 command_stride=STRIDE)
    shm = FakeShm([math.radians(45.0)])      # asked for further out still
    supervisor.enable(shm)
    supervisor.service(shm)
    assert bus.commands[-1][0] == pytest.approx(math.radians(30.0), abs=1e-6)


def _kondo_past_pulse(pulse):
    """A real KondoIcsActuator, no taught limit, resting at raw ``pulse``."""
    from actuo.drivers.actuators.kondo_ics import (
        KondoIcsActuator, KondoIcsCalibration)
    actuator = KondoIcsActuator(104, KondoIcsCalibration())
    bus = FakeBus([actuator.pulse_to_rad(pulse)])
    bus.actuators[0] = actuator
    return bus, actuator


def test_a_krs_servo_resting_past_its_pulse_envelope_comes_back():
    """j0 on the RCB-4mini rig sagged to pulse 3496 when limp, 4 pulses past
    the KRS envelope, and had no taught limit: torque was refused ("the pose
    to hold cannot be commanded") and nothing could ever move it back."""
    bus, actuator = _kondo_past_pulse(3496)
    supervisor = ServoSupervisor(bus, ["j0"], 0.05, command_stride=STRIDE)
    shm = FakeShm([bus.last_state(104).position])   # adopted where it rests

    ok, reason = supervisor.enable(shm)
    assert ok, reason
    assert "OUTSIDE its envelope" in supervisor.pop_filter_event()
    supervisor.service(shm)
    assert actuator.rad_to_pulse(bus.commands[-1][0]) == pytest.approx(3500, abs=1e-6)


def test_a_joint_far_outside_is_not_pulled_back_without_force():
    """Coming back is a move, and further than the jump threshold it is one
    the module manager must not make on its own."""
    bus, actuator = _kondo_past_pulse(3300)          # ~6.8 deg past the edge
    supervisor = ServoSupervisor(bus, ["j0"], 0.05, command_stride=STRIDE)
    shm = FakeShm([bus.last_state(104).position])

    ok, reason = supervisor.enable(shm)
    assert not ok and "OUTSIDE its envelope" in reason and "force" in reason
    assert bus.positions_sent == []
    ok, reason = supervisor.enable(shm, force=True)
    assert ok, reason


def test_the_edge_is_held_until_the_controller_commands_inside():
    """Back inside, the controller still holds the pose it adopted outside.
    Passing that through was refused by the actuator one cycle later -- an
    exception inside service(), which is the whole servicer going down."""
    bus, actuator = _kondo_past_pulse(3496)
    supervisor = ServoSupervisor(bus, ["j0"], 0.05, command_stride=STRIDE)
    outside = bus.last_state(104).position
    shm = FakeShm([outside])
    assert supervisor.enable(shm)[0]
    supervisor.service(shm)
    edge = bus.commands[-1][0]
    # The servo reaches the edge; the controller has not moved yet.
    bus._states[104] = ActuatorState(position=edge, temperature=30.0, error=0)
    for _ in range(3):
        supervisor.service(shm)
        assert bus.commands[-1][0] == pytest.approx(edge)
        assert actuator.position_refusal(bus.commands[-1][0]) is None
    # A trajectory starts inside: from here the controller is in charge.
    inside = actuator.pulse_to_rad(3600)
    shm._c[0] = inside
    supervisor.service(shm)
    assert bus.commands[-1][0] == pytest.approx(inside)


def test_force_does_not_override_an_impossible_pose():
    """force means "yes, I accept the robot moving", not "send it anyway".

    Overriding an unrepresentable value does not get the requested motion --
    the servo takes it modulo its range and does something else. There is
    nothing for an operator to consent to.
    """
    bus = FakeBus([0.0])
    bus.actuators[0]._refuse = "cannot express that"
    supervisor = ServoSupervisor(bus, ["head"], 0.05,
                                 command_stride=STRIDE)
    shm = FakeShm([0.0])

    ok, reason = supervisor.enable(shm, force=True)
    assert not ok
    assert "cannot be commanded" in reason
    assert bus.positions_sent == []


# --------------------------------------------------------------------------
# VELOCITY mode. Added 2026-07-30 to remove the ~130 ms cascade lag: commanding
# velocity leaves only ONE position loop in the path (ros2_control's) instead of
# JTC's setpoints being chased by the driver's own position controller.
#
# The safety rule changes shape, and that is the point of these tests. In
# position mode the hazard is a command far from the robot. In velocity mode
# there IS no commanded position, so that hazard does not exist -- the hazard is
# a NON-ZERO commanded speed, which starts the joint moving the moment torque
# arrives.
# --------------------------------------------------------------------------

def _velocity_bus(positions):
    """FakeBus, but recording the VELOCITY field instead of position."""
    bus = FakeBus(positions)

    def write_batch(commands):
        bus.commands.append([None if c is None else c.velocity
                             for c in commands])
    bus.write_batch = write_batch
    return bus


def _velocity_supervisor(bus, stride=3):
    from actuo.core.contracts import OperatingMode
    return ServoSupervisor(bus, ["j0", "j1"], jump_threshold=0.05,
                           command_stride=stride,
                           mode=OperatingMode.VELOCITY)


class FullShm:
    """A command block written out in full, unlike FakeShm.

    FakeShm takes one POSITION per joint and zeroes the other slots, which is
    exactly right for the position-mode tests and useless here: these tests turn
    on the velocity slot holding something different from the position slot.
    """

    def __init__(self, rows):
        flat = []
        for row in rows:
            flat.extend(row)
        self._c = np.array(flat, dtype=float)
        self.reads = 0

    def read_command(self):
        self.reads += 1
        return self._c


def _shm(rows, stride=3):
    """Command block: one row per joint of [position, velocity, effort]."""
    return FullShm(rows)


def test_velocity_mode_reads_the_VELOCITY_slot_not_the_position_one():
    """The whole change, in one assertion.

    Reading slot 0 would send the position as though it were a speed -- a joint
    at 1.5 rad would be commanded 1.5 rad/s forever, which looks like a
    plausible number and is a runaway.
    """
    bus = _velocity_bus([0.0, 0.0])
    sup = _velocity_supervisor(bus)
    # Raised so this test is about the SLOT and not about the enable guard --
    # the default 0.05 rad/s correctly refuses these speeds, which is what
    # test_velocity_mode_REFUSES_while_the_controller_asks_for_motion covers.
    sup.velocity_threshold = 1.0
    shm = _shm([[1.5, 0.25, 0.0],        # j0: pos 1.5, vel 0.25
                [-2.0, -0.10, 0.0]])     # j1: pos -2.0, vel -0.10
    ok, _ = sup.enable(shm)
    assert ok
    sup.service(shm)
    assert bus.commands[-1] == pytest.approx([0.25, -0.10])


def test_position_mode_still_reads_slot_zero():
    """The existing robot must be untouched by adding a mode."""
    bus = FakeBus([0.0, 0.0])
    sup = ServoSupervisor(bus, ["j0", "j1"], jump_threshold=5.0,
                          command_stride=3)
    shm = _shm([[1.5, 0.25, 0.0], [-2.0, -0.10, 0.0]])
    assert sup.enable(shm)[0]
    sup.service(shm)
    assert bus.commands[-1] == pytest.approx([1.5, -2.0])


def test_velocity_mode_has_NO_position_jump_hazard():
    """A command 100 rad from the robot is irrelevant when it is a speed.

    In position mode this same block would be refused outright. Here the
    position slot is not read at all, so it cannot throw anything.
    """
    bus = _velocity_bus([0.0, 0.0])
    sup = _velocity_supervisor(bus)
    shm = _shm([[100.0, 0.0, 0.0], [-100.0, 0.0, 0.0]])
    ok, reason = sup.enable(shm)
    assert ok, reason


def test_velocity_mode_REFUSES_while_the_controller_asks_for_motion():
    """The hazard that replaces it: torque on + non-zero speed = it just goes."""
    bus = _velocity_bus([0.0, 0.0])
    sup = _velocity_supervisor(bus)
    shm = _shm([[0.0, 0.8, 0.0], [0.0, 0.0, 0.0]])
    ok, reason = sup.enable(shm)
    assert not ok
    assert "asking for motion" in reason
    assert "j0" in reason and "rad/s" in reason
    # ...and it must NOT tell the operator to re-sync, which is meaningless here.
    assert "re-sync" not in reason.lower() or "nothing to re-sync" in reason
    assert bus.commands == [], "refused but still wrote to the wire"


def test_a_wheel_with_no_position_is_reported_as_such_not_as_nan_within_rad():
    """A wheel senses no position, so measured is NaN by design.

    The enable report used to say "at +nan -> commanding +0.0000 ... within
    0.0500 rad", which read as a broken reading (2026-10-26, the RCB-4mini
    emulator). It must say no position is sensed and give the speed in rad/s.
    """
    bus = _velocity_bus([float("nan"), float("nan")])
    sup = _velocity_supervisor(bus)
    ok, reason = sup.enable(_shm([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    assert ok, reason
    assert "nan" not in reason.lower()
    assert "no position sensed" in reason
    assert "rad/s" in reason and " rad of its" not in reason


def test_the_velocity_threshold_is_what_decides():
    bus = _velocity_bus([0.0, 0.0])
    sup = _velocity_supervisor(bus)
    sup.velocity_threshold = 1.0
    assert sup.enable(_shm([[0.0, 0.8, 0.0], [0.0, 0.0, 0.0]]))[0]


def test_velocity_mode_never_asks_an_actuator_about_a_POSITION_it_did_not_get():
    """position_refusal must not be handed a speed.

    It answers "is this angle reachable on the wire". Asking it about 0.3 rad/s
    is a question about the wrong quantity, and a refusal derived from it would
    be confident nonsense.
    """
    bus = _velocity_bus([0.0, 0.0])
    bus.actuators = [FakeActuator(104, refuse="angle out of range"),
                     FakeActuator(105, refuse="angle out of range")]
    sup = _velocity_supervisor(bus)
    ok, reason = sup.enable(_shm([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    assert ok, reason
    # ...while position mode DOES ask, and refuses.
    pbus = FakeBus([0.0, 0.0])
    pbus.actuators = [FakeActuator(104, refuse="angle out of range"),
                      FakeActuator(105, refuse="angle out of range")]
    psup = ServoSupervisor(pbus, ["j0", "j1"], jump_threshold=5.0,
                           command_stride=3)
    ok, reason = psup.enable(_shm([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    assert not ok and "cannot be commanded" in reason


def test_all_off_still_never_reads_the_command_block_in_velocity_mode():
    """Property 1 of the module docstring must survive the new mode."""
    bus = _velocity_bus([0.0, 0.0])
    sup = _velocity_supervisor(bus)
    shm = _shm([[0.0, 5.0, 0.0], [0.0, 5.0, 0.0]])
    sup.service(shm)
    assert bus.releases == 1
    assert shm.reads == 0, "read the command block while fully OFF"
    assert bus.commands == []


def test_an_unsupported_mode_is_refused_at_construction():
    from actuo.core.contracts import OperatingMode
    with pytest.raises(ValueError, match="unsupported servo mode"):
        ServoSupervisor(FakeBus([0.0]), ["j0"], jump_threshold=0.05,
                        command_stride=3, mode=OperatingMode.PWM)


# --------------------------------------------------------------------------
# The envelope fence in VELOCITY mode.
#
# CubeMarsActuator checks every commanded ANGLE against the taught envelope, but
# that check lives in the position encoder -- so switching to a velocity command
# interface removed it without anyone noticing. Observed on hardware: joint2 sat
# 1.2 deg outside its envelope and nothing complained, where position mode would
# have refused.
# --------------------------------------------------------------------------

class _Calib:
    def __init__(self, limit):
        self.position_limit = limit


def _fenced_bus(positions, limit=(-1.0, 1.0)):
    bus = _velocity_bus(positions)
    for a in bus.actuators:
        a.calibration = _Calib(limit)

    def write_batch(commands):
        bus.commands.append([None if c is None else c.velocity
                             for c in commands])
    bus.write_batch = write_batch
    return bus


def test_velocity_is_zeroed_when_it_would_drive_further_out_of_bounds():
    bus = _fenced_bus([1.5, -1.5])          # both already outside [-1, 1]
    sup = _velocity_supervisor(bus)
    sup.velocity_threshold = 10.0
    # j0 is above the top and asked to go up; j1 below the bottom, asked to go
    # down. Both must be stopped.
    shm = _shm([[0.0, +2.0, 0.0], [0.0, -2.0, 0.0]])
    assert sup.enable(shm)[0]
    sup.service(shm)
    assert bus.commands[-1] == pytest.approx([0.0, 0.0])


def test_a_joint_outside_can_still_come_BACK():
    """Blocking all motion would strand it there with no recovery but power off.

    Being outside the envelope is also exactly the state that deactivates
    arm_controller, so "cannot move at all" is the worst possible answer.
    """
    bus = _fenced_bus([1.5, -1.5])
    sup = _velocity_supervisor(bus)
    sup.velocity_threshold = 10.0
    shm = _shm([[0.0, -2.0, 0.0], [0.0, +2.0, 0.0]])   # both heading back in
    assert sup.enable(shm)[0]
    sup.service(shm)
    assert bus.commands[-1] == pytest.approx([-2.0, 2.0])


def test_inside_the_envelope_nothing_is_touched():
    bus = _fenced_bus([0.0, 0.0])
    sup = _velocity_supervisor(bus)
    sup.velocity_threshold = 10.0
    shm = _shm([[0.0, 1.25, 0.0], [0.0, -0.75, 0.0]])
    assert sup.enable(shm)[0]
    sup.service(shm)
    assert bus.commands[-1] == pytest.approx([1.25, -0.75])


def test_an_unknown_position_refuses_to_move():
    """Cannot judge, so does not move. Guessing 'probably fine' is how a joint
    gets driven further into a stop."""
    bus = _fenced_bus([0.0, 0.0])
    sup = _velocity_supervisor(bus)
    sup.velocity_threshold = 10.0
    shm = _shm([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    assert sup.enable(shm)[0]
    bus._states = {}                      # every joint's state has gone away
    sup.service(shm)
    assert bus.commands[-1] == pytest.approx([0.0, 0.0])


def test_a_joint_with_no_taught_envelope_is_left_alone():
    """No limit configured means nothing to enforce -- not 'stop'."""
    bus = _fenced_bus([5.0, 5.0])
    for a in bus.actuators:
        a.calibration = _Calib(None)
    sup = _velocity_supervisor(bus)
    sup.velocity_threshold = 10.0
    shm = _shm([[0.0, 3.0, 0.0], [0.0, 3.0, 0.0]])
    assert sup.enable(shm)[0]
    sup.service(shm)
    assert bus.commands[-1] == pytest.approx([3.0, 3.0])


def test_a_position_command_outside_the_envelope_is_held_at_the_edge_and_said():
    """It used to be passed through for the actuator to refuse -- by raising
    inside service(), which took the servicer and every joint's torque down.
    That happened twice on 2026-10-17, from ordinary trajectories ending at a
    start pose recorded a few pulses outside. Now the command is held at the
    nearest edge, and that is announced, never silent.
    """
    bus = FakeBus([0.5, 0.5])
    for a in bus.actuators:
        a.calibration = _Calib((-1.0, 1.0))
    sup = ServoSupervisor(bus, ["j0", "j1"], jump_threshold=5.0,
                          command_stride=3)
    shm = _shm([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    assert sup.enable(shm)[0]
    shm._c[0] = 2.0                     # the controller asks for outside
    sup.service(shm)
    assert bus.commands[-1] == pytest.approx([1.0, 0.5])
    said = sup.pop_filter_event()
    assert "outside the envelope" in said and "j0" in said
    sup.service(shm)
    assert sup.pop_filter_event() is None   # said once, not every cycle


# --------------------------------------------------------------------------
# The command-filter seam. actuo supplies the hook and knows nothing of what it
# filters for; the robot's package supplies the filter, e.g. a self-collision
# check (see actuo.core.contracts.CommandFilter).
# --------------------------------------------------------------------------


class FakeFilter:
    """Clamps every value to `ceiling`, and records what it was shown."""

    def __init__(self, ceiling):
        self.ceiling = ceiling
        self.seen = []

    def filter(self, command):
        import numpy as np

        from actuo.core.contracts import FilterResult
        self.seen.append(list(command))
        applied = np.minimum(np.asarray(command, dtype=float), self.ceiling)
        gated = bool((applied != np.asarray(command, dtype=float)).any())
        return FilterResult(applied=applied, gated=gated,
                            reason="clamped" if gated else "")


def test_the_filter_decides_what_reaches_the_wire():
    bus = FakeBus([0.0, 0.0])
    filt = FakeFilter(ceiling=0.5)
    supervisor = ServoSupervisor(bus, ["a", "b"], 1.0, command_stride=STRIDE,
                                 command_filter=filt)
    shm = FakeShm([0.0, 0.0])
    assert supervisor.enable(shm)[0]

    supervisor.service(FakeShm([2.0, 0.1]))

    assert bus.positions_sent[-1] == [0.5, 0.1], "the filter's value, not the command"
    assert supervisor.filtered_cycles == 1


def test_a_limp_joint_is_shown_where_it_IS_not_where_it_is_commanded():
    """The filter judges the pose the robot will END UP in.

    A released joint is not going where the controller says -- it is staying
    where it sagged to. Showing the command instead would have the filter judge
    a pose the robot is not going to be in, which errs in both directions: it
    can stop motion over an imaginary contact and miss a real one.
    """
    bus = FakeBus([0.0, 1.234])          # joint 'b' is measured at 1.234
    filt = FakeFilter(ceiling=99.0)
    supervisor = ServoSupervisor(bus, ["a", "b"], 1.0, command_stride=STRIDE,
                                 command_filter=filt)
    shm = FakeShm([0.0, 0.0])
    assert supervisor.enable(shm, joints=["a"])[0]   # only 'a' holds torque

    # controller commands 0.3 for the held joint and 0 for the limp one
    supervisor.service(FakeShm([0.3, 0.0]))

    assert filt.seen[-1] == [0.3, 1.234]


def test_a_filter_is_refused_in_a_mode_where_the_command_is_not_a_pose():
    """Handing speeds to something that asks "would this pose collide" is
    comparing units."""
    bus = FakeBus([0.0])
    with pytest.raises(ValueError, match="judges a POSE"):
        ServoSupervisor(bus, ["a"], 1.0, command_stride=STRIDE,
                        mode=OperatingMode.VELOCITY,
                        command_filter=FakeFilter(ceiling=1.0))


def test_no_filter_leaves_the_old_path_untouched():
    bus = FakeBus([0.0])
    supervisor = ServoSupervisor(bus, ["a"], 1.0, command_stride=STRIDE)
    shm = FakeShm([0.0])
    assert supervisor.enable(shm)[0]

    supervisor.service(FakeShm([0.42]))

    assert bus.positions_sent[-1] == [0.42]
    assert supervisor.filtered_cycles == 0


class LatchingFilter:
    """Stops above `ceiling` and STAYS stopped until reset -- like the real gate."""

    def __init__(self, ceiling):
        self.ceiling = ceiling
        self.latched = False
        self.safe = None
        self.reset_with = None

    def filter(self, command):
        import numpy as np

        from actuo.core.contracts import FilterResult
        command = np.asarray(command, dtype=float)
        if self.latched:
            return FilterResult(applied=self.safe.copy(), gated=True,
                                reason="latched")
        if (command > self.ceiling).any():
            self.latched = True
            applied = self.safe.copy() if self.safe is not None else command
            return FilterResult(applied=applied, gated=True, reason="collision")
        self.safe = command.copy()
        return FilterResult(applied=command.copy())

    def reset(self, command):
        import numpy as np

        self.reset_with = list(np.asarray(command, dtype=float))
        self.safe = np.asarray(command, dtype=float).copy()
        self.latched = False


def _latched_supervisor():
    bus = FakeBus([0.0])
    filt = LatchingFilter(ceiling=1.0)
    sup = ServoSupervisor(bus, ["a"], 1.0, command_stride=STRIDE,
                          command_filter=filt)
    shm = FakeShm([0.0])
    assert sup.enable(shm)[0]
    sup.service(FakeShm([0.4]))          # allowed; 0.4 becomes the safe pose
    sup.service(FakeShm([5.0]))          # collides -> latched, holding 0.4
    assert filt.latched
    return sup, bus, filt


def test_release_adopts_the_held_pose_not_the_command():
    """Adopting the CONTROLLER's command would tell the filter that the pose it
    stopped for is fine."""
    sup, _, filt = _latched_supervisor()

    ok, message = sup.release_filter()

    assert ok
    assert filt.reset_with == [0.4], "released with the held pose"
    assert filt.latched is False
    assert "still asks for the pose that stopped this" in message


def test_release_does_not_move_the_robot_by_itself():
    sup, bus, _ = _latched_supervisor()
    before = len(bus.commands)

    sup.release_filter()

    assert len(bus.commands) == before, "a release must not emit a frame"


def test_a_still_colliding_command_simply_latches_again():
    """The honest outcome, reported by doing it rather than predicting it."""
    sup, _, filt = _latched_supervisor()
    sup.release_filter()

    sup.service(FakeShm([5.0]))          # controller has not moved

    assert filt.latched is True
    assert sup.filtered_cycles >= 2


def test_release_without_a_filter_says_so_instead_of_pretending():
    bus = FakeBus([0.0])
    sup = ServoSupervisor(bus, ["a"], 1.0, command_stride=STRIDE)
    ok, message = sup.release_filter()
    assert not ok and "no command filter" in message


def test_release_before_the_filter_has_ever_run_is_refused():
    """There is no held pose yet, and inventing one is how a release becomes a
    command."""
    bus = FakeBus([0.0])
    sup = ServoSupervisor(bus, ["a"], 1.0, command_stride=STRIDE,
                          command_filter=LatchingFilter(ceiling=1.0))
    ok, message = sup.release_filter()
    assert not ok and "has not run yet" in message


# --------------------------------------------------------------------------
# A robot whose wheels and arm joints share one board -- so one supervisor
# services two different quantities in the same packet.
# --------------------------------------------------------------------------


class RecordingBus(FakeBus):
    """Keeps the ActuatorCommands themselves; a mixed robot's batch carries two
    different quantities and FakeBus only remembers positions."""

    def write_batch(self, commands):
        self.commands.append(list(commands))


def mixed(measured=(1.70, 1.34), commanded=(1.70, 0.0), threshold=0.05):
    """joint1 is an arm joint (POSITION); wheel_fl is a wheel (VELOCITY only)."""
    bus = RecordingBus(list(measured))
    bus.actuators[1] = FakeActuator(105, modes=(OperatingMode.VELOCITY,),
                                    has_position_sense=False)
    bus._states[105] = ActuatorState(position=None, temperature=None, error=0)
    shm = FakeShm(list(commanded))
    sup = ServoSupervisor(bus, ["joint1", "wheel_fl"], threshold, STRIDE)
    return sup, bus, shm


def test_each_joint_takes_the_mode_its_actuator_can_actually_do():
    sup, _, _ = mixed()
    assert sup.modes == [OperatingMode.POSITION, OperatingMode.VELOCITY]


def test_the_servicer_and_the_supervisor_share_ONE_mode_rule():
    """Two copies that disagreed would hand the pose filter a vector
    mis-aligned by one joint, silently -- so there is one function."""
    from actuo.drivers.servo import resolve_modes

    sup, bus, _ = mixed()
    assert resolve_modes(bus.actuators, sup.joint_names,
                         OperatingMode.POSITION) == sup.modes


def test_an_actuator_with_several_modes_but_not_the_asked_one_is_an_error():
    """Then the answer would have to be guessed, and guessing which quantity a
    controller is commanding is not a recoverable kind of wrong."""
    from actuo.drivers.servo import resolve_modes

    bus = RecordingBus([0.0])
    bus.actuators[0] = FakeActuator(
        104, modes=(OperatingMode.VELOCITY, OperatingMode.CURRENT))
    with pytest.raises(ValueError, match="cannot be inferred"):
        resolve_modes(bus.actuators, ["joint1"], OperatingMode.POSITION)


def test_a_wheel_reads_the_VELOCITY_slot_of_the_same_command_block():
    """The shm block is stride-3 [pos, vel, eff] per joint and the C++ side
    fills only the slots the URDF declares -- so a mixed robot needs no new
    layout, just each joint reading its own slot."""
    sup, bus, shm = mixed()
    shm._c[0 * STRIDE + 0] = 1.70          # joint1's position
    assert sup.enable(shm)[0] is True      # wheel commanded 0: nothing to jump
    shm._c[1 * STRIDE + 1] = 3.3           # ...then the controller spins it up
    sup.service(shm)
    sent = bus.commands[-1]
    assert sent[0].mode is OperatingMode.POSITION
    assert sent[0].position == pytest.approx(1.70)
    assert sent[1].mode is OperatingMode.VELOCITY
    assert sent[1].velocity == pytest.approx(3.3)


def test_a_wheel_with_no_position_reports_NaN_rather_than_a_plausible_zero():
    sup, _, shm = mixed()
    wheel = sup.inspect(shm)[1]
    assert math.isnan(wheel.measured)
    assert math.isnan(wheel.mismatch)      # a speed minus an angle is not a number
    assert wheel.mode is OperatingMode.VELOCITY


def test_the_wheels_hazard_is_a_non_zero_speed_the_arms_is_a_stale_pose():
    """Same enable, two different questions, in one robot."""
    sup, _, shm = mixed(measured=(1.70, 0.0), commanded=(1.70, 0.0))
    sup.disable(["wheel_fl"])              # released, so enabling it is a transition
    shm._c[1 * STRIDE + 1] = 2.0           # the controller is asking the wheel to spin
    ok, reason = sup.enable(shm)
    assert ok is False
    assert "wheel_fl" in reason and "asking for motion" in reason
    assert "trajectory controller" not in reason   # wrong remedy for a wheel

    sup2, _, shm2 = mixed(measured=(0.50, 0.0), commanded=(1.70, 0.0))
    ok, reason = sup2.enable(shm2)
    assert ok is False
    assert "joint1" in reason and "throw the robot" in reason


def test_both_refusals_are_reported_together_not_just_the_first():
    """They have opposite remedies -- re-sync the controller vs stop asking for
    motion -- so reporting one sends the operator to fix the wrong end."""
    sup, _, shm = mixed(measured=(0.50, 0.0), commanded=(1.70, 0.0))
    sup.disable(["wheel_fl"])
    shm._c[1 * STRIDE + 1] = 2.0
    ok, reason = sup.enable(shm)
    assert ok is False
    assert "joint1" in reason and "wheel_fl" in reason
    assert "AND" in reason


def test_a_pose_filter_is_shown_the_arm_joints_and_NOT_the_wheels():
    """Handing a collision filter a wheel's rad/s in a vector labelled "pose"
    is comparing units. It sees the POSITION joints, in their own order."""
    seen = []

    from actuo.core.contracts import FilterResult

    class Recorder:
        def filter(self, command):
            seen.append(list(command))
            return FilterResult(applied=np.asarray(command), gated=False)

        def reset(self, safe):
            pass

    sup, bus, shm = mixed()
    sup = ServoSupervisor(bus, ["joint1", "wheel_fl"], 0.05, STRIDE,
                          command_filter=Recorder())
    shm._c[0 * STRIDE + 0] = 1.70
    sup.enable(shm)
    shm._c[1 * STRIDE + 1] = 3.3
    sup.service(shm)
    assert seen == [[pytest.approx(1.70)]]           # one value, not two
    # ...and the wheel still got its speed, untouched by the filter.
    assert bus.commands[-1][1].velocity == pytest.approx(3.3)


def test_a_filter_on_a_robot_with_no_pose_joints_is_refused():
    bus = RecordingBus([0.0])
    bus.actuators[0] = FakeActuator(104, modes=(OperatingMode.VELOCITY,),
                                    has_position_sense=False)
    with pytest.raises(ValueError, match="POSITION"):
        ServoSupervisor(bus, ["wheel_fl"], 0.05, STRIDE,
                        command_filter=object())


def test_a_joint_with_no_POSE_comes_up_driven():
    """A wheel has no stale pose to be thrown to, so the state this class
    exists to protect against does not exist for it -- and gating it made a
    base that would not move until someone pressed a button whose entire
    purpose does not apply to it."""
    sup, bus, shm = mixed()
    assert sup.enabled_joints == ["wheel_fl"]
    assert sup.default_enabled == ["wheel_fl"]
    assert sup.all_off is False


def test_the_POSE_joints_still_come_up_released():
    sup, _, _ = mixed()
    assert "joint1" not in sup.enabled_joints


def test_a_wheel_is_commanded_from_the_first_cycle_without_enabling():
    sup, bus, shm = mixed()
    shm._c[1 * STRIDE + 1] = 3.3
    sup.service(shm)                       # nothing was enabled
    sent = bus.commands[-1]
    assert sent[0] is None, "the arm joint must still be released"
    assert sent[1] is not None and sent[1].velocity == pytest.approx(3.3)


def test_a_released_wheel_STAYS_released_until_asked_back():
    """Releasing is still available and still means something -- limp is
    free-rolling, which is how you push the robot by hand."""
    sup, bus, shm = mixed()
    sup.disable(["wheel_fl"])
    sup.service(shm)
    assert sup.all_off is True
    # Nothing was commanded at all: with every joint off, service() takes the
    # branch that does not even read the command block.
    assert bus.commands == []
    assert bus.releases == 1


def test_reset_to_default_brings_the_wheels_back_but_not_the_arms():
    """What a link recovery does. The arms stay released because the robot may
    have moved; the wheels come back because a base that stops answering after
    a cable hiccup is its own hazard."""
    sup, _, shm = mixed()
    sup.enable(shm)                        # everything on
    sup.disable()                          # everything off
    sup.reset_to_default()
    assert sup.enabled_joints == ["wheel_fl"]


def test_an_already_driven_wheel_is_not_re_judged_on_enable():
    """The guard is about a TRANSITION -- a joint going from released to
    driven. A wheel that is already turning cannot be started by enabling it,
    so refusing the whole request over its speed would block the arms for a
    reason that does not apply."""
    sup, _, shm = mixed(measured=(1.70, 0.0), commanded=(1.70, 0.0))
    shm._c[1 * STRIDE + 1] = 2.0           # the wheel is being driven, hard
    ok, _ = sup.enable(shm)
    assert ok is True
    assert "joint1" in sup.enabled_joints


def test_idle_releases_are_paced_but_the_first_one_is_immediate(monkeypatch):
    """At 200 Hz an EXIT every cycle collided with the MIT motors' own answers.
    While all off, the release repeats at release_period; going off (or coming
    up) sends at once, and holding torque is untouched."""
    clock = [10.0]
    monkeypatch.setattr("actuo.drivers.servo.time.monotonic", lambda: clock[0])
    sup, bus, shm = make(AT, AT)
    sup.release_period = 0.02
    sup.service(shm)                       # start-up: immediate
    assert bus.releases == 1
    for _ in range(3):                     # 5 ms cycles inside the period
        clock[0] += 0.005
        sup.service(shm)
    assert bus.releases == 1
    clock[0] += 0.005                      # 20 ms since the last
    sup.service(shm)
    assert bus.releases == 2
    assert shm.reads == 0, "idle read the command block"
    ok, _ = sup.enable(shm)                # enable reads it once, for the jump check
    assert ok
    reads_at_enable = shm.reads
    clock[0] += 0.001
    sup.service(shm)                       # held: a command every cycle
    assert len(bus.positions_sent) == 1
    sup.disable()
    clock[0] += 0.001                      # 1 ms later: released at once
    sup.service(shm)
    assert bus.releases == 3
    assert shm.reads == reads_at_enable + 1, "only the held cycle read it"


# --------------------------------------------------------------------------
# Unowned slots (shm layout version 2): a NaN command means no controller owns the joint
# --------------------------------------------------------------------------


def test_enabling_a_joint_no_controller_commands_is_refused():
    sup, bus, shm = make([0.1, 0.2, 0.3], [0.1, math.nan, 0.3])
    ok, reason = sup.enable(shm, ["joint2"])
    assert not ok
    assert "no controller commands joint2" in reason
    assert sup.enabled_joints == []


def test_a_held_joint_whose_controller_stops_keeps_its_last_target():
    sup, bus, shm = make([0.1, 0.2, 0.3], [0.1, 0.25, 0.3])
    assert sup.enable(shm)[0]
    sup.service(shm)
    shm._c[1 * STRIDE] = math.nan          # joint2's module deactivated
    sup.service(shm)
    assert bus.commands[-1] == pytest.approx([0.1, 0.25, 0.3])


def test_an_unowned_held_joint_never_sent_holds_where_it_is():
    sup, bus, shm = make([0.1, 0.2, 0.3], [0.1, 0.2, 0.3])
    assert sup.enable(shm)[0]
    shm._c[2 * STRIDE] = math.nan          # before anything was sent
    sup.service(shm)
    assert bus.commands[-1] == pytest.approx([0.1, 0.2, 0.3])


def test_an_unowned_wheel_is_stopped_not_left_driving():
    bus = FakeBus([0.0])
    bus.actuators = [FakeActuator(104, modes=(OperatingMode.VELOCITY,),
                                  has_position_sense=False)]
    bus._states = {104: ActuatorState(position=None, temperature=30.0, error=0)}
    velocities = []
    bus.write_batch = lambda commands: velocities.append(
        [None if c is None else c.velocity for c in commands])
    sup = ServoSupervisor(bus, ["wheel"], 0.05, STRIDE)
    shm = FakeShm([0.0])
    shm._c[1] = 2.0                        # driving at 2 rad/s
    sup.service(shm)
    shm._c[1] = math.nan                   # its controller went away
    sup.service(shm)
    assert velocities == [[2.0], [0.0]]


def test_released_joint_forgets_what_it_held():
    sup, bus, shm = make([0.1, 0.2, 0.3], [0.1, 0.25, 0.3])
    assert sup.enable(shm)[0]
    sup.service(shm)
    sup.disable(["joint2"])
    assert 1 not in sup._hold


def test_enabling_an_absent_joint_is_refused():
    sup, bus, shm = make([0.1, 0.2, 0.3], [0.1, 0.2, 0.3])
    bus.present = lambda device_id: device_id != 105
    ok, reason = sup.enable(shm, ["joint2"])
    assert not ok and "joint2 not answering" in reason


def test_an_unowned_joint_holds_what_was_sent_not_what_was_asked():
    """The controller asked for the pose j0 was resting at, outside the
    envelope; the fence sent the edge. When the controller let go, holding
    the raw ask handed the actuator a pose it refuses (servicer down)."""
    bus, actuator = _kondo_past_pulse(3496)
    supervisor = ServoSupervisor(bus, ["j0"], 0.05, command_stride=STRIDE)
    shm = FakeShm([bus.last_state(104).position])
    assert supervisor.enable(shm)[0]
    supervisor.service(shm)
    edge = bus.commands[-1][0]
    shm._c[0] = math.nan            # the hardware component went away
    supervisor.service(shm)
    assert bus.commands[-1][0] == pytest.approx(edge)
    assert actuator.position_refusal(bus.commands[-1][0]) is None


def test_a_leaping_command_is_moved_at_max_velocity_and_said(monkeypatch):
    """A trajectory controller that aborted on a bad reading held that
    reading -- 1.4 rad from where the joint was -- and the servo went there
    in one step (2026-10-17). With max_velocity the wire never sees the leap."""
    import actuo.drivers.servo as servo_module
    clock = [100.0]
    monkeypatch.setattr(servo_module.time, "monotonic", lambda: clock[0])
    bus = FakeBus([0.0, 0.0])
    sup = ServoSupervisor(bus, ["j0", "j1"], 0.05, STRIDE,
                          max_velocity={"j0": 2.0})
    shm = FakeShm([0.0, 0.0])
    assert sup.enable(shm)[0]
    sup.service(shm)                     # first send: the adopted pose
    shm._c[0] = 1.4
    shm._c[STRIDE] = 1.4                 # j1 has no limit
    sent = []
    for _ in range(5):
        clock[0] += 0.02
        sup.service(shm)
        sent.append(bus.commands[-1])
    assert [round(c[0], 4) for c in sent] == [0.04, 0.08, 0.12, 0.16, 0.2]
    assert all(c[1] == 1.4 for c in sent)
    assert "max_velocity" in sup.pop_filter_event()


def test_max_velocity_must_name_real_joints():
    with pytest.raises(KeyError, match="unknown joints"):
        ServoSupervisor(FakeBus([0.0]), ["j0"], 0.05, STRIDE, max_velocity={"elbow": 1.0})
