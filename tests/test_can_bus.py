"""CanBus orchestration tests -- layer integration without hardware.

Uses a fake broadcast transport that emits real CubeMars status bytes, so this
exercises Transport -> Bus -> Actuator routing, batching, and the watchdogs
while running anywhere (no python-can, no CAN adapter, no ROS).
"""

from __future__ import annotations

import math
import os
import struct
import sys

import pytest

_HERE = os.path.dirname(__file__)

from actuo.core.contracts import (  # noqa: E402
    Actuator,
    ActuatorCommand,
    ActuatorState,
    Calibration,
    Capabilities,
    Frame,
    OperatingMode,
    Transport,
)
from actuo.core.diagnostics import DiagLevel  # noqa: E402
from actuo.drivers.actuators.cubemars import (  # noqa: E402
    CubeMarsActuator,
    CubeMarsCalibration,
    CubeMarsLimits,
)
from actuo.drivers.buses.can_bus import CanBus, OverspeedError, StaleFeedbackError  # noqa: E402

MOTORS = ((104, 39.0), (105, 9.0), (106, 6.0))  # the real robot's three motors


class FakeCanTransport(Transport):
    """Broadcast transport: frames appear unsolicited, nothing replies."""

    def __init__(self):
        self._open = False
        self.sent = []
        self.inbox = []

    def open(self):
        self._open = True

    def close(self):
        self._open = False

    @property
    def is_open(self):
        return self._open

    def write(self, frames):
        self.sent.append(list(frames))  # record each burst separately

    def recv(self, timeout):
        return self.inbox.pop(0) if self.inbox else None

    def queue_status(self, node_id, deg=0.0, erpm=0.0, amps=0.0, temp=30, error=0):
        self.inbox.append(
            Frame(
                payload=struct.pack(
                    ">hhhbB",
                    int(round(deg * 10)),
                    int(round(erpm / 10.0)),
                    int(round(amps * 100)),
                    temp,
                    error,
                ),
                address=(9 << 8) | node_id,  # some packet id; low byte identifies
                meta={"extended": True},
            )
        )


def build_bus(transport, speed_erpm=(220.0, 50.0, 35.0)):
    actuators = []
    for (node_id, ratio), erpm in zip(MOTORS, speed_erpm):
        # kt present, as the shipped config now has it: without it CURRENT is
        # (correctly) absent from capabilities and cannot be exercised here.
        calib = CubeMarsCalibration(gear_ratio=ratio, torque_constant=0.105)
        actuators.append(
            CubeMarsActuator(
                node_id,
                calib,
                limits=CubeMarsLimits(
                    max_velocity_erpm=erpm, max_acceleration_erpm_s=erpm
                ),
            )
        )
    return CanBus(transport, actuators, stale_after=0.5)


def test_write_batch_is_one_burst_not_a_loop_of_singles():
    transport = FakeCanTransport()
    transport.open()
    bus = build_bus(transport)

    bus.write_batch(
        [
            ActuatorCommand(mode=OperatingMode.POSITION, position=math.radians(d))
            for d in (10.0, 20.0, 5.0)
        ]
    )

    assert len(transport.sent) == 1          # a single transaction...
    assert len(transport.sent[0]) == 3       # ...carrying all three motors
    assert [f.address & 0xFF for f in transport.sent[0]] == [104, 105, 106]


def test_write_batch_rejects_wrong_command_count():
    transport = FakeCanTransport()
    transport.open()
    bus = build_bus(transport)
    with pytest.raises(ValueError, match="expected 3 commands"):
        bus.write_batch([ActuatorCommand(mode=OperatingMode.POSITION, position=0.0)])


def test_a_none_slot_releases_that_motor_in_the_same_burst():
    """Per-joint servo: hold two, free one, in ONE bus transaction.

    Two bursts would leave the joints in different states for the gap between
    them -- which is exactly when the freed limb starts to fall.
    """
    transport = FakeCanTransport()
    transport.open()
    bus = build_bus(transport)

    bus.write_batch(
        [
            ActuatorCommand(mode=OperatingMode.POSITION, position=math.radians(10.0)),
            None,                                  # joint2 goes limp
            ActuatorCommand(mode=OperatingMode.POSITION, position=math.radians(5.0)),
        ]
    )

    assert len(transport.sent) == 1              # still one transaction
    frames = transport.sent[0]
    assert len(frames) == 3                      # every motor addressed
    assert [f.address & 0xFF for f in frames] == [104, 105, 106]

    # The released motor got the SAME frame release_all would have sent it, and
    # the others did not.
    released = build_bus(FakeCanTransport()).actuators[1].encode_release()
    assert frames[1].payload == released.payload
    assert frames[1].address == released.address
    assert frames[0].payload != released.payload
    assert frames[2].payload != released.payload


def test_overspeed_fires_on_the_real_joint3_runaway():
    """A real runaway (2026-07-17), pinned as a number.

    joint3, an AK80-6 on a CubeMars arm, was handed a 35 ERPM envelope and ran ~188 deg/s at the output. With
    21 pole pairs and a 6:1 gear that is 35 * 6 * 21 / 6 ~= 3950 ERPM, ~113x.
    """
    transport = FakeCanTransport()
    transport.open()
    bus = build_bus(transport)

    # The guard only judges a joint that is being commanded a POSITION, so put
    # the bus in that mode first -- as the servicer does every cycle.
    bus.write_batch([ActuatorCommand(mode=OperatingMode.POSITION,
                                     position=math.radians(d))
                     for d in (10.0, 20.0, 150.0)])
    runaway_erpm = 188.0 * 21 * 6.0 / 6.0        # deg/s -> ERPM, ~3948
    transport.queue_status(104, deg=10.0, erpm=0.0)
    transport.queue_status(105, deg=20.0, erpm=0.0)
    transport.queue_status(106, deg=150.0, erpm=runaway_erpm)
    bus.poll(timeout=0.0)

    with pytest.raises(OverspeedError) as excinfo:
        bus.check_speed()
    message = str(excinfo.value)
    assert "106" in message
    assert "35" in message                        # the envelope it was given
    assert "not honouring" in message
    # The multiple is stated, because "too fast" is not actionable and "113x" is
    assert "113x" in message or "112x" in message


def test_overspeed_is_silent_on_healthy_motion():
    """It must never fire on a normal move, or its threshold gets widened away."""
    transport = FakeCanTransport()
    transport.open()
    bus = build_bus(transport)

    bus.write_batch([ActuatorCommand(mode=OperatingMode.POSITION,
                                     position=math.radians(10.0))] * 3)
    # Each motor at 3x its envelope: sloppy trapezoid overshoot, not a runaway.
    for node, erpm in ((104, 220.0 * 3), (105, 50.0 * 3), (106, 35.0 * 3)):
        transport.queue_status(node, deg=10.0, erpm=erpm)
    bus.poll(timeout=0.0)
    bus.check_speed()                              # must not raise


def test_overspeed_ignores_a_joint_that_was_not_commanded_a_position():
    """CURRENT mode commands no speed at all, so there is no envelope to break.

    Judging it against max_velocity_erpm compares a measurement to a limit that
    was never sent. This fired on the first real current-mode push -- 14 deg/s
    of correct behaviour, aborted as "9x" -- and a guard that fires on correct
    behaviour is one that gets switched off.
    """
    transport = FakeCanTransport()
    transport.open()
    bus = build_bus(transport)

    bus.write_batch([None, None,
                     ActuatorCommand(mode=OperatingMode.CURRENT, current=0.6)])
    transport.queue_status(104, deg=10.0, erpm=0.0)
    transport.queue_status(105, deg=20.0, erpm=0.0)
    transport.queue_status(106, deg=-83.0, erpm=300.0)      # 8.5x the envelope
    bus.poll(timeout=0.0)
    bus.check_speed()                    # must not raise: no speed was asked for


def test_overspeed_ignores_a_released_joint():
    """A limp joint is being back-driven; its speed is the operator's business."""
    transport = FakeCanTransport()
    transport.open()
    bus = build_bus(transport)

    bus.release_all()
    transport.queue_status(106, deg=-83.0, erpm=9999.0)
    bus.poll(timeout=0.0)
    bus.check_speed()


def test_release_all_delegates_so_it_cannot_drift_from_the_batch_path():
    transport = FakeCanTransport()
    transport.open()
    bus = build_bus(transport)

    bus.release_all()
    all_none = FakeCanTransport()
    all_none.open()
    other = build_bus(all_none)
    other.write_batch([None, None, None])

    assert len(transport.sent) == 1
    assert [(f.address, f.payload) for f in transport.sent[0]] == \
           [(f.address, f.payload) for f in all_none.sent[0]]


def test_read_batch_routes_each_frame_to_its_actuator():
    transport = FakeCanTransport()
    transport.open()
    bus = build_bus(transport)

    transport.queue_status(105, deg=20.0)
    transport.queue_status(104, deg=10.0)
    transport.queue_status(106, deg=-5.0)

    states = bus.read_batch(timeout=0.2)

    # returned in actuator (joint) order regardless of arrival order
    assert states[0].position == pytest.approx(math.radians(10.0))
    assert states[1].position == pytest.approx(math.radians(20.0))
    assert states[2].position == pytest.approx(math.radians(-5.0))


def test_read_batch_raises_when_a_motor_never_reports():
    transport = FakeCanTransport()
    transport.open()
    bus = build_bus(transport)
    transport.queue_status(104)
    transport.queue_status(105)
    # 106 stays silent -- a wiring/node-id fault, not a zero position
    with pytest.raises(StaleFeedbackError, match="106"):
        bus.read_batch(timeout=0.05)


def test_foreign_frames_are_counted_not_silently_dropped():
    transport = FakeCanTransport()
    transport.open()
    bus = build_bus(transport)
    transport.inbox.append(Frame(payload=b"\x00" * 8, address=0x777, meta={}))
    bus.poll(timeout=0.01)
    assert bus.unmatched_frames == 1


def test_error_frames_are_never_mistaken_for_node_ids():
    """Regression: an error frame's id is a bitmask, not an address.

    A probe on real hardware once reported "motors are talking on node id 12" for
    1701 frames whose id 0x0C was actually CAN_ERR_PROT|CAN_ERR_CRTL. The bus
    must classify them as errors and must NOT offer them as evidence about ids.
    """
    transport = FakeCanTransport()
    transport.open()
    bus = build_bus(transport)
    for _ in range(5):
        transport.inbox.append(
            Frame(payload=bytes(8), address=0x0C, meta={"error_frame": True})
        )
    bus.poll(timeout=0.01)

    assert bus.error_frames == 5
    assert bus.unmatched_addresses == {}   # 0x0C must not look like "node 12"
    assert bus.unmatched_frames == 0


def test_error_frame_on_a_matching_id_is_not_decoded_as_state():
    """A garbled id can collide with a real node id; that must not become a pose."""
    transport = FakeCanTransport()
    transport.open()
    bus = build_bus(transport)
    transport.inbox.append(
        Frame(payload=bytes(8), address=104, meta={"error_frame": True})
    )
    bus.poll(timeout=0.01)
    assert bus.error_frames == 1
    assert bus.last_state(104) is None      # never invented a position


def test_poll_counts_distinguishes_matched_feedback_from_drained_frames():
    transport = FakeCanTransport()
    transport.open()
    bus = build_bus(transport)
    transport.inbox.append(Frame(payload=bytes(8), address=0x777, meta={}))
    transport.inbox.append(
        Frame(payload=bytes(8), address=0x0C, meta={"error_frame": True})
    )

    consumed, matched = bus.poll_counts(timeout=0.01)
    assert consumed == 2
    assert matched == 0
    assert bus.poll(timeout=0.01) == 0


def test_check_fresh_flags_a_motor_that_went_quiet(monkeypatch):
    transport = FakeCanTransport()
    transport.open()
    bus = build_bus(transport)

    for node_id, _ in MOTORS:
        transport.queue_status(node_id)
    bus.read_batch(timeout=0.2)
    bus.check_fresh()  # all fresh right now

    # jump time past stale_after without any new frames
    import actuo.drivers.buses.can_bus as can_bus_mod

    real_monotonic = can_bus_mod.time.monotonic
    monkeypatch.setattr(
        can_bus_mod.time, "monotonic", lambda: real_monotonic() + 5.0
    )
    with pytest.raises(StaleFeedbackError, match="silent for"):
        bus.check_fresh()


def test_error_code_survives_to_the_supervisor():
    transport = FakeCanTransport()
    transport.open()
    bus = build_bus(transport)
    for node_id, _ in MOTORS:
        transport.queue_status(node_id, error=4)  # under-voltage
    states = bus.read_batch(timeout=0.2)
    assert all(s.error == 4 for s in states)


def _diag(items, name):
    for item in items:
        if item.name == name:
            return item
    raise AssertionError("diagnostic item {!r} not found".format(name))


def test_diagnose_reports_error_frame_trend_unmatched_ids_and_fault_name():
    transport = FakeCanTransport()
    transport.open()
    bus = build_bus(transport)
    bus.set_joint_names(["joint1", "joint2", "joint3"])

    # Prime state cache and baseline counters.
    transport.queue_status(104, deg=1.0, error=0)
    transport.queue_status(105, deg=2.0, error=0)
    transport.queue_status(106, deg=3.0, error=5)  # encoder fault
    bus.read_batch(timeout=0.2)
    bus.diagnose()  # baseline snapshot for trend deltas

    # One CAN error frame + one unmatched data frame after baseline.
    transport.inbox.append(
        Frame(payload=bytes(8), address=0x0C, meta={"error_frame": True})
    )
    transport.inbox.append(Frame(payload=bytes(8), address=0x777, meta={}))
    bus.poll(timeout=0.01)

    items = bus.diagnose()
    can_error = _diag(items, "bus/can_error_frames")
    unmatched = _diag(items, "bus/unmatched_addresses")
    faulted = _diag(items, "servo/joint3")

    assert can_error.level == DiagLevel.WARN
    assert can_error.values["error_frames_delta"] == "1"
    assert unmatched.level == DiagLevel.WARN
    assert unmatched.values["0x777_total"] == "1"
    assert "silence_age_s" in faulted.values
    assert faulted.level == DiagLevel.ERROR
    assert faulted.values["fault_name"] == "encoder fault"


class _HookFaultActuator(Actuator):
    def __init__(self, actuator_id, fault_names):
        super().__init__(actuator_id, Calibration(counts_per_rev=1.0))
        self._fault_names = dict(fault_names)

    @property
    def capabilities(self):
        return Capabilities(modes=(OperatingMode.POSITION,))

    def encode_command(self, command):
        return Frame(payload=b"\x00", address=self.id)

    def matches(self, frame):
        return frame.address == self.id

    def decode_state(self, frame):
        return ActuatorState(
            position=0.0, error=int(frame.payload[0]), aux={}
        )

    def encode_mode(self, mode):
        self.capabilities.require(mode)
        return []

    def fault_name(self, code):
        return self._fault_names.get(
            int(code), "unknown fault code {}".format(int(code))
        )


def test_diagnose_uses_the_matched_actuators_fault_names_not_cubemars():
    transport = FakeCanTransport()
    transport.open()
    actuator = _HookFaultActuator(42, {5: "custom overcurrent"})
    bus = CanBus(transport, [actuator], stale_after=0.5)

    transport.inbox.append(Frame(payload=bytes([5]), address=42, meta={}))
    bus.poll(timeout=0.01)
    item = _diag(bus.diagnose(), "servo/id_42")

    assert item.level == DiagLevel.ERROR
    assert item.values["fault_name"] == "custom overcurrent"


def test_diagnose_reports_unknown_fault_code_when_an_actuator_has_no_name():
    transport = FakeCanTransport()
    transport.open()
    actuator = _HookFaultActuator(77, {})
    bus = CanBus(transport, [actuator], stale_after=0.5)

    transport.inbox.append(Frame(payload=bytes([5]), address=77, meta={}))
    bus.poll(timeout=0.01)
    item = _diag(bus.diagnose(), "servo/id_77")

    assert item.values["fault_name"] == "unknown fault code 5"


def test_solicit_asks_only_the_actuators_that_have_to_be_asked():
    """A mixed bus is the case this exists for.

    On a 12-joint CubeMars biped, six motors upload at 50 Hz unprompted and six say nothing until addressed.
    A bus that only listens watches the second six go stale and reports a fault
    about hardware that is healthy and waiting.
    """
    from actuo.drivers.actuators.cubemars_mit import (
        CubeMarsMitActuator, EXIT_MOTOR_MODE, MitCalibration)
    from actuo.drivers.actuators.cubemars_v3 import (
        CubeMarsV3Actuator, CubeMarsV3Calibration)

    streaming = CubeMarsV3Actuator(5, CubeMarsV3Calibration(
        gear_ratio=9.0, p_max=12.56, v_max=30.0, t_max=32.0, kp=1.0, kd=1.0))
    asked = CubeMarsMitActuator(1, MitCalibration(
        gear_ratio=64.0, p_max=12.56, v_max=8.0, t_max=144.0, kp=1.0, kd=1.0))
    assert streaming.answers_only_when_addressed is False
    assert asked.answers_only_when_addressed is True

    transport = FakeCanTransport()
    bus = CanBus(transport, [streaming, asked])
    assert bus.solicit() == 1
    sent = transport.sent[-1]
    assert len(sent) == 1
    assert sent[0].address == 1
    assert sent[0].payload == EXIT_MOTOR_MODE
    assert sent[0].meta["extended"] is False


def test_solicit_sends_nothing_on_an_all_streaming_bus():
    actuator = CubeMarsActuator(5, CubeMarsCalibration(gear_ratio=9.0))
    transport = FakeCanTransport()
    bus = CanBus(transport, [actuator])
    before = len(transport.sent)
    assert bus.solicit() == 0
    assert len(transport.sent) == before


def _mit_and_servo_bus():
    """One 2.0 MIT motor beside one servo-mode motor, as on a CubeMars biped's legs."""
    from actuo.drivers.actuators.cubemars_mit import (
        ENTER_MOTOR_MODE, CubeMarsMitActuator, MitCalibration)
    mit = CubeMarsMitActuator(4, MitCalibration(
        gear_ratio=64.0, p_max=12.56, v_max=8.0, t_max=144.0, kp=500.0, kd=5.0))
    servo = CubeMarsActuator(105, CubeMarsCalibration(gear_ratio=9.0),
                             limits=CubeMarsLimits(50.0, 50.0))
    transport = FakeCanTransport()
    bus = CanBus(transport, [mit, servo])
    return bus, transport, bytes(ENTER_MOTOR_MODE)


def test_a_mit_motor_is_armed_before_its_first_command(monkeypatch):
    # Nothing on the servicer's path armed MIT motors: six legs reported
    # "enabled" while limp and ignored every target. The bus arms them itself
    # on the first command after a release, and only them. The command
    # follows in a second burst of the SAME cycle, after the ENTER is answered:
    # waiting a cycle let the drive run its stored command for 10 ms.
    bus, transport, enter = _mit_and_servo_bus()
    clock = [100.0]
    monkeypatch.setattr("actuo.drivers.buses.can_bus.time.monotonic", lambda: clock[0])
    hold = ActuatorCommand(mode=OperatingMode.POSITION, position=0.0)

    bus.write_batch([hold, hold])
    burst, second = transport.sent[-2], transport.sent[-1]
    assert [f.payload for f in burst].count(enter) == 1
    assert burst[0].payload == enter and burst[0].address == 4
    assert len(burst) == 2                     # ENTER for the MIT, command for the servo
    assert len(second) == 1 and second[0].address == 4
    assert second[0].payload != enter          # the command itself

    clock[0] += 0.01                           # next cycle: the command, no ENTER
    bus.write_batch([hold, hold])
    burst = transport.sent[-1]
    assert enter not in [f.payload for f in burst] and len(burst) == 2
    assert any(f.address == 4 for f in burst)

    clock[0] += 0.05                           # still within the re-arm period
    bus.write_batch([hold, hold])
    assert enter not in [f.payload for f in transport.sent[-1]]

    clock[0] += 0.06                           # past it: armed again, instead of commanded
    sent = len(transport.sent)
    bus.write_batch([hold, hold])
    assert [f.payload for f in transport.sent[-1]].count(enter) == 1
    assert len(transport.sent) == sent + 1     # a re-arm holds a fresh command: one burst


def test_a_burst_never_addresses_one_mit_motor_twice(monkeypatch):
    # A MIT motor answers on the id it is addressed on. Two frames to it in one
    # burst put the second on the wire while it answers the first: same id,
    # different data, a bit error on the bus.
    bus, transport, _enter = _mit_and_servo_bus()
    clock = [100.0]
    monkeypatch.setattr("actuo.drivers.buses.can_bus.time.monotonic", lambda: clock[0])
    hold = ActuatorCommand(mode=OperatingMode.POSITION, position=0.0)
    for _ in range(40):                        # arm, command, re-arms, a release
        bus.write_batch([hold, hold])
        clock[0] += 0.01
    bus.write_batch([None, hold])
    bus.write_batch([hold, hold])
    for burst in transport.sent:
        addressed = [f.address for f in burst if not f.meta.get("extended", False)]
        assert addressed.count(4) <= 1, burst


def test_a_release_means_the_next_command_arms_again(monkeypatch):
    bus, transport, enter = _mit_and_servo_bus()
    clock = [100.0]
    monkeypatch.setattr("actuo.drivers.buses.can_bus.time.monotonic", lambda: clock[0])
    hold = ActuatorCommand(mode=OperatingMode.POSITION, position=0.0)
    bus.write_batch([hold, hold])
    clock[0] += 0.01
    bus.write_batch([None, hold])         # EXIT: the motor is out of MIT mode
    clock[0] += 0.01
    bus.write_batch([hold, hold])         # well inside the period, still re-armed
    assert [f.payload for f in transport.sent[-2]].count(enter) == 1


def test_a_commanded_mit_motor_is_released_limp_first(monkeypatch):
    # A 2.0 MIT drive keeps its last command through EXIT and runs it on the
    # next ENTER. Zero gains go first, so what it keeps asks for nothing.
    from actuo.drivers.actuators.cubemars_mit import EXIT_MOTOR_MODE
    bus, transport, _enter = _mit_and_servo_bus()
    mit = bus.actuators[0]
    hold = ActuatorCommand(mode=OperatingMode.POSITION, position=0.3)
    bus.write_batch([hold, hold])
    bus.write_batch([None, hold])
    limp, exit_ = transport.sent[-2][0], transport.sent[-1][0]
    kept = mit.decode_command(limp)
    assert kept["kp"] == 0.0 and kept["kd"] == 0.0
    assert abs(kept["torque"]) < 0.1
    assert exit_.payload == bytes(EXIT_MOTOR_MODE) and len(transport.sent[-1]) == 1

    sent = len(transport.sent)
    bus.write_batch([None, hold])         # already out: a plain EXIT, one burst
    assert len(transport.sent) == sent + 1
    assert transport.sent[-1][0].payload == bytes(EXIT_MOTOR_MODE)


def test_solicit_never_disarms_a_commanded_mit_motor():
    # The poll's frame is a release, i.e. EXIT on MIT. Sent to a joint that is
    # being commanded, it disarmed the legs every 20 ms. It asks only the
    # joints nobody is commanding.
    from actuo.drivers.actuators.cubemars_mit import EXIT_MOTOR_MODE
    bus, transport, _enter = _mit_and_servo_bus()
    hold = ActuatorCommand(mode=OperatingMode.POSITION, position=0.0)
    bus.write_batch([hold, hold])
    before = len(transport.sent)
    assert bus.solicit() == 0
    assert len(transport.sent) == before

    bus.write_batch([None, hold])         # released: it is polled again
    assert bus.solicit() == 1
    assert transport.sent[-1][0].payload == bytes(EXIT_MOTOR_MODE)
