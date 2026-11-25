"""Re-labelling a joint's position: the FEETECH homing-offset write and the
supervisor around it. A fake servo keeps its registers and reads
``present = (raw - homing_offset) mod 4096``, as an STS3215 does."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("feetech_cli")

from actuo.core.contracts import ActuatorState, Capabilities, Frame, OperatingMode  # noqa: E402
from actuo.drivers import feetech_protocol as fp  # noqa: E402
from actuo.drivers.actuators.feetech_sts import (  # noqa: E402
    FeetechStsActuator, FeetechStsCalibration, ServoFacts, model_by_name)
from actuo.drivers.buses.feetech_bus import FeetechBus  # noqa: E402
from actuo.drivers.servo import ServoSupervisor  # noqa: E402
from actuo.core.contracts import Bus as _ContractBus


def _sm(value, bit):
    return (abs(value) | (1 << bit)) if value < 0 else value


def _from_sm(value, bit):
    return -(value & ((1 << bit) - 1)) if value & (1 << bit) else value


class Servo:
    def __init__(self, raw, offset=0):
        self.raw = raw
        self.offset = offset
        self.lock = 1
        self.torque = 1
        self.goal = None
        self.log = []

    def read(self, address, length):
        values = {31: _sm(self.offset, 11), 56: (self.raw - self.offset) % 4096,
                  40: self.torque, 55: self.lock}
        return int(values.get(address, 0)).to_bytes(length, "little")

    def write(self, address, data):
        value = int.from_bytes(data, "little")
        self.log.append((address, value))
        if address == 31:
            if self.lock:
                return          # a locked EEPROM ignores the write
            self.offset = _from_sm(value, 11)
        elif address == 55:
            self.lock = value
        elif address == 40:
            self.torque = value
        elif address == 42:
            self.goal = value
            self.torque = 1     # a goal write energises an STS


class Line:
    def __init__(self, servos):
        self.servos = servos

    def transact(self, frames, timeout):
        replies = []
        for frame in frames:
            body = frame.payload
            servo = self.servos.get(body[2])
            if servo is None:
                replies.append(None)
                continue
            if body[4] == fp.READ:
                data = servo.read(body[5], body[6])
            else:
                servo.write(body[5], bytes(body[6:-1]))
                data = b""
            replies.append(Frame(payload=fp.build_packet(body[2], 0, data),
                                 address=body[2]))
        return replies


def _bus(servo, direction=1):
    actuator = FeetechStsActuator(1, FeetechStsCalibration(direction=direction),
                                  model_by_name("STS3215"))
    actuator.facts = ServoFacts(homing_offset=servo.offset)
    return FeetechBus(Line({1: servo}), [actuator])


def test_the_present_pose_reads_as_the_declared_angle():
    servo = Servo(raw=2500)
    bus = _bus(servo)
    result = bus.set_origin(0, 0.0, torque=True)
    assert servo.offset == 452 and result["new_offset"] == 452
    assert (servo.raw - servo.offset) % 4096 == 2048
    assert servo.lock == 1 and servo.torque == 1 and servo.goal == 2048
    # Torque off before the offset moves the frame; goal before torque returns.
    order = [a for a, _ in servo.log]
    assert order.index(40) < order.index(31) < order.index(42)
    assert bus.actuators[0].facts.homing_offset == 452


def test_direction_and_torque_off_afterwards():
    servo = Servo(raw=2048, offset=10)
    bus = _bus(servo, direction=-1)
    bus.set_origin(0, math.pi / 2, torque=False)
    assert (servo.raw - servo.offset) % 4096 == 2048 - 1024
    assert servo.torque == 0


def test_an_offset_the_register_cannot_hold_is_refused_unwritten():
    servo = Servo(raw=100)
    bus = _bus(servo)
    with pytest.raises(ValueError):
        bus.set_origin(0, 0.95 * math.pi, torque=True)
    assert servo.offset == 0 and all(a != 31 for a, _ in servo.log)


def test_a_write_without_effect_is_an_error():
    servo = Servo(raw=2500)
    bus = _bus(servo)
    servo.write = lambda address, data: servo.log.append((address, 0))   # deaf
    with pytest.raises(RuntimeError):
        bus.set_origin(0, 0.0, torque=True)


# --- the supervisor -----------------------------------------------------------


class FakeBus:
    # The presence contract, as actuo.core.contracts.Bus defines it for a bus
    # that never takes a servo away.
    hot_plug = False
    present = _ContractBus.present
    absence_reason = _ContractBus.absence_reason
    def __init__(self, position, outcome=None):
        self.actuators = [type("A", (), {
            "id": 1, "capabilities": Capabilities(modes=(OperatingMode.POSITION,)),
            "position_refusal": lambda self, p: None})()]
        self.position = position
        self.outcome = outcome
        self.batches = []

    def last_state(self, device_id):
        return ActuatorState(position=self.position, error=0)

    def write_batch(self, commands):
        self.batches.append(commands)

    def release_all(self):
        pass

    def set_origin(self, index, reference, torque):
        if self.outcome is not None:
            raise self.outcome
        self.position = reference
        return {"old_offset": 0.0, "new_offset": 100.0, "reported": 2148.0}


class Shm:
    def __init__(self, value):
        self.c = [value, math.nan, math.nan]

    def read_command(self):
        return self.c


def test_supervisor_holds_the_new_reading_until_resynced():
    bus = FakeBus(0.7)
    shm = Shm(0.7)
    sup = ServoSupervisor(bus, ["joint0"], 0.05, 3)
    assert sup.enable(shm)[0]
    ok, reason = sup.set_origin("joint0", 0.2)
    assert ok, reason
    sup.service(shm)              # the controller still says 0.7, old frame
    assert bus.batches[-1][0].position == pytest.approx(0.2)
    assert sup.inspect(shm)[0].resyncing


def test_supervisor_refusals_and_failures():
    bus = FakeBus(0.7, outcome=ValueError("too far"))
    shm = Shm(0.7)
    sup = ServoSupervisor(bus, ["joint0"], 0.05, 3)
    assert sup.enable(shm)[0]
    ok, reason = sup.set_origin("joint0", 0.2)
    assert not ok and "too far" in reason
    assert sup.enabled_joints == ["joint0"]           # unchanged frame: held
    bus.outcome = RuntimeError("deaf")
    ok, reason = sup.set_origin("joint0", 0.2)
    assert not ok and "left limp" in reason
    assert sup.enabled_joints == []
