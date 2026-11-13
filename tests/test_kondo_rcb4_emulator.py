"""The RCB-4mini emulator drives the real bus: cold start, servo on, a move."""

import math

from actuo.core.contracts import ActuatorCommand, Frame, OperatingMode, Transport

from actuo.drivers.actuators.kondo_ics import KondoIcsActuator, KondoIcsCalibration
from actuo.drivers.buses.kondo_rcb4_bus import KondoRcb4Bus
from actuo.drivers.kondo_rcb4_emulator import FRAME_S, Rcb4Emulator


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class EmulatorTransport(Transport):
    """Hands each request body straight to the emulator, no pty."""

    def __init__(self, emulator):
        self.emulator = emulator
        self.bitrate = 1250000
        self.current_bitrate = 1250000

    def open(self):
        pass

    def close(self):
        pass

    @property
    def is_open(self):
        return True

    def reopen(self, bitrate):
        self.current_bitrate = bitrate

    def write(self, frames):
        self.transact(frames, 0.0)

    def recv(self, timeout):
        return None

    def transact(self, frames, timeout):
        out = []
        for item in frames:
            answer = self.emulator.reply(item.payload)
            out.append(None if answer is None else Frame(payload=answer))
        return out


def test_a_servo_reaches_its_target_after_the_move_time():
    clock = Clock()
    emulator = Rcb4Emulator([2, 5], clock=clock)
    bus = KondoRcb4Bus(
        EmulatorTransport(emulator),
        [KondoIcsActuator(i, KondoIcsCalibration(), velocity=1) for i in (2, 5)],
        joint_names=["j2", "j5"], assign_settle_s=0.0)
    bus.verify_servos()
    assert [s.position for s in bus.read_batch()] == [0.0, 0.0]

    target = math.radians(30.0)
    command = ActuatorCommand(mode=OperatingMode.POSITION, position=target)
    bus.write_batch([command, command])     # switches the servos on
    bus.write_batch([command, command])     # and moves them
    clock.now += 2 * FRAME_S
    # A reading that jumps is held until reading_confirm reads agree.
    for _ in range(bus.reading_confirm):
        positions = [s.position for s in bus.read_batch()]
    assert all(abs(p - target) < math.radians(0.1) for p in positions)


def test_a_released_servo_does_not_move():
    clock = Clock()
    emulator = Rcb4Emulator([2], clock=clock)
    emulator.reply(bytes([0x10, 0x04, 0, 0, 0, 0, 1]) + (9000).to_bytes(2, "little"))
    clock.now += 1.0
    assert emulator.pulse(2) == 7500
