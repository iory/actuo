"""FEETECH STS/HLS plugin and bus: wire bytes, units, start-up facts.

Hardware-free: a fake transport answers like servos on the line. What these
cannot check was checked on an arm of five STS3215 (ids 1-5) plus an air relay board
(id 19) at 1 Mbit/s on 2026-11-23: verify_servos read every model and homing
offset, and one SYNC_READ of the five took 1.5 ms with no misses.
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("feetech_cli")

from actuo.core.contracts import (  # noqa: E402
    ActuatorCommand, CapabilityError, Frame, OperatingMode)
from actuo.core.diagnostics import DiagLevel  # noqa: E402
from actuo.drivers import feetech_protocol as fp  # noqa: E402
from actuo.drivers.actuators.feetech_sts import (  # noqa: E402
    FACTS_BLOCK_ADDR, FACTS_BLOCK_LEN, STATE_BLOCK_ADDR, STATE_BLOCK_LEN,
    FeetechStsActuator, FeetechStsCalibration, ServoFacts, model_by_name)
from actuo.drivers.buses.feetech_bus import FeetechBus  # noqa: E402


def sm(value, sign_bit):
    return (abs(value) | (1 << sign_bit)) if value < 0 else value


def status(servo_id, data, error=0):
    return Frame(payload=fp.build_packet(servo_id, error, bytes(data)),
                 address=servo_id)


def facts_block(model=777, homing_offset=0, mode=0, protection_current=0):
    block = bytearray(FACTS_BLOCK_LEN)

    def put(address, size, value):
        start = address - FACTS_BLOCK_ADDR
        block[start:start + size] = int(value).to_bytes(size, "little")

    put(3, 2, model)
    put(28, 2, protection_current)
    put(31, 2, sm(homing_offset, 11))
    put(33, 1, mode)
    return bytes(block)


def state_block(position=2048, velocity=0, load=0, voltage=120, temp=30,
                status_reg=0, current=0):
    block = bytearray(STATE_BLOCK_LEN)

    def put(address, size, value):
        start = address - STATE_BLOCK_ADDR
        block[start:start + size] = int(value).to_bytes(size, "little")

    put(56, 2, sm(position, 15))
    put(58, 2, sm(velocity, 15))
    put(60, 2, sm(load, 10))
    put(62, 1, voltage)
    put(63, 1, temp)
    put(65, 1, status_reg)
    put(69, 2, sm(current, 15))
    return bytes(block)


def actuator(servo_id=1, model="STS3215", direction=1, offset_counts=0.0,
             facts=True):
    a = FeetechStsActuator(
        servo_id, FeetechStsCalibration(direction=direction,
                                        zero_offset_counts=2048 + offset_counts),
        model_by_name(model))
    if facts:
        a.facts = ServoFacts(protection_current=500 if a.is_hls else 0)
    return a


class FakeLine:
    """Servos on a wire: answers READ / WRITE / SYNC_READ, records writes."""

    def __init__(self, servos):
        self.servos = servos            # id -> {"facts": bytes, "state": bytes}
        self.written = []
        self.pending = []
        self.memory = {i: {} for i in servos}

    def reset_input(self):
        self.pending.clear()

    def write(self, frames):
        for frame in frames:
            self.written.append(frame.payload)
            body = frame.payload
            if body[4] == fp.SYNC_READ:
                address, length = body[5], body[6]
                for servo_id in body[7:-1]:
                    if servo_id in self.servos:
                        self.pending.append(status(servo_id, self.servos[servo_id]["state"]))

    def recv(self, timeout):
        return self.pending.pop(0) if self.pending else None

    def transact(self, frames, timeout):
        replies = []
        for frame in frames:
            self.written.append(frame.payload)
            servo_id, instruction = frame.payload[2], frame.payload[4]
            if servo_id not in self.servos:
                replies.append(None)
                continue
            if instruction == fp.READ:
                address, length = frame.payload[5], frame.payload[6]
                if address == FACTS_BLOCK_ADDR:
                    data = self.servos[servo_id]["facts"]
                elif address == 33:
                    data = bytes([self.memory[servo_id].get(33, 0)])
                else:
                    data = bytes(length)
                replies.append(status(servo_id, data))
            else:
                address = frame.payload[5]
                for k, byte in enumerate(frame.payload[6:-1]):
                    self.memory[servo_id][address + k] = byte
                replies.append(status(servo_id, b""))
        return replies


# --- actuator -----------------------------------------------------------------
def test_state_is_decoded_in_radians_with_direction():
    a = actuator(direction=-1)
    s = a.decode_block(state_block(position=2048 + 1024, velocity=-512,
                                   load=-300, voltage=119, current=100))
    assert s.position == pytest.approx(-math.pi / 2)
    assert s.velocity == pytest.approx(512 * 2 * math.pi / 4096)
    assert s.voltage == pytest.approx(11.9)
    # present_load is opposite to the torque; flipped, then by direction.
    assert s.aux["load"] == pytest.approx(-0.3)
    assert s.aux["current_a"] == pytest.approx(0.65)
    assert s.error is None


def test_outside_position_mode_the_homing_offset_is_applied_by_the_plugin():
    a = actuator()
    a.facts = ServoFacts(homing_offset=100, wire_mode=2)
    s = a.decode_block(state_block(position=2148))
    assert s.position == pytest.approx(0.0)


def test_a_position_goal_is_two_little_endian_bytes_at_42():
    a = actuator()
    address, data = a.goal_payload(ActuatorCommand(OperatingMode.POSITION,
                                                   position=math.pi / 2))
    assert address == 42
    assert int.from_bytes(data, "little") == 2048 + 1024


def test_an_hls_goal_carries_its_current_and_speed():
    a = actuator(model="HLS3960M")
    address, data = a.goal_payload(ActuatorCommand(OperatingMode.POSITION,
                                                   position=0.0))
    assert address == 41 and len(data) == 7
    assert int.from_bytes(data[1:3], "little") == 2048
    assert int.from_bytes(data[3:5], "little") == 500       # protection_current
    assert int.from_bytes(data[5:7], "little") == 60


def test_an_hls_has_no_pwm_mode():
    with pytest.raises(CapabilityError, match="constant current"):
        actuator(model="HLS3960M").require_mode(OperatingMode.PWM)


def test_pwm_duty_is_sign_magnitude_at_44_and_signed_like_the_torque():
    a = actuator(direction=1)
    address, data = a.goal_payload(ActuatorCommand(OperatingMode.PWM, pwm=0.25))
    value = int.from_bytes(data, "little")
    assert address == 44
    # A positive raw duty turns towards lower counts, so +torque is negative raw.
    assert value == 250 | (1 << 10)


def test_a_goal_outside_one_turn_is_refused():
    a = actuator()
    assert a.position_refusal(math.pi * 0.99) is None
    assert "outside 0..4095" in a.position_refusal(math.pi * 1.01)
    with pytest.raises(ValueError):
        a.goal_payload(ActuatorCommand(OperatingMode.POSITION, position=4.0))


def test_a_different_model_is_refused():
    a = actuator(facts=False)
    with pytest.raises(CapabilityError, match="model_number 4618"):
        a.decode_facts(status(1, facts_block(model=4618)))


# --- bus ----------------------------------------------------------------------
def make_bus(states, facts=None):
    servos = {i: {"facts": (facts or {}).get(i, facts_block()), "state": s}
              for i, s in states.items()}
    line = FakeLine(servos)
    actuators = [actuator(i, facts=False) for i in sorted(states)]
    return FeetechBus(line, actuators), line


def test_verify_reads_facts_and_puts_a_pwm_servo_back_in_position_mode():
    bus, line = make_bus({1: state_block(), 2: state_block()},
                         facts={2: facts_block(mode=2, homing_offset=-10)})
    items = bus.verify_servos()
    assert bus.actuators[1].facts.homing_offset == -10
    assert bus.actuators[1].facts.wire_mode == 0
    assert line.memory[2][33] == 0 and line.memory[2][40] == 0   # mode, torque off
    assert line.memory[1][48] == 232 and line.memory[1][49] == 3  # torque_limit 1000
    assert "position" in items[1].message


def test_verify_stops_on_a_silent_servo():
    bus, line = make_bus({1: state_block()})
    bus.actuators.append(actuator(9, facts=False))
    with pytest.raises(RuntimeError, match="did not answer"):
        bus.verify_servos()


def test_one_sync_read_reads_every_servo_and_a_silent_one_is_a_miss():
    bus, line = make_bus({1: state_block(position=2048), 2: state_block(position=3072)})
    bus.verify_servos()
    del line.servos[2]
    line.written.clear()
    states = bus.read_batch()
    assert [p[4] for p in line.written] == [fp.SYNC_READ]
    assert states[0].position == pytest.approx(0.0)
    assert states[1] is None and bus.missed == 1


def test_write_batch_sends_goals_then_enables_and_releases_in_sync_writes():
    bus, line = make_bus({1: state_block(), 2: state_block()})
    bus.verify_servos()
    line.written.clear()
    bus.write_batch([ActuatorCommand(OperatingMode.POSITION, position=0.0), None])
    packets = [(p[2], p[4], p[5]) for p in line.written]
    assert packets == [(fp.BROADCAST_ID, fp.SYNC_WRITE, 40),     # release id 2
                       (fp.BROADCAST_ID, fp.SYNC_WRITE, 42),     # goal id 1
                       (fp.BROADCAST_ID, fp.SYNC_WRITE, 40)]     # enable id 1


# --- hot plug -------------------------------------------------------------------
def make_hot_bus(states, configured, facts=None, **kwargs):
    """A hot-plug bus configured for ``configured`` ids; ``states`` are plugged in."""
    servos = {i: {"facts": (facts or {}).get(i, facts_block()), "state": s}
              for i, s in states.items()}
    line = FakeLine(servos)
    actuators = [actuator(i, facts=False) for i in configured]
    return FeetechBus(line, actuators, hot_plug=True, **kwargs), line


def plug(line, servo_id, facts=None, state=None):
    line.servos[servo_id] = {"facts": facts or facts_block(),
                             "state": state or state_block()}
    line.memory[servo_id] = {}


def sync_write_items(packet):
    """``{id: data}`` of one SYNC_WRITE packet."""
    size = packet[6]
    body = packet[7:-1]
    return {body[k]: bytes(body[k + 1:k + 1 + size])
            for k in range(0, len(body), size + 1)}


def test_hot_plug_start_up_reports_a_missing_servo_and_carries_on():
    bus, _ = make_hot_bus({1: state_block()}, [1, 2])
    items = bus.verify_servos()
    assert bus.present(1) and not bus.present(2)
    assert items[1].level == DiagLevel.WARN
    assert "start-up" in bus.absence_reason(2)
    assert bus.absence_reason(1) == ""


def test_without_hot_plug_a_missing_servo_still_stops_the_start_up():
    bus, _ = make_bus({1: state_block()})
    bus.actuators.append(actuator(2, facts=False))
    with pytest.raises(RuntimeError, match="did not answer"):
        bus.verify_servos()


def test_an_absent_servo_is_left_out_of_the_sync_read():
    """Kept in, every cycle would wait the whole timeout for its reply."""
    bus, line = make_hot_bus({1: state_block()}, [1, 2], presence_period=100)
    bus.verify_servos()
    line.written.clear()
    bus.read_batch()
    sync = [p for p in line.written if p[4] == fp.SYNC_READ]
    assert len(sync) == 1 and list(sync[0][7:-1]) == [1]


def test_a_servo_plugged_in_is_probed_checked_and_then_read():
    bus, line = make_hot_bus({1: state_block()}, [1, 2], presence_period=2)
    bus.verify_servos()
    plug(line, 2, facts=facts_block(mode=2), state=state_block(position=3072))
    bus.read_batch()                      # first read: no probe yet
    assert not bus.present(2)
    bus.read_batch()                      # second read: probe, start-up check
    assert bus.present(2)
    assert line.memory[2][33] == 0        # put back in position mode
    assert line.memory[2][48] == 232 and line.memory[2][49] == 3  # torque limit
    states = bus.read_batch()
    assert states[1].position == pytest.approx(math.pi / 2)


def test_a_servo_unplugged_goes_absent_after_the_misses_and_gets_no_goal():
    bus, line = make_hot_bus({1: state_block(), 2: state_block()}, [1, 2],
                             absent_after_misses=3, presence_period=100)
    bus.verify_servos()
    bus.read_batch()
    del line.servos[2]
    bus.read_batch()
    bus.read_batch()
    assert bus.present(2)                 # two lost replies are not a lost module
    bus.read_batch()
    assert not bus.present(2) and "3 reads" in bus.absence_reason(2)

    line.written.clear()
    bus.write_batch([ActuatorCommand(OperatingMode.POSITION, position=0.0)] * 2)
    writes = [(p[5], sync_write_items(p)) for p in line.written
              if p[4] == fp.SYNC_WRITE]
    goals = [items for address, items in writes if address == 42]
    torque = [items for address, items in writes if address == 40]
    assert [sorted(g) for g in goals] == [[1]]          # no goal to id 2
    assert {2: b"\x00"} in torque                       # id 2: torque off only
    assert {1: b"\x01"} in torque


def test_a_servo_of_another_model_plugged_in_stays_absent_with_the_reason():
    bus, line = make_hot_bus({1: state_block()}, [1, 2], presence_period=1)
    bus.verify_servos()
    plug(line, 2, facts=facts_block(model=4618))
    bus.read_batch()
    assert not bus.present(2)
    assert "4618" in bus.absence_reason(2)


def test_absent_servos_are_probed_in_turn():
    bus, line = make_hot_bus({1: state_block()}, [1, 2, 3], presence_period=1)
    bus.verify_servos()
    line.written.clear()
    bus.read_batch()
    bus.read_batch()
    probed = [p[2] for p in line.written if p[4] == fp.READ]
    assert probed == [2, 3]


def test_the_config_turns_hot_plug_on_for_a_feetech_bus(tmp_path):
    from actuo.drivers.config import build_bus

    path = tmp_path / "robot.yaml"
    path.write_text(
        "bus:\n"
        "  transport: {plugin: serial, channel: /dev/null, bitrate: 1000000}\n"
        "  hot_plug: true\n"
        "  absent_after_misses: 5\n"
        "joints:\n"
        "  - name: j1\n"
        "    actuator: {plugin: feetech_sts, id: 1, model: STS3215}\n"
        "  - name: j3\n"
        "    actuator: {plugin: feetech_sts, id: 3, model: HLS3950M}\n")
    bus, names = build_bus(str(path))
    assert isinstance(bus, FeetechBus) and names == ["j1", "j3"]
    assert bus.hot_plug and bus.absent_after_misses == 5


def test_watched_ids_are_pinged_in_turn_and_reported(tmp_path):
    bus, line = make_hot_bus({1: state_block()}, [1], presence_period=1,
                             watch_ids=[1, 5, 7])
    assert bus.watch_ids == (5, 7)                  # its own id is not "watched"
    bus.verify_servos()
    plug(line, 5)
    line.written.clear()
    bus.read_batch()
    bus.read_batch()
    pings = [p[2] for p in line.written if p[4] == fp.PING]
    assert pings == [5, 7]
    assert bus.watched(5) and not bus.watched(7)
    assert bus.answering_ids() == [1, 5]
    del line.servos[5]
    bus.read_batch()                                # 5 again: gone
    assert bus.answering_ids() == [1]


def test_bus_overrides_reach_the_feetech_bus(tmp_path):
    from actuo.drivers.config import build_bus

    path = tmp_path / "robot.yaml"
    path.write_text(
        "bus:\n"
        "  transport: {plugin: serial, channel: /dev/null, bitrate: 1000000}\n"
        "joints:\n"
        "  - name: j1\n"
        "    actuator: {plugin: feetech_sts, id: 1, model: STS3215}\n")
    bus, _ = build_bus(str(path), bus_overrides={"hot_plug": True, "watch_ids": [2, 3]})
    assert bus.hot_plug and bus.watch_ids == (2, 3)


def test_an_hls_origin_that_needs_a_negative_offset_takes_the_largest_positive_one():
    from actuo.drivers.buses.feetech_bus import hls_homing_offset

    assert hls_homing_offset(4087, 2048, 2047, 4096) == (2039, 2048)   # positive: exact
    assert hls_homing_offset(2, 2048, 2047, 4096) == (2047, 2051)      # 3 counts away
