"""Feetech SCS0009 plugin tests: packet layer, big-endian decode, batch, bus.

ROS-free and hardware-free -- a loopback fake stands in for the serial wire. The
one thing these cannot check is the real endianness/register map; that was
verified on the actual Amazing Hand (left hand, ids 11-18 read cleanly). These
lock the wire format so it does not silently drift.
"""

from __future__ import annotations

import math
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))

from actuo.core.contracts import (  # noqa: E402
    ActuatorCommand, CapabilityError, Frame, OperatingMode)
from actuo.drivers.actuators import feetech_scs as f  # noqa: E402
from actuo.drivers.actuators.feetech_scs import (  # noqa: E402
    Scs0009Actuator, ScsCalibration)
from actuo.drivers.buses.serial_bus import SerialBus  # noqa: E402


# --- packet layer -----------------------------------------------------------

def test_read_packet_bytes_are_exact():
    # FF FF ID LEN INSTR addr len CHECKSUM ; LEN = params(2) + 2 = 4
    packet = f.build_read(1, f.REG_PRESENT_POSITION, 8)
    assert packet == bytes.fromhex("ffff0104023808b8")


def test_checksum_matches_feetech_formula():
    params = bytes([f.REG_PRESENT_POSITION, 8])
    assert f.build_read(1, f.REG_PRESENT_POSITION, 8)[-1] == \
        f.checksum(1, 4, f.READ, params)


def test_parse_status_round_trips_and_rejects_corruption():
    data = bytes([0x02, 0x00, 0, 0, 0, 0, 0, 0x19])
    good = f.build_packet(3, 0x00, data)
    assert f.parse_status(good) == (3, 0, data)
    # flip one data byte, leave the checksum -> rejected (returns None, no raise)
    bad = bytearray(good)
    bad[5] ^= 0xFF
    assert f.parse_status(bytes(bad)) is None
    # truncated -> None
    assert f.parse_status(good[:-1]) is None


def test_two_byte_values_are_big_endian():
    assert f.u16_be(512) == bytes([0x02, 0x00])
    assert f.from_u16_be(bytes([0x02, 0x00])) == 512
    # not little-endian
    assert f.u16_be(512) != bytes([0x00, 0x02])


# --- actuator ---------------------------------------------------------------

def _actuator(direction=1, limit=None):
    return Scs0009Actuator(11, ScsCalibration(direction=direction,
                                              position_limit=limit))


def test_position_decode_matches_rustypot_formula():
    a = _actuator()
    # raw 895 (0x037F), big-endian, present block starts at data[0:2]
    data = bytes([0x03, 0x7F, 0, 0, 0, 0, 0, 21])
    state = a.decode_state(Frame(payload=f.build_packet(11, 0, data), address=11))
    expected = math.radians(300.0) * ((895 & 0x3FF) - 511) / 1024.0
    assert state.position == pytest.approx(expected)
    assert state.temperature == 21.0
    assert state.aux["raw_position"] == 895.0


def test_centre_count_is_zero_radians():
    a = _actuator()
    data = bytes([0x01, 0xFF, 0, 0, 0, 0, 0, 20])  # 0x01FF = 511 = centre
    state = a.decode_state(Frame(payload=f.build_packet(11, 0, data), address=11))
    assert state.position == pytest.approx(0.0, abs=1e-9)


def test_goal_position_is_big_endian_and_centred():
    a = _actuator()
    frame = a.encode_command(ActuatorCommand(mode=OperatingMode.POSITION,
                                             position=0.0))
    parsed = f.parse_status(frame.payload)  # WRITE packet shares the framing
    # WRITE params = [addr, hi, lo]; 0 rad -> centre 511 = 0x01FF
    assert frame.payload[4] == f.WRITE
    assert frame.payload[5] == f.REG_GOAL_POSITION
    assert frame.payload[6:8] == bytes([0x01, 0xFF])


def test_command_outside_limit_is_refused():
    a = _actuator(limit=(math.radians(-30), math.radians(30)))
    with pytest.raises(ValueError):
        a.encode_command(ActuatorCommand(mode=OperatingMode.POSITION,
                                         position=math.radians(90)))


def test_non_position_mode_is_a_capability_error():
    a = _actuator()
    with pytest.raises(CapabilityError):
        a.encode_command(ActuatorCommand(mode=OperatingMode.CURRENT, current=0.1))


def test_release_is_torque_disable():
    a = _actuator()
    frame = a.encode_release()
    assert frame.payload[4] == f.WRITE
    assert frame.payload[5] == f.REG_TORQUE_ENABLE
    assert frame.payload[6] == 0


def test_matches_only_its_own_id():
    a = _actuator()
    mine = f.build_packet(11, 0, bytes(8))
    other = f.build_packet(12, 0, bytes(8))
    assert a.matches(Frame(payload=mine)) is True
    assert a.matches(Frame(payload=other)) is False
    assert a.matches(Frame(payload=b"\x00\x01\x02")) is False


# --- bus (loopback fake) ----------------------------------------------------

class LoopbackTransport:
    """A fake serial wire: answers each READ with a canned present block and
    records the frames written, so batch framing can be asserted."""

    def __init__(self, positions):
        self.positions = dict(positions)     # id -> raw count
        self.written = []
        self._open = False

    def open(self):
        self._open = True

    def close(self):
        self._open = False

    @property
    def is_open(self):
        return self._open

    def write(self, frames):
        self.written.extend(frames)

    def recv(self, timeout):
        return None

    def transact(self, frames, timeout):
        replies = []
        for frame in frames:
            servo_id = frame.payload[2]
            raw = self.positions.get(servo_id)
            if raw is None:
                replies.append(None)
                continue
            data = bytes(f.u16_be(raw)) + bytes([0, 0, 0, 0, 0, 22])
            replies.append(Frame(payload=f.build_packet(servo_id, 0, data),
                                 address=servo_id))
        return replies


def _bus(ids):
    acts = [Scs0009Actuator(i, ScsCalibration(direction=1)) for i in ids]
    t = LoopbackTransport({i: 511 for i in ids})
    return SerialBus(t, acts), t


def test_read_batch_decodes_every_servo():
    bus, _t = _bus([11, 12, 13])
    states = bus.read_batch(0.05)
    assert len(states) == 3
    assert all(abs(s.position) < 1e-9 for s in states)   # all at centre
    assert bus.missed == 0


def test_read_batch_keeps_last_state_on_a_missed_reply():
    bus, t = _bus([11, 12])
    bus.read_batch(0.05)
    del t.positions[12]                # 12 stops answering
    states = bus.read_batch(0.05)
    assert states[0] is not None       # 11 still fresh
    assert states[1] is not None       # 12 keeps its last state, not faked 0
    assert bus.missed == 1


def test_write_batch_is_a_single_sync_write():
    bus, t = _bus([11, 12, 13])
    cmds = [ActuatorCommand(mode=OperatingMode.POSITION, position=0.0)
            for _ in range(3)]
    bus.write_batch(cmds)
    # one frame on the wire, addressed to broadcast, SYNC_WRITE of goal_position
    assert len(t.written) == 1
    payload = t.written[0].payload
    assert payload[2] == f.BROADCAST_ID
    assert payload[4] == f.SYNC_WRITE
    assert payload[5] == f.REG_GOAL_POSITION
    assert payload[6] == 2                       # 2 bytes per servo
    # three servos each contribute id + 2 bytes
    assert payload[7] == 11 and payload[10] == 12 and payload[13] == 13


def test_write_batch_none_releases_that_servo_in_the_same_burst():
    bus, t = _bus([11, 12])
    bus.write_batch([ActuatorCommand(mode=OperatingMode.POSITION, position=0.0),
                     None])
    # a release write for 12 plus one sync-write for the held 11
    kinds = [(fr.payload[4], fr.payload[2]) for fr in t.written]
    assert (f.WRITE, 12) in kinds                # 12 torque-off
    assert any(k == f.SYNC_WRITE for k, _ in kinds)


def test_set_mode_rejects_unsupported():
    bus, _t = _bus([11])
    bus.set_mode(OperatingMode.POSITION)         # ok
    with pytest.raises(CapabilityError):
        bus.set_mode(OperatingMode.CURRENT)


# --- auto limit finder: following-error stop detection ----------------------

import importlib.util  # noqa: E402


def _load_find_limits():
    path = os.path.join(_HERE, "..", "actuo", "cli", "hand_find_limits.py")
    spec = importlib.util.spec_from_file_location("actuo_hand_find_limits", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_fl = _load_find_limits()


def _finger(stop_raw, direction):
    """A fake servo whose present tracks goal but cannot pass ``stop_raw``."""
    def step_fn(goal):
        return min(goal, stop_raw) if direction > 0 else max(goal, stop_raw)
    return step_fn


def test_find_stop_detects_a_mechanical_stop_high():
    # Detected stop lands within a step/noise of the true stop (the last count
    # where a > noise advance happened), not necessarily exactly on it.
    stop, note = _fl.find_stop(
        _finger(900, +1), start=500, direction=+1, step=4, bound=1001,
        follow_error=15, settle=5, noise=2, max_steps=400)
    assert stop is not None and abs(stop - 900) <= 4, note
    assert "stop" in note


def test_find_stop_detects_a_mechanical_stop_low():
    stop, note = _fl.find_stop(
        _finger(150, -1), start=500, direction=-1, step=4, bound=20,
        follow_error=15, settle=5, noise=2, max_steps=400)
    assert stop is not None and abs(stop - 150) <= 4, note


def test_find_stop_uses_the_soft_bound_when_no_mechanical_stop():
    # No mechanical stop before the servo's own angle limit: it tracks all the
    # way, so the soft bound itself is the usable limit for this direction.
    stop, note = _fl.find_stop(
        _finger(2000, +1), start=500, direction=+1, step=4, bound=1001,
        follow_error=15, settle=5, noise=2, max_steps=400)
    assert stop is not None and abs(stop - 1001) <= 4
    assert "soft limit" in note
