"""Presence as one contract for every bus: hot_plug, watched ids, refusals.

What is asserted is what the layers above rely on whatever the bus: a servo
that stops answering becomes absent and is only ever released, one that comes
back is present again, an id the bus does not drive is reported while it
answers and is never written, and a bus that cannot do either says so when it
is built rather than looking configured for it.
"""

import struct

import pytest

from actuo.core.contracts import (
    ActuatorCommand, Bus, CapabilityError, Frame, OperatingMode, Transport,
    require_presence)
from actuo.drivers import dynamixel_protocol as dxl
from actuo.drivers.buses.presence import ABSENT, WATCHED, PresenceTracker


# --------------------------------------------------------------------------
# The shared bookkeeping
# --------------------------------------------------------------------------

def test_a_servo_is_absent_after_misses_in_a_row_and_left_out_of_the_read():
    tracker = PresenceTracker([1, 2], hot_plug=True, absent_after_misses=3)
    tracker.mark_present(1)
    tracker.mark_present(2)
    tracker.missed(2)
    tracker.answered(2)                  # one lost packet is not a module
    tracker.missed(2)
    tracker.missed(2)
    assert tracker.present(2)
    tracker.missed(2)

    assert not tracker.present(2)
    assert tracker.absence_reason(2) == "no reply to 3 reads in a row"
    assert tracker.polled([1, 2]) == [1]


def test_probes_take_absent_servos_and_watched_ids_in_turn():
    tracker = PresenceTracker([1, 2], hot_plug=True, presence_period=2,
                              watch_ids=[7, 1])
    tracker.mark_present(1)

    assert tracker.watch_ids == (7,)     # its own servo is not watched
    due = [tracker.next_probe() for _ in range(6)]
    assert due == [None, (ABSENT, 2), None, (WATCHED, 7), None, (ABSENT, 2)]


def test_answering_ids_are_present_servos_and_watched_ids_that_answered():
    tracker = PresenceTracker([1, 2], hot_plug=True, watch_ids=[7, 8])
    tracker.mark_present(1)
    tracker.watched_answered(8, True)
    tracker.watched_answered(7, False)

    assert tracker.answering_ids() == [1, 8]


def test_without_hot_plug_a_servo_is_never_taken_away():
    tracker = PresenceTracker([1], hot_plug=False, absent_after_misses=1)
    tracker.answered(1)
    for _ in range(5):
        tracker.missed(1)

    assert tracker.present(1)
    assert tracker.consecutive_misses(1) == 5


# --------------------------------------------------------------------------
# A bus that cannot, refuses
# --------------------------------------------------------------------------

def test_a_bus_without_presence_refuses_hot_plug_and_watch_before_it_is_built():
    from actuo.drivers.buses.kondo_h7_bus import KondoH7Bus

    with pytest.raises(CapabilityError, match="hot_plug"):
        require_presence(KondoH7Bus, True, ())
    with pytest.raises(CapabilityError, match="watch_ids"):
        require_presence(KondoH7Bus, False, (4,))
    require_presence(KondoH7Bus, False, ())


def test_the_default_watch_refuses_ids_it_cannot_see():
    class Plain(Bus):
        def set_mode(self, mode):
            pass

        def write_batch(self, commands):
            pass

        def read_batch(self, timeout):
            return []

        def last_state(self, device_id):
            return None

    bus = Plain(None, [type("A", (), {"id": 1})()])
    bus.watch([1])                       # its own servo: nothing to watch
    with pytest.raises(CapabilityError, match="cannot tell"):
        bus.watch([1, 3])
    assert not bus.present(1)
    assert bus.absence_reason(1) == "has not reported"


def test_build_bus_refuses_hot_plug_on_a_bus_that_cannot(tmp_path):
    from actuo.drivers.config import build_bus

    config = tmp_path / "robot.yaml"
    config.write_text(
        "bus:\n"
        "  plugin: serial_bus\n"
        "  transport: {plugin: serial, channel: /dev/null, bitrate: 1000000}\n"
        "  hot_plug: true\n"
        "joints:\n"
        "  - name: j1\n"
        "    actuator: {plugin: dynamixel, id: 1, model: XL430-W250-T}\n")
    with pytest.raises(CapabilityError, match="SerialBus cannot take servos away"):
        build_bus(str(config))


# --------------------------------------------------------------------------
# DYNAMIXEL: PING with the model number
# --------------------------------------------------------------------------

class DxlLine(Transport):
    """Protocol 2.0 servos on a line: they answer PING, READ and SYNC_READ.

    ``servos`` maps id -> model number; taking one out of it unplugs it.
    """

    BLOCK = dxl.to_le(0, 2) + dxl.to_le(0, 4) + dxl.to_le(2048, 4)

    def __init__(self, servos):
        self.servos = dict(servos)
        self.written = []
        self._inbox = []

    def open(self):
        pass

    def close(self):
        pass

    @property
    def is_open(self):
        return True

    def reset_input(self):
        self._inbox.clear()

    def _status(self, dxl_id, params):
        return Frame(payload=dxl.build_packet(dxl.PROTOCOL_2, dxl_id, dxl.STATUS,
                                              b"\x00" + params), address=dxl_id)

    def _answer(self, payload):
        dxl_id, instruction, params = payload[4], payload[7], payload[8:-2]
        if instruction == dxl.PING and dxl_id in self.servos:
            number = self.servos[dxl_id]
            return [self._status(dxl_id, bytes([number & 0xFF, number >> 8, 45]))]
        if instruction == 0x82:          # SYNC_READ: one status per id asked
            return [self._status(i, self.BLOCK) for i in params[4:] if i in self.servos]
        if instruction == 0x02 and dxl_id in self.servos:
            return [self._status(dxl_id, self.BLOCK)]
        return []

    def write(self, frames):
        for frame in frames:
            self.written.append(frame.payload)
            self._inbox.extend(self._answer(frame.payload))

    def recv(self, timeout):
        return self._inbox.pop(0) if self._inbox else None

    def transact(self, frames, timeout):
        out = []
        for frame in frames:
            self.written.append(frame.payload)
            replies = self._answer(frame.payload)
            out.append(replies[0] if replies else None)
        return out


def dxl_bus(servos, ids=(1, 2), **kwargs):
    from actuo.drivers.actuators.dynamixel import (
        DynamixelActuator, DynamixelCalibration)
    from actuo.drivers.buses.dynamixel_bus import DynamixelBus
    from actuo.drivers.dynamixel_models import load_model_table

    model = load_model_table().by_name("XL430-W250-T")
    actuators = [DynamixelActuator(i, DynamixelCalibration(
        counts_per_rev=model.counts_per_rev, zero_offset_counts=model.center_counts),
        model) for i in ids]
    line = DxlLine({i: (model.number if n is None else n) for i, n in servos.items()})
    kwargs.setdefault("presence_period", 1)
    kwargs.setdefault("absent_after_misses", 2)
    return DynamixelBus(line, actuators, hot_plug=True, **kwargs), line, model


def test_dynamixel_starts_without_a_missing_servo_and_takes_it_back():
    bus, line, model = dxl_bus({1: None})
    items = bus.verify_servos()

    assert bus.present(1) and not bus.present(2)
    assert [i.message for i in items] == ["not answering"]
    line.servos[2] = model.number
    for _ in range(3):
        bus.read_batch()
    assert bus.present(2)


def test_dynamixel_an_unplugged_servo_is_left_out_and_only_released():
    bus, line, _ = dxl_bus({1: None, 2: None})
    bus.verify_servos()
    del line.servos[2]
    bus.read_batch()
    bus.read_batch()
    assert not bus.present(2)

    line.written.clear()
    bus.read_batch()
    sync_reads = [p for p in line.written if p[7] == 0x82]
    assert sync_reads and all(2 not in p[8 + 4:-2] for p in sync_reads)
    bus.write_batch([ActuatorCommand(mode=OperatingMode.POSITION, position=0.0)] * 2)
    goal = [p for p in line.written if p[7] == 0x83 and p[8] == 116]  # goal_position
    assert goal and all(2 not in p[8 + 4:-2:5] for p in goal)


def test_dynamixel_a_servo_of_another_model_stays_absent_and_says_so():
    bus, _, model = dxl_bus({1: None, 2: 1200})
    bus.verify_servos()

    assert not bus.present(2)
    assert "model number 1200" in bus.absence_reason(2)


def test_dynamixel_watched_ids_are_pinged_and_reported():
    bus, line, model = dxl_bus({1: None, 2: None, 7: None}, watch_ids=[7, 8])
    bus.verify_servos()
    for _ in range(4):
        bus.read_batch()

    assert bus.answering_ids() == [1, 2, 7]
    del line.servos[7]
    for _ in range(4):
        bus.read_batch()
    assert bus.answering_ids() == [1, 2]


# --------------------------------------------------------------------------
# CAN: presence from status frames as they arrive
# --------------------------------------------------------------------------

class CanLine(Transport):
    """Hands back the frames queued in ``inbox``; records what is written."""

    def __init__(self):
        self.inbox = []
        self.written = []

    def open(self):
        pass

    def close(self):
        pass

    @property
    def is_open(self):
        return True

    def write(self, frames):
        self.written.extend(frames)

    def recv(self, timeout):
        return self.inbox.pop(0) if self.inbox else None

    def transact(self, frames, timeout):
        raise NotImplementedError


def status_frame(node_id):
    """A CubeMars servo-mode status upload (packet 41) from ``node_id``."""
    return Frame(payload=struct.pack(">hhhbb", 0, 0, 0, 25, 0),
                 address=(41 << 8) | node_id)


def cubemars(node_id):
    from actuo.drivers.actuators.cubemars import CubeMarsActuator, CubeMarsCalibration

    return CubeMarsActuator(node_id, CubeMarsCalibration(pole_pairs=21), use_pos_spd=False)


def test_can_a_motor_that_goes_quiet_is_absent_and_gets_its_release_only(monkeypatch):
    from actuo.drivers.buses import can_bus

    now = [100.0]
    monkeypatch.setattr(can_bus.time, "monotonic", lambda: now[0])
    line = CanLine()
    bus = can_bus.CanBus(line, [cubemars(1), cubemars(2)], stale_after=0.5,
                         hot_plug=True)
    line.inbox = [status_frame(1)]
    states = bus.read_batch(timeout=0.01)    # 2 never reported: no raise

    assert states[1] is None
    assert bus.present(1) and not bus.present(2)
    assert bus.absence_reason(2) == "no status frame ever received"
    now[0] += 1.0
    assert not bus.present(1)
    assert bus.absence_reason(1).startswith("silent for 1.00 s")

    line.inbox = [status_frame(1)]
    bus.read_batch(timeout=0.01)
    hold = ActuatorCommand(mode=OperatingMode.POSITION, position=0.0)
    bus.write_batch([hold, hold])
    sent = {int(f.address) & 0xFF: int(f.address) >> 8 for f in line.written}
    release = int(cubemars(2).encode_release().address) >> 8
    assert sent[2] == release and sent[1] != release


def test_can_watched_motors_count_while_their_status_frames_arrive():
    line = CanLine()
    from actuo.drivers.buses.can_bus import CanBus

    bus = CanBus(line, [cubemars(1)], hot_plug=True, watch_ids=[5])
    line.inbox = [status_frame(1), status_frame(5), status_frame(9)]
    bus.read_batch(timeout=0.01)
    bus.poll()

    assert bus.answering_ids() == [1, 5]
    assert bus.unmatched_addresses == {(41 << 8) | 9: 1}   # 9 is not watched
    assert all(int(f.address) & 0xFF != 5 for f in line.written)


def test_can_motors_that_answer_only_when_addressed_cannot_be_watched():
    from actuo.drivers.actuators.cubemars_mit import CubeMarsMitActuator
    from actuo.drivers.buses.can_bus import CanBus

    mit = CubeMarsMitActuator.__new__(CubeMarsMitActuator)
    mit.id = 1
    with pytest.raises(CapabilityError, match="answers only when addressed"):
        CanBus(CanLine(), [mit], watch_ids=[2])


# --------------------------------------------------------------------------
# RCB-4mini: watched ICS numbers are polled by the board, never driven
# --------------------------------------------------------------------------

def test_rcb4_watched_ids_are_assigned_never_commanded_and_followed():
    from test_kondo_rcb4_bus import StepClock, make_bus, FakeRcb4, table_of
    from rcb4.protocol import rcb4mini as protocol

    board = FakeRcb4({2: 9000, 5: 7000, 7: 8000})
    bus = make_bus(board, hot_plug=True, watch_ids=[7, 9], presence_period=2,
                   absent_timeout_s=0.05, clock=StepClock())
    bus.verify_servos()
    # Watched numbers stay assigned while silent (9), so a later plug is seen.
    assert set(table_of(board)) == {2, 5, 7, 9}
    for _ in range(4):
        bus.read_batch()
    assert bus.answering_ids() == [2, 5, 7]

    bus.write_batch([ActuatorCommand(mode=OperatingMode.POSITION, position=0.0)] * 2)
    multi = protocol.build_multi_servo([2], [7500], 1)[0]
    mask7 = protocol.build_multi_servo([7], [7500], 1)[1:6]
    commands = [p for p in board.sent if p[0] == multi]
    assert commands and all(not any(a & b for a, b in zip(p[1:6], mask7))
                            for p in commands)

    del board.servos[7]
    board.servos[9] = 8000
    for _ in range(6):
        bus.read_batch()
    assert bus.answering_ids() == [2, 5, 9]


def test_rcb4_watched_ids_are_fixed_at_connect():
    from test_kondo_rcb4_bus import FakeRcb4, make_bus

    bus = make_bus(FakeRcb4({2: 9000, 5: 7000}), hot_plug=True, watch_ids=[7])
    bus.verify_servos()
    bus.watch([7])                       # the same ids: fine
    with pytest.raises(CapabilityError, match="at connect"):
        bus.watch([7, 9])

