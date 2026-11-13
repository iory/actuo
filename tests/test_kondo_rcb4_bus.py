"""Tests for the Kondo RCB-4mini bus, against a simulated board (no hardware).

``FakeRcb4`` keeps the board's RAM and behaves the way an RCB-4mini was seen to
(see references.md, "Kondo RCB-4mini"): it answers only at the COM rate its
config selects, acknowledges a config write at the old rate and then switches,
and writes a servo's pulse into its record only once the ICS number is assigned
and the ICS rate matches the servo's.
"""

import math
import struct

import pytest

from actuo.core.contracts import ActuatorCommand, Frame, OperatingMode, Transport
from actuo.core.diagnostics import DiagLevel

from actuo.drivers.actuators.kondo_ics import (
    CENTER_PULSE, KondoIcsActuator, KondoIcsCalibration, PULSE_PER_DEG,
    SERVO_OFF_PULSE, SERVO_ON_PULSE)
from actuo.drivers.buses.kondo_rcb4_bus import KondoRcb4Bus
from actuo.drivers.kondo_h7_protocol import OP_MULTI_SERVO_SINGLE_VELOCITY
from rcb4.protocol import rcb4mini as protocol
from rcb4.protocol.framing import ProtocolError

FACTORY_CONFIG = 0x8313   # read from an RCB-4mini without a project


class FakeRcb4(Transport):
    """An RCB-4mini behind a serial port, as far as the bus can tell."""

    def __init__(self, servos, config=FACTORY_CONFIG, table=None,
                 servo_ics_bitrate=1250000, firmware=protocol.FIRMWARE_VERSION_REPLY,
                 bitrate=1250000, mute=False):
        self.bitrate = bitrate
        self.current_bitrate = bitrate
        self.ram = bytearray(0x400)
        self._put(protocol.CONFIG_ADDRESS, struct.pack("<H", config))
        table = table or {}
        for slot in range(protocol.ICS_SLOT_COUNT):
            self._put(protocol.ICS_ASSIGN_ADDRESS + 2 * slot, struct.pack(
                "<H", table.get(slot, protocol.ICS_UNASSIGNED)))
        #: ICS number -> pulse of the servos actually wired up.
        self.servos = dict(servos)
        self.servo_ics_code = protocol.ICS_BITRATE_CODES[servo_ics_bitrate]
        self.firmware = firmware
        self.mute = mute
        #: Refuse this many of the next requests with ``[0x0b, 0x15]``.
        self.refuse = 0
        #: Leave this many of the next requests unanswered.
        self.drop = 0
        self.sent = []
        self.reopened = []

    def _put(self, address, data):
        self.ram[address:address + len(data)] = data

    def _word(self, address):
        return struct.unpack("<H", self.ram[address:address + 2])[0]

    @property
    def board_bitrate(self):
        return protocol.com_bitrate_of(self._word(protocol.CONFIG_ADDRESS))

    def _ics_cycle(self):
        """What the board's servo loop does between host packets."""
        ics_code = ((self._word(protocol.CONFIG_ADDRESS) & protocol.ICS_BITRATE_MASK)
                    >> protocol.ICS_BITRATE_SHIFT)
        for slot, pulse in self.servos.items():
            record = protocol.servo_record_address(slot)
            assigned = self._word(protocol.ICS_ASSIGN_ADDRESS + 2 * slot) == record
            if assigned and ics_code == self.servo_ics_code:
                self._put(record + 4, struct.pack("<H", pulse))

    # -- Transport contract --------------------------------------------------

    def open(self):
        pass

    def close(self):
        pass

    @property
    def is_open(self):
        return True

    def reopen(self, bitrate):
        self.current_bitrate = bitrate
        self.reopened.append(bitrate)

    def write(self, frames):
        self.transact(frames, 0.0)

    def recv(self, timeout):
        return None

    def transact(self, frames, timeout):
        return [self._reply(item.payload) for item in frames]

    def _reply(self, payload):
        self.sent.append(bytes(payload))
        if self.mute or self.current_bitrate != self.board_bitrate:
            return None
        if self.drop > 0:
            self.drop -= 1
            return None
        if self.refuse > 0:
            self.refuse -= 1
            return Frame(payload=bytes([0x0B, protocol.NAK_BYTE]))
        self._ics_cycle()
        if payload == protocol.build_ack():
            return Frame(payload=bytes([0xFE, 0x06]))
        if payload == protocol.build_version():
            return Frame(payload=self.firmware)
        if payload[:2] == bytes([protocol.OP_MOVE, protocol.SUB_RAM_TO_COM]):
            address = payload[5] | payload[6] << 8
            size = payload[7]
            return Frame(payload=bytes([0x00]) + bytes(self.ram[address:address + size]))
        if payload[:2] == bytes([protocol.OP_MOVE, protocol.SUB_COM_TO_RAM]):
            address = payload[2] | payload[3] << 8
            self._put(address, payload[5:])
            return Frame(payload=bytes([0x00, 0x06]))
        return Frame(payload=bytes([payload[0], 0x06]))


def servo(servo_id, **calibration):
    return KondoIcsActuator(servo_id, KondoIcsCalibration(**calibration))


def make_bus(board, ids=(2, 5), **kwargs):
    kwargs.setdefault("assign_settle_s", 0.0)
    return KondoRcb4Bus(board, [servo(i) for i in ids],
                        joint_names=["j{}".format(i) for i in ids], **kwargs)


def table_of(board):
    return {slot: board._word(protocol.ICS_ASSIGN_ADDRESS + 2 * slot)
            for slot in range(protocol.ICS_SLOT_COUNT)
            if board._word(protocol.ICS_ASSIGN_ADDRESS + 2 * slot)
            != protocol.ICS_UNASSIGNED}


# --------------------------------------------------------------------------
# Cold start
# --------------------------------------------------------------------------


def test_cold_start_from_a_factory_board():
    board = FakeRcb4({2: 9000, 5: 7000})
    assert board.board_bitrate == 115200
    bus = make_bus(board)

    items = bus.verify_servos()

    # Tried 1.25M, found the board at 115200, switched it, came back at 1.25M.
    assert board.reopened == [115200, 1250000]
    assert board.board_bitrate == 1250000
    assert table_of(board) == {2: protocol.servo_record_address(2),
                               5: protocol.servo_record_address(5)}
    names = {item.name for item in items}
    assert names == {"bus/com_bitrate", "bus/config", "bus/ics_assign"}
    assert all(item.level == DiagLevel.WARN for item in items)
    config = board._word(protocol.CONFIG_ADDRESS)
    assert not config & (protocol.CONFIG_RUN_ROM | protocol.CONFIG_VECTOR_JUMP)


def test_a_stopped_rom_program_is_not_reported_again():
    board = FakeRcb4({2: 9000, 5: 7000})
    bus = make_bus(board)
    bus.verify_servos()

    items = bus.verify_servos()

    assert "bus/config" not in {item.name for item in items}


def test_assigning_sends_the_servo_off_first():
    board = FakeRcb4({2: 9000})
    board._put(protocol.servo_record_address(2) + 6, struct.pack("<H", CENTER_PULSE))
    bus = make_bus(board, ids=(2,))

    bus.verify_servos()

    assign_index = next(
        i for i, p in enumerate(board.sent)
        if p[:4] == bytes([0x00, 0x02, protocol.ICS_ASSIGN_ADDRESS, 0x00]))
    # No pulse, OFF, and no interpolation left to rewrite the OFF.
    clear = protocol.build_ram_write(
        protocol.servo_record_address(2) + 4, struct.pack("<HHB", 0, SERVO_OFF_PULSE, 0))
    assert clear in board.sent[:assign_index]


def test_a_stale_pulse_is_not_taken_for_a_servo():
    board = FakeRcb4({2: 9000})
    # Servo 5 is not wired, but its record still holds a pulse from before.
    board._put(protocol.servo_record_address(5) + 4, struct.pack("<H", 8000))
    bus = make_bus(board)

    with pytest.raises(RuntimeError, match=r"j5 \(id 5\)"):
        bus.verify_servos()


def test_a_missing_servo_is_unassigned_again():
    board = FakeRcb4({2: 9000})
    bus = make_bus(board)

    items = bus.verify_servos(allow_missing=True)

    assert table_of(board) == {2: protocol.servo_record_address(2)}
    missing = [item for item in items if item.name == "bus/startup_missing"]
    assert missing and missing[0].values["missing_ids"] == "5"


def test_servos_at_another_ics_rate_do_not_answer():
    board = FakeRcb4({2: 9000, 5: 7000})
    board.servo_ics_code = 1   # wired servos run at a rate the bus does not set
    bus = make_bus(board)
    with pytest.raises(RuntimeError, match="did not answer"):
        bus.verify_servos()


def test_a_project_assignment_is_left_alone():
    project_config = protocol.with_ics_bitrate(
        protocol.with_com_bitrate(FACTORY_CONFIG, 1250000), 1250000)
    table = {2: protocol.servo_record_address(2), 5: protocol.servo_record_address(5)}
    board = FakeRcb4({2: 9000, 5: 7000}, config=project_config, table=table)
    bus = make_bus(board)

    # The project's program is stopped (it would command the same servos);
    # its assignment, already each number's own record, is not touched.
    assert [item.name for item in bus.verify_servos()] == ["bus/config"]
    writes = [p for p in board.sent if p[:2] == bytes([0x00, 0x02])]
    assert writes == [protocol.build_ram_write(
        protocol.CONFIG_ADDRESS,
        struct.pack("<H", protocol.flat_config(project_config)))]


def test_a_project_mapping_to_another_record_is_mapped_back():
    # ICS 2 pointed at ICS 3's record, ICS 9 (not configured) at ICS 2's: the
    # board would drive servo 9 with servo 2's target.
    table = {2: protocol.servo_record_address(3), 9: protocol.servo_record_address(2)}
    board = FakeRcb4({2: 9000}, config=protocol.with_com_bitrate(FACTORY_CONFIG, 1250000),
                     table=table)
    bus = make_bus(board, ids=(2,))

    items = bus.verify_servos()

    assert table_of(board) == {2: protocol.servo_record_address(2),
                               9: protocol.servo_record_address(9)}
    assert board._word(protocol.servo_record_address(9) + 6) == SERVO_OFF_PULSE
    (remap,) = [item for item in items if item.name == "bus/ics_remap"]
    assert remap.values["remapped_ids"] == "2, 9"
    bus.read_batch()
    assert bus.last_state(2).aux["pulse"] == 9000


def test_the_ics_rate_is_set_under_a_project_too():
    # The project drives ICS 7 at 115200; this config says 1.25 M. The board
    # is run at the config's rate, and servo 7, not at it, stops answering.
    table = {7: protocol.servo_record_address(7)}
    board = FakeRcb4({2: 9000, 7: 7500},
                     config=protocol.with_com_bitrate(FACTORY_CONFIG, 1250000),
                     table=table)
    bus = make_bus(board, ids=(2,))

    items = bus.verify_servos()

    config = board._word(protocol.CONFIG_ADDRESS)
    assert config & protocol.ICS_BITRATE_MASK == 2 << protocol.ICS_BITRATE_SHIFT
    (flat,) = [item for item in items if item.name == "bus/config"]
    assert "ICS rate to 1250000" in flat.message


def test_the_interpolation_message_and_the_output_cycle_are_put_back():
    project_config = protocol.with_ics_bitrate(
        protocol.with_com_bitrate(FACTORY_CONFIG, 1250000), 1250000)
    project_config = (project_config & ~protocol.FRAME_MASK) | 0x0030 | 0x0004
    board = FakeRcb4({2: 9000}, config=project_config)
    bus = make_bus(board, ids=(2,))

    items = bus.verify_servos()

    config = board._word(protocol.CONFIG_ADDRESS)
    assert config & protocol.FRAME_MASK == protocol.FLAT_FRAME
    assert not config & protocol.CONFIG_SERVO_RESPONSE
    (flat,) = [item for item in items if item.name == "bus/config"]
    assert "interpolation-end message off" in flat.message
    assert "output cycle to 15 ms" in flat.message


def test_another_board_is_refused():
    board = FakeRcb4({2: 9000}, firmware=b"\xfdARMH7 v0.6.5")
    bus = make_bus(board, ids=(2,))
    with pytest.raises(RuntimeError, match="RCB-4mini"):
        bus.verify_servos()


def test_a_board_that_answers_at_no_rate():
    board = FakeRcb4({2: 9000}, mute=True)
    bus = make_bus(board, ids=(2,))
    with pytest.raises(RuntimeError, match="answers at none"):
        bus.verify_servos()
    assert board.current_bitrate == 1250000


# --------------------------------------------------------------------------
# Cycle
# --------------------------------------------------------------------------


def started(servos, ids=(2, 5), **kwargs):
    board = FakeRcb4(servos)
    bus = make_bus(board, ids=ids, **kwargs)
    bus.verify_servos()
    board.sent.clear()
    return board, bus


def test_read_batch_decodes_every_servo():
    pulse = int(round(CENTER_PULSE + 30 * PULSE_PER_DEG))
    board, bus = started({2: pulse, 5: CENTER_PULSE})

    states = bus.read_batch()

    assert math.degrees(states[0].position) == pytest.approx(30.0, abs=0.05)
    assert states[1].position == pytest.approx(0.0, abs=1e-9)
    assert len(board.sent) == 1          # 2 and 5 share one record read
    assert bus.last_state(2) is states[0]
    assert bus.state_updates == 1


def test_read_batch_without_an_answer_keeps_nothing_invented():
    board, bus = started({2: 9000, 5: 7000})
    board.current_bitrate = 115200   # board stops answering

    states = bus.read_batch()

    assert states == [None, None]
    assert bus.missed == 1
    assert bus.state_updates == 0


def test_write_batch_is_one_packet_with_release_in_it():
    board, bus = started({2: 9000, 5: 7000}, move_time_ms=None)
    command = ActuatorCommand(mode=OperatingMode.POSITION, position=0.0)

    bus.write_batch([command, None])

    # The servo ON packet comes first; the targets are the last packet.
    assert len(board.sent) == 2
    packet = board.sent[-1]
    assert packet[0] == OP_MULTI_SERVO_SINGLE_VELOCITY
    pulses = struct.unpack("<2H", packet[7:11])
    assert pulses == (CENTER_PULSE, SERVO_OFF_PULSE)
    assert bus.servo_on_states() == {2: True, 5: False}


def test_a_released_servo_is_switched_on_before_its_first_target():
    board, bus = started({2: 9000, 5: 7000}, move_time_ms=None)
    command = ActuatorCommand(mode=OperatingMode.POSITION, position=0.0)

    bus.write_batch([command, None])
    bus.write_batch([command, None])

    # First call: servo ON where it stands, then the target. Second call: the
    # target only, since the servo is already on.
    assert len(board.sent) == 3
    first = struct.unpack("<2H", board.sent[0][7:11])
    assert first == (SERVO_ON_PULSE, SERVO_OFF_PULSE)
    for packet in board.sent[1:]:
        assert struct.unpack("<2H", packet[7:11]) == (CENTER_PULSE, SERVO_OFF_PULSE)


def test_a_servo_released_again_is_switched_on_again():
    board, bus = started({2: 9000, 5: 7000}, move_time_ms=None)
    command = ActuatorCommand(mode=OperatingMode.POSITION, position=0.0)

    bus.write_batch([command, command])
    bus.write_batch([None, command])
    board.sent.clear()
    bus.write_batch([command, command])

    assert len(board.sent) == 2
    assert struct.unpack("<2H", board.sent[0][7:11]) == (SERVO_ON_PULSE, CENTER_PULSE)


def test_different_move_bytes_are_refused_not_sent():
    board, bus = started({2: 9000, 5: 7000}, move_time_ms=None)
    bus.actuators[1].velocity = 5
    command = ActuatorCommand(mode=OperatingMode.POSITION, position=0.0)

    with pytest.raises(ValueError, match="move byte"):
        bus.write_batch([command, command])

    assert board.sent == []


def test_different_configured_move_bytes_are_refused_at_build():
    board = FakeRcb4({2: 9000, 5: 7000})
    actuators = [KondoIcsActuator(2, KondoIcsCalibration(), velocity=1),
                 KondoIcsActuator(5, KondoIcsCalibration(), velocity=5)]
    with pytest.raises(ValueError, match="move byte"):
        KondoRcb4Bus(board, actuators)


def wheel(servo_id, rotation_checked=None):
    from actuo.drivers.actuators.kondo_ics_wheel import (
        KondoIcsWheelActuator, KondoIcsWheelCalibration)
    return KondoIcsWheelActuator(servo_id, KondoIcsWheelCalibration(
        deadband_deg=6.0, saturate_deg=20.0, rev_s_per_deg=0.0528),
        rotation_checked=rotation_checked)


def test_a_wheel_whose_rotation_mode_was_not_checked_is_refused():
    board = FakeRcb4({2: 9000})
    with pytest.raises(ValueError, match="rotation_checked"):
        KondoRcb4Bus(board, [wheel(2)])


def started_with_wheel():
    """Servo 2 a joint, servo 5 a checked wheel whose pot reads 11000."""
    board = FakeRcb4({2: 9000, 5: 11000})
    # Move byte 1 for the joint too: a wheel's is always 1, and the board
    # takes one move byte for every servo.
    joint = KondoIcsActuator(2, KondoIcsCalibration(), velocity=1)
    bus = KondoRcb4Bus(board, [joint, wheel(5, "test")],
                       joint_names=["j2", "w5"], assign_settle_s=0.0)
    bus.verify_servos()
    board.sent.clear()
    return board, bus


def reference_of(board, slot):
    return board._word(protocol.servo_record_address(slot) + 6)


def test_a_wheel_is_switched_on_at_stop_not_at_its_pot_reading():
    board, bus = started_with_wheel()
    joint = ActuatorCommand(mode=OperatingMode.POSITION, position=0.0)
    speed = ActuatorCommand(mode=OperatingMode.VELOCITY, velocity=0.0)

    bus.write_batch([joint, speed])

    # SERVO_ON_PULSE would copy the pot (11000, i.e. a fast speed) into the
    # wheel's reference. The wheel's reference is written to stop instead,
    # and only the joint gets SERVO_ON_PULSE.
    assert reference_of(board, 5) == CENTER_PULSE
    on_packets = [struct.unpack("<2H", packet[7:11]) for packet in board.sent
                  if packet[0] == OP_MULTI_SERVO_SINGLE_VELOCITY]
    assert on_packets[0] == (SERVO_ON_PULSE, CENTER_PULSE)
    assert on_packets[-1] == (CENTER_PULSE, CENTER_PULSE)
    assert all(pulses[1] != SERVO_ON_PULSE for pulses in on_packets)
    assert bus.servo_on_states() == {2: True, 5: True}


def test_hold_all_holds_a_wheel_at_stop():
    board, bus = started_with_wheel()

    bus.hold_all()

    assert reference_of(board, 5) == CENTER_PULSE
    packet = [p for p in board.sent if p[0] == OP_MULTI_SERVO_SINGLE_VELOCITY][-1]
    assert struct.unpack("<2H", packet[7:11]) == (SERVO_ON_PULSE, CENTER_PULSE)


def test_supply_voltage_is_reported_with_its_threshold():
    board, bus = started({2: 9000, 5: 7000}, battery_voltage_warn_below=10.0)
    # 451 counts * 25 / 1024 = 11.01 V
    board._put(protocol.BATTERY_ADC_ADDRESS, struct.pack("<H", 451))

    power = [i for i in bus.diagnose() if i.name == "bus/power"][0]

    assert power.level == DiagLevel.OK
    assert power.values["battery_voltage_v"] == "11.011"
    bus.set_battery_voltage_threshold(11.5)
    power = [i for i in bus.diagnose() if i.name == "bus/power"][0]
    assert power.level == DiagLevel.WARN


def test_ids_outside_the_board_are_refused():
    with pytest.raises(ValueError):
        make_bus(FakeRcb4({}), ids=(35,))


def test_a_refused_read_is_sent_again():
    board, bus = started({2: 9000, 5: 7000})
    board.refuse = 2

    states = bus.read_batch()

    assert states[0] is not None and states[1] is not None
    assert len(board.sent) == 3
    assert bus.naks == 2
    assert bus.missed == 0


def test_a_board_that_keeps_refusing_is_a_miss_not_a_state():
    board, bus = started({2: 9000, 5: 7000}, nak_retries=1)
    board.refuse = 10

    states = bus.read_batch()

    assert states == [None, None]
    assert len(board.sent) == 2
    assert bus.missed == 1
    assert bus.naks == 2


def test_a_refused_write_is_sent_again():
    board, bus = started({2: 9000, 5: 7000})
    command = ActuatorCommand(mode=OperatingMode.POSITION, position=0.0)
    bus.write_batch([command, command])   # servos on
    board.sent.clear()
    board.refuse = 1

    bus.write_batch([command, command])

    assert len(board.sent) == 2 and board.sent[0] == board.sent[1]
    assert bus.naks == 1


def test_an_unanswered_ram_write_is_sent_again():
    board, bus = started({2: 9000, 5: 7000})
    board.sent.clear()
    board.drop = 2

    bus._write_config(bus.read_config())

    assert len(board.sent) == 4          # read dropped twice, then read, write
    assert bus.unanswered == 2


def test_a_ram_write_never_answered_is_an_error():
    board, bus = started({2: 9000, 5: 7000}, nak_retries=1)
    board.mute = True

    with pytest.raises(ProtocolError, match="did not acknowledge"):
        bus._write_ram(protocol.CONFIG_ADDRESS, b"\x00\x00")
    assert bus.unanswered == 2


def test_an_unanswered_cycle_read_is_a_miss_not_resent():
    board, bus = started({2: 9000, 5: 7000})
    board.sent.clear()
    board.drop = 1

    states = bus.read_batch()

    assert states == [None, None]
    assert len(board.sent) == 1
    assert bus.missed == 1


def test_move_time_is_refused_while_its_unit_is_unknown(monkeypatch):
    from actuo.drivers.buses import kondo_rcb4_bus
    monkeypatch.setattr(kondo_rcb4_bus, "FRAME_SCALE_MS", None)
    with pytest.raises(ValueError, match="not been measured"):
        make_bus(FakeRcb4({}), move_time_ms=50.0)


# --------------------------------------------------------------------------
# Hot plug
# --------------------------------------------------------------------------


class StepClock:
    """A clock that moves on by one control period each time it is read."""

    def __init__(self, step=0.03):
        self.now = 0.0
        self.step = step

    def __call__(self):
        self.now += self.step
        return self.now


def hot(servos, ids=(2, 5), **kwargs):
    kwargs.setdefault("presence_period", 2)
    kwargs.setdefault("absent_timeout_s", 0.05)
    kwargs.setdefault("clock", StepClock())
    return started(servos, ids=ids, hot_plug=True, **kwargs)


def test_hot_plug_starts_without_a_missing_servo_and_keeps_it_assigned():
    board = FakeRcb4({2: 9000})
    bus = make_bus(board, hot_plug=True)

    items = bus.verify_servos()          # no raise without allow_missing

    assert "bus/startup_missing" in {item.name for item in items}
    # Still assigned, so the board keeps polling 5 and notices it later.
    assert table_of(board) == {2: protocol.servo_record_address(2),
                               5: protocol.servo_record_address(5)}
    assert bus.present(2) and not bus.present(5)


def test_an_unplugged_servo_becomes_absent_and_keeps_its_last_pose():
    board, bus = hot({2: 9000, 5: 7000})
    bus.read_batch()
    assert bus.present(5)

    del board.servos[5]                  # unplugged
    for _ in range(4):
        bus.read_batch()

    assert not bus.present(5)
    assert bus.present(2)
    # Nothing decoded from a zero pulse: the last pose is the real one.
    assert bus.last_state(5).aux["pulse"] == 7000


def test_a_servo_plugged_back_in_is_present_again():
    board, bus = hot({2: 9000, 5: 7000})
    del board.servos[5]
    for _ in range(4):
        bus.read_batch()
    assert not bus.present(5)

    board.servos[5] = 7100
    bus.read_batch()
    assert bus.present(5)
    # 100 pulses (0.059 rad) from where it was last seen: held until three
    # reads agree (reading_jump_rad, reading_confirm).
    bus.read_batch()
    bus.read_batch()

    assert bus.last_state(5).position == pytest.approx(
        math.radians((7100 - CENTER_PULSE) / PULSE_PER_DEG), abs=1e-6)


def test_a_cleared_pulse_that_is_rewritten_never_reads_absent():
    # Every servo answers: however the clears fall, none is ever absent.
    board, bus = hot({2: 9000, 5: 7000}, presence_period=1)
    for _ in range(10):
        bus.read_batch()
        assert bus.present(2) and bus.present(5)


def test_presence_check_clears_only_the_measured_pulse():
    board, bus = hot({2: 9000, 5: 7000}, presence_period=1)
    bus.read_batch()
    clears = [p for p in board.sent if p[:2] == bytes([0x00, 0x02])]
    assert clears == [
        protocol.build_ram_write(protocol.servo_record_address(i) + 4, b"\x00\x00")
        for i in (2, 5)]


def test_without_hot_plug_presence_is_having_reported():
    board, bus = started({2: 9000, 5: 7000})
    assert not bus.present(2)
    bus.read_batch()
    assert bus.present(2)
    assert not [p for p in board.sent if p[:2] == bytes([0x00, 0x02])]


# --------------------------------------------------------------------------
# Port detection
# --------------------------------------------------------------------------


def ported(servos, ids=(1, 2), **kwargs):
    kwargs.setdefault("move_time_ms", None)
    board = FakeRcb4(servos)
    bus = make_bus(board, ids=ids, port_detect=True, **kwargs)
    items = bus.verify_servos()
    board.sent.clear()
    return board, bus, items


HOLD = ActuatorCommand(mode=OperatingMode.POSITION, position=0.0)


def test_port_detect_finds_each_servo_on_its_port():
    # ID 1 on SIO1-4 (ICS 2) and ID 2 on SIO5-8 (ICS 5), as on the test setup this was measured on.
    board, bus, items = ported({2: 9000, 5: 7000})

    assert bus.ics_number(1) == 2 and bus.ics_number(2) == 5
    assert bus.where(2) == "ICS 5 (SIO5-8)"
    # Without hot_plug the numbers nothing answered on are given back.
    assert table_of(board) == {2: protocol.servo_record_address(2),
                               5: protocol.servo_record_address(5)}
    ports = [item for item in items if item.name == "bus/ics_ports"][0]
    assert ports.values == {"j1": "ICS 2 (SIO1-4)", "j2": "ICS 5 (SIO5-8)"}
    states = bus.read_batch()
    assert states[0].aux["pulse"] == 9000 and states[1].aux["pulse"] == 7000


def test_port_detect_sends_each_target_to_its_port_and_off_to_the_other():
    board, bus, _ = ported({2: 9000, 5: 7000})

    bus.write_batch([HOLD, HOLD])

    # ICS 2, 3, 4, 5 in order: 3 and 4 have no servo of these ids.
    assert struct.unpack("<4H", board.sent[-1][7:15]) == (
        CENTER_PULSE, SERVO_OFF_PULSE, SERVO_OFF_PULSE, CENTER_PULSE)


def test_port_detect_without_hot_plug_needs_every_servo():
    board = FakeRcb4({2: 9000})
    bus = make_bus(board, ids=(1, 2), port_detect=True)
    with pytest.raises(RuntimeError, match=r"j2 \(id 2\)"):
        bus.verify_servos()


def test_port_detect_ids_are_servo_ids():
    make_bus(FakeRcb4({}), ids=(17,), port_detect=True)   # ICS 34 only
    with pytest.raises(ValueError, match="0..17"):
        make_bus(FakeRcb4({}), ids=(18,), port_detect=True)


def test_two_servos_with_one_id_are_refused_without_hot_plug():
    board = FakeRcb4({4: 7000, 5: 7100})
    bus = make_bus(board, ids=(2,), port_detect=True)
    with pytest.raises(RuntimeError, match="same ID"):
        bus.verify_servos()


def test_two_servos_with_one_id_are_driven_on_neither_with_hot_plug():
    board, bus, items = ported({4: 7000, 5: 7100}, ids=(2,), hot_plug=True)

    assert "bus/ics_port_conflict" in {item.name for item in items}
    assert not bus.present(2)
    bus.write_batch([HOLD])
    assert struct.unpack("<2H", board.sent[-1][7:11]) == (
        SERVO_OFF_PULSE, SERVO_OFF_PULSE)


def test_a_servo_plugged_in_later_is_found_on_its_port():
    board, bus, _ = ported({2: 9000}, hot_plug=True, presence_period=2,
                           clock=StepClock())
    assert not bus.present(2) and bus.ics_number(2) is None

    board.servos[5] = 7000
    bus.read_batch()

    assert bus.present(2) and bus.ics_number(2) == 5


def test_a_servo_moved_to_the_other_port_is_followed_and_comes_back_limp():
    board, bus, _ = ported({2: 9000, 4: 7000}, hot_plug=True,
                           presence_period=2, absent_timeout_s=0.05,
                           clock=StepClock())
    bus.read_batch()
    bus.write_batch([HOLD, HOLD])
    assert bus.servo_on_states()[2]

    del board.servos[4]                  # unplugged from SIO1-4 ...
    board.servos[5] = 7000               # ... and plugged into SIO5-8
    seen = []
    for _ in range(8):
        bus.read_batch()
        seen.append(bus.present(2))

    # It read absent on the way, so the servicer released it.
    assert not all(seen) and seen[-1]
    assert bus.ics_number(2) == 5
    assert not bus.servo_on_states()[2]
    board.sent.clear()
    bus.write_batch([HOLD, HOLD])
    # Switched on where it stands on its new number before any target.
    assert struct.unpack("<4H", board.sent[0][7:15]) == (
        CENTER_PULSE, SERVO_OFF_PULSE, SERVO_OFF_PULSE, SERVO_ON_PULSE)
    ports = [i for i in bus.diagnose() if i.name == "bus/ics_ports"][0]
    assert "1 moved" in ports.message


def test_a_zero_read_soon_after_the_clear_is_not_an_absence():
    # A loop catching up after a stall reads twice right after the clear,
    # before the board's next frame: the servo must not read as gone.
    clock = StepClock(step=0.001)
    board, bus = hot({2: 9000, 5: 7000}, presence_period=1,
                     absent_timeout_s=0.1, clock=clock)
    real_cycle = board._ics_cycle
    board._ics_cycle = lambda: None        # the board has not got to them yet
    for _ in range(5):
        bus.read_batch()
        assert bus.present(2) and bus.present(5)
    clock.now += 0.2                       # ...and now it never does
    bus.read_batch()
    assert not bus.present(2) and not bus.present(5)
    board._ics_cycle = real_cycle
    bus.read_batch()
    assert bus.present(2) and bus.present(5)


class ManualClock:
    """A clock the test moves by hand."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_a_pulled_servo_is_noticed_when_checks_come_just_inside_the_timeout():
    # Checks every 5 reads of 19 ms (95 ms) against a 100 ms timeout. Clearing
    # a number still at zero restarted its timeout every check, so the servo
    # never read as gone (0.4-3.5 s late on the stack, by chance of jitter).
    clock = ManualClock()
    board, bus = hot({2: 9000, 5: 7000}, presence_period=5,
                     absent_timeout_s=0.1, clock=clock)

    def cycle():
        clock.now += 0.019
        bus.read_batch()

    for _ in range(10):
        cycle()
    del board.servos[5]
    reads = 0
    while bus.present(5) and reads < 50:
        cycle()
        reads += 1

    assert not bus.present(5)
    assert reads <= 12          # one period to the next clear, then the timeout
    assert bus.present(2)


def test_an_absent_servo_says_why():
    clock = ManualClock()
    board, bus = hot({2: 9000, 5: 7000}, presence_period=1,
                     absent_timeout_s=0.1, clock=clock)
    assert bus.absence_reason(5) == ""
    del board.servos[5]
    for _ in range(10):
        clock.now += 0.02
        bus.read_batch()
    assert not bus.present(5)
    assert "ICS 5: pulse still 0" in bus.absence_reason(5)


def test_a_glitch_of_one_or_two_reads_is_dropped_and_a_real_move_is_taken():
    board, bus = started({2: 9000, 5: 7000}, reading_jump_rad=0.05, reading_confirm=3)
    bus.read_batch()
    before = bus.last_state(2).position

    for glitch_reads in (1, 2):          # both lengths were seen on real hardware
        board.servos[2] = 4000
        for _ in range(glitch_reads):
            bus.read_batch()
            assert bus.last_state(2).position == before
        board.servos[2] = 9000
        bus.read_batch()
        assert bus.last_state(2).position == before
    assert bus.dropped_readings == 3 and bus.held_readings == 3

    board.servos[2] = 4000               # three reads agree: it really moved
    for _ in range(3):
        bus.read_batch()
    assert bus.last_state(2).aux["pulse"] == 4000


def test_a_held_run_that_jumps_again_starts_over():
    board, bus = started({2: 9000, 5: 7000}, reading_jump_rad=0.05, reading_confirm=3)
    bus.read_batch()
    before = bus.last_state(2).position
    for pulse in (4000, 4000, 6000, 6000):   # two glitches of two reads each
        board.servos[2] = pulse
        bus.read_batch()
    assert bus.last_state(2).position == before
    bus.read_batch()                          # third agreeing read of 6000
    assert bus.last_state(2).aux["pulse"] == 6000
    assert bus.dropped_readings == 2


def test_reading_checks_can_be_switched_off():
    board, bus = started({2: 9000, 5: 7000}, reading_jump_rad=None)
    bus.read_batch()
    board.servos[2] = 4000
    bus.read_batch()
    assert bus.last_state(2).aux["pulse"] == 4000


# --------------------------------------------------------------------------
# Board trim
# --------------------------------------------------------------------------


def test_a_board_trim_is_cleared_and_its_servo_switched_off():
    board = FakeRcb4({2: 9000, 5: 7000})
    trim2 = protocol.servo_record_address(2) + 2
    board._put(trim2, struct.pack("<h", 2000))
    board._put(protocol.servo_record_address(2) + 6, struct.pack("<H", CENTER_PULSE))
    bus = make_bus(board)

    items = bus.verify_servos()

    assert struct.unpack("<h", board.ram[trim2:trim2 + 2])[0] == 0
    # Off first: without its trim, a held reference would be driven 2000
    # pulses away at once.
    assert board._word(protocol.servo_record_address(2) + 6) == SERVO_OFF_PULSE
    (trim,) = [item for item in items if item.name == "bus/trim"]
    assert trim.level == DiagLevel.WARN
    assert "ICS 2 (trim +2000" in trim.message
    assert bus.servo_on_states()[2] is False


def test_no_trim_no_report():
    board = FakeRcb4({2: 9000, 5: 7000})
    bus = make_bus(board)

    items = bus.verify_servos()

    assert "bus/trim" not in {item.name for item in items}


def test_a_trim_on_a_number_this_config_does_not_poll_is_cleared_too():
    board = FakeRcb4({2: 9000, 5: 7000})
    trim7 = protocol.servo_record_address(7) + 2
    board._put(trim7, struct.pack("<h", -2700))
    bus = make_bus(board)

    items = bus.verify_servos()

    assert struct.unpack("<h", board.ram[trim7:trim7 + 2])[0] == 0
    assert board._word(protocol.servo_record_address(7) + 6) == SERVO_OFF_PULSE
    assert "ICS 7 (trim -2700" in [i for i in items if i.name == "bus/trim"][0].message


# --------------------------------------------------------------------------
# Board mixing
# --------------------------------------------------------------------------


def put_mixing(board, slot, source, ratio, which=0):
    offset = (14, 17)[which]
    board._put(protocol.servo_record_address(slot) + offset,
               struct.pack("<HB", source, ratio))


def test_the_ics_switch_is_turned_on():
    board = FakeRcb4({2: 9000, 5: 7000}, config=FACTORY_CONFIG & ~0x0001)
    bus = make_bus(board)

    items = bus.verify_servos()

    assert board._word(protocol.CONFIG_ADDRESS) & 0x0001
    config_items = [i for i in items if i.name == "bus/config"]
    assert config_items and "ICS port on" in config_items[0].message


def test_a_record_left_random_gets_its_head_and_servo_id():
    # A board without a project: the records hold whatever the RAM powered up
    # as, and the board addresses the servo ID in byte 1.
    board = FakeRcb4({2: 9000, 5: 7000})
    for slot in (2, 5):
        board._put(protocol.servo_record_address(slot), bytes([0x59, 0xC5]))
    bus = make_bus(board)

    bus.verify_servos()

    for slot in (2, 5):
        record = protocol.servo_record_address(slot)
        assert bytes(board.ram[record:record + 2]) == bytes([0, slot >> 1])


def test_a_board_mixing_is_disabled_and_its_servo_switched_off():
    board = FakeRcb4({2: 9000, 5: 7000})
    put_mixing(board, 5, 0x4022, 0x05, which=1)   # an analog input, x5
    board._put(protocol.servo_record_address(5) + 6, struct.pack("<H", CENTER_PULSE))
    bus = make_bus(board)

    items = bus.verify_servos()

    record = protocol.servo_record_address(5)
    assert board.ram[record + 19] == 0
    assert board._word(record + 6) == SERVO_OFF_PULSE
    (mixing,) = [item for item in items if item.name == "bus/mixing"]
    assert "ICS 5 (source 0x4022 ratio 0x05)" in mixing.message
    assert bus.servo_on_states()[5] is False


def test_a_mixing_with_no_source_is_left_alone():
    # What every record on a humanoid's RCB-4mini read: source $FFFF, ratio 1.
    board = FakeRcb4({2: 9000, 5: 7000})
    for slot in range(protocol.ICS_SLOT_COUNT):
        put_mixing(board, slot, 0xFFFF, 0x01, which=0)
        put_mixing(board, slot, 0xFFFF, 0x01, which=1)
    bus = make_bus(board)

    items = bus.verify_servos()

    assert "bus/mixing" not in {item.name for item in items}
    assert board.ram[protocol.servo_record_address(5) + 16] == 0x01


# --------------------------------------------------------------------------
# Stretch and hold_relax
# --------------------------------------------------------------------------

from actuo.drivers.actuators.kondo_ics import HoldRelax  # noqa: E402


def stretched_bus(board, stretch=40, relax=None, **kwargs):
    kwargs.setdefault("assign_settle_s", 0.0)
    gripper = KondoIcsActuator(2, KondoIcsCalibration(), stretch=stretch,
                               hold_relax=relax)
    return KondoRcb4Bus(board, [gripper, servo(5)], joint_names=["grip", "j5"],
                        **kwargs)


def test_stretch_is_set_at_connect_and_reported():
    board = FakeRcb4({2: 9000, 5: 7000})
    bus = stretched_bus(board)

    items = bus.verify_servos()

    assert protocol.build_stretch({2: 40}) in board.sent
    assert "bus/stretch" in {item.name for item in items}
    assert {i.name: i.level for i in bus.diagnose()}["bus/stretch"] == DiagLevel.OK


def test_stretch_is_set_again_when_a_servo_comes_back():
    board = FakeRcb4({2: 9000, 5: 7000})
    bus = stretched_bus(board, hot_plug=True, presence_period=2,
                        absent_timeout_s=0.05, clock=StepClock())
    bus.verify_servos()
    del board.servos[2]
    for _ in range(4):
        bus.read_batch()
    assert not bus.present(2)
    board.sent.clear()

    board.servos[2] = 9000
    bus.read_batch()

    assert bus.present(2)
    assert board.sent.count(protocol.build_stretch({2: 40})) == 1
    bus.read_batch()
    assert board.sent.count(protocol.build_stretch({2: 40})) == 1


def target_pulse(board):
    return struct.unpack("<2H", board.sent[-1][7:11])[0]


def test_a_servo_held_away_from_its_command_is_held_lighter():
    # The servo sits at 9000 pulses (+50.6 deg) and is commanded +80 deg.
    board = FakeRcb4({2: 9000, 5: 7000})
    clock = StepClock(step=0.1)
    relax = HoldRelax(error_rad=math.radians(3.0), seconds=1.0,
                      squeeze_rad=math.radians(1.0))
    bus = stretched_bus(board, relax=relax, clock=clock, move_time_ms=None)
    bus.verify_servos()
    bus.read_batch()
    measured = bus.last_state(2).position
    command = ActuatorCommand(mode=OperatingMode.POSITION, position=math.radians(80.0))
    commanded_pulse = int(round(CENTER_PULSE + 80.0 * PULSE_PER_DEG))

    bus.write_batch([command, None])
    assert target_pulse(board) == commanded_pulse
    for _ in range(12):                      # 1.2 s at the same command
        bus.write_batch([command, None])

    held = int(round(CENTER_PULSE + (math.degrees(measured) + 1.0) * PULSE_PER_DEG))
    assert target_pulse(board) == pytest.approx(held, abs=1)
    assert set(bus.relaxed()) == {2}
    assert "grip" in {i.name: i.message for i in bus.diagnose()}["bus/hold_relax"]

    # A new command is driven at once.
    other = ActuatorCommand(mode=OperatingMode.POSITION, position=math.radians(70.0))
    bus.write_batch([other, None])
    assert target_pulse(board) == int(round(CENTER_PULSE + 70.0 * PULSE_PER_DEG))
    assert bus.relaxed() == {}


def test_a_servo_that_gets_there_is_not_relaxed():
    board = FakeRcb4({2: 9000, 5: 7000})
    relax = HoldRelax(error_rad=math.radians(3.0), seconds=1.0,
                      squeeze_rad=math.radians(1.0))
    bus = stretched_bus(board, relax=relax, clock=StepClock(step=0.1),
                        move_time_ms=None)
    bus.verify_servos()
    bus.read_batch()
    command = ActuatorCommand(mode=OperatingMode.POSITION,
                              position=bus.last_state(2).position + math.radians(2.0))

    for _ in range(20):
        bus.write_batch([command, None])

    assert bus.relaxed() == {}


def test_needs_write_asks_for_the_lighter_hold_with_the_command_unchanged():
    # The bus node writes only when the command changes; a blocked gripper's
    # command does not, so the bus has to ask for the write that relaxes it.
    board = FakeRcb4({2: 9000, 5: 7000})
    clock = StepClock(step=0.1)
    relax = HoldRelax(error_rad=math.radians(3.0), seconds=1.0,
                      squeeze_rad=math.radians(1.0))
    bus = stretched_bus(board, relax=relax, clock=clock, move_time_ms=None)
    bus.verify_servos()
    bus.read_batch()
    command = ActuatorCommand(mode=OperatingMode.POSITION, position=math.radians(80.0))
    bus.write_batch([command, None])         # the one write the node makes
    assert not bus.needs_write()

    while not bus.needs_write():
        assert clock.now < 3.0
        bus.read_batch()

    bus.write_batch([command, None])
    assert set(bus.relaxed()) == {2}
    assert not bus.needs_write()
