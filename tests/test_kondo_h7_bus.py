"""Tests for the Kondo ARM-H7 bus, against a fake transport (no hardware).

The fake answers reads with a synthetic servo vector, so the whole
command/read cycle -- batch packet out, strided read in, per-servo decode -- is
exercised end to end in-process.
"""

import math
import struct

import pytest

from actuo.core.contracts import ActuatorCommand, Frame, OperatingMode, Transport
from actuo.core.diagnostics import DiagLevel

from actuo.drivers.actuators.kondo_ics import (
    CENTER_PULSE, KondoIcsActuator, KondoIcsCalibration, PULSE_PER_DEG,
    SERVO_OFF_PULSE, SERVO_ON_PULSE)
from actuo.drivers.buses.kondo_h7_bus import (
    KondoDiscovery, KondoH7Bus, compare_discovery)
from actuo.drivers.kondo_h7_protocol import (
    CONTROL_SLOTS, OP_MREADV, OP_MULTI_SERVO_MULTI_VELOCITIES, OP_MWRITEV,
    OP_VERSION, ProtocolError, SERVO_SLOTS,
    SERVO_STRUCT_SIZE, SERVO_VECTOR_LENGTH, SYSTEM_SLOTS, SymbolMap)

SERVO_VECTOR = 0x24000100
SYSTEM_BASE = 0x24000000
CONTROL_BASE = 0x24001000
SYMBOLS = SymbolMap(
    "v-test",
    {"servo_vector": SERVO_VECTOR, "SysB": SYSTEM_BASE, "Control": CONTROL_BASE},
)
SYMBOLS_NO_CONTROL = SymbolMap(
    "v-test", {"servo_vector": SERVO_VECTOR, "SysB": SYSTEM_BASE})


class FakeBoard(Transport):
    """A board that records what it was sent and answers reads from a vector.

    ``pulses`` is indexed by servo id, mirroring the firmware's flat
    ``servo_vector``.
    """

    def __init__(self, pulses=None, bytes_values=None, slot_values=None,
                 system_values=None,
                 device_port=None, answer=True, answer_device_port=True,
                 version="v-test"):
        self.sent = []
        self.pulses = pulses or {}
        self.bytes_values = bytes_values or {}
        self.slot_values = {name: dict(values)
                            for name, values in (slot_values or {}).items()}
        self.system_values = {
            str(name): values for name, values in (system_values or {}).items()
        }
        self.device_port = list(device_port) if device_port is not None else [0] * 64
        self.answer = answer
        self.answer_device_port = answer_device_port
        self.version = version

    def open(self):
        pass

    def close(self):
        pass

    @property
    def is_open(self):
        return True

    def write(self, frames):
        self.transact(frames, 0.0)

    def recv(self, timeout):
        return None

    def transact(self, frames, timeout):
        replies = []
        for item in frames:
            self.sent.append(item.payload)
            replies.append(self._reply(item.payload) if self.answer else None)
        return replies

    def _reply(self, payload):
        if payload[0] == OP_VERSION:
            return Frame(payload=self.version.encode("utf-8"))
        if payload[0] != OP_MREADV:
            return Frame(payload=b"\x06")           # ack-ish, ignored
        address = int.from_bytes(payload[1:5], "little")
        count = payload[5]
        element = payload[6]
        stride = int.from_bytes(payload[7:9], "little")

        # Control.device_port (64-byte block, count 1, no stride).
        if (address == CONTROL_BASE + CONTROL_SLOTS["device_port"].offset
                and count == 1 and element == CONTROL_SLOTS["device_port"].size
                and stride == 0):
            if not self.answer_device_port:
                return None
            return Frame(payload=bytes(self.device_port))

        # SystemStruct reads: one slot, one struct instance.
        if count == 1 and stride == 0:
            for name, slot in SYSTEM_SLOTS.items():
                if address == SYSTEM_BASE + slot.offset and element == slot.size:
                    values = self.system_values.get(name)
                    if values is None:
                        if slot.length == 1:
                            values = [0]
                        else:
                            values = [0] * slot.length
                    return Frame(payload=slot.encode(values))

        # ServoStruct slot reads: one slot, 36 entries, struct stride.
        if count == SERVO_VECTOR_LENGTH and stride == SERVO_STRUCT_SIZE:
            slot_offset = address - SERVO_VECTOR
            slot_name = None
            for name, slot in SERVO_SLOTS.items():
                if slot.offset == slot_offset and slot.size == element:
                    slot_name = name
                    break
            if slot_name is None:
                raise RuntimeError(
                    "test fake cannot map servo slot at offset {} size {}"
                    .format(slot_offset, element))
            source = self.slot_values.get(slot_name)
            if source is None:
                source = self.pulses if element == 2 else self.bytes_values
            values = [source.get(i, 0) for i in range(count)]
            fmt = {1: "B", 2: "H"}[element]
            return Frame(payload=struct.pack("<" + fmt * count, *values))

        return Frame(payload=b"")


def make_bus(ids=(0, 1, 5), pulses=None, bytes_values=None, slot_values=None,
             system_values=None, joint_names=None, battery_voltage_warn_below=None,
             device_port=None, answer=True, answer_device_port=True,
             symbols=SYMBOLS, version="v-test", **bus_kwargs):
    actuators = [KondoIcsActuator(i, KondoIcsCalibration()) for i in ids]
    transport = FakeBoard(
        pulses=pulses, bytes_values=bytes_values, slot_values=slot_values,
        system_values=system_values,
        device_port=device_port, answer=answer,
        answer_device_port=answer_device_port, version=version)
    if joint_names is None:
        joint_names = ["joint{}".format(i) for i in range(len(ids))]
    return KondoH7Bus(
        transport, actuators, symbols,
        joint_names=joint_names,
        battery_voltage_warn_below=battery_voltage_warn_below,
        **bus_kwargs), transport


# --------------------------------------------------------------------------
# Batch write
# --------------------------------------------------------------------------


def test_write_batch_sends_exactly_one_packet():
    """N joints, one bus transaction -- the whole reason this bus exists."""
    bus, transport = make_bus()
    bus.write_batch([ActuatorCommand(mode=OperatingMode.POSITION, position=0.0)] * 3)
    assert len(transport.sent) == 1
    assert transport.sent[0][0] == 0x10


def test_write_batch_carries_every_joints_pulse_in_id_order():
    bus, transport = make_bus(ids=(5, 0, 1))
    angles = [math.radians(10.0), math.radians(-10.0), 0.0]
    bus.write_batch([ActuatorCommand(mode=OperatingMode.POSITION, position=a)
                     for a in angles])
    body = transport.sent[0]
    pulses = struct.unpack("<3H", body[7:])
    # ids ascend 0, 1, 5 -> the angles commanded for those ids, reordered
    assert pulses[0] == pytest.approx(CENTER_PULSE - 10.0 * PULSE_PER_DEG, abs=1)
    assert pulses[1] == CENTER_PULSE
    assert pulses[2] == pytest.approx(CENTER_PULSE + 10.0 * PULSE_PER_DEG, abs=1)


def test_a_none_command_releases_that_joint_in_the_same_packet():
    """Released and held joints must not straddle two transactions, or the
    robot spends a cycle half-commanded."""
    bus, transport = make_bus(ids=(0, 1))
    bus.write_batch([None,
                     ActuatorCommand(mode=OperatingMode.POSITION, position=0.0)])
    assert len(transport.sent) == 1
    pulses = struct.unpack("<2H", transport.sent[0][7:])
    assert pulses[0] == SERVO_OFF_PULSE
    assert pulses[1] == CENTER_PULSE


def test_move_time_overrides_the_per_actuator_default():
    """20 ms of board-side interpolation is 2 units of the firmware's
    10 ms move-duration byte."""
    bus, transport = make_bus(move_time_ms=20.0)
    bus.write_batch([ActuatorCommand(mode=OperatingMode.POSITION, position=0.0)] * 3)
    assert transport.sent[0][6] == 2


def test_an_arm_joint_and_a_wheel_ride_in_the_SAME_packet():
    """A wheeled humanoid's wheels hang off the same board as its arms, so they
    belong in one transaction -- and the bus must build it without ever asking
    which servo is which (no per-servo-type ``switch`` in shared code). The wheel's field is a SPEED and its move byte is 1;
    the joint's field is a pose and its move byte is the bus's tuned 20 ms.
    Both come out of the same loop.
    """
    from actuo.drivers.actuators.kondo_ics_wheel import (
        KondoIcsWheelActuator, KondoIcsWheelCalibration)

    joint = KondoIcsActuator(0, KondoIcsCalibration())
    wheel = KondoIcsWheelActuator(20, KondoIcsWheelCalibration(
        deadband_deg=6.0, saturate_deg=20.0, rev_s_per_deg=0.0528))
    bus = KondoH7Bus(FakeBoard(), [joint, wheel], SYMBOLS,
                     joint_names=["shoulder", "wheel_fl"], move_time_ms=20.0)

    speed = 2.0 * math.pi * 0.0528 * 10.0          # 10 command deg by the fit
    bus.write_batch([
        ActuatorCommand(mode=OperatingMode.POSITION, position=math.radians(10.0)),
        ActuatorCommand(mode=OperatingMode.VELOCITY, velocity=speed),
    ])

    body = bus.transport.sent[0]
    assert len(bus.transport.sent) == 1
    # Per-servo durations, so the interleaved opcode: [op][bitmap 5][vel,pulse]*
    assert body[0] == OP_MULTI_SERVO_MULTI_VELOCITIES
    joint_vel, joint_pulse, wheel_vel, wheel_pulse = struct.unpack(
        "<BHBH", body[6:])
    assert joint_vel == 2                  # 20 ms / 10 ms per unit
    assert wheel_vel == 1                  # a speed field has nothing to ramp
    assert joint_pulse == pytest.approx(
        CENTER_PULSE + 10.0 * PULSE_PER_DEG, abs=1)
    assert wheel_pulse == pytest.approx(
        CENTER_PULSE + 10.0 * PULSE_PER_DEG, abs=1)


def _wheel(servo_id=20):
    from actuo.drivers.actuators.kondo_ics_wheel import (
        KondoIcsWheelActuator, KondoIcsWheelCalibration)
    return KondoIcsWheelActuator(servo_id, KondoIcsWheelCalibration(
        deadband_deg=6.0, saturate_deg=20.0, rev_s_per_deg=0.0528))


def test_a_position_servo_configured_as_a_wheel_is_refused_at_startup():
    """The board knows which kind each servo is; this is checked, not trusted.
    A speed written into a position servo's field steps it to that pulse."""
    bus = KondoH7Bus(FakeBoard(), [_wheel(20)], SYMBOLS,
                     joint_names=["wheel_fl"])          # board says rotation=0
    with pytest.raises(RuntimeError, match="rotation"):
        bus.check_rotation_modes()


def test_a_wheel_configured_as_a_joint_is_refused_at_startup():
    """The dangerous direction: a pose command lands in a SPEED field, so the
    wheel spins up and nothing a position controller ever sends means stop."""
    bus = KondoH7Bus(FakeBoard(slot_values={"rotation": {0: 1}}),
                     [KondoIcsActuator(0, KondoIcsCalibration())], SYMBOLS,
                     joint_names=["shoulder"])
    with pytest.raises(RuntimeError, match="rotation"):
        bus.check_rotation_modes()


def test_matching_rotation_modes_pass_quietly():
    bus = KondoH7Bus(FakeBoard(slot_values={"rotation": {20: 1}}),
                     [KondoIcsActuator(0, KondoIcsCalibration()), _wheel(20)],
                     SYMBOLS, joint_names=["shoulder", "wheel_fl"])
    bus.check_rotation_modes()


def test_a_wheel_commanded_a_position_is_refused_by_the_wheel():
    """The bus does not check modes; the servo does. A trajectory controller
    aimed at a wheel must fail loudly, not drive it to a pose it has none of."""
    from actuo.core.contracts import CapabilityError
    from actuo.drivers.actuators.kondo_ics_wheel import (
        KondoIcsWheelActuator, KondoIcsWheelCalibration)

    wheel = KondoIcsWheelActuator(20, KondoIcsWheelCalibration(
        deadband_deg=6.0, saturate_deg=20.0, rev_s_per_deg=0.0528))
    bus = KondoH7Bus(FakeBoard(), [wheel], SYMBOLS, joint_names=["wheel_fl"])
    with pytest.raises(CapabilityError):
        bus.write_batch([ActuatorCommand(mode=OperatingMode.POSITION,
                                         position=0.0)])


def test_write_batch_refuses_a_wrong_length_command_vector():
    bus, _ = make_bus()
    with pytest.raises(ValueError):
        bus.write_batch([ActuatorCommand(mode=OperatingMode.POSITION,
                                         position=0.0)] * 2)


def test_set_mode_refuses_a_mode_the_servos_do_not_have():
    bus, _ = make_bus()
    bus.set_mode(OperatingMode.POSITION)
    with pytest.raises(Exception):
        bus.set_mode(OperatingMode.VELOCITY)


# --------------------------------------------------------------------------
# Batch read
# --------------------------------------------------------------------------


def test_read_batch_uses_one_round_trip_per_slot():
    bus, transport = make_bus(pulses={0: 7500, 1: 8000, 5: 7000})
    bus.read_batch()
    assert len(transport.sent) == 1
    assert transport.sent[0][0] == OP_MREADV


def test_read_batch_gives_each_joint_its_own_servos_value():
    """The reply covers all 36 servo slots; each actuator must get *its* index,
    not its position in the joint list."""
    bus, _ = make_bus(ids=(0, 1, 5),
                      pulses={0: 7500, 1: 8000, 5: 7000})
    states = bus.read_batch()
    assert states[0].position == pytest.approx(0.0)
    assert states[1].position == pytest.approx(
        (8000 - CENTER_PULSE) / PULSE_PER_DEG * math.pi / 180.0, abs=1e-4)
    assert states[2].position == pytest.approx(
        (7000 - CENTER_PULSE) / PULSE_PER_DEG * math.pi / 180.0, abs=1e-4)


def test_last_state_answers_without_touching_the_bus():
    """The torque layer asks every bus this, so every bus must answer it.

    KondoH7Bus did not, and nothing caught it until a real board was already
    talking: the servicer came up, published one servo state, and died in
    ServoSupervisor.inspect with an AttributeError. `last_state` is now part of
    the Bus contract in actuo.core rather than a habit three buses happened to
    share.
    """
    bus, transport = make_bus(ids=(0, 1, 5), pulses={0: 7500, 1: 8000, 5: 7000})
    assert bus.last_state(0) is None, "nothing has been read yet"

    bus.read_batch()
    sent_after_read = len(transport.sent)
    assert bus.last_state(1).position == pytest.approx(
        (8000 - CENTER_PULSE) / PULSE_PER_DEG * math.pi / 180.0, abs=1e-4)
    assert len(transport.sent) == sent_after_read, "asking must not poll"
    assert bus.last_state(4) is None, "a servo not on this bus is unknown"


def test_read_batch_reads_every_requested_slot():
    bus, transport = make_bus(pulses={0: 7500}, bytes_values={0: 99},
                              state_slots=("current_angle", "temperature"))
    states = bus.read_batch()
    assert len(transport.sent) == 2
    assert states[0].temperature == pytest.approx(50.0)


def test_a_silent_board_yields_no_state_and_counts_a_miss():
    """Never a fabricated zero: an unanswered read leaves the state absent."""
    bus, _ = make_bus(answer=False)
    states = bus.read_batch()
    assert states == [None, None, None]
    assert bus.missed == 1


def test_a_previously_read_state_is_kept_when_a_cycle_is_missed():
    bus, transport = make_bus(pulses={0: 8000, 1: 8000, 5: 8000})
    first = bus.read_batch()
    transport.answer = False
    second = bus.read_batch()
    assert second[0].position == first[0].position
    assert bus.missed == 1


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


def test_scan_servo_flags_and_ports_use_single_strided_reads():
    bus, transport = make_bus(
        slot_values={
            "flag": {0: 1, 5: 1},
            "port": {0: 1, 5: 6},
        }
    )
    flags = bus.scan_servo_flags()
    ports = bus.scan_servo_ports()
    assert len(transport.sent) == 2
    assert flags[0] is True
    assert flags[1] is False
    assert ports[0] == 1
    assert ports[5] == 6


def test_scan_ics_channels_per_port_decodes_presence_timeout_and_two_ports():
    device_port = [0] * 64
    device_port[0] = 128 + 1            # id 0, port 1
    device_port[1] = 128 + 64 + 1       # id 1, port 1, timeout bit
    device_port[31] = 128 + 32          # id 31, port 6
    bus, _ = make_bus(device_port=device_port)
    per_port = bus.scan_ics_channels_per_port()
    assert per_port[1] == [0, 1]
    assert per_port[6] == [31]
    assert per_port[2] == []


def test_device_port_index_is_the_servo_id_not_half_of_it():
    """The packing measured on a humanoid's ARM-H7 board (v0.6.5), pinned.

    This is the exact table the hardware returned for the two arms: the right
    arm's even ids on J4, the left arm's odd ids on J3. Under rcb4's
    ``index // 2`` reading, indices 0 and 1 are the same servo on two different
    UARTs -- a self-contradiction. Under ``index == id`` they are two servos,
    which is what ``ServoStruct.port`` independently said for all 13 of them.
    """
    device_port = [0] * 64
    for index in (0, 2, 4, 6):
        device_port[index] = 128 + 8        # J4
    for index in (1, 3, 5):
        device_port[index] = 128 + 4        # J3
    device_port[33] = 128 + 16              # J5, head -- an id >= 32
    device_port[35] = 128 + 16
    bus, _ = make_bus(device_port=device_port)
    per_port = bus.scan_ics_channels_per_port()
    assert per_port[4] == [0, 2, 4, 6]
    assert per_port[3] == [1, 3, 5]
    # Ids >= 32 are representable: the old `index // 2` reading could not reach
    # them at all, and this board has two.
    assert per_port[5] == [33, 35]


def test_scan_ics_channels_refuses_an_implausible_port_bitmask():
    device_port = [0] * 64
    device_port[0] = 128 + 3
    bus, _ = make_bus(device_port=device_port)
    with pytest.raises(ProtocolError) as excinfo:
        bus.scan_ics_channels_per_port()
    assert "bitmask" in str(excinfo.value)


def test_scan_ics_channels_keeps_an_all_blank_table_as_empty_scan():
    bus, _ = make_bus(device_port=[0] * 64)
    per_port = bus.scan_ics_channels_per_port()
    assert per_port[1] == []
    assert per_port[6] == []


def test_scan_ics_channels_refuses_ids_outside_the_servo_vector(monkeypatch):
    device_port = [0] * 64
    device_port[6] = 128 + 1
    bus, _ = make_bus(device_port=device_port)
    import actuo.drivers.buses.kondo_h7_bus as bus_module
    monkeypatch.setattr(bus_module, "SERVO_VECTOR_LENGTH", 2)
    with pytest.raises(ProtocolError) as excinfo:
        bus.scan_ics_channels_per_port()
    assert "outside 0..1" in str(excinfo.value)


def test_discover_reads_expected_slots_and_builds_expected_sets():
    device_port = [0] * 64
    device_port[0] = 128 + 64 + 1
    device_port[31] = 128 + 32
    bus, transport = make_bus(
        slot_values={
            "flag": {0: 1, 31: 1},
            "port": {0: 1, 31: 6},
        },
        device_port=device_port,
        version="v-test",
    )
    discovery = bus.discover()
    assert isinstance(discovery, KondoDiscovery)
    assert discovery.firmware_version == "v-test"
    assert discovery.present_ids == (0, 31)
    assert discovery.ports == {0: 1, 31: 6}
    assert discovery.per_port[1] == (0,)
    assert discovery.per_port[6] == (31,)
    assert discovery.timed_out_ids == (0,)
    assert discovery.ics_channel_scan_error is None
    assert [payload[0] for payload in transport.sent] == [
        OP_VERSION, OP_MREADV, OP_MREADV, OP_MREADV
    ]


def test_discover_uses_verified_flag_and_port_when_control_symbol_is_missing():
    bus, _ = make_bus(
        symbols=SYMBOLS_NO_CONTROL,
        slot_values={"flag": {2: 1}, "port": {2: 4}},
    )
    discovery = bus.discover()
    assert discovery.present_ids == (2,)
    assert discovery.ports == {2: 4}
    assert discovery.per_port is None
    assert discovery.timed_out_ids is None
    assert "Control" in discovery.ics_channel_scan_error


def test_discover_uses_verified_flag_and_port_when_device_port_is_implausible():
    device_port = [0] * 64
    device_port[0] = 128 + 3
    bus, _ = make_bus(
        slot_values={"flag": {0: 1}, "port": {0: 1}},
        device_port=device_port,
    )
    discovery = bus.discover()
    assert discovery.present_ids == (0,)
    assert discovery.ports == {0: 1}
    assert discovery.per_port is None
    assert discovery.timed_out_ids is None
    assert "bitmask" in discovery.ics_channel_scan_error


def test_discover_uses_verified_flag_and_port_when_device_port_read_times_out():
    bus, _ = make_bus(
        slot_values={"flag": {5: 1}, "port": {5: 6}},
        answer_device_port=False,
    )
    discovery = bus.discover()
    assert discovery.present_ids == (5,)
    assert discovery.ports == {5: 6}
    assert discovery.per_port is None
    assert discovery.timed_out_ids is None
    assert "did not answer Control.device_port" in discovery.ics_channel_scan_error


def test_scan_ics_channels_still_raises_when_device_port_read_times_out():
    bus, _ = make_bus(answer_device_port=False)
    with pytest.raises(ProtocolError) as excinfo:
        bus.scan_ics_channels_per_port()
    assert "did not answer Control.device_port" in str(excinfo.value)


def test_discover_distinguishes_empty_ics_scan_from_unavailable_scan():
    bus_empty, _ = make_bus(device_port=[0] * 64)
    empty = bus_empty.discover()
    assert empty.per_port is not None
    assert all(not ids for ids in empty.per_port.values())
    assert empty.timed_out_ids == ()
    assert empty.ics_channel_scan_error is None

    bus_unavailable, _ = make_bus(answer_device_port=False)
    unavailable = bus_unavailable.discover()
    assert unavailable.per_port is None
    assert unavailable.timed_out_ids is None
    assert "did not answer Control.device_port" in unavailable.ics_channel_scan_error


def test_device_port_timeout_bit_marks_only_its_own_servo():
    """One entry per servo, so a timeout belongs to exactly one id.

    Previously this table was read two-entries-per-servo and the timeout of
    either entry was attributed to the shared id -- a conservative rule that had
    to exist because the two entries could disagree. With ``index == id`` there
    is nothing to reconcile: id 32's timeout is id 32's, and id 33 beside it
    stays healthy. ``timed_out_ids`` is also the one fact ``ServoStruct.port``
    cannot supply, which is why this source is still read at all.
    """
    device_port = [0] * 64
    device_port[32] = 128 + 64 + 1      # id 32, port 1, timed out
    device_port[33] = 128 + 1           # id 33, port 1, healthy
    bus, _ = make_bus(
        slot_values={"flag": {32: 1, 33: 1}, "port": {32: 1, 33: 1}},
        device_port=device_port,
    )
    discovery = bus.discover()
    assert discovery.per_port[1] == (32, 33)
    assert discovery.timed_out_ids == (32,)


def test_compare_servo_ports_with_inferred_ics_ports_reports_agreement():
    bus, _ = make_bus()
    disagreements = bus.compare_servo_ports_with_inferred_ics_ports(
        {0: 1, 16: 1, 31: 6}, {0: 1, 16: 1, 31: 6})
    assert disagreements == {}


def test_compare_servo_ports_with_inferred_ics_ports_reports_disagreeing_ids():
    bus, _ = make_bus()
    disagreements = bus.compare_servo_ports_with_inferred_ics_ports(
        {0: 1, 16: 2, 31: 6}, {0: 1, 16: 1, 31: 6})
    assert disagreements == {16: (2, 1)}


def test_compare_discovery_reports_missing_unexpected_and_timed_out():
    discovery = KondoDiscovery(
        present_ids=(0, 2, 5),
        ports={0: 1, 2: 2, 5: 6},
        per_port={1: (0,), 2: (2,), 6: (5,)},
        timed_out_ids=(5,),
        firmware_version="v-test",
    )
    missing, unexpected, timed_out = compare_discovery(discovery, [0, 1, 2])
    assert missing == {1}
    assert unexpected == {5}
    assert timed_out == {5}


def test_compare_discovery_preserves_unavailable_timed_out_information():
    discovery = KondoDiscovery(
        present_ids=(0,),
        ports={0: 1},
        per_port=None,
        timed_out_ids=None,
        firmware_version="v-test",
        ics_channel_scan_error="device_port unavailable",
    )
    missing, unexpected, timed_out = compare_discovery(discovery, [0])
    assert missing == set()
    assert unexpected == set()
    assert timed_out is None


# --------------------------------------------------------------------------
# Startup verification and diagnostics
# --------------------------------------------------------------------------


def _diag(items, name):
    for item in items:
        if item.name == name:
            return item
    raise AssertionError("diagnostic item {!r} not found".format(name))


def test_verify_servos_raises_for_startup_missing_and_names_joint_and_port_findings():
    bus, _ = make_bus(
        ids=(6, 7),
        joint_names=("rarm_joint0", "rarm_joint1"),
        slot_values={
            "flag": {7: 1},
            "port": {7: 1},
        },
    )
    bus.actuators[0].ics_channel = "J1"
    bus.actuators[1].ics_channel = "J1"
    with pytest.raises(RuntimeError) as excinfo:
        bus.verify_servos()
    message = str(excinfo.value)
    assert "rarm_joint0" in message
    assert "id 6" in message
    assert "config J1" in message
    assert "[7]" in message


def test_verify_servos_allow_missing_returns_warn_instead_of_raising():
    bus, _ = make_bus(
        ids=(6, 7),
        joint_names=("rarm_joint0", "rarm_joint1"),
        slot_values={
            "flag": {7: 1},
            "port": {7: 1},
        },
    )
    bus.actuators[0].ics_channel = "J1"
    items = bus.verify_servos(allow_missing=True)
    missing = _diag(items, "bus/startup_missing")
    assert missing.level == DiagLevel.WARN
    assert "rarm_joint0" in missing.message


def test_diagnose_reports_runtime_disappearance_as_error_named_by_joint():
    bus, transport = make_bus(
        ids=(32,),
        joint_names=("head_neck_y",),
        slot_values={"flag": {32: 1}, "port": {32: 6}},
    )
    bus.actuators[0].ics_channel = "J6"
    bus.verify_servos()
    transport.slot_values["flag"] = {}  # absent now
    transport.slot_values["port"] = {}
    items = bus.diagnose()
    discovery = _diag(items, "bus/discovery")
    assert discovery.level == DiagLevel.ERROR
    assert "head_neck_y" in discovery.message
    servo = _diag(items, "servo/head_neck_y")
    assert servo.level == DiagLevel.ERROR
    assert "disappeared" in servo.message


def test_diagnose_on_firmware_mismatch_returns_only_firmware_item():
    """When firmware and symbol map disagree, mapped-address reads are invalid."""
    bus, transport = make_bus(ids=(0,), joint_names=("rarm_joint0",))
    original_reply = transport._reply

    def reply(payload):
        if payload[0] == OP_VERSION:
            return Frame(payload=b"v-other")
        return original_reply(payload)

    transport._reply = reply
    items = bus.diagnose()
    assert [item.name for item in items] == ["bus/firmware"]
    assert items[0].level == DiagLevel.ERROR
    assert len(transport.sent) == 1
    assert transport.sent[0][0] == OP_VERSION


def test_diagnose_on_unreadable_firmware_returns_only_firmware_item():
    bus, transport = make_bus(ids=(0,), joint_names=("rarm_joint0",))
    original_reply = transport._reply

    def reply(payload):
        if payload[0] == OP_VERSION:
            return None
        return original_reply(payload)

    transport._reply = reply
    items = bus.diagnose()
    assert [item.name for item in items] == ["bus/firmware"]
    assert items[0].level == DiagLevel.ERROR
    assert len(transport.sent) == 1
    assert transport.sent[0][0] == OP_VERSION


def test_diagnose_marks_servo_non_ok_when_current_angle_read_is_unavailable():
    bus, transport = make_bus(
        ids=(0,),
        joint_names=("rarm_joint0",),
        slot_values={
            "flag": {0: 1},
            "port": {0: 1},
            "current_angle": {0: 7500},
            "timeout_count": {0: 0},
            "rxerror_count": {0: 0},
        },
    )
    bus.verify_servos()
    original_reply = transport._reply
    current_angle_address = SERVO_VECTOR + SERVO_SLOTS["current_angle"].offset

    def reply(payload):
        if (payload[0] == OP_MREADV
                and int.from_bytes(payload[1:5], "little") == current_angle_address):
            return None
        return original_reply(payload)

    transport._reply = reply
    servo = _diag(bus.diagnose(), "servo/rarm_joint0")
    assert servo.level != DiagLevel.OK
    assert "Measured pulse unavailable this cycle" in servo.message


@pytest.mark.parametrize("slot_name", ["timeout_count", "rxerror_count"])
def test_diagnose_marks_servo_non_ok_when_counter_read_is_unavailable(slot_name):
    bus, transport = make_bus(
        ids=(0,),
        joint_names=("rarm_joint0",),
        slot_values={
            "flag": {0: 1},
            "port": {0: 1},
            "current_angle": {0: 7500},
            "timeout_count": {0: 0},
            "rxerror_count": {0: 0},
        },
    )
    bus.verify_servos()
    original_reply = transport._reply
    slot_address = SERVO_VECTOR + SERVO_SLOTS[slot_name].offset

    def reply(payload):
        if (payload[0] == OP_MREADV
                and int.from_bytes(payload[1:5], "little") == slot_address):
            return None
        return original_reply(payload)

    transport._reply = reply
    servo = _diag(bus.diagnose(), "servo/rarm_joint0")
    assert servo.level != DiagLevel.OK
    assert "{} unavailable this cycle".format(slot_name) in servo.message


def test_diagnose_warns_when_rxerror_count_increases_but_not_when_flat_nonzero():
    bus, transport = make_bus(
        ids=(0,),
        joint_names=("rarm_joint0",),
        slot_values={
            "flag": {0: 1},
            "port": {0: 1},
            "current_angle": {0: 7500},
            "timeout_count": {0: 0},
            "rxerror_count": {0: 2},
        },
    )
    bus.verify_servos()
    first = bus.diagnose()
    assert _diag(first, "servo/rarm_joint0").level == DiagLevel.OK
    second = bus.diagnose()
    assert _diag(second, "servo/rarm_joint0").level == DiagLevel.OK
    transport.slot_values["rxerror_count"][0] = 3
    third = bus.diagnose()
    assert _diag(third, "servo/rarm_joint0").level == DiagLevel.WARN


def test_diagnose_omits_current_and_temperature_values_when_flags_are_off():
    bus, _ = make_bus(
        ids=(0,),
        joint_names=("rarm_joint0",),
        slot_values={
            "flag": {0: 1},
            "port": {0: 1},
            "current_angle": {0: 7500},
            "timeout_count": {0: 0},
            "rxerror_count": {0: 0},
            "current": {0: 30},
            "temperature": {0: 80},
        },
        system_values={
            "servo_current_read_flag": [0],
            "servo_temperature_read_flag": [0],
        },
    )
    bus.verify_servos()
    servo = _diag(bus.diagnose(), "servo/rarm_joint0")
    assert "current_a" not in servo.values
    assert "temperature_c" not in servo.values
    assert "stale by construction" in servo.message


def test_diagnose_power_without_threshold_reports_value_without_claimed_limit():
    bus, _ = make_bus(
        ids=(0,),
        joint_names=("rarm_joint0",),
        slot_values={"flag": {0: 1}, "port": {0: 1}},
        system_values={"battery_voltage": [12.34]},
    )
    bus.verify_servos()
    power = _diag(bus.diagnose(), "bus/power")
    assert power.level == DiagLevel.OK
    assert power.values["battery_voltage_v"] == "12.340"
    assert "no low-voltage threshold" in power.message


# --------------------------------------------------------------------------
# Servo on / off and board switches
# --------------------------------------------------------------------------


def test_release_all_and_hold_all_are_single_packets():
    bus, transport = make_bus()
    bus.release_all()
    assert struct.unpack("<3H", transport.sent[-1][7:])[0] == SERVO_OFF_PULSE
    bus.hold_all()
    assert struct.unpack("<3H", transport.sent[-1][7:])[0] == SERVO_ON_PULSE
    assert all(bus.servo_on_states().values())


def test_current_reading_flag_targets_the_system_struct():
    bus, transport = make_bus()
    bus.set_current_reading(True)
    body = transport.sent[-1]
    assert body[1:5] == (SYSTEM_BASE + 49).to_bytes(4, "little")
    assert body[-1] == 1


def test_ics_comm_stop_writes_all_six_uarts():
    bus, transport = make_bus()
    bus.set_ics_comm(False)
    body = transport.sent[-1]
    assert body[1:5] == (SYSTEM_BASE + 64).to_bytes(4, "little")
    assert body[-6:] == bytes([1] * 6)


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------


def test_a_servo_id_outside_the_boards_vector_is_refused():
    actuators = [KondoIcsActuator(SERVO_VECTOR_LENGTH, KondoIcsCalibration())]
    with pytest.raises(ValueError):
        KondoH7Bus(FakeBoard(), actuators, SYMBOLS)


def test_a_bus_that_reads_nothing_is_refused():
    with pytest.raises(ValueError):
        make_bus(state_slots=())


def test_verify_firmware_refuses_a_mismatched_board():
    """The board reports a version string; the symbol map claims one. If they
    disagree, every address this bus uses points at the wrong memory."""
    bus, transport = make_bus()

    def reply(payload):
        return Frame(payload=b"v-other")

    transport._reply = reply
    with pytest.raises(Exception) as excinfo:
        bus.verify_firmware()
    assert "firmware mismatch" in str(excinfo.value)


def test_verify_servos_blames_the_firmware_not_the_servos():
    """A wrong firmware makes every struct address wrong, so the servo scan
    reads the wrong memory and finds nothing. Reporting that as "these servos
    were not recognised" sends the operator to check id dials and wiring that
    are fine -- the exact plausible-but-wrong output this driver must not
    produce. The version check therefore runs before any struct read is
    interpreted."""
    bus, transport = make_bus()

    def reply(payload):
        return Frame(payload=b"v-other")

    transport._reply = reply
    with pytest.raises(Exception) as excinfo:
        bus.verify_servos()
    message = str(excinfo.value)
    assert "firmware mismatch" in message
    assert "not recognised" not in message


def test_scan_ics_channels_names_missing_control_symbol():
    bus, _ = make_bus(symbols=SYMBOLS_NO_CONTROL)
    with pytest.raises(ProtocolError) as excinfo:
        bus.scan_ics_channels_per_port()
    assert "Control" in str(excinfo.value)
    assert "actuo-kondo-symbols" in str(excinfo.value)


# --------------------------------------------------------------------------
# ServoStruct.feedback -- the byte that silently stops a servo being driven.
# --------------------------------------------------------------------------


def test_clearing_the_feedback_flag_touches_only_the_servos_that_have_it():
    """Measured on a cold board: the four wheels came up with feedback=1 and
    every arm servo with 0. Commands land either way -- ref_angle updates and
    the frame is byte-identical to rcb4's -- and a servo with the byte set does
    not move. Clearing it turned the wheel on the very next identical command.
    """
    bus, transport = make_bus(ids=(0, 1, 5),
                              slot_values={"feedback": {1: 1, 5: 1}})

    cleared = bus.clear_feedback_flags()

    assert cleared == [1, 5], "servo 0 had nothing to clear"
    writes = [payload for payload in transport.sent if payload[0] == OP_MWRITEV]
    assert len(writes) == 2, "one write per servo that had the byte set"


def test_nothing_is_written_when_no_servo_has_the_flag():
    """This modifies the board, so it must not happen out of habit."""
    bus, transport = make_bus(ids=(0, 1, 5))
    before = len(transport.sent)

    assert bus.clear_feedback_flags() == []

    writes = [p for p in transport.sent[before:] if p[0] == OP_MWRITEV]
    assert writes == []


def test_a_servo_slot_write_addresses_the_right_entry():
    """servo_vector[id].feedback, not servo_vector[0]. Off by one struct is a
    different servo's byte, and this board has 36 of them."""
    from actuo.drivers.kondo_h7_protocol import SERVO_STRUCT_SIZE, servo_slot_write
    import struct as _struct

    payload = servo_slot_write(0x24000000, "feedback", 5, 0)
    address = _struct.unpack("<I", payload[1:5])[0]
    assert address == 0x24000000 + 5 * SERVO_STRUCT_SIZE + 116


def test_a_servo_slot_write_refuses_an_id_off_the_vector():
    from actuo.drivers.kondo_h7_protocol import ProtocolError, servo_slot_write
    with pytest.raises(ProtocolError):
        servo_slot_write(0x24000000, "feedback", 99, 0)
