"""DynamixelBus orchestration, driven through a fake transport.

What is worth asserting about a bus is transaction *shape*, not that bytes come
back: whether N joints cost one transaction or N, whether a released servo can
still be carried in a position write, and whether a silent servo corrupts the
joints around it. Those are the properties that decide whether the robot is
controllable at rate and safe when something drops.
"""

import math
from dataclasses import replace

import pytest

from actuo.core.contracts import (
    ActuatorCommand, CapabilityError, Frame, OperatingMode, Transport)
from actuo.core.diagnostics import DiagLevel

from actuo.drivers import dynamixel_protocol as dxl
from actuo.drivers.actuators.dynamixel import (
    DynamixelActuator, DynamixelCalibration)
from actuo.drivers.buses.dynamixel_bus import DynamixelBus
from actuo.drivers.dynamixel_models import load_model_table


class FakeBus(Transport):
    """Records what was written and replays queued replies.

    Deliberately dumb: it does not model a servo. A fake that "answers
    correctly" mostly tests the fake, so this one only lets the tests assert
    what went onto the wire and hand back chosen replies.
    """

    def __init__(self, protocol=dxl.PROTOCOL_2):
        self.protocol = protocol
        self.written = []
        self.inbox = []
        self.resets = 0
        self._open = False

    def open(self):
        self._open = True

    def close(self):
        self._open = False

    @property
    def is_open(self):
        return self._open

    def reset_input(self):
        self.resets += 1

    def write(self, frames):
        self.written.extend(f.payload for f in frames)

    def recv(self, timeout):
        return self.inbox.pop(0) if self.inbox else None

    def transact(self, frames, timeout):
        self.written.extend(f.payload for f in frames)
        replies = []
        for _ in frames:
            replies.append(self.inbox.pop(0) if self.inbox else None)
        return replies


def make(model_name, ids, **kwargs):
    table = load_model_table()
    model = table.by_name(model_name)
    actuators = [
        DynamixelActuator(
            i,
            DynamixelCalibration(counts_per_rev=model.counts_per_rev,
                                 zero_offset_counts=model.center_counts),
            model)
        for i in ids
    ]
    transport = FakeBus(model.protocol)
    return DynamixelBus(transport, actuators, **kwargs), transport, actuators


def status_2(dxl_id, block):
    return Frame(payload=dxl.build_packet(dxl.PROTOCOL_2, dxl_id, dxl.STATUS,
                                          b"\x00" + block), address=dxl_id)


def status_2_with_error(dxl_id, error, block):
    """Like :func:`status_2`, but with a chosen packet-level error byte --
    needed to simulate the Alert bit (0x80) that triggers a hardware-fault
    follow-up read, and to build that follow-up's own reply (a 1-byte block:
    the Hardware Error Status register value)."""
    return Frame(payload=dxl.build_packet(
        dxl.PROTOCOL_2, dxl_id, dxl.STATUS, bytes([error]) + block),
        address=dxl_id)


def status_1(dxl_id, block):
    return Frame(payload=dxl.build_packet(dxl.PROTOCOL_1, dxl_id, 0x00, block),
                 address=dxl_id)


X_BLOCK = dxl.to_le(0, 2) + dxl.to_le(0, 4) + dxl.to_le(2048, 4)
AX_BLOCK = (dxl.to_le(512, 2) + dxl.to_le(0, 2) + dxl.to_le(0, 2)
            + bytes([120, 40]))


# --- batch is the default, not an optimisation -------------------------------
def test_three_joints_cost_one_write_transaction():
    bus, transport, _ = make("XL430-W250-T", [1, 2, 3])
    command = ActuatorCommand(mode=OperatingMode.POSITION, position=0.0)
    bus.write_batch([command] * 3)

    # Two transactions regardless of joint count: the goals, then the torque
    # enable that makes a DYNAMIXEL act on them. What must NOT scale with the
    # number of joints is the number of transactions.
    assert len(transport.written) == 2
    packet = transport.written[0]
    assert packet[4] == dxl.BROADCAST_ID
    body = dxl.remove_stuffing(packet[7:-2])
    assert body[0] == dxl.SYNC_WRITE
    assert dxl.from_le(body[1:3]) == 116        # goal_position
    assert dxl.from_le(body[3:5]) == 4          # four bytes each
    assert [body[5], body[10], body[15]] == [1, 2, 3]



def test_a_released_joint_is_not_carried_in_the_position_write():
    """Commanding a pose to a servo that was asked to go limp would re-grip it."""
    bus, transport, _ = make("XL430-W250-T", [1, 2])
    command = ActuatorCommand(mode=OperatingMode.POSITION, position=0.0)
    bus.write_batch([command, None])

    release, sync, torque = transport.written
    assert dxl.remove_stuffing(release[7:-2])[1:] == bytes([64, 0, 0])
    body = dxl.remove_stuffing(sync[7:-2])
    assert body[5] == 1 and len(body) == 5 + 5   # only joint 1 rides along
    # ...and neither does it ride in the torque enable that follows.
    body = dxl.remove_stuffing(torque[7:-2])
    assert dxl.from_le(body[1:3]) == 64
    assert body[5] == 1 and len(body) == 5 + 2


def test_mixed_layouts_become_one_transaction_per_register_group():
    """Two control-table layouts cannot share a SYNC_WRITE -- a sync write names
    ONE address for every id in it. Each group must still be one transaction,
    so a two-layout bus costs two writes, never one per joint."""
    table = load_model_table()
    base = table.by_name("XL430-W250-T")
    # A hypothetical sibling whose goal_position sits elsewhere. Synthesised
    # rather than taken from the shipped table, because inventing a real
    # model's addresses is exactly what the table exists to prevent.
    moved_registers = dict(base.table.registers)
    moved_registers["goal_position"] = replace(
        moved_registers["goal_position"], address=564)
    other = replace(base, name="SYNTHETIC",
                    table=replace(base.table, registers=moved_registers))

    actuators = [
        DynamixelActuator(1, DynamixelCalibration(
            counts_per_rev=base.counts_per_rev,
            zero_offset_counts=base.center_counts), base),
        DynamixelActuator(2, DynamixelCalibration(
            counts_per_rev=other.counts_per_rev,
            zero_offset_counts=other.center_counts), other),
    ]
    bus = DynamixelBus(FakeBus(), actuators)
    bus.write_batch([ActuatorCommand(mode=OperatingMode.POSITION,
                                     position=0.0)] * 2)

    # Two goal groups (one per layout) plus the shared torque enable, and NOT
    # one transaction per joint.
    assert len(bus.transport.written) == 3
    addresses = sorted(
        dxl.from_le(dxl.remove_stuffing(p[7:-2])[1:3])
        for p in bus.transport.written)
    assert addresses == [64, 116, 564]


def test_torque_enable_is_also_batched():
    bus, transport, _ = make("XL430-W250-T", [1, 2, 3])
    bus.enable_all()
    assert len(transport.written) == 1
    body = dxl.remove_stuffing(transport.written[0][7:-2])
    assert dxl.from_le(body[1:3]) == 64          # torque_enable
    assert [body[5], body[7], body[9]] == [1, 2, 3]


def _sync_writes(transport):
    """{address: body} for the SYNC_WRITE packets only.

    Releases are per-servo WRITE frames (instruction 0x03) whose id lives in
    the header, not the body, so parsing them with the sync-write layout gives
    a plausible-looking wrong address -- which is how the first version of
    these tests "found" a torque write that was not there.
    """
    out = {}
    for payload in transport.written:
        body = dxl.remove_stuffing(payload[7:-2])
        if body[0] != dxl.SYNC_WRITE:
            continue
        out[dxl.from_le(body[1:3])] = body
    return out


def test_a_commanded_servo_is_torque_enabled_not_just_given_a_goal():
    """The regression that made the arm's first playback do nothing.

    A DYNAMIXEL with torque_enable = 0 ACCEPTS a goal position and ignores it.
    The write succeeds, the servo answers, Present Position keeps reporting the
    truth -- and the arm stays limp while the servicer reports torque=ON and
    the trajectory controller reports "Motion completed successfully". So a
    goal is not enough: the enable is part of the command.
    """
    bus, transport, _ = make("XL430-W250-T", [1, 2])
    bus.write_batch([ActuatorCommand(mode=OperatingMode.POSITION,
                                     position=0.0)] * 2)
    writes = _sync_writes(transport)
    assert 116 in writes, "goal position was not written"
    assert 64 in writes, "torque was never enabled -- the servo ignores the goal"
    assert writes[64][5:] == bytes([1, 1, 2, 1])     # both ids, value 1
    # Goal BEFORE enable: a servo enabled first drives to the goal it still
    # held, which is the start-up slam this stack exists to prevent.
    order = [dxl.remove_stuffing(p[7:-2])[1] for p in transport.written]
    assert order.index(116) < order.index(64)


def test_torque_enable_is_reasserted_every_batch():
    """Self-healing: a servo that drops torque on its own must come back.

    An overload shutdown or a power blip clears the register behind our back.
    Enabling only on the transition would leave that joint silently limp for
    the rest of the run, with nothing upstream any the wiser.
    """
    bus, transport, _ = make("XL430-W250-T", [1])
    command = [ActuatorCommand(mode=OperatingMode.POSITION, position=0.0)]
    bus.write_batch(command)
    transport.written.clear()
    bus.write_batch(command)
    assert 64 in _sync_writes(transport)


def test_a_released_servo_is_not_torque_enabled():
    bus, transport, _ = make("XL430-W250-T", [1, 2])
    bus.write_batch([ActuatorCommand(mode=OperatingMode.POSITION,
                                     position=0.0), None])
    writes = _sync_writes(transport)
    assert writes[64][5:] == bytes([1, 1]), \
        "the released servo rode in the torque enable"


# --- reads -------------------------------------------------------------------
def test_protocol2_reads_every_joint_in_one_round_trip():
    bus, transport, _ = make("XL430-W250-T", [1, 2])
    transport.inbox = [status_2(1, X_BLOCK), status_2(2, X_BLOCK)]

    states = bus.read_batch(timeout=0.01)
    assert bus.reads_per_cycle == 1
    assert len(transport.written) == 1
    body = dxl.remove_stuffing(transport.written[0][7:-2])
    assert body[0] == dxl.SYNC_READ
    assert all(s is not None and s.position == pytest.approx(0.0) for s in states)


def test_sync_read_replies_are_matched_by_id_not_by_arrival_order():
    """One silent servo must cost one missing state, not shift every state after
    it onto the wrong joint -- which would move the wrong joints."""
    bus, transport, _ = make("XL430-W250-T", [1, 2, 3])
    moved = dxl.to_le(0, 2) + dxl.to_le(0, 4) + dxl.to_le(3072, 4)
    transport.inbox = [status_2(3, moved), status_2(1, X_BLOCK)]  # 2 is silent

    states = bus.read_batch(timeout=0.01)
    assert states[0].position == pytest.approx(0.0)     # id 1
    assert states[1] is None                            # id 2 never answered
    assert states[2].position == pytest.approx(math.pi / 2, abs=1e-3)  # id 3
    assert bus.missed == 1


def test_a_silent_servo_keeps_its_last_state_and_is_counted():
    bus, transport, _ = make("XL430-W250-T", [1])
    transport.inbox = [status_2(1, X_BLOCK)]
    first = bus.read_batch(timeout=0.01)[0]

    transport.inbox = []                                # nothing answers now
    second = bus.read_batch(timeout=0.001)[0]
    assert second is first, "stale state is held, never replaced with a fake 0"
    assert bus.missed == 1


# --- hardware fault diagnosis -------------------------------------------------
def test_healthy_reads_never_pay_for_the_fault_follow_up():
    """The common case costs nothing extra: one transaction, not two."""
    bus, transport, _ = make("XL430-W250-T", [1])
    transport.inbox = [status_2(1, X_BLOCK)]
    state = bus.read_batch(timeout=0.01)[0]
    assert state.error is None
    assert len(transport.written) == 1


def test_an_alert_bit_is_followed_up_and_named():
    """The regression this exists for: 'error code 128' told an operator
    nothing. Once the Hardware Error Status register is read, the SPECIFIC
    fault (Overload, here) replaces the vague Alert bit."""
    bus, transport, _ = make("XL430-W250-T", [5])
    transport.inbox = [
        status_2_with_error(5, dxl.ERROR_ALERT_BIT, X_BLOCK),   # cyclic read
        status_2_with_error(5, 0, bytes([0x20])),                # addr 70 = Overload
    ]
    state = bus.read_batch(timeout=0.01)[0]

    assert state.error == 0x20, "the specific fault, not the vague Alert bit"
    assert state.aux["packet_error"] == float(dxl.ERROR_ALERT_BIT), \
        "the original wire-level byte must survive the overwrite"
    # Exactly one follow-up transaction, addressed to register 70.
    assert len(transport.written) == 2
    follow_up = dxl.remove_stuffing(transport.written[1][7:-2])
    assert follow_up[0] == dxl.READ
    assert dxl.from_le(follow_up[1:3]) == 70


def test_a_hardware_fault_name_covers_more_than_one_bit_at_once():
    """Overheating AND Overload together must not collapse to one name."""
    bus, transport, _ = make("XL430-W250-T", [5])
    transport.inbox = [
        status_2_with_error(5, dxl.ERROR_ALERT_BIT, X_BLOCK),
        status_2_with_error(5, 0, bytes([0x24])),   # 0x04 overheating | 0x20 overload
    ]
    state = bus.read_batch(timeout=0.01)[0]
    assert state.error == 0x24

    from actuo.drivers.actuators.dynamixel import ERROR_NAMES
    assert ERROR_NAMES[0x24] == "overheating, overload"


def test_a_failed_follow_up_leaves_the_alert_bit_visible_not_hidden():
    """No reply to the diagnostic read must not erase the fault it was asked
    about -- the Alert bit stays, which is still enough to refuse an enable."""
    bus, transport, _ = make("XL430-W250-T", [5])
    transport.inbox = [
        status_2_with_error(5, dxl.ERROR_ALERT_BIT, X_BLOCK),
        None,                                       # the follow-up times out
    ]
    state = bus.read_batch(timeout=0.01)[0]
    assert state.error == dxl.ERROR_ALERT_BIT
    assert "packet_error" not in state.aux, "nothing to preserve -- it never changed"


def test_no_alert_bit_means_no_follow_up_even_with_a_nonzero_error():
    """Only the Alert bit triggers the follow-up. A different packet-level
    error (e.g. a malformed instruction) is not "check Hardware Error
    Status" and must not be misread as one."""
    bus, transport, _ = make("XL430-W250-T", [5])
    transport.inbox = [status_2_with_error(5, 0x02, X_BLOCK)]   # instruction error
    state = bus.read_batch(timeout=0.01)[0]
    assert state.error == 0x02
    assert len(transport.written) == 1, "no follow-up was owed here"


def test_protocol1_read_cost_is_reported_not_hidden():
    """An AX bus has no SYNC_READ: N joints cost N round trips, and the bus says
    so rather than letting a slow loop be a mystery."""
    bus, transport, _ = make("AX-12A", [1, 2, 3])
    assert bus.reads_per_cycle == 3
    transport.inbox = [status_1(i, AX_BLOCK) for i in (1, 2, 3)]
    states = bus.read_batch(timeout=0.01)
    assert len(transport.written) == 3
    assert all(s is not None for s in states)


def test_sync_read_flushes_stale_bytes_first():
    """Otherwise last cycle's leftover packet is reported as this cycle's state."""
    bus, transport, _ = make("XL430-W250-T", [1])
    transport.inbox = [status_2(1, X_BLOCK)]
    bus.read_batch(timeout=0.01)
    assert transport.resets == 1


def _diag(items, name):
    for item in items:
        if item.name == name:
            return item
    raise AssertionError("diagnostic item {!r} not found".format(name))


def test_diagnose_reports_per_servo_fault_bits_and_comm_failures():
    bus, transport, _ = make("XL430-W250-T", [5, 6], use_sync_read=False)
    transport.inbox = [
        status_2_with_error(5, dxl.ERROR_ALERT_BIT, X_BLOCK),  # id 5 query
        None,                                                   # id 6 query
        status_2_with_error(5, 0, bytes([0x20])),              # follow-up read
    ]
    states = bus.read_batch(timeout=0.01)
    assert states[0] is not None
    assert states[1] is None

    items = bus.diagnose()
    bus_reads = _diag(items, "bus/reads")
    faulted = _diag(items, "servo/id_5")
    silent = _diag(items, "servo/id_6")

    assert bus_reads.level == DiagLevel.WARN
    assert bus_reads.values["missed_total"] == "1"

    assert faulted.level == DiagLevel.ERROR
    assert faulted.values["hardware_error_code"] == "32"
    assert "overload" in faulted.values["hardware_error_name"]

    assert silent.level == DiagLevel.STALE
    assert silent.values["comm_failures_total"] == "1"
    assert silent.values["consecutive_misses"] == "1"


def test_diagnose_escalates_sustained_silence_to_error_after_first_reply():
    bus, transport, _ = make(
        "XL430-W250-T",
        [5],
        silence_error_misses=3,
        silence_error_age_s=10.0,
    )
    transport.inbox = [status_2(5, X_BLOCK)]
    assert bus.read_batch(timeout=0.01)[0] is not None

    transport.inbox = []
    bus.read_batch(timeout=0.001)
    bus.read_batch(timeout=0.001)
    bus.read_batch(timeout=0.001)

    servo = _diag(bus.diagnose(), "servo/id_5")
    assert servo.level == DiagLevel.ERROR
    assert "sustained silence" in servo.message
    assert servo.values["consecutive_misses"] == "3"


# --- refusals ----------------------------------------------------------------
def test_a_bus_cannot_mix_protocol_versions():
    table = load_model_table()
    ax, xl = table.by_name("AX-12A"), table.by_name("XL430-W250-T")
    actuators = [
        DynamixelActuator(1, DynamixelCalibration(
            counts_per_rev=ax.counts_per_rev), ax),
        DynamixelActuator(2, DynamixelCalibration(
            counts_per_rev=xl.counts_per_rev), xl),
    ]
    with pytest.raises(ValueError, match="two protocol versions"):
        DynamixelBus(FakeBus(), actuators)


def test_set_mode_checks_every_servo_before_writing_any():
    """A half-switched bus is worse than a refused one."""
    bus, transport, _ = make("AX-12A", [1, 2])
    with pytest.raises(CapabilityError):
        bus.set_mode(OperatingMode.VELOCITY)
    assert transport.written == []


def test_write_batch_rejects_a_wrong_length_command_vector():
    bus, _, _ = make("XL430-W250-T", [1, 2])
    with pytest.raises(ValueError):
        bus.write_batch([ActuatorCommand(mode=OperatingMode.POSITION,
                                         position=0.0)])
