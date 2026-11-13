"""Kondo RCB-4mini bus: one servo packet per cycle, few record reads per cycle.

Drives the ICS servos hanging off an RCB-4mini with the same actuator model as
the ARM-H7 (:class:`~actuo.drivers.actuators.kondo_ics.KondoIcsActuator`): the
servos and the batch move packet are the same, the board's memory is not.

* ``write_batch`` sends **one** ``0x10`` packet for every servo, as on the H7.
  Never ``0x11`` (a move byte per servo): it stops or resets this board, so
  servos with different move bytes are refused.
* ``read_batch`` has no strided read to lean on. It reads whole servo records
  instead, grouping ids that sit close together into one read of up to 126
  bytes (:func:`~rcb4.protocol.rcb4mini.plan_record_reads`), then
  slices each servo's members into a one-servo frame for the actuator.

Cold start (:meth:`KondoRcb4Bus.verify_servos`)
-----------------------------------------------
Whatever a HeartToHeart project in the board's ROM set, the bus runs the board
flat: as a board with no project, plus what this config asks for. The settings
live in RAM and come back from ROM at every power-up, so they are put right on
every connect and reconnect, not once:

* **COM bit rate.** A board without a HeartToHeart project powers up at
  115200. The bus connects at that, switches the board to the transport's
  rate, and reopens the port.
* **Config word.** The board powers up running the program in its ROM, which
  can command the same servos as the bus. The bus stops it and its vector
  jumps, as kxreus does around every servo command, switches the
  interpolation-end message off, sets the output cycle to the no-project
  15 ms and the ICS rate to ``ics_bitrate``.
* **Trim and mixing**, on all 35 servo records: both change what a servo is
  driven to, so both are zeroed and the servo switched off.
* **ICS assignment.** The board talks only to ICS numbers its assignment table
  maps to a servo record, and a board without a project maps none. The bus
  maps any number the project pointed at another record back to its own,
  assigns the configured ids it finds unassigned, and keeps the project's
  assignment where it already is the number's own record.

All are writes to the board, so all are reported as diagnostics rather than
done silently. None touches ROM: power-cycling restores the board.

Servo settings from the config
------------------------------
* ``stretch``: a servo's holding gain, sent through the board
  (:func:`~rcb4.protocol.rcb4mini.build_stretch`) at connect and,
  with ``hot_plug``, again whenever the servo answers after being absent,
  since a servo that lost power is back at its own.
* ``hold_relax``: a servo held away from its command (a gripper closed on an
  object) is held lighter after a while; see
  :class:`~actuo.drivers.actuators.kondo_ics.HoldRelax`. While it is, the bus
  sends that servo another target than its command, and ``diagnose`` names
  it (``bus/hold_relax``).

Port detection (``port_detect``)
--------------------------------
The board numbers a servo by where it is plugged in as well as by its ID:
ICS number = 2 x ID + port, port 0 for SIO1-4 and 1 for SIO5-8. With
``port_detect`` each joint's ``id`` is the servo's ID, both of its ICS numbers
are assigned, and the one whose record the board keeps rewriting is its port.
With ``hot_plug`` this is looked at again at every presence check, so a servo
moved to the other side of the board is followed; it reads absent for one
presence period first, so the servicer releases it and it comes back limp
rather than snapping to its old target.
"""

from __future__ import annotations

import math
import struct
import time
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

from actuo.core.contracts import (
    ActuatorCommand, ActuatorState, Bus, Frame, OperatingMode)
from actuo.core.diagnostics import DiagItem, DiagLevel

from actuo.drivers.actuators.kondo_ics import (
    CENTER_PULSE, PULSE_PER_DEG, SERVO_OFF_PULSE, SERVO_ON_PULSE)
from rcb4.protocol.rcb4mini import (
    BATTERY_ADC_ADDRESS, BATTERY_VOLTS_PER_COUNT, CONFIG_ADDRESS, CONFIG_RUN_ROM,
    CONFIG_SERVO_RESPONSE, CONFIG_VECTOR_JUMP, FIRMWARE_VERSION_REPLY,
    FLAT_FRAME, FRAME_MASK, ICS_ASSIGN_ADDRESS, ICS_BITRATE_MASK, ICS_SLOT_COUNT,
    ICS_UNASSIGNED, ProtocolError, SERVO_RECORD_SIZE,
    build_ack, build_multi_servo, build_ram_read, build_ram_write,
    build_stretch, build_version, com_bitrate_of, decode_servo_records, flat_config,
    ics_numbers_of,
    is_ack, is_nak, is_stretch_ack, is_write_ack, max_ics_id,
    parse_ram_read, plan_record_reads, port_name, servo_record_address,
    with_com_bitrate, with_ics_bitrate)

#: Milliseconds per unit of the move-duration byte on the RCB-4mini. Not
#: measured yet, so ``move_time_ms`` cannot be converted and is refused; each
#: actuator's own move byte is used instead.
FRAME_SCALE_MS: Optional[float] = None

#: State members read every cycle by default: the measured pulse only.
DEFAULT_STATE_SLOTS = ("current_angle",)

#: Rates tried when the board does not answer at the transport's rate. 115200
#: is what a board without a HeartToHeart project powers up at.
DEFAULT_CONNECT_BITRATES = (115200,)

#: Time for the board to exchange a frame with a newly assigned servo and
#: write its position into the record.
DEFAULT_ASSIGN_SETTLE_S = 0.1

#: Times a refused request is sent again before the cycle gives up on it.
#: Every request this bus sends is idempotent (absolute targets, RAM reads and
#: writes of fixed values), so resending one cannot apply it twice. Bounded,
#: because a board that keeps refusing is a fault to report, not to hammer.
DEFAULT_NAK_RETRIES = 3

#: Servo record bytes cleared before assigning: the measured pulse (+4), the
#: reference (+6) and the interpolation speed (+8). See
#: :meth:`KondoRcb4Bus._assign_ics`.
_ASSIGN_CLEAR_OFFSET = 4
#: What goes there: no pulse, OFF, no interpolation. An interpolation speed
#: left running would keep rewriting the reference each cycle ([21]).
_OFF_CLEAR = struct.pack("<HHB", 0, SERVO_OFF_PULSE, 0)

#: Servo record offset of the reference pulse (``0x8000`` = off).
_REFERENCE_OFFSET = 6

#: Config word bit 0, the board's ICS switch (rcb4's ``ConfigData.EnableSio``).
#: Off at power-up on a board without a project, and with it off no servo
#: answers on either SIO port, whatever is assigned (measured on an RCB-4mini:
#: 0 of 22 servos with the bit clear, 22 of 22 with it set and the records
#: prepared). A project sets it; this bus sets it in :meth:`_flatten_config`.
_CONFIG_ICS_SWITCH = 0x0001

#: The first two bytes of a servo record: a head byte, which must be 0, and the
#: servo ID the board addresses on the ICS line. A HeartToHeart project's
#: startup routine copies them from ROM; on a board without one (ROM erased)
#: they power up as whatever the RAM held, and the board then talks to another
#: ID or to none, so no servo answers. Written before every assignment
#: (:meth:`KondoRcb4Bus._prepare_record`). The ID is the ICS number >> 1 (the
#: low bit is the port). Measured on an RCB-4mini without a project: with the
#: head byte at 1 no servo on an odd ICS number answered, and at 2 the board
#: stopped answering until power-cycled.
_RECORD_HEAD_OFFSET = 0
_RECORD_HEAD = 0

#: Servo record offset of the board's per-servo trim, a signed pulse count.
#: The board reports ``servo pulse - trim`` and drives ``reference + trim``
#: (measured, references.md). A HeartToHeart project in the board's ROM sets
#: it at power-up.
_TRIM_OFFSET = 2

#: Servo record offsets of the two mixings: a uint16 naming the source data,
#: then a ratio byte (sign and factor). The board drives
#: ``reference + mixing 1 + mixing 2``, each ``source x ratio``, every cycle
#: (firmware reference, references.md [21]). A project sets them.
_MIXING_OFFSETS = (14, 17)

#: A mixing source naming no data. Every mixing on a humanoid's board read this,
#: ratio 1, with no mixing seen (measured). Read as a source it is neither a
#: RAM address (the RAM ends at ``$048F``) nor an ICS number (bits 8-13 would
#: be 63, past 35), so it names nothing (inferred from [21]).
_MIXING_NO_SOURCE = 0xFFFF

#: Hot-plug presence check: every this many ``read_batch`` calls the measured
#: pulse of each answering servo is cleared, and a servo whose pulse still
#: reads 0 ``DEFAULT_ABSENT_TIMEOUT_S`` after the clear is taken as gone.
#: The board rewrites the pulse of every servo that answers once per frame,
#: limp or held, and leaves a silent one at 0 (measured, see references.md):
#: 6000 clears of three servos at 115200 were all rewritten within 16 ms.
#: The timeout is on the clock, not a count of reads: a control loop that
#: catches up after a stall reads twice within a few milliseconds of the
#: clear, and a count of 2 took a servo that never stopped answering for gone
#: (11 times in 41 s on one servo, measured).
DEFAULT_PRESENCE_PERIOD = 5
DEFAULT_ABSENT_TIMEOUT_S = 0.1

#: Largest change between two reads taken without confirmation, rad. On the
#: rig, real motion moved a servo at most 0.012 rad per 20 ms read, and the
#: smallest glitch seen was 0.034 rad off (references.md); 0.05 sits between.
#: A real move faster than this only costs one read of latency.
DEFAULT_READING_JUMP_RAD = 0.05

#: Consecutive reads that must agree before a jump is believed. 2 was not
#: enough: glitches of 0.07 and 0.39 rad lasted two reads on the rig, and the
#: second one became a trajectory controller's start pose (2026-10-17).
DEFAULT_READING_CONFIRM = 3

#: Two commands closer than this, rad, are the same command to ``hold_relax``.
#: A trajectory controller holding its last point sends the same value every
#: cycle; 1e-4 rad is 0.17 pulse.
_SAME_COMMAND_RAD = 1e-4


class KondoRcb4Bus(Bus):
    """Drives a set of :class:`KondoIcsActuator` through one RCB-4mini.

    Parameters
    ----------
    transport : KondoRcb4Transport
        Its ``bitrate`` is the rate the board is switched to.
    actuators : sequence of Actuator
        In URDF joint order. Each ``id`` is the servo's ICS number (0..34).
    state_slots : sequence of str
        Servo record members to read each cycle, from
        :data:`~rcb4.protocol.rcb4mini.SERVO_SLOTS`.
    move_time_ms : float, optional
        Board-side interpolation time per packet. ``None`` uses each
        actuator's own move byte. Set it from the control period: every packet
        restarts the interpolation.
    read_timeout : float
        Seconds to wait for one reply.
    joint_names : sequence of str, optional
        Joint name per actuator, for operator-facing messages.
    ics_bitrate : int
        Servo-side bit rate set on the board when the bus assigns servos.
    connect_bitrates : sequence of int
        Rates tried when the board does not answer at the transport's rate.
    assign_settle_s : float
        Wait between assigning servos and checking that they answered.
    nak_retries : int
        Times a refused request is sent again. A RAM read or write that gets
        no answer at all is also sent again up to this many times: the board
        drops one now and then, and both are safe to repeat.
    battery_voltage_warn_below : float, optional
        Supply voltage below which ``diagnose`` reports WARN. ``None`` claims
        no low-voltage level.
    hot_plug : bool
        Servos may be absent at start and may come and go while running. A
        configured servo that does not answer stays assigned (so the board
        keeps polling its number and notices it when it is plugged in) and
        :meth:`present` reports it. Without it a missing servo is an error.
    presence_period : int
        With ``hot_plug``, ``read_batch`` calls between presence checks.
    absent_timeout_s : float
        With ``hot_plug``, seconds after a clear that a still-zero pulse
        makes a servo absent.
    reading_jump_rad : float, optional
        A reading further than this from the last one is held until
        ``reading_confirm`` reads in a row agree (see :meth:`_accept`).
        ``None`` takes every reading.
    reading_confirm : int
        Reads in a row, the jumping one included, that must agree.
    clock : callable
        Monotonic seconds; replaceable for tests.
    port_detect : bool
        Each actuator ``id`` is the servo's ICS ID (0..17), not its ICS
        number, and the bus finds which port it is plugged into. See the
        module docstring.
    """

    def __init__(self, transport, actuators,
                 state_slots: Sequence[str] = DEFAULT_STATE_SLOTS,
                 move_time_ms: Optional[float] = None,
                 read_timeout: float = 0.1,
                 joint_names: Optional[Sequence[str]] = None,
                 ics_bitrate: int = 1_250_000,
                 connect_bitrates: Sequence[int] = DEFAULT_CONNECT_BITRATES,
                 assign_settle_s: float = DEFAULT_ASSIGN_SETTLE_S,
                 nak_retries: int = DEFAULT_NAK_RETRIES,
                 battery_voltage_warn_below: Optional[float] = None,
                 hot_plug: bool = False,
                 presence_period: int = DEFAULT_PRESENCE_PERIOD,
                 absent_timeout_s: float = DEFAULT_ABSENT_TIMEOUT_S,
                 port_detect: bool = False,
                 reading_jump_rad: Optional[float] = DEFAULT_READING_JUMP_RAD,
                 reading_confirm: int = DEFAULT_READING_CONFIRM,
                 clock=time.monotonic) -> None:
        super().__init__(transport, actuators)
        #: Set at startup, armed on the first write_batch (see _arm_failsafe).
        self._failsafe_pending = False
        self.hot_plug = bool(hot_plug)
        self.port_detect = bool(port_detect)
        if int(presence_period) < 1 or not float(absent_timeout_s) > 0:
            raise ValueError("presence_period must be >= 1 and "
                             "absent_timeout_s > 0")
        self.presence_period = int(presence_period)
        self.absent_timeout_s = float(absent_timeout_s)
        self.reading_jump_rad = (None if reading_jump_rad is None
                                 else float(reading_jump_rad))
        if int(reading_confirm) < 2:
            raise ValueError("reading_confirm must be >= 2 (1 would take every jump)")
        self.reading_confirm = int(reading_confirm)
        # Device id -> readings held back since the last accepted one.
        self._pending: Dict[int, List[ActuatorState]] = {}
        self._held_readings = 0
        self._dropped_readings = 0
        self._clock = clock
        # ICS number -> when its pulse was last cleared / whether it answers.
        self._cleared_at: Dict[int, float] = {}
        # ICS number -> rewritten by the board since its last clear.
        self._rewritten: Dict[int, bool] = {}
        # ICS number -> when a nonzero pulse was last read / why it went absent.
        self._last_rewrite: Dict[int, float] = {}
        self._absent_why: Dict[int, str] = {}
        self._present: Dict[int, bool] = {}
        self._reads_since_check = 0
        self.state_slots = tuple(state_slots)
        if not self.state_slots:
            raise ValueError(
                "a bus that reads no state slot reports no state; name at "
                "least 'current_angle'")
        for actuator in actuators:
            if (getattr(actuator, "expects_rotation", False)
                    and not getattr(actuator, "rotation_checked", None)):
                # On the ARM-H7 the bus reads the servo's rotation setting
                # before driving a wheel. This board has no record of it and
                # cannot relay the servo's EEPROM (references.md), so the
                # check is made off the board, with the servo on the adapter,
                # and the config has to say so.
                raise ValueError(
                    "servo id {} is a continuous-rotation actuator, but its "
                    "config has no `rotation_checked`. The RCB-4mini cannot "
                    "read whether a servo is in rotation mode, and a position "
                    "servo driven as a wheel swings to wherever the speed "
                    "command points. Read the rotation flag with the servo on "
                    "the Dual USB Adapter in ICS mode (rcb4 ics.py, "
                    "read_rotation) and record how and when in "
                    "`rotation_checked`.".format(actuator.id))
            if self.port_detect:
                if not 0 <= int(actuator.id) <= max_ics_id():
                    raise ValueError(
                        "servo id {} is outside the ICS IDs the board has "
                        "numbers for (0..{})".format(actuator.id, max_ics_id()))
            elif not 0 <= int(actuator.id) < ICS_SLOT_COUNT:
                raise ValueError(
                    "servo id {} is outside the board's ICS numbers (0..{})"
                    .format(actuator.id, ICS_SLOT_COUNT - 1))
        self._ids = [int(a.id) for a in actuators]
        if len(set(self._ids)) != len(self._ids):
            raise ValueError("servo ids {} are not unique".format(self._ids))
        # Actuator id -> the ICS numbers it may answer on. Without port
        # detection the id is the ICS number.
        self._candidates: Dict[int, List[int]] = {
            i: (ics_numbers_of(i) if self.port_detect else [i]) for i in self._ids}
        claimed: Dict[int, int] = {}
        for servo_id, slots in self._candidates.items():
            for slot in slots:
                if slot in claimed:
                    raise ValueError(
                        "ICS number {} is claimed by servo ids {} and {}".format(
                            slot, claimed[slot], servo_id))
                claimed[slot] = servo_id
        #: Every ICS number the board is asked to poll, ascending.
        self._slots = sorted(claimed)
        # Actuator id -> ICS number it is driven on; None while unknown.
        self._slot_of: Dict[int, Optional[int]] = {
            i: (None if self.port_detect else i) for i in self._ids}
        # Ids answering on both ports: two servos share the ID, one per port,
        # and nothing says which is this joint, so neither is driven.
        self._ambiguous: set = set()
        # Id -> reads it still reads absent after moving to the other port.
        self._settling: Dict[int, int] = {}
        self._port_moves = 0
        # Validates the slot names too.
        self._reads = (plan_record_reads(self._slots, self.state_slots)
                       if self._slots else [])
        self.move_time_ms = move_time_ms
        self._bus_move_byte()  # refuse an unconvertible move_time_ms now
        move_bytes = {int(a.id): self._velocity_for(a) for a in actuators}
        if len(set(move_bytes.values())) > 1:
            raise ValueError(
                "servos have different move bytes ({}); the RCB-4mini is only "
                "driven with one move byte for every servo (see write_batch)."
                .format(move_bytes))
        self.read_timeout = float(read_timeout)
        self.ics_bitrate = int(ics_bitrate)
        self.connect_bitrates = tuple(int(b) for b in connect_bitrates)
        self.assign_settle_s = float(assign_settle_s)
        self.nak_retries = int(nak_retries)
        self.set_battery_voltage_threshold(battery_voltage_warn_below)
        self._states: Dict[int, ActuatorState] = {}
        self._missed = 0
        self._naks = 0
        self._unanswered = 0
        # Cycles in which at least one read answered; see KondoH7Bus for why
        # `missed` alone cannot tell a frozen robot from a healthy one.
        self._state_updates = 0
        self._servo_on: Dict[int, bool] = {i: False for i in self._ids}
        # Servo id -> stretch its config asks for.
        self._stretch: Dict[int, int] = {
            int(a.id): int(a.stretch) for a in actuators
            if getattr(a, "stretch", None) is not None}
        # ICS numbers whose stretch was set since they last answered.
        self._stretch_set: set = set()
        self._stretch_failures = 0
        # Servo id -> its HoldRelax.
        self._relax_config = {
            int(a.id): a.hold_relax for a in actuators
            if getattr(a, "hold_relax", None) is not None}
        # Servo id -> (command, when it was first sent).
        self._relax_since: Dict[int, Tuple[float, float]] = {}
        # Servo id -> (command, target sent instead) while held lighter.
        self._relaxed: Dict[int, Tuple[float, float]] = {}
        self._relax_count = 0
        self._joint_name_by_id: Dict[int, str] = {}
        if joint_names is not None:
            self.set_joint_names(joint_names)

    def set_joint_names(self, joint_names: Sequence[str]) -> None:
        """Name each configured servo id for operator-facing messages."""
        if len(joint_names) != len(self.actuators):
            raise ValueError(
                "joint_names has {} entries for {} actuators".format(
                    len(joint_names), len(self.actuators)))
        self._joint_name_by_id = {
            int(actuator.id): str(name)
            for actuator, name in zip(self.actuators, joint_names)}

    def set_battery_voltage_threshold(self, warn_below: Optional[float]) -> None:
        """Set/clear the optional low-voltage warning threshold in volts."""
        self._battery_voltage_warn_below = (
            None if warn_below is None else float(warn_below))

    def _joint_name(self, servo_id: int) -> str:
        return self._joint_name_by_id.get(int(servo_id), "id {}".format(servo_id))

    # -- housekeeping ------------------------------------------------------

    @property
    def frame_scale_ms(self) -> Optional[float]:
        """Milliseconds per unit of the move-duration byte."""
        return FRAME_SCALE_MS

    @property
    def state_updates(self) -> int:
        """Cycles in which at least one record read answered."""
        return self._state_updates

    @property
    def missed(self) -> int:
        """Reads that did not answer. Never turned into a fabricated state."""
        return self._missed

    @property
    def naks(self) -> int:
        """Requests the board refused, including ones a resend then got through."""
        return self._naks

    @property
    def unanswered(self) -> int:
        """RAM reads and writes the board did not answer, including ones a
        resend then got through."""
        return self._unanswered

    def _command(self, payload: bytes, timeout: Optional[float] = None,
                 resend_unanswered: bool = False) -> Optional[Frame]:
        """Send one packet and consume its reply (the board answers everything).

        A refusal is sent again up to ``nak_retries`` times. If the board still
        refuses, the refusal is returned, and the caller treats it as the
        wrong reply it is. With ``resend_unanswered`` a request that gets no
        reply is sent again too, within the same number of tries; only for
        requests that do the same thing when repeated.
        """
        wait = self.read_timeout if timeout is None else timeout
        for _ in range(1 + self.nak_retries):
            reply = self.transport.transact([Frame(payload=payload)], wait)[0]
            if reply is None:
                if not resend_unanswered:
                    return None
                self._unanswered += 1
                continue
            if not is_nak(reply.payload):
                return reply
            self._naks += 1
        return reply

    def _ack(self) -> bool:
        reply = self._command(build_ack())
        return reply is not None and is_ack(reply.payload)

    def _read_ram(self, address: int, size: int) -> bytes:
        reply = self._command(build_ram_read(address, size),
                              resend_unanswered=True)
        if reply is None:
            raise ProtocolError(
                "board did not answer a {}-byte read at {:#06x}".format(
                    size, address))
        return parse_ram_read(reply.payload, size)

    def _write_ram(self, address: int, data: bytes) -> None:
        reply = self._command(build_ram_write(address, data),
                              resend_unanswered=True)
        if reply is None or not is_write_ack(reply.payload):
            raise ProtocolError(
                "board did not acknowledge a {}-byte write at {:#06x}: {!r}"
                .format(len(data), address,
                        None if reply is None else reply.payload))

    def read_config(self) -> int:
        """The board's system config word."""
        return struct.unpack("<H", self._read_ram(CONFIG_ADDRESS, 2))[0]

    def _write_config(self, config: int) -> None:
        self._write_ram(CONFIG_ADDRESS, struct.pack("<H", config))

    def read_ics_assign_table(self) -> List[int]:
        """The board's ICS assignment table, one entry per ICS number."""
        data = self._read_ram(ICS_ASSIGN_ADDRESS, 2 * ICS_SLOT_COUNT)
        return list(struct.unpack("<{}H".format(ICS_SLOT_COUNT), data))

    def battery_voltage(self) -> float:
        """Supply voltage in volts, from analog input 0."""
        count = struct.unpack("<H", self._read_ram(BATTERY_ADC_ADDRESS, 2))[0]
        return count * BATTERY_VOLTS_PER_COUNT

    def firmware_version(self) -> bytes:
        """The board's reply to the version request, opcode echo included."""
        reply = self._command(build_version())
        if reply is None:
            raise ProtocolError("board did not answer the version request")
        return bytes(reply.payload)

    def verify_firmware(self) -> None:
        """Refuse to continue unless this is the RCB-4mini firmware.

        Every address used here is the RCB-4mini's fixed layout, so on another
        board this bus would read the right number of bytes from the wrong
        place.
        """
        version = self.firmware_version()
        if version != FIRMWARE_VERSION_REPLY:
            raise RuntimeError(
                "board firmware reports {!r}, not the RCB-4mini's {!r}. This "
                "bus only knows the RCB-4mini's memory layout; an ARM-H7 is "
                "driven by kondo_h7_bus.".format(version, FIRMWARE_VERSION_REPLY))

    # -- Bus contract ------------------------------------------------------

    def _bus_move_byte(self) -> Optional[int]:
        """The move-duration byte from ``move_time_ms``, or ``None``."""
        if self.move_time_ms is None:
            return None
        if FRAME_SCALE_MS is None:
            raise ValueError(
                "move_time_ms is set, but the RCB-4mini's move-duration unit has "
                "not been measured, so it cannot be converted. Leave it unset "
                "to use each servo's own move byte.")
        return max(1, min(255, int(round(self.move_time_ms / FRAME_SCALE_MS))))

    def _velocity_for(self, actuator) -> int:
        return actuator.move_byte(self._bus_move_byte())

    def set_mode(self, mode: OperatingMode) -> None:
        """Capability-check every servo. ICS is position-only."""
        for actuator in self.actuators:
            actuator.capabilities.require(mode)

    def write_batch(self, commands: Sequence[Optional[ActuatorCommand]]) -> None:
        """Push every servo's target in one packet.

        The first call also hands a relay on the link its failsafe
        (:meth:`_arm_failsafe`).

        A ``None`` command releases that servo; released and held servos ride
        in the same packet, as on the H7.

        A servo that was released and now gets a target is first switched on
        where it stands (``SERVO_ON_PULSE``) in a packet of its own: from the
        OFF reference this board never interpolates toward a target, so the
        servo would stay limp (measured, see references.md).

        A WHEEL is switched on differently, because ``SERVO_ON_PULSE`` copies
        the measured pulse into the reference, and a wheel reads its reference
        as a speed: the pot's arbitrary position would become a speed up to
        full. Its record's reference is written to stop (``CENTER_PULSE``) in
        RAM instead (:meth:`_stop_wheel`), and the speed follows in the batch.

        Every ICS number the bus polls is in the packet. With ``port_detect``
        the ones a servo is not driven on get OFF, and so does a servo whose
        port is not known, answers on both, or has just moved: its target
        is not sent anywhere, and :meth:`present` is False for it.
        """
        if self._failsafe_pending:
            self._failsafe_pending = False
            self.transport.set_failsafe(self.failsafe_frames())

        if len(commands) != len(self.actuators):
            raise ValueError("write_batch expects {} commands, got {}".format(
                len(self.actuators), len(commands)))
        targets: Dict[int, int] = {slot: SERVO_OFF_PULSE for slot in self._slots}
        velocities: List[int] = []
        turning_on: List[int] = []
        driven: Dict[int, bool] = {}
        for actuator, command in zip(self.actuators, commands):
            velocities.append(self._velocity_for(actuator))
            slot = self._drive_slot(actuator.id)
            driven[int(actuator.id)] = command is not None and slot is not None
            if not driven[int(actuator.id)]:
                self._relax_since.pop(int(actuator.id), None)
                self._relaxed.pop(int(actuator.id), None)
                continue
            targets[slot] = actuator.batch_pulse(self._relax(actuator, command))
            if not self._servo_on[actuator.id]:
                if getattr(actuator, "expects_rotation", False):
                    self._stop_wheel(slot)
                else:
                    turning_on.append(slot)
        ids = list(targets)
        pulses = list(targets.values())
        if not ids:
            return
        if len(set(velocities)) != 1:
            # The per-servo-velocity packet (0x11) stops or resets this
            # board; see references.md. Refuse instead of sending it.
            raise ValueError(
                "servos have different move bytes ({}); the RCB-4mini is only "
                "driven with one move byte for every servo. Give every "
                "joint the same `velocity`, or set move_time_ms.".format(
                    dict(zip(self._ids, velocities))))
        if turning_on:
            self._command(build_multi_servo(
                ids,
                [SERVO_ON_PULSE if slot in turning_on else pulse
                 for slot, pulse in zip(ids, pulses)],
                velocities[0]))
        self._command(build_multi_servo(ids, pulses, velocities[0]))
        self._servo_on.update(driven)

    def _relax(self, actuator, command: ActuatorCommand) -> ActuatorCommand:
        """``command``, or the lighter hold that replaces it (``hold_relax``)."""
        servo_id = int(actuator.id)
        config = self._relax_config.get(servo_id)
        state = self._states.get(servo_id)
        if (config is None or command.position is None or state is None
                or state.position is None):
            return command
        commanded = command.position
        measured = state.position
        now = self._clock()
        held = self._relaxed.get(servo_id)
        if held is not None:
            if abs(commanded - held[0]) <= _SAME_COMMAND_RAD:
                return replace(command, position=held[1])
            # A new command: drive it.
            del self._relaxed[servo_id]
        since = self._relax_since.get(servo_id)
        if (since is None or abs(commanded - since[0]) > _SAME_COMMAND_RAD
                or abs(commanded - measured) <= config.error_rad):
            self._relax_since[servo_id] = (commanded, now)
            return command
        if now - since[1] < config.seconds:
            return command
        target = measured + math.copysign(config.squeeze_rad, commanded - measured)
        self._relaxed[servo_id] = (commanded, target)
        self._relax_count += 1
        return replace(command, position=target)

    def needs_write(self) -> bool:
        """Whether a write with unchanged commands would now send something else.

        True once a servo watched by ``hold_relax`` has been held short of its
        command for its ``seconds``: the lighter hold goes out with the next
        write. A loop that skips writes while the commands stand still (the
        bus node does, see its main loop) has to write then.
        """
        now = self._clock()
        for servo_id, (commanded, since) in self._relax_since.items():
            if servo_id in self._relaxed:
                continue
            config = self._relax_config[servo_id]
            state = self._states.get(servo_id)
            if (state is not None and state.position is not None
                    and abs(commanded - state.position) > config.error_rad
                    and now - since >= config.seconds):
                return True
        return False

    def relaxed(self) -> Dict[int, Tuple[float, float]]:
        """Servo id -> (command, target sent instead), for servos held lighter."""
        return dict(self._relaxed)

    def _apply_stretch(self, servo_ids: Sequence[int]) -> Optional[DiagItem]:
        """Set the configured stretch of ``servo_ids`` that are driven.

        Returns
        -------
        DiagItem or None
            What was set, or ``None`` if nothing was.

        Raises
        ------
        ProtocolError
            If the board does not acknowledge it.
        """
        wanted: Dict[int, int] = {}
        for servo_id in servo_ids:
            slot = self._drive_slot(servo_id)
            if servo_id in self._stretch and slot is not None:
                wanted[slot] = self._stretch[servo_id]
        if not wanted:
            return None
        reply = self._command(build_stretch(wanted))
        if reply is None or not is_stretch_ack(reply.payload):
            raise ProtocolError(
                "board did not acknowledge setting the stretch of ICS {}: {!r}"
                .format(sorted(wanted), None if reply is None else reply.payload))
        self._stretch_set.update(wanted)
        by_id = {i: self._drive_slot(i) for i in servo_ids if self._drive_slot(i) in wanted}
        return DiagItem(
            "bus/stretch", DiagLevel.WARN,
            "Set the stretch of {}. Reported because this CHANGED the servos; "
            "it reverts when they lose power.".format(", ".join(
                "{} (ICS {}) to {}".format(self._joint_name(i), slot, wanted[slot])
                for i, slot in sorted(by_id.items()))),
            {"stretch": ", ".join("{}={}".format(slot, value)
                                  for slot, value in sorted(wanted.items()))})

    def _restretch(self) -> None:
        """Set the stretch of servos that answer again (``hot_plug``).

        A refusal is counted and tried again next cycle; ``diagnose`` names
        the servos whose stretch is not set.
        """
        due = [i for i in self._stretch
               if self._drive_slot(i) is not None
               and self._present.get(self._drive_slot(i))
               and self._drive_slot(i) not in self._stretch_set]
        if not due:
            return
        try:
            self._apply_stretch(due)
        except ProtocolError:
            self._stretch_failures += 1

    def _drive_slot(self, device_id) -> Optional[int]:
        """ICS number to send ``device_id``'s target to, or ``None`` when it
        must not be driven (port unknown, two servos with its ID, or it just
        moved to the other port)."""
        device_id = int(device_id)
        if device_id in self._ambiguous or device_id in self._settling:
            return None
        return self._slot_of[device_id]

    def ics_number(self, device_id) -> Optional[int]:
        """ICS number ``device_id`` was last found on; ``None`` if never."""
        return self._slot_of.get(int(device_id))

    def where(self, device_id) -> str:
        """``ICS 5 (SIO5-8)`` for operator messages; '' if not known."""
        slot = self.ics_number(device_id)
        if slot is None:
            return ""
        return "ICS {} ({})".format(slot, port_name(slot))

    def read_batch(self, timeout: Optional[float] = None) -> List[ActuatorState]:
        """Read the configured members of every servo's record.

        A read that does not answer drops its servos from this cycle rather
        than filling them in; they keep their previous state (``None`` before
        the first) and ``missed`` counts up.
        """
        wait = self.read_timeout if timeout is None else float(timeout)
        raw: Dict[int, Dict[str, int]] = {}
        for address, size in self._reads:
            reply = self._command(build_ram_read(address, size), wait)
            if reply is None:
                self._missed += 1
                continue
            try:
                data = parse_ram_read(reply.payload, size)
            except ProtocolError:
                # Out of step: decoding anyway would hand one servo's angle to
                # another. Drop it and let the counter show it.
                self._missed += 1
                continue
            raw.update(decode_servo_records(
                address, data, self._slots, self.state_slots))
        if raw:
            self._state_updates += 1

        if self.hot_plug:
            for slot in self._slots:
                values = raw.get(slot)
                if values is None:
                    continue
                if values.get("current_angle") == 0:
                    # Cleared by the presence check and not rewritten: either
                    # the servo is gone or the board has not got to it yet.
                    # Not a position either way, so nothing is decoded.
                    cleared = self._cleared_at.get(slot)
                    now = self._clock()
                    if cleared is None or now - cleared >= self.absent_timeout_s:
                        if self._present.get(slot):
                            self._absent_why[slot] = self._why_absent(slot, now)
                        self._present[slot] = False
                        # Unplugged, it loses the stretch it was set to.
                        self._stretch_set.discard(slot)
                    del raw[slot]
                else:
                    self._present[slot] = True
                    self._rewritten[slot] = True
                    self._last_rewrite[slot] = self._clock()
            for servo_id in list(self._settling):
                self._settling[servo_id] -= 1
                if self._settling[servo_id] <= 0:
                    del self._settling[servo_id]
            if self.port_detect:
                self._resolve_ports(self._present)
            if self._stretch:
                self._restretch()

        states: List[ActuatorState] = []
        for actuator in self.actuators:
            servo_id = int(actuator.id)
            slot = self._slot_of[servo_id]
            values = None
            if slot is not None and servo_id not in self._ambiguous:
                values = raw.get(slot)
            if values is not None:
                frame = self._state_frame(servo_id, values)
                if actuator.matches(frame):
                    self._accept(actuator.id, actuator.decode_state(frame))
            states.append(self._states.get(actuator.id))
        if self.hot_plug:
            self._reads_since_check += 1
            if self._reads_since_check >= self.presence_period:
                self._reads_since_check = 0
                # Only numbers the board has rewritten since their last
                # clear: clearing one that is still at zero would restart its
                # timeout, and with presence_period x cycle close to the
                # timeout that postponed noticing a pulled servo by 0.4-3.5 s
                # (measured on the stack; the clear kept landing just before
                # the timeout ran out).
                self._clear_pulses([s for s in self._slots if self._present.get(s)
                                    and self._rewritten.get(s, True)])
        return states

    def _accept(self, device_id, state: ActuatorState) -> None:
        """Take a decoded reading, unless it jumps implausibly far.

        The board occasionally reports a moving or held servo somewhere it is
        not -- 0.03 to 1.38 rad away, for one read or two, never while the
        servo was still. One is not harmless: a trajectory controller that sees
        it fails its path tolerance, aborts, and holds "the current position"
        -- that reading -- or starts its next trajectory from it, and the servo
        really went there (1.4 rad and 0.26 rad, 2026-10-17). So a reading more
        than ``reading_jump_rad`` from the last accepted one is held back until
        ``reading_confirm`` reads in a row agree with each other; a real move
        costs that many reads of latency, a glitch is dropped. Both are counted
        (``held_readings``, ``dropped_readings``).
        """
        limit = self.reading_jump_rad
        last = self._states.get(device_id)
        if (limit is None or last is None or last.position is None
                or state.position is None
                or abs(state.position - last.position) <= limit):
            dropped = self._pending.pop(device_id, None)
            if dropped:
                self._dropped_readings += len(dropped)
            self._states[device_id] = state
            return
        pending = self._pending.setdefault(device_id, [])
        if pending and abs(state.position - pending[-1].position) > limit:
            # Disagrees with the run being held: that run was a glitch.
            self._dropped_readings += len(pending)
            pending.clear()
        pending.append(state)
        self._held_readings += 1
        if len(pending) >= self.reading_confirm:
            # Enough reads agree: the servo really is over there.
            del self._pending[device_id]
            self._states[device_id] = state

    @property
    def held_readings(self) -> int:
        """Readings held back for jumping more than ``reading_jump_rad``."""
        return self._held_readings

    @property
    def dropped_readings(self) -> int:
        """Held readings the next read did not confirm: glitches."""
        return self._dropped_readings

    def _resolve_ports(self, present: Dict[int, bool]) -> None:
        """Drive each servo on the one ICS number of its ID that answers.

        One answering: that is its port. None: it keeps the last one (and
        reads absent there). Both: two servos share the ID, and neither is
        driven. A servo found on another port than before reads absent for
        ``presence_period`` reads, so whatever held it releases it first.
        """
        for servo_id in self._ids:
            answering = [s for s in self._candidates[servo_id] if present.get(s)]
            if len(answering) > 1:
                self._ambiguous.add(servo_id)
                continue
            self._ambiguous.discard(servo_id)
            if not answering or answering[0] == self._slot_of[servo_id]:
                continue
            moved = self._slot_of[servo_id] is not None
            self._slot_of[servo_id] = answering[0]
            self._servo_on[servo_id] = False
            if moved:
                self._port_moves += 1
                self._settling[servo_id] = self.presence_period

    def _clear_pulses(self, ids: Sequence[int]) -> None:
        """Zero the measured pulse of ``ids``; the board rewrites it for every
        servo that answers. A write the board does not take is counted as
        missed and retried at the next check, never raised: this runs inside
        the control cycle."""
        for slot in ids:
            try:
                self._write_ram(servo_record_address(slot) + _ASSIGN_CLEAR_OFFSET,
                                b"\x00\x00")
            except ProtocolError:
                self._missed += 1
                continue
            # Taken after the board acknowledged, so never earlier than the
            # clear itself: the timeout can only come out long, not short.
            self._cleared_at[slot] = self._clock()
            self._rewritten[slot] = False

    def _why_absent(self, slot: int, now: float) -> str:
        cleared = self._cleared_at.get(slot)
        seen = self._last_rewrite.get(slot)
        return ("ICS {}: pulse still 0 {} after the presence check cleared it "
                "(timeout {:.0f} ms); last rewritten {} ago".format(
                    slot,
                    "with no clear on record" if cleared is None
                    else "{:.0f} ms".format(1000 * (now - cleared)),
                    1000 * self.absent_timeout_s,
                    "never" if seen is None else "{:.0f} ms".format(1000 * (now - seen))))

    def absence_reason(self, device_id) -> str:
        """Why :meth:`present` is False for ``device_id``; '' if it is not."""
        device_id = int(device_id)
        if self.present(device_id):
            return ""
        if device_id in self._ambiguous:
            return "answers on both ICS {}: two servos share ID {}".format(
                " and ".join(str(s) for s in self._candidates[device_id]), device_id)
        if device_id in self._settling:
            return "just moved to ICS {}; held absent for {} more reads".format(
                self._slot_of[device_id], self._settling[device_id])
        slot = self._slot_of.get(device_id)
        if slot is None:
            return "never answered on ICS {}".format(
                " or ".join(str(s) for s in self._candidates[device_id]))
        return self._absent_why.get(slot, "ICS {}: not answering".format(slot))

    def present(self, device_id) -> bool:
        """Whether a servo is answering.

        With ``hot_plug``: its pulse was rewritten since the last presence
        check. Without it a servo is never taken away, so this is whether it
        has ever reported. With ``port_detect`` a servo is also absent while
        :meth:`write_batch` does not drive it (see there).
        """
        if self.port_detect and self._drive_slot(device_id) is None:
            return False
        if self.hot_plug:
            return self._present.get(self._slot_of[int(device_id)], False)
        return device_id in self._states

    def last_state(self, device_id) -> Optional[ActuatorState]:
        """Most recent decoded state for one servo; ``None`` before its first."""
        return self._states.get(device_id)

    def _state_frame(self, servo_id: int, values: Dict[str, int]) -> Frame:
        """One servo's members as the frame :class:`KondoIcsActuator` decodes."""
        slots = tuple(name for name in self.state_slots if name in values)
        payload = struct.pack("<" + "i" * len(slots),
                              *(values[name] for name in slots))
        return Frame(payload=payload, address=servo_id,
                     meta={"kind": "servo_state", "slots": slots})

    def release_all(self) -> None:
        """Servo OFF for every joint, in one packet."""
        if not self._ids:
            return
        self._command(build_multi_servo(
            self._slots, [SERVO_OFF_PULSE] * len(self._slots),
            self._velocity_for(self.actuators[0])))
        for servo_id in self._ids:
            self._servo_on[servo_id] = False

    def hold_all(self) -> None:
        """Servo ON for every joint, each holding wherever it currently is.

        A servo :meth:`write_batch` would not drive stays OFF.
        """
        if not self._ids:
            return
        driven = {i: self._drive_slot(i) for i in self._ids}
        held = {slot for slot in driven.values() if slot is not None}
        # A wheel "held where it is" is a wheel standing still: see
        # write_batch for why it is not sent SERVO_ON_PULSE.
        wheels = {self._drive_slot(a.id) for a in self.actuators
                  if getattr(a, "expects_rotation", False)} & held
        for slot in wheels:
            self._stop_wheel(slot)
        self._command(build_multi_servo(
            self._slots,
            [CENTER_PULSE if slot in wheels
             else SERVO_ON_PULSE if slot in held else SERVO_OFF_PULSE
             for slot in self._slots],
            self._velocity_for(self.actuators[0])))
        for servo_id, slot in driven.items():
            self._servo_on[servo_id] = slot is not None

    def _stop_wheel(self, slot: int) -> None:
        """Switch a wheel on at standstill: its record's reference := stop.

        Measured on an RCB-4mini with four continuous-rotation servos: after
        this write the servo stands still (its pot reading did not move), and
        the next batch packet's speed is taken from it as from any other held
        reference. Without it the board ignores a move to a free servo: its
        reference stays at the OFF value. The OFF reference is written the
        same way at assignment (``_assign_ics``).
        """
        self._write_ram(servo_record_address(slot) + _REFERENCE_OFFSET,
                        struct.pack("<H", CENTER_PULSE))

    def servo_on_states(self) -> Dict[int, bool]:
        """Last servo on/off state this bus commanded, per servo id."""
        return dict(self._servo_on)

    # -- cold start --------------------------------------------------------

    def _connect(self) -> Optional[DiagItem]:
        """Get the board answering at the transport's rate.

        Returns
        -------
        DiagItem or None
            A WARN item if the board had to be switched, else ``None``.

        Raises
        ------
        RuntimeError
            If the board answers at no rate, or stops answering after the
            switch.
        """
        target = self.transport.bitrate
        if self.transport.current_bitrate != target:
            self.transport.reopen(target)
        if self._ack():
            return None
        for rate in self.connect_bitrates:
            if rate == target:
                continue
            self.transport.reopen(rate)
            if not self._ack():
                continue
            config = self.read_config()
            # The board acknowledges at the old rate, then switches.
            self._write_config(with_com_bitrate(config, target))
            self.transport.reopen(target)
            if not self._ack():
                raise RuntimeError(
                    "switched the board's COM port from {} to {} bit/s, and it "
                    "does not answer at {}".format(rate, target, target))
            return DiagItem(
                "bus/com_bitrate", DiagLevel.WARN,
                "Switched the board's COM port from {} to {} bit/s in RAM. "
                "Reported because this CHANGED the board; it reverts at "
                "power-off.".format(rate, target),
                {"from": str(rate), "to": str(target)})
        self.transport.reopen(target)
        raise RuntimeError(
            "board answers at none of {} bit/s. Check power, the cable, and "
            "that the adapter's LED is green (serial mode, not ICS)."
            .format(", ".join(str(r) for r in (target,) + self.connect_bitrates)))

    def _flatten_config(self) -> Optional[DiagItem]:
        """Put the board's config word back to flat, whatever a project set.

        The ROM program and vector jumps stopped, so the board's own program
        does not command the servos alongside this bus; the interpolation-end
        message off, so no unasked bytes come out of COM; the output cycle at
        the no-project 15 ms; the ICS rate at ``ics_bitrate``, the rate this
        config says its servos are set to; and the ICS switch on, without
        which no servo answers. A servo the project drives at
        another rate stops answering, and is reported missing like any other.

        Returns
        -------
        DiagItem or None
            A WARN item naming what was changed, else ``None``.
        """
        config = self.read_config()
        wanted = with_ics_bitrate(flat_config(config), self.ics_bitrate)
        wanted |= _CONFIG_ICS_SWITCH
        if wanted == config:
            return None
        changed = []
        if config & (CONFIG_RUN_ROM | CONFIG_VECTOR_JUMP):
            changed.append("stopped the ROM program and vector jumps")
        if config & CONFIG_SERVO_RESPONSE:
            changed.append("switched the interpolation-end message off")
        if config & FRAME_MASK != FLAT_FRAME:
            changed.append("set the output cycle to 15 ms")
        if config & ICS_BITRATE_MASK != wanted & ICS_BITRATE_MASK:
            changed.append("set the ICS rate to {} bit/s".format(self.ics_bitrate))
        if not config & _CONFIG_ICS_SWITCH:
            changed.append("switched the ICS port on")
        self._write_config(wanted)
        what = "; ".join(changed)
        return DiagItem(
            "bus/config", DiagLevel.WARN,
            "{} in RAM (config {:#06x} -> {:#06x}), so the board runs as with no "
            "project. Reported because this CHANGED the board; it reverts at "
            "power-off.".format(what[:1].upper() + what[1:], config, wanted),
            {"config": "{:#06x}".format(wanted)})

    def _read_present(self, ids: Sequence[int]) -> Dict[int, bool]:
        """Whether each servo's record holds a measured pulse."""
        present: Dict[int, bool] = {}
        for address, size in plan_record_reads(ids, ("current_angle",)):
            data = self._read_ram(address, size)
            for servo_id, values in decode_servo_records(
                    address, data, ids, ("current_angle",)).items():
                present[servo_id] = values["current_angle"] != 0
        return present

    def _probe_present(self, slots: Sequence[int]) -> Dict[int, bool]:
        """Clear the measured pulse of ``slots`` and see which the board
        rewrites, i.e. which have a servo answering."""
        for slot in slots:
            self._write_ram(servo_record_address(slot) + _ASSIGN_CLEAR_OFFSET,
                            b"\x00\x00")
        time.sleep(self.assign_settle_s)
        return self._read_present(slots) if slots else {}

    def _port_items(self) -> List[DiagItem]:
        """Where each servo was found, and any ID answering on both ports."""
        found = {
            self._joint_name(i): "ICS {} ({})".format(
                self._slot_of[i], port_name(self._slot_of[i]))
            for i in self._ids
            if self._slot_of[i] is not None and i not in self._ambiguous}
        items = [DiagItem(
            "bus/ics_ports", DiagLevel.OK,
            "Port of each answering servo, from which of its two ICS numbers "
            "the board keeps rewriting ({} moved since start).".format(
                self._port_moves),
            found)]
        if self._ambiguous:
            both = sorted(self._ambiguous)
            items.append(DiagItem(
                "bus/ics_port_conflict", DiagLevel.ERROR,
                "Two servos answer with the same ID, one on each port: {}. "
                "Nothing says which is the joint, so neither is driven. Give "
                "one of them another ID, or configure the joint by its ICS "
                "number without port_detect.".format(
                    "; ".join("{} (id {}: ICS {})".format(
                        self._joint_name(i), i,
                        " and ".join(str(s) for s in self._candidates[i]))
                        for i in both)),
                {"ids": ", ".join(str(i) for i in both)}))
        return items

    def _prepare_record(self, slot: int) -> None:
        """Set up ICS number ``slot``'s record before it is assigned.

        Head byte and servo ID (:data:`_RECORD_HEAD_OFFSET`), the measured
        pulse cleared so a value left from an earlier run is not taken for an
        answer, the reference OFF since the board sends it to the servo the
        moment the number is assigned, and no interpolation. Trim and mixings
        are left to :meth:`flatten_board`, which reports what it changes.
        """
        record = servo_record_address(slot)
        self._write_ram(record + _RECORD_HEAD_OFFSET,
                        bytes([_RECORD_HEAD, slot >> 1]))
        self._write_ram(record + _ASSIGN_CLEAR_OFFSET, _OFF_CLEAR)

    def _assign_ics(self, table: List[int]) -> Tuple[List[int], List[DiagItem]]:
        """Map every ICS number to its own record; assign the configured ones.

        A project may map an ICS number to some other record. This bus reads
        and drives each servo through its own record (``$0090`` + 20 x
        number), so any such entry -- configured or not, since two numbers
        sharing a record would drive one servo with the other's target -- is
        remapped to the number's own record, switched off.

        Returns
        -------
        assigned : list of int
            Ids this call assigned and found answering.
        items : list of DiagItem
            What was changed on the board.
        """
        items: List[DiagItem] = []
        new_table = list(table)
        remapped = [i for i in range(ICS_SLOT_COUNT)
                    if table[i] not in (ICS_UNASSIGNED, servo_record_address(i))]
        if remapped:
            for slot in remapped:
                self._prepare_record(slot)
                new_table[slot] = servo_record_address(slot)
            self._write_ram(ICS_ASSIGN_ADDRESS,
                            struct.pack("<{}H".format(ICS_SLOT_COUNT), *new_table))
            for servo_id in self._ids:
                if any(slot in remapped for slot in self._candidates[servo_id]):
                    self._servo_on[servo_id] = False
            items.append(DiagItem(
                "bus/ics_remap", DiagLevel.WARN,
                "Remapped ICS number(s) {} from the records the board's project "
                "gave them ({}) to their own, servo off, in RAM. Reported because "
                "this CHANGED the board; it reverts at power-off.".format(
                    remapped, ", ".join("{:#06x}".format(table[i]) for i in remapped)),
                {"remapped_ids": ", ".join(str(i) for i in remapped)}))
        unassigned = [i for i in self._slots if new_table[i] == ICS_UNASSIGNED]
        if not unassigned:
            return [], items

        for servo_id in unassigned:
            self._prepare_record(servo_id)
        for servo_id in unassigned:
            new_table[servo_id] = servo_record_address(servo_id)
        self._write_ram(ICS_ASSIGN_ADDRESS,
                        struct.pack("<{}H".format(ICS_SLOT_COUNT), *new_table))
        time.sleep(self.assign_settle_s)

        present = self._read_present(unassigned)
        answered = [i for i in unassigned if present.get(i)]
        silent = [i for i in unassigned if not present.get(i)]
        if silent and not self.hot_plug:
            # Leave the board as found for servos that are not there. With
            # hot_plug they stay assigned: the board polls only assigned
            # numbers, so unassigning one is never noticing it plugged in.
            for servo_id in silent:
                new_table[servo_id] = ICS_UNASSIGNED
            self._write_ram(ICS_ASSIGN_ADDRESS,
                            struct.pack("<{}H".format(ICS_SLOT_COUNT), *new_table))
        if answered:
            items.append(DiagItem(
                "bus/ics_assign", DiagLevel.WARN,
                "Assigned ICS number(s) {} in RAM: the board's project left them "
                "unassigned, so it was not driving them. Reported because this "
                "CHANGED the board; it reverts at power-off.".format(answered),
                {"assigned_ids": ", ".join(str(i) for i in answered)}))
        return answered, items

    def verify_servos(self, allow_missing: bool = False) -> List[DiagItem]:
        """Connect, check the firmware, run the board flat, assign servos.

        Called at startup and after every reconnect: every board setting it
        fixes lives in RAM, so a board that lost power needs it again.

        A servo the project assigned is taken as present; one this call
        assigned must have answered. The board keeps a project-assigned
        servo's last pulse in its record after the servo goes quiet, so that
        check is weaker -- it is what the board offers.

        Parameters
        ----------
        allow_missing : bool
            If False (default), configured ids that do not answer raise --
            unless the bus is ``hot_plug``, where a missing servo is expected.
            If True, they are returned as a WARN diagnostic instead.
        """
        allow_missing = allow_missing or self.hot_plug
        items: List[DiagItem] = []
        switched = self._connect()
        if switched is not None:
            items.append(switched)
        self.verify_firmware()
        flattened = self._flatten_config()
        if flattened is not None:
            items.append(flattened)
        items.extend(self._flatten_records())
        table = self.read_ics_assign_table()
        answered, assign_items = self._assign_ics(table)
        items.extend(assign_items)

        if self.port_detect:
            # A project-assigned record may hold a pulse from before, so
            # every number is cleared and has to be rewritten to count.
            present = self._probe_present(self._slots)
            self._slot_of = {i: None for i in self._ids}
            self._ambiguous = set()
            self._settling = {}
            self._resolve_ports(present)
            missing = [i for i in self._ids if self._slot_of[i] is None]
            items.extend(self._port_items())
            if self._ambiguous and not self.hot_plug:
                raise RuntimeError(items[-1].message)
        else:
            missing = [i for i in self._ids
                       if table[i] == ICS_UNASSIGNED and i not in answered]
        if self.hot_plug:
            # Start (or restart, after a reconnect) the presence picture from
            # what the records hold now.
            if not self.port_detect:
                present = self._read_present(self._slots) if self._slots else {}
            self._present = {i: bool(present.get(i)) for i in self._slots}
            self._cleared_at = {}
            self._rewritten = {}
            self._last_rewrite = {}
            self._absent_why = {}
            self._reads_since_check = 0
        self._stretch_set = set()
        stretched = self._apply_stretch(
            [i for i in self._stretch if i not in missing
             and (not self.hot_plug or self._present.get(self._drive_slot(i)))])
        if stretched is not None:
            items.append(stretched)
        if missing:
            message = (
                "Configured servos did not answer: {}. Check the servo ids, "
                "wiring, power to the servos, and that they are set to {} "
                "bit/s.".format(
                    "; ".join("{} (id {})".format(self._joint_name(i), i)
                              for i in missing),
                    self.ics_bitrate))
            if not allow_missing:
                raise RuntimeError(message)
            items.append(DiagItem(
                "bus/startup_missing", DiagLevel.WARN, message,
                {"missing_ids": ", ".join(str(i) for i in missing)}))
        items.extend(self._arm_failsafe())
        return items

    def failsafe_frames(self) -> List[Frame]:
        """What a relay on the link sends if the host goes silent: the wheels
        stopped, in one packet. Arm joints are left out, so they keep holding
        their last pose (released, an arm would fall)."""
        wheels = sorted({self._drive_slot(a.id) for a in self.actuators
                         if getattr(a, "expects_rotation", False)} - {None})
        if not wheels:
            return []
        return [Frame(payload=build_multi_servo(
            wheels, [CENTER_PULSE] * len(wheels),
            self._velocity_for(self.actuators[0])))]

    def _arm_failsafe(self) -> List[DiagItem]:
        """Disarm a relay's failsafe now; arm it on the first command cycle.

        Not armed here: after startup the servicer waits for ros2_control's
        segment, and that wait is silence the relay would answer with a
        failsafe on every start -- a warning that fires on every healthy run
        is one people learn to ignore. Nothing drives the wheels before the
        first command cycle, so nothing needs stopping before it either.
        Disarmed explicitly, because a relay keeps what a previous run (or a
        reconnect) gave it.
        """
        if not self.transport.set_failsafe([]):
            return []
        frames = self.failsafe_frames()
        self._failsafe_pending = True
        return [DiagItem(
            "bus/failsafe", DiagLevel.OK,
            "The link has a relay; from the first command cycle it holds {} "
            "failsafe packet(s): {}".format(
                len(frames), "the wheels stop if the host goes silent"
                if frames else "nothing to stop (no wheels)"),
            {"packets": str(len(frames))})]

    def _flatten_records(self) -> List[DiagItem]:
        """Zero the board's trim and mixings on every ICS number.

        Both change what a servo is driven to behind the host's back. A trim
        shifts what the board reports and what it drives by the same pulse
        count, so a joint commanded to 0 sits ``trim`` away from the servo's
        neutral while reading 0 -- 67.5 deg on one humanoid's left hinge, left there
        by a project in the board's ROM. A mixing adds another value (a gyro,
        an analog input, another servo) times a ratio to the drive every
        cycle. Joint zeros belong in this package's config (``offset_deg``),
        where they can be read, so the board's are removed. A mixing is
        disabled by its ratio := 0, which makes it add nothing whatever its
        source. The servo is switched off first: with its trim gone, a held
        reference would be driven ``trim`` away at once.

        Every number, not only the ones this config polls: a module plugged in
        later may be configured later, and its servo would come up with what
        the project left on its number.

        RAM only; the project restores them at the next power-up, and this
        runs again at every connect.
        """
        trims = {}
        mixings = {}
        for slot in range(ICS_SLOT_COUNT):
            record = servo_record_address(slot)
            data = self._read_ram(record, SERVO_RECORD_SIZE)
            trim = struct.unpack_from("<h", data, _TRIM_OFFSET)[0]
            active = [(offset, struct.unpack_from("<HB", data, offset))
                      for offset in _MIXING_OFFSETS]
            active = [(offset, source, ratio) for offset, (source, ratio) in active
                      if source != _MIXING_NO_SOURCE and ratio != 0]
            if trim == 0 and not active:
                continue
            self._write_ram(record + _REFERENCE_OFFSET, _OFF_CLEAR[2:])
            if trim != 0:
                self._write_ram(record + _TRIM_OFFSET, b"\x00\x00")
                trims[slot] = trim
            for offset, source, ratio in active:
                self._write_ram(record + offset + 2, b"\x00")
                mixings.setdefault(slot, []).append((source, ratio))
        changed = set(trims) | set(mixings)
        for servo_id in self._ids:
            if any(slot in changed for slot in self._candidates[servo_id]):
                self._servo_on[servo_id] = False
        items = []
        if trims:
            items.append(DiagItem(
                "bus/trim", DiagLevel.WARN,
                "Cleared the board's trim in RAM on {} and switched those servos "
                "off: the board shifts what it reports and drives by its trim, so "
                "the joints' zeros were not the servos' neutrals. Joint zeros "
                "belong in offset_deg. Reported because this CHANGED the board; "
                "the board's project restores them at power-up.".format(
                    ", ".join("ICS {} (trim {:+d} pulses, {:+.1f} deg)".format(
                        slot, trim, trim / PULSE_PER_DEG)
                        for slot, trim in sorted(trims.items()))),
                {"cleared": ", ".join("{}:{}".format(slot, trim)
                                      for slot, trim in sorted(trims.items()))}))
        if mixings:
            where = ", ".join(
                "ICS {} (source {})".format(slot, ", ".join(
                    "{:#06x} ratio {:#04x}".format(source, ratio)
                    for source, ratio in found))
                for slot, found in sorted(mixings.items()))
            items.append(DiagItem(
                "bus/mixing", DiagLevel.WARN,
                "Disabled the board's mixing in RAM (ratio := 0) on {} and "
                "switched those servos off: the board adds a mixing to what it "
                "drives every cycle. Reported because this CHANGED the board; "
                "the board's project restores them at power-up.".format(where),
                {"disabled": ", ".join(str(slot) for slot in sorted(mixings))}))
        return items

    def diagnose(self) -> List[DiagItem]:
        """Link health: the rate in use and the read counters."""
        values = {
            "com_bitrate": str(self.transport.current_bitrate),
            "missed": str(self._missed),
            "naks": str(self._naks),
            "state_updates": str(self._state_updates),
            "held_readings": str(self._held_readings),
            "dropped_readings": str(self._dropped_readings),
        }
        try:
            config = self.read_config()
        except ProtocolError as error:
            values["error"] = str(error)
            return [DiagItem(
                "bus/link", DiagLevel.ERROR,
                "Board did not answer a config read; check power and cable.",
                values)]
        values["config"] = "{:#06x}".format(config)
        try:
            values["config_com_bitrate"] = str(com_bitrate_of(config))
        except ProtocolError as error:
            values["error"] = str(error)
        items = [DiagItem("bus/link", DiagLevel.OK, "Board answering.", values),
                 self._diagnose_power()]
        if self.port_detect:
            items.extend(self._port_items())
        if self._stretch:
            items.append(self._diagnose_stretch())
        if self._relax_config:
            items.append(self._diagnose_relax())
        return items

    def _diagnose_stretch(self) -> DiagItem:
        """``bus/stretch``: the configured stretch, and where it is not set."""
        values = {"failures": str(self._stretch_failures),
                  "configured": ", ".join("{}={}".format(self._joint_name(i), v)
                                          for i, v in sorted(self._stretch.items()))}
        unset = [i for i in self._stretch
                 if self.present(i) and self._drive_slot(i) not in self._stretch_set]
        if unset:
            values["unset"] = ", ".join(self._joint_name(i) for i in unset)
            return DiagItem(
                "bus/stretch", DiagLevel.WARN,
                "Stretch not set on {}: they hold at the servo's own stretch."
                .format(values["unset"]), values)
        return DiagItem("bus/stretch", DiagLevel.OK,
                        "Configured stretch set on every answering servo.", values)

    def _diagnose_relax(self) -> DiagItem:
        """``bus/hold_relax``: which servos are held lighter than commanded."""
        values = {"times_relaxed": str(self._relax_count)}
        now = self._clock()
        for servo_id in sorted(self._relax_config):
            since = self._relax_since.get(servo_id)
            state = self._states.get(servo_id)
            values["watch_" + self._joint_name(servo_id)] = (
                "not watched" if since is None else
                "command {:+.2f} deg for {:.1f} s, at {}".format(
                    math.degrees(since[0]), now - since[1],
                    "?" if state is None or state.position is None
                    else "{:+.2f} deg".format(math.degrees(state.position))))
        if not self._relaxed:
            return DiagItem("bus/hold_relax", DiagLevel.OK,
                            "No servo is held lighter than commanded.", values)
        values["relaxed"] = ", ".join(self._joint_name(i) for i in sorted(self._relaxed))
        return DiagItem(
            "bus/hold_relax", DiagLevel.OK,
            "Held lighter than commanded: {}.".format("; ".join(
                "{} commanded {:+.1f} deg, held at {:+.1f} deg".format(
                    self._joint_name(i), math.degrees(c), math.degrees(h))
                for i, (c, h) in sorted(self._relaxed.items()))),
            values)

    def _diagnose_power(self) -> DiagItem:
        """``bus/power``, in the same shape as the ARM-H7 bus reports it."""
        try:
            voltage_v = self.battery_voltage()
        except ProtocolError as error:
            return DiagItem(
                "bus/power", DiagLevel.STALE,
                "Battery voltage could not be read this cycle; no voltage "
                "claim is made.", {"error": str(error)})
        values = {"battery_voltage_v": "{:.3f}".format(voltage_v)}
        if self._battery_voltage_warn_below is None:
            return DiagItem(
                "bus/power", DiagLevel.OK,
                "Battery voltage measured; no low-voltage threshold is "
                "configured, so no low-voltage level is claimed.", values)
        values["warn_below_v"] = "{:.3f}".format(self._battery_voltage_warn_below)
        if voltage_v < self._battery_voltage_warn_below:
            return DiagItem(
                "bus/power", DiagLevel.WARN,
                "Battery voltage is below the configured warning threshold; "
                "check supply and load.", values)
        return DiagItem(
            "bus/power", DiagLevel.OK,
            "Battery voltage is above the configured warning threshold.", values)
