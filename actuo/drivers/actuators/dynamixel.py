"""DYNAMIXEL servo -- one model-agnostic actuator plugin, driven by a data table.

There is deliberately no per-model class here, and no ``if model == ...``. The
control-table addresses, resolution, units and available modes all arrive as a
:class:`~actuo.drivers.dynamixel_models.DynamixelModel` loaded from YAML, so
supporting a new servo is a row of data with an e-Manual URL. That is the same
discipline the CubeMars plugin follows for ``pole_pairs``/``kt``, and it is the
direct fix for an earlier design that hardcoded ``4096/360``, centre ``2048``
in a layer that claimed to be generic.

What this class owns: units, calibration, capabilities, and the bytes of a
single servo's command/status. What it does not know: whether the wire is a
DXHUB's FT232H, some other USB-serial adapter, or a loopback in a test.

Three traps, all of which produce plausible-looking wrong numbers
----------------------------------------------------------------
* **Travel is not a revolution.** An AX-12A has 1024 counts across 300 deg. Its
  counts-per-*revolution* is 1228.8. Using 1024 scales every angle by 0.83 and
  still looks like a robot pose.
* **Protocol 1.0 does not use two's complement** for present velocity/load: bit
  10 is a direction flag and bits 9..0 are magnitude. Decoding as signed gives
  a sign that flips in the wrong place and speeds that are wrong near zero.
* **Little-endian**, unlike the Feetech SCS servos on the neighbouring plugin.

Capabilities are computed, not declared
---------------------------------------
A mode listed in the model table is only offered if the numbers needed to
*honour* it are also present: VELOCITY needs a documented rev/min per LSB,
CURRENT needs a torque constant, PWM needs a documented full scale. A mode
whose unit is unknown is withheld and :meth:`unsupported_reason` says exactly
which field is missing -- rather than being emulated with a guessed scale,
which would make a broken mode look like a working one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from actuo.core.contracts import (
    Actuator, ActuatorCommand, ActuatorState, Calibration, Capabilities,
    CapabilityError, ChannelKind, ChannelSpec, Frame, OperatingMode)
from actuo.core.registry import register_actuator

from actuo.drivers import dynamixel_protocol as dxl
from actuo.drivers.dynamixel_models import DIRECTION_BIT_10, DynamixelModel

#: Protocol 1.0 packs a direction flag into bit 10 of velocity and load.
_DIRECTION_BIT = 1 << 10
_MAGNITUDE_MASK = _DIRECTION_BIT - 1

#: rad/s <-> rev/min.
_RPM_PER_RAD_S = 60.0 / (2.0 * math.pi)

#: X-series (protocol 2.0) Hardware Error Status bits (control table address
#: 70), per https://emanual.robotis.com/docs/en/dxl/x/xl430-w250/#hardware-error-status70.
#: Bits 1, 6, 7 are unused/reserved on this family.
#:
#: This is a DIFFERENT thing from the status-PACKET error byte every reply
#: already carries (``dxl.ERROR_ALERT_BIT`` etc.): the packet byte says "check
#: Hardware Error Status", this register says WHICH bit is set. Distinguishing
#: them is the entire point -- "error code 128" (the Alert bit, decimal) is what
#: this driver reported for a genuine Overload Error on 2026-07-26, and gave
#: the operator nothing to act on beyond a stack trace.
HARDWARE_ERROR_BITS = {
    0x01: "input_voltage",
    0x04: "overheating",
    0x08: "motor_encoder",
    0x10: "electrical_shock",
    0x20: "overload",
}


@dataclass
class DynamixelCalibration(Calibration):
    """Per-servo calibration. ``counts_per_rev`` and ``zero_offset_counts``
    come from the model row unless the robot config overrides them, so a joint's
    zero is expressed once, in the robot's own file.

    ``homing_offset_counts`` is a DYNAMIXEL-specific hardware fact, not a
    generic actuator concept -- it belongs here, not in the shared core
    ``Calibration``, which knows nothing of any one protocol. It is the value written to this
    servo's EEPROM Homing Offset register (control table address 20), if
    any; 0.0 for every servo that has never had it touched (the default).
    It shifts the valid window `position_refusal` checks a goal against --
    see that method's docstring for why this exists and what happens
    without it."""

    counts_per_rev: float = 4096.0
    zero_offset_counts: float = 2048.0
    homing_offset_counts: float = 0.0


@register_actuator("dynamixel")
class DynamixelActuator(Actuator):
    """One DYNAMIXEL servo of any model the table knows.

    Parameters
    ----------
    actuator_id : int
        The servo's bus id (0..252). This bus is single-drop-per-id, so the id
        is the whole address -- unlike a gateway board, where the address is
        (port, id). Which shape an address has is deliberately the actuator's
        business, never a widened type in the core (actuator ids stay opaque
        and are never narrowed at a shared layer).
    calibration : DynamixelCalibration
    model : DynamixelModel
        Loaded from ``data/dynamixel_models.yaml``.
    profile_velocity : int, optional
        Protocol 2.0 only: the servo-side trapezoid velocity limit, in the
        model's raw units. ``None`` leaves whatever the servo already has.
    """

    def __init__(self, actuator_id: int, calibration: DynamixelCalibration,
                 model: DynamixelModel,
                 profile_velocity: Optional[int] = None) -> None:
        super().__init__(actuator_id, calibration)
        self.model = model
        self.profile_velocity = profile_velocity
        self._modes, self._withheld = self._resolve_modes(model, calibration)

    # --- capabilities --------------------------------------------------------
    @staticmethod
    def _resolve_modes(model: DynamixelModel, calibration: Calibration
                       ) -> Tuple[Tuple[OperatingMode, ...], Dict[OperatingMode, str]]:
        """Offer only the modes whose unit conversion is actually known."""
        offered: List[OperatingMode] = []
        withheld: Dict[OperatingMode, str] = {}
        for mode in model.modes:
            if mode is OperatingMode.VELOCITY and model.velocity_unit_rpm is None:
                withheld[mode] = (
                    "model {} has no documented velocity_unit_rpm, so rad/s "
                    "cannot be converted to a wire value".format(model.name))
            elif mode is OperatingMode.CURRENT and (
                    calibration.torque_constant is None
                    and model.torque_constant is None):
                withheld[mode] = (
                    "model {} has no torque_constant (N*m/A), so a torque "
                    "command has no wire value".format(model.name))
            elif mode is OperatingMode.PWM:
                withheld[mode] = (
                    "PWM full scale is not transcribed for model {}; a duty "
                    "cycle cannot be scaled to goal_pwm counts".format(model.name))
            else:
                offered.append(mode)
        if not offered:
            raise CapabilityError(
                "model {} ends up with no usable mode: {}".format(
                    model.name, "; ".join(withheld.values())))
        return tuple(offered), withheld

    def unsupported_reason(self, mode: OperatingMode) -> str:
        """Why a mode the servo physically has is not offered here."""
        if mode in self._modes:
            return ""
        if mode in self._withheld:
            return self._withheld[mode]
        return "model {} does not have mode {!r} at all".format(
            self.model.name, mode.value)

    def require_mode(self, mode: OperatingMode) -> None:
        """Capability check whose message names the missing datum, not just the
        mode. "unsupported" sends a user hunting hardware; "no documented
        velocity_unit_rpm" tells them which YAML field to fill in."""
        if mode not in self._modes:
            raise CapabilityError("id {} ({}): {}".format(
                self.id, self.model.name, self.unsupported_reason(mode)))

    @property
    def capabilities(self) -> Capabilities:
        table = self.model.table
        return Capabilities(
            modes=self._modes,
            has_position_sense=True,
            has_velocity_sense=("present_velocity" in table.present_block
                                and self.model.velocity_unit_rpm is not None),
            has_current_sense=(self.model.current_unit_ma is not None
                               and self.torque_constant is not None),
            has_temperature_sense=self.model.has_temperature_sense,
            multiturn=False,
        )

    @property
    def torque_constant(self) -> Optional[float]:
        """N*m per amp: the robot config wins over the model default."""
        if self.calibration.torque_constant is not None:
            return self.calibration.torque_constant
        return self.model.torque_constant

    @property
    def channels(self) -> Tuple[ChannelSpec, ...]:
        specs = [ChannelSpec("position", "rad", ChannelKind.COMMAND),
                 ChannelSpec("position", "rad", ChannelKind.STATE)]
        caps = self.capabilities
        if caps.has_velocity_sense:
            specs.append(ChannelSpec("velocity", "rad/s", ChannelKind.STATE))
        if caps.has_current_sense:
            specs.append(ChannelSpec("current", "N*m", ChannelKind.STATE))
        if caps.has_temperature_sense:
            specs.append(ChannelSpec("temperature", "degC", ChannelKind.STATE))
        return tuple(specs)

    # --- units ---------------------------------------------------------------
    @property
    def protocol(self) -> int:
        return self.model.protocol

    def position_refusal(self, position_rad: float) -> Optional[str]:
        """Why this angle cannot be commanded, or None. See the base contract.

        Two independent reasons, and the second one is the one that bit:

        1. Outside the taught envelope, when the robot config supplied one.
        2. **Outside the range a goal_position can express.** The servo's goal
           register holds one turn of counts, and only starts at 0 when its
           EEPROM Homing Offset is 0 (the factory default). Homing Offset ADDS to the raw absolute
           encoder value to produce present_position (and goal_position uses
           the same shifted coordinate space), so a servo whose Homing Offset
           was written to move its wrap seam -- see dynamixel_models.yaml's
           homing_offset register comment -- has its valid window shifted by
           that same amount, not fixed at 0..resolution-1. Getting this wrong
           has two, independent-looking failure modes on record: an angle
           that converts outside 0..resolution-1 is not a far-away target, it
           is a DIFFERENT target the servo takes modulo its range (2026-07-26,
           -6.7058 rad -> raw 4392, commanded "hold where it is", the servo
           read ~296 and the joint slammed ~340 degrees); and, found while
           bringing up a trained policy on this same joint after ITS Homing
           Offset had been shifted by +512 to fix exactly that, checking
           against the un-shifted window refused a perfectly valid,
           reachable pose (raw 4244, comfortably inside the shifted
           512..4607) because the code did not know the window had moved.

        A joint that reaches the first failure mode does not need a wider
        bound -- it needs its `offset_deg` taught, so its travel lands inside
        one turn. A joint whose Homing Offset was deliberately shifted needs
        that fact recorded in its config (`homing_offset_counts`), not a
        wider bound either -- the window is still exactly one turn wide, it
        just does not start at 0 for this specific servo.
        """
        if self.calibration.position_limit is not None:
            low, high = self.calibration.position_limit
            if position_rad < low or position_rad > high:
                return ("id {}: {:.4f} rad is outside the taught limit "
                        "[{:.4f}, {:.4f}]".format(
                            self.id, position_rad, low, high))
        if self.protocol == dxl.PROTOCOL_1:
            # Protocol 1.0 is left to :meth:`position_counts`'s clamp. Its
            # travel is a fraction of a turn against hard stops, so an
            # over-range count cannot wrap into a different pose the way a
            # full-turn 2.0 goal does -- the failure this refusal exists for
            # does not arise. Not changed here on principle: this fix came out
            # of a 2.0 fault, and quietly altering an untested path while
            # fixing a tested one is how a second incident gets made.
            return None
        raw = int(round(self.calibration.rad_to_counts(position_rad)))
        offset = self.calibration.homing_offset_counts
        low_raw, high_raw = offset, offset + self.model.resolution - 1
        if not low_raw <= raw <= high_raw:
            return (
                "id {}: {:.4f} rad converts to goal count {}, outside the "
                "{:.0f}..{:.0f} a {} with homing_offset_counts={:.0f} can "
                "express -- the servo would take it modulo its range and "
                "drive somewhere else. Teach this joint's offset_deg so its "
                "travel lands inside one turn (or, if this servo's Homing "
                "Offset register was deliberately shifted, check "
                "homing_offset_counts matches it).".format(
                    self.id, position_rad, raw, low_raw, high_raw,
                    self.model.name, offset))
        return None

    def position_counts(self, position_rad: float) -> int:
        """Convert a commanded angle, refusing anything it cannot express.

        Refuses rather than clamps. A clamp turns "the trajectory asked for
        something impossible" into "the arm stopped somewhere near the end and
        nobody knows why"; the taught limit exists to be a wall, not a silent
        saturation (this is the same rule the Feetech and CubeMars plugins use).

        The one exception is protocol 1.0, which is clamped below rather than
        refused -- kept because a 1.0 servo's travel is a fraction of a turn and
        its goal register is unsigned, so the arithmetic can land one count out
        at the ends. It is guarded by the same refusal first.
        """
        refusal = self.position_refusal(position_rad)
        if refusal is not None:
            raise ValueError(refusal)
        raw = int(round(self.calibration.rad_to_counts(position_rad)))
        if self.protocol == dxl.PROTOCOL_1:
            # A 1.0 servo's goal_position is unsigned and bounded by its travel;
            # a negative or over-range count wraps into a wildly different pose.
            return max(0, min(self.model.resolution - 1, raw))
        return raw

    def velocity_counts(self, velocity_rad_s: float) -> int:
        """rad/s at the output shaft -> the model's raw velocity unit."""
        if self.model.velocity_unit_rpm is None:
            raise CapabilityError(
                "id {} ({}): no documented velocity_unit_rpm".format(
                    self.id, self.model.name))
        motor_rpm = (velocity_rad_s * self.calibration.direction
                     * self.calibration.gear_ratio * _RPM_PER_RAD_S)
        return int(round(motor_rpm / self.model.velocity_unit_rpm))

    def _decode_signed(self, raw: int, size: int) -> int:
        """Apply the table's sign convention to a raw velocity/load word."""
        if self.model.table.signed_convention == DIRECTION_BIT_10:
            magnitude = raw & _MAGNITUDE_MASK
            return -magnitude if raw & _DIRECTION_BIT else magnitude
        limit = 1 << (8 * size - 1)
        return raw - (1 << (8 * size)) if raw >= limit else raw

    # --- commands ------------------------------------------------------------
    def goal_register(self, mode: OperatingMode) -> str:
        """Which control-table register carries a command in ``mode``."""
        self.require_mode(mode)
        if mode is OperatingMode.POSITION:
            return "goal_position"
        if mode is OperatingMode.VELOCITY:
            return "goal_velocity"
        if mode is OperatingMode.CURRENT:
            return "goal_current"
        return "goal_pwm"

    def goal_payload(self, command: ActuatorCommand) -> Tuple[str, bytes]:
        """The register name and its bytes, for the bus to pack into a SYNC_WRITE.

        Returned as (register, bytes) rather than a whole frame because a
        sync-write addresses one register for many servos: the bus groups by
        this name, which is how a mixed-model bus stays a single transaction per
        group instead of a loop of singles.
        """
        register = self.goal_register(command.mode)
        entry = self.model.table[register]
        if command.mode is OperatingMode.POSITION:
            if command.position is None:
                raise ValueError("POSITION mode needs command.position (rad)")
            value = self.position_counts(command.position)
        elif command.mode is OperatingMode.VELOCITY:
            if command.velocity is None:
                raise ValueError("VELOCITY mode needs command.velocity (rad/s)")
            value = self.velocity_counts(command.velocity)
        elif command.mode is OperatingMode.CURRENT:
            if command.current is None:
                raise ValueError("CURRENT mode needs command.current (N*m)")
            amps = command.current / float(self.torque_constant)
            value = int(round(amps * 1000.0 / self.model.current_unit_ma))
        else:  # pragma: no cover - PWM is withheld by _resolve_modes today
            raise CapabilityError(self.unsupported_reason(command.mode))
        return register, dxl.to_le(value, entry.size)

    def encode_command(self, command: ActuatorCommand) -> Frame:
        register, data = self.goal_payload(command)
        payload = dxl.build_write(self.protocol, self.id,
                                  self.model.table[register].address, data)
        return Frame(payload=payload, address=self.id)

    def encode_mode(self, mode: OperatingMode) -> List[Frame]:
        """Frames that put the servo into ``mode``.

        On protocol 2.0 the operating mode register only accepts a write while
        torque is disabled, so torque-off is part of the transition rather than
        something the caller is expected to remember. Protocol 1.0 servos in
        this table have a single mode and need no frames at all.
        """
        self.require_mode(mode)
        value = self.model.modes[mode]
        if value is None or not self.model.table.has("operating_mode"):
            return []
        return [self.encode_torque_enable(False),
                Frame(payload=dxl.build_write(
                    self.protocol, self.id,
                    self.model.table["operating_mode"].address,
                    dxl.to_le(value, self.model.table["operating_mode"].size)),
                    address=self.id)]

    def encode_torque_enable(self, enabled: bool) -> Frame:
        entry = self.model.table["torque_enable"]
        payload = dxl.build_write(self.protocol, self.id, entry.address,
                                  bytes([1 if enabled else 0]))
        return Frame(payload=payload, address=self.id)

    def encode_release(self) -> Frame:
        """Servo off: torque_enable = 0, so the joint is back-drivable.

        A real register write, not "stop sending commands" -- a DYNAMIXEL holds
        its last goal position indefinitely, so silence is not release.
        """
        return self.encode_torque_enable(False)

    def encode_profile_velocity(self) -> List[Frame]:
        """Optional servo-side speed limit; empty when the config sets none."""
        if self.profile_velocity is None or not self.model.table.has(
                "profile_velocity"):
            return []
        entry = self.model.table["profile_velocity"]
        return [Frame(payload=dxl.build_write(
            self.protocol, self.id, entry.address,
            dxl.to_le(int(self.profile_velocity), entry.size)), address=self.id)]

    # --- hardware fault diagnosis ---------------------------------------------
    #
    # Deliberately NOT part of encode_query()/decode_state(): those decode a
    # FIXED-length cyclic block (table.block_length), and Hardware Error Status
    # (one byte at address 70) sits nowhere near it on the X-series map (it is
    # not in `present_block`, on purpose -- see the yaml). Folding it into the
    # cyclic read would mean reading everything in between too, for a register
    # that is 0x00 the overwhelming majority of the time. So it is a separate,
    # explicit, one-shot query -- issued by the bus only when the cheap signal
    # (the status packet's Alert bit) says it is worth asking.
    @property
    def has_hardware_error_status(self) -> bool:
        return self.model.table.has("hardware_error_status")

    def encode_hardware_error_query(self) -> List[Frame]:
        """READ the Hardware Error Status register, if this model has one."""
        if not self.has_hardware_error_status:
            raise CapabilityError(
                "id {} ({}): no documented hardware_error_status register -- "
                "add it to dynamixel_models.yaml with its e-Manual address "
                "before asking for it".format(self.id, self.model.name))
        entry = self.model.table["hardware_error_status"]
        payload = dxl.build_read(self.protocol, self.id, entry.address,
                                 entry.size)
        return [Frame(payload=payload, address=self.id)]

    def decode_hardware_error_status(self, frame: Frame) -> int:
        """Parse a reply to :meth:`encode_hardware_error_query` into the raw byte."""
        parsed = dxl.parse_status(self.protocol, frame.payload)
        if parsed is None or parsed[0] != self.id:
            raise ValueError(
                "id {}: frame is not this servo's status".format(self.id))
        entry = self.model.table["hardware_error_status"]
        if len(parsed[2]) < entry.size:
            raise ValueError(
                "id {}: short hardware_error_status reply ({} of {} bytes)"
                .format(self.id, len(parsed[2]), entry.size))
        return dxl.from_le(parsed[2][:entry.size])

    @staticmethod
    def hardware_error_names(byte: int) -> List[str]:
        """Named bits set in a Hardware Error Status byte. Empty = healthy.

        An unrecognised bit is reported as ``bit<N>`` rather than dropped --
        the whole reason this exists is to never let a real fault go unnamed
        because the table did not happen to list it.
        """
        names = [name for bit, name in sorted(HARDWARE_ERROR_BITS.items())
                if byte & bit]
        unnamed = byte & ~sum(HARDWARE_ERROR_BITS)
        names.extend("bit{}".format(i) for i in range(8) if unnamed & (1 << i))
        return names

    def fault_name(self, code: int) -> str:
        code = int(code)
        if code in ERROR_NAMES:
            return ERROR_NAMES[code]
        decoded = self.hardware_error_names(code)
        if decoded:
            return ", ".join(decoded)
        return "unknown fault code {}".format(code)

    # --- reads ---------------------------------------------------------------
    def encode_query(self) -> List[Frame]:
        """READ the cyclic block. This bus is request/response; nothing streams."""
        table = self.model.table
        payload = dxl.build_read(self.protocol, self.id,
                                 table.block_address, table.block_length)
        return [Frame(payload=payload, address=self.id)]

    def matches(self, frame: Frame) -> bool:
        """Whether ``frame`` is this servo's status packet.

        Protocol 2.0 status packets carry instruction 0x55, so an echoed command
        is rejected structurally by ``parse_status``. Protocol 1.0 has no such
        field -- a status packet and an instruction packet have the same shape --
        so the parameter count is checked against the block length as well. That
        is a weaker guard than 2.0's, which is why ``actuo-dxl-probe`` explicitly
        tests the adapter for echo: this stack has twice been bitten by decoding
        a frame it sent itself (once as an apparent CubeMars "encoder fault" that
        was really an echoed command).
        """
        parsed = dxl.parse_status(self.protocol, frame.payload)
        if parsed is None or parsed[0] != self.id:
            return False
        if self.protocol == dxl.PROTOCOL_1:
            return len(parsed[2]) == self.model.table.block_length
        return True

    def decode_state(self, frame: Frame) -> ActuatorState:
        parsed = dxl.parse_status(self.protocol, frame.payload)
        if parsed is None or parsed[0] != self.id:
            raise ValueError(
                "id {}: frame is not this servo's status".format(self.id))
        return self.decode_block(parsed[2], error=parsed[1])

    def decode_block(self, data: bytes, error: int = 0) -> ActuatorState:
        """Decode the cyclic read block. Also used by the bus for SYNC_READ,
        where the bytes arrive without a per-servo status packet around them."""
        table = self.model.table
        if len(data) < table.block_length:
            raise ValueError("id {}: short status block ({} of {} bytes)".format(
                self.id, len(data), table.block_length))

        def field(name: str) -> Optional[int]:
            if name not in table.present_block:
                return None
            offset = table.block_offset(name)
            return dxl.from_le(data[offset:offset + table[name].size])

        aux: Dict[str, float] = {}

        raw_position = field("present_position")
        if raw_position is None:
            raise ValueError(
                "control table {!r} does not read present_position each cycle; "
                "a state with no measured angle is not a state".format(table.name))
        if self.protocol == dxl.PROTOCOL_1:
            position_counts = float(raw_position)
        else:
            position_counts = float(self._decode_signed(
                raw_position, table["present_position"].size))
        aux["raw_position"] = position_counts
        position = self.calibration.counts_to_rad(position_counts)

        velocity = None
        raw_velocity = field("present_velocity")
        if raw_velocity is not None:
            counts = self._decode_signed(
                raw_velocity, table["present_velocity"].size)
            aux["raw_velocity"] = float(counts)
            if self.model.velocity_unit_rpm is not None:
                rpm = counts * self.model.velocity_unit_rpm
                velocity = (rpm / _RPM_PER_RAD_S * self.calibration.direction
                            / self.calibration.gear_ratio)

        current = None
        raw_load = field("present_load")
        if raw_load is not None:
            counts = self._decode_signed(raw_load, table["present_load"].size)
            if self.model.current_unit_ma is not None:
                amps = counts * self.model.current_unit_ma / 1000.0
                aux["current_a"] = amps
                if self.torque_constant is not None:
                    current = amps * self.torque_constant
            elif self.model.load_unit_percent is not None:
                # Load is a fraction of rated torque, NOT a current and NOT a
                # torque in N*m. It is reported under its own name so nobody can
                # mistake it for one (an earlier design reported voltage as
                # temperature exactly this way).
                aux["load_percent"] = counts * self.model.load_unit_percent
            else:
                aux["raw_load"] = float(counts)

        voltage = None
        raw_voltage = field("present_voltage")
        if raw_voltage is None:
            raw_voltage = field("present_input_voltage")
        if raw_voltage is not None and self.model.voltage_divisor is not None:
            voltage = raw_voltage / self.model.voltage_divisor

        temperature = None
        raw_temperature = field("present_temperature")
        if raw_temperature is not None:
            temperature = float(raw_temperature)

        return ActuatorState(
            position=position,
            velocity=velocity,
            current=current,
            temperature=temperature,
            voltage=voltage,
            error=error or None,
            aux=aux,
        )


def _all_hardware_error_names() -> Dict[int, str]:
    """{byte value: name}, for every value Hardware Error Status can take.

    Built from :data:`HARDWARE_ERROR_BITS` by enumeration rather than hand
    listed, because a fault is not always one bit -- Overheating and Overload
    can both be set -- and hand-writing 2**5 combinations is exactly how one of
    them silently goes unnamed the day it is the one that matters.
    """
    names: Dict[int, str] = {0: "none"}
    for value in range(1, 256):
        parts = DynamixelActuator.hardware_error_names(value)
        if parts:
            names[value] = ", ".join(parts)
    # The bare packet-level Alert bit, for when it fired but the follow-up
    # read that would explain it could not be done (transport hiccup, or a
    # model with no hardware_error_status row). Safe as a special case: bits
    # 6/7 are unused on this family, so a genuine Hardware Error Status value
    # can never equal 0x80 and collide with this entry.
    names[dxl.ERROR_ALERT_BIT] = "hardware fault (detail unavailable)"
    return names


#: Passed to ``ServoSupervisor(error_names=...)`` for a DYNAMIXEL bus -- the
#: same role ``cubemars.ERROR_NAMES`` plays for a CAN bus. This dict, plus the
#: bus writing the decoded byte into ``ActuatorState.error`` (see
#: ``DynamixelBus._diagnose_hardware_faults``), is what makes an
#: Overload/Overheating/etc fault show up as a name in the servo state a ROS 2
#: layer publishes, instead of an opaque "code 128" -- the same visibility the
#: CubeMars plugin gives CAN faults.
ERROR_NAMES = _all_hardware_error_names()
