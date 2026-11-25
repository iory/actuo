"""FEETECH STS / SMS / HLS serial bus servo (STS3215, HLS3960M, ...).

The register map, the sign-magnitude encoding and the current scale come from
feetech-cli (``feetech_cli.registers``, ``feetech_cli.protocol``), the vendor
library this plugin sits on; framing is ``actuo.drivers.feetech_protocol``.
This module owns what feetech-cli does not: units in radians, calibration,
which modes a model offers, and how a command becomes register bytes.

Two series share the STS control table but not every register:

* **STS / SMS** (STS3215, model 777): ``goal_position`` at 42; in operating
  mode 2 (open-loop PWM) ``goal_time`` at 44 is the duty in 0.1 %, sign in bit
  10, capped by ``torque_limit`` (checked on an STS3215, firmware 3.10).
* **HLS** (HLS3950M / HLS3960M, model 4618): 44 is the goal *current* (6.5 mA
  per count) a position command carries, and 46 its speed (0.732 rpm per
  count); FEETECH's ``HLSCL::WritePosEx`` writes acceleration, position,
  current and speed as one 7-byte block from 41
  (https://github.com/ftservo/FTServo_Arduino/blob/main/src/HLSCL.cpp). An HLS
  position command written the STS way leaves 0 in 44 and the servo limp with
  torque on. Its operating mode 2 is constant current, not PWM, so PWM is not
  offered on it.

Runtime facts a config cannot know -- the servo's EEPROM homing offset, the
operating mode it is in, an HLS servo's protection current -- are read off the
servo by :meth:`FeetechBus.verify_servos` into :class:`ServoFacts` before the
first command.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from feetech_cli.protocol import describe_error
from feetech_cli.protocol import from_sign_magnitude
from feetech_cli.protocol import to_sign_magnitude
from feetech_cli.registers import CENTER_POSITION
from feetech_cli.registers import CONTROL_TABLE
from feetech_cli.registers import CURRENT_MA_PER_COUNT
from feetech_cli.registers import LOAD_FULL_SCALE
from feetech_cli.registers import RESOLUTION

from actuo.core.contracts import (
    Actuator, ActuatorCommand, ActuatorState, Calibration, Capabilities,
    CapabilityError, ChannelKind, ChannelSpec, Frame, OperatingMode)
from actuo.core.registry import register_actuator
from actuo.drivers import feetech_protocol as fp

#: ``operating_mode`` (EEPROM 33) values used here.
WIRE_MODE_POSITION = 0
WIRE_MODE_PWM = 2

#: ``goal_time`` count that is 100 % PWM duty, and the bit carrying its sign.
PWM_FULL_SCALE = 1000
PWM_SIGN_BIT = 10

#: ``torque_limit`` (SRAM 48) count that is full torque (0.1 % units).
TORQUE_LIMIT_FULL_SCALE = 1000

#: Speed sent with every HLS position command: what FEETECH's HLS examples use
#: (60 * 0.732 rpm = 4.6 rad/s,
#: https://github.com/ftservo/FTServo_Arduino/blob/main/examples/HLSCL/SyncWritePos/SyncWritePos.ino).
#: The trajectory above already ramps slower than that.
HLS_GOAL_SPEED = 60

#: present_position .. present_current, contiguous: one READ per cycle.
STATE_BLOCK_ADDR = CONTROL_TABLE["present_position"].address
STATE_BLOCK_LEN = (CONTROL_TABLE["present_current"].address
                   + CONTROL_TABLE["present_current"].size - STATE_BLOCK_ADDR)

#: model_number .. operating_mode, contiguous: what verify_servos reads once.
FACTS_BLOCK_ADDR = CONTROL_TABLE["model_number"].address
FACTS_BLOCK_LEN = (CONTROL_TABLE["operating_mode"].address
                   + CONTROL_TABLE["operating_mode"].size - FACTS_BLOCK_ADDR)

#: The 7 bytes an HLS position command writes from ``acceleration``.
HLS_GOAL_ADDR = CONTROL_TABLE["acceleration"].address
HLS_GOAL_LEN = 7

COUNTS_PER_RAD = RESOLUTION / (2.0 * math.pi)


@dataclass(frozen=True)
class FeetechModel:
    """One servo model: its model_number and which register layout it has."""

    name: str
    model_number: int
    series: str          # "sts" or "hls"


#: Models this plugin knows. The name is what a robot config states; the
#: number is what the servo reads back, and the two must agree before anything
#: is written (FeetechBus.verify_servos). Numbers: feetech-cli's
#: CURRENT_MA_PER_COUNT sources (777 on an STS3215; 4618 read on an HLS3960M).
MODELS: Dict[str, FeetechModel] = {
    "STS3215": FeetechModel("STS3215", 777, "sts"),
    "HLS3950M": FeetechModel("HLS3950M", 4618, "hls"),
    "HLS3960M": FeetechModel("HLS3960M", 4618, "hls"),
}


def model_by_name(name: str) -> FeetechModel:
    if name not in MODELS:
        raise KeyError("unknown FEETECH model {!r}; known: {}".format(
            name, sorted(MODELS)))
    return MODELS[name]


def _register_bytes(name: str, value: int) -> bytes:
    """Encode ``value`` for register ``name``, little-endian, sign-magnitude
    where the register is."""
    reg = CONTROL_TABLE[name]
    if reg.sign_bit is not None:
        value = to_sign_magnitude(int(value), reg.sign_bit)
    return int(value).to_bytes(reg.size, "little")


def _field(block: bytes, block_addr: int, name: str) -> int:
    """Decode register ``name`` out of a block read from ``block_addr``."""
    reg = CONTROL_TABLE[name]
    start = reg.address - block_addr
    value = int.from_bytes(block[start:start + reg.size], "little")
    if reg.sign_bit is not None:
        return from_sign_magnitude(value, reg.sign_bit)
    return value


@dataclass
class FeetechStsCalibration(Calibration):
    """4096 counts per turn, 0 rad at the centre count unless offset."""

    counts_per_rev: float = float(RESOLUTION)
    zero_offset_counts: float = float(CENTER_POSITION)


@dataclass
class ServoFacts:
    """What the servo itself says, read once at start-up.

    Attributes
    ----------
    homing_offset : int
        EEPROM homing offset. In position mode present_position already has it
        applied; outside position mode the servo reports its raw count without
        it (checked on an STS3215, firmware 3.10), so the plugin applies it.
    wire_mode : int
        ``operating_mode`` the servo is in.
    protection_current : int
        HLS only: the goal current every position command carries. 0 would
        leave the servo limp with torque on.
    """

    homing_offset: int = 0
    wire_mode: int = WIRE_MODE_POSITION
    protection_current: int = 0


@register_actuator("feetech_sts")
class FeetechStsActuator(Actuator):
    """One STS/SMS/HLS servo.

    Parameters
    ----------
    actuator_id : int
        Bus id.
    calibration : FeetechStsCalibration
    model : FeetechModel
        From the robot config; checked against the servo's model_number.
    """

    preferred_bus = "feetech_bus"

    answers_only_when_addressed = True

    def __init__(self, actuator_id: int, calibration: FeetechStsCalibration,
                 model: FeetechModel) -> None:
        super().__init__(actuator_id, calibration)
        self.model = model
        self.facts: Optional[ServoFacts] = None
        entry = CURRENT_MA_PER_COUNT.get(model.model_number)
        #: (A per count, "supply" | "phase"), or None when undocumented.
        self.current_scale: Optional[Tuple[float, str]] = (
            None if entry is None else (entry[0] * 1e-3, entry[1]))

    # --- capabilities --------------------------------------------------------
    @property
    def is_hls(self) -> bool:
        return self.model.series == "hls"

    @property
    def capabilities(self) -> Capabilities:
        modes = ((OperatingMode.POSITION,) if self.is_hls
                 else (OperatingMode.POSITION, OperatingMode.PWM))
        return Capabilities(
            modes=modes,
            has_position_sense=True,
            has_velocity_sense=True,
            # present_current is amperes, not N*m: no torque constant is known,
            # so it is reported as aux['current_a'], never as a faked torque.
            has_current_sense=False,
            has_temperature_sense=True,
            multiturn=False,
        )

    def require_mode(self, mode: OperatingMode) -> None:
        if mode not in self.capabilities.modes:
            why = ("operating mode 2 of an HLS is constant current, not PWM"
                   if self.is_hls and mode is OperatingMode.PWM
                   else "not a mode of this servo")
            raise CapabilityError("id {} ({}): mode {!r}: {}".format(
                self.id, self.model.name, mode.value, why))

    @property
    def channels(self) -> Tuple[ChannelSpec, ...]:
        specs = list(super().channels)
        specs.append(ChannelSpec("load", "duty", ChannelKind.STATE,
                                 "PWM duty -1..1, signed like the joint torque"))
        specs.append(ChannelSpec("current_a", "A", ChannelKind.STATE))
        specs.append(ChannelSpec("voltage", "V", ChannelKind.STATE))
        if OperatingMode.PWM in self.capabilities.modes:
            specs.append(ChannelSpec("pwm", "duty", ChannelKind.COMMAND))
        return tuple(specs)

    def wire_mode_of(self, mode: OperatingMode) -> int:
        self.require_mode(mode)
        return WIRE_MODE_PWM if mode is OperatingMode.PWM else WIRE_MODE_POSITION

    # --- start-up facts ------------------------------------------------------
    def encode_facts_query(self) -> List[Frame]:
        return [Frame(payload=fp.build_read(self.id, FACTS_BLOCK_ADDR,
                                            FACTS_BLOCK_LEN), address=self.id)]

    def decode_facts(self, frame: Frame) -> ServoFacts:
        """Read model, homing offset, mode and protection current, refusing a
        servo that is not the configured model."""
        parsed = fp.parse_status(frame.payload)
        if parsed is None or parsed[0] != self.id:
            raise ValueError("id {}: not this servo's status".format(self.id))
        data = parsed[2]
        if len(data) < FACTS_BLOCK_LEN:
            raise ValueError("id {}: short block ({} of {} bytes)".format(
                self.id, len(data), FACTS_BLOCK_LEN))
        model_number = _field(data, FACTS_BLOCK_ADDR, "model_number")
        if model_number != self.model.model_number:
            raise CapabilityError(
                "id {} reads model_number {}, but the config says {} ({}). "
                "Refusing to drive it with another model's registers.".format(
                    self.id, model_number, self.model.name,
                    self.model.model_number))
        self.facts = ServoFacts(
            homing_offset=_field(data, FACTS_BLOCK_ADDR, "homing_offset"),
            wire_mode=_field(data, FACTS_BLOCK_ADDR, "operating_mode"),
            protection_current=_field(data, FACTS_BLOCK_ADDR,
                                      "protection_current"))
        return self.facts

    def _require_facts(self) -> ServoFacts:
        if self.facts is None:
            raise RuntimeError(
                "id {}: start-up facts were never read (FeetechBus."
                "verify_servos); an HLS goal or a non-position reading would "
                "be wrong".format(self.id))
        return self.facts

    # --- commands ------------------------------------------------------------
    def position_refusal(self, position_rad: float) -> Optional[str]:
        if self.calibration.position_limit is not None:
            low, high = self.calibration.position_limit
            if position_rad < low or position_rad > high:
                return ("id {}: {:.4f} rad is outside the taught limit "
                        "[{:.4f}, {:.4f}]".format(self.id, position_rad, low,
                                                  high))
        raw = int(round(self.calibration.rad_to_counts(position_rad)))
        if not 0 <= raw <= RESOLUTION - 1:
            return ("id {}: {:.4f} rad converts to goal count {}, outside "
                    "0..{}; teach offset_deg so the joint's travel lands "
                    "inside one turn".format(self.id, position_rad, raw,
                                             RESOLUTION - 1))
        return None

    def position_counts(self, position_rad: float) -> int:
        refusal = self.position_refusal(position_rad)
        if refusal is not None:
            raise ValueError(refusal)
        return int(round(self.calibration.rad_to_counts(position_rad)))

    def position_goal(self, counts: int) -> Tuple[int, bytes]:
        """(address, bytes) of a position goal in raw counts."""
        if self.is_hls:
            current = self._require_facts().protection_current
            data = (bytes([0]) + _register_bytes("goal_position", counts)
                    + int(current).to_bytes(2, "little")
                    + int(HLS_GOAL_SPEED).to_bytes(2, "little"))
            return HLS_GOAL_ADDR, data
        return (CONTROL_TABLE["goal_position"].address,
                _register_bytes("goal_position", counts))

    def pwm_goal(self, duty: float) -> Tuple[int, bytes]:
        """(address, bytes) of a PWM duty, -1..1 signed like the joint torque."""
        self.require_mode(OperatingMode.PWM)
        # A positive raw duty turns the horn towards lower counts.
        raw = -self.calibration.direction * float(duty)
        magnitude = min(int(round(abs(raw) * PWM_FULL_SCALE)), PWM_FULL_SCALE)
        value = magnitude | (1 << PWM_SIGN_BIT) if raw < 0 else magnitude
        return CONTROL_TABLE["goal_time"].address, value.to_bytes(2, "little")

    def goal_payload(self, command: ActuatorCommand) -> Tuple[int, bytes]:
        """(address, bytes) the bus groups into one SYNC_WRITE per address."""
        self.require_mode(command.mode)
        if command.mode is OperatingMode.PWM:
            if command.pwm is None:
                raise ValueError("PWM mode needs command.pwm (duty -1..1)")
            return self.pwm_goal(command.pwm)
        if command.position is None:
            raise ValueError("POSITION mode needs command.position (rad)")
        return self.position_goal(self.position_counts(command.position))

    def encode_command(self, command: ActuatorCommand) -> Frame:
        address, data = self.goal_payload(command)
        return Frame(payload=fp.build_write(self.id, address, data),
                     address=self.id)

    def encode_write(self, register: str, value: int) -> Frame:
        """An acknowledged WRITE of one named register."""
        return Frame(payload=fp.build_write(
            self.id, CONTROL_TABLE[register].address,
            _register_bytes(register, value)), address=self.id)

    def encode_read(self, register: str) -> Frame:
        reg = CONTROL_TABLE[register]
        return Frame(payload=fp.build_read(self.id, reg.address, reg.size),
                     address=self.id)

    def decode_register(self, register: str, frame: Frame) -> int:
        parsed = fp.parse_status(frame.payload)
        reg = CONTROL_TABLE[register]
        if parsed is None or parsed[0] != self.id or len(parsed[2]) < reg.size:
            raise ValueError("id {}: no valid {} reply".format(self.id, register))
        return _field(parsed[2], reg.address, register)

    def encode_mode(self, mode: OperatingMode) -> List[Frame]:
        """Torque off, then ``operating_mode``. Both are acknowledged writes.

        operating_mode is an EEPROM register; it is written with the EEPROM
        still locked, which the servo applies (read back on firmware 3.10).
        """
        return [self.encode_write("torque_enable", 0),
                self.encode_write("operating_mode", self.wire_mode_of(mode))]

    def encode_torque_enable(self, enabled: bool) -> Frame:
        return self.encode_write("torque_enable", 1 if enabled else 0)

    def encode_release(self) -> Frame:
        """Servo off: torque_enable = 0, so the joint is back-drivable."""
        return self.encode_torque_enable(False)

    # --- reads ---------------------------------------------------------------
    def encode_query(self) -> List[Frame]:
        return [Frame(payload=fp.build_read(self.id, STATE_BLOCK_ADDR,
                                            STATE_BLOCK_LEN), address=self.id)]

    def matches(self, frame: Frame) -> bool:
        parsed = fp.parse_status(frame.payload)
        return parsed is not None and parsed[0] == self.id

    def decode_state(self, frame: Frame) -> ActuatorState:
        parsed = fp.parse_status(frame.payload)
        if parsed is None or parsed[0] != self.id:
            raise ValueError("id {}: frame is not this servo's status".format(
                self.id))
        return self.decode_block(parsed[2], error=parsed[1])

    def decode_block(self, data: bytes, error: int = 0) -> ActuatorState:
        """Decode the present_position..present_current block."""
        if len(data) < STATE_BLOCK_LEN:
            raise ValueError("id {}: short status block ({} of {} bytes)".format(
                self.id, len(data), STATE_BLOCK_LEN))
        facts = self._require_facts()

        def field(name: str) -> int:
            return _field(data, STATE_BLOCK_ADDR, name)

        counts = field("present_position")
        if facts.wire_mode != WIRE_MODE_POSITION:
            # Outside position mode the servo reports its raw count, without
            # the homing offset (checked on an STS3215, firmware 3.10).
            counts = (counts - facts.homing_offset) % RESOLUTION
        direction = self.calibration.direction
        aux = {
            "raw_position": float(counts),
            # present_load has the opposite sign of the torque the servo
            # exerts (checked holding an arm against gravity, both ways), so it
            # is flipped to be signed like that torque.
            "load": -direction * field("present_load") / float(LOAD_FULL_SCALE),
            "status": float(field("status")),
            "moving": float(field("moving")),
            "wire_mode": float(facts.wire_mode),
        }
        if self.current_scale is not None:
            aux["current_a"] = field("present_current") * self.current_scale[0]
            aux["current_is_supply"] = (
                1.0 if self.current_scale[1] == "supply" else 0.0)
        return ActuatorState(
            position=self.calibration.counts_to_rad(counts),
            velocity=direction * field("present_velocity") / COUNTS_PER_RAD,
            temperature=float(field("present_temperature")),
            voltage=field("present_voltage") * 0.1,
            error=error or None,
            aux=aux,
        )

    def fault_name(self, code: int) -> str:
        code = int(code)
        if code == 0:
            return "none"
        return ", ".join(describe_error(code))
