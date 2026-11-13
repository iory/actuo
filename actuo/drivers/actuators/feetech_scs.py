"""Feetech SCS0009 serial servo (Amazing Hand). Pure plugin -- no actuo.core edit.

The Amazing Hand is 8x SCS0009 on one TTL serial bus. These are register-mapped
servos speaking the Feetech / Dynamixel-1.0 packet form -- the very family whose
"leaky register abstraction" sank an earlier design, which assumed every servo
was register-mapped and so could not fit Kondo ICS or CAN. Here it stays
contained: the actuator owns the register map and its (big-endian!) scaling, the
transport moves opaque packet bytes, and the bus batches with SYNC_WRITE. Nothing
about registers reaches a shared layer.

Protocol provenance: see ``actuo/drivers/references.md`` ("Feetech SCS0009"),
sourced from rustypot's ``src/servo/feetech/scs0009.rs`` -- the library the
upstream Amazing Hand demo drives these servos with.

Two SCS-specific traps, both handled here and both easy to get backwards:

* **Big-endian.** SCS-series 2-byte registers are high-byte-first on the wire,
  the opposite of the STS/SMS series. Reading them little-endian yields plausible
  but wrong angles.
* **A command frame is not a status frame.** This bus is request/response, so an
  actuator only decodes a READ *reply* (``matches``), never an echoed command.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from actuo.core.contracts import (
    Actuator, ActuatorCommand, ActuatorState, Calibration, Capabilities,
    CapabilityError, ChannelKind, ChannelSpec, Frame, OperatingMode)
from actuo.core.registry import register_actuator

# --- protocol constants (references.md [11]) --------------------------------
HEADER = b"\xff\xff"
BROADCAST_ID = 0xFE

PING = 0x01
READ = 0x02
WRITE = 0x03
SYNC_WRITE = 0x83

# Register addresses (byte offset in the servo's memory table).
REG_ID = 5
REG_BAUDRATE = 6
REG_MIN_ANGLE_LIMIT = 9
REG_MAX_ANGLE_LIMIT = 11
REG_MAX_TORQUE_LIMIT = 16
REG_TORQUE_ENABLE = 40
REG_GOAL_POSITION = 42
REG_GOAL_SPEED = 46
REG_PRESENT_POSITION = 56
REG_PRESENT_LOAD = 60
REG_PRESENT_TEMPERATURE = 63

# One READ covers present_position(56,57) speed(58,59) load(60,61) voltage(62)
# temperature(63): eight contiguous bytes from 56.
PRESENT_BLOCK_ADDR = REG_PRESENT_POSITION
PRESENT_BLOCK_LEN = 8

# Position: 10-bit raw, centre 511, full scale 1024 counts = 300 deg.
SCS_RESOLUTION = 1024
SCS_RANGE_DEG = 300.0
SCS_CENTER_COUNTS = 511
# counts per full 2*pi rev, so the generic Calibration's counts<->rad math lands
# on rustypot's radians(300) * (raw - 511) / 1024.
COUNTS_PER_REV = SCS_RESOLUTION * 360.0 / SCS_RANGE_DEG   # 1228.8

# present_speed: radians(300)/1024 per count, sign at bit 15, magnitude & 0x3ff.
_SPEED_PER_COUNT = math.radians(SCS_RANGE_DEG) / SCS_RESOLUTION


def checksum(servo_id: int, length: int, instruction: int,
             params: bytes) -> int:
    """Feetech/DXL-1.0 checksum: ~(id+len+instr+sum(params)) & 0xFF."""
    return (~(servo_id + length + instruction + sum(params))) & 0xFF


def build_packet(servo_id: int, instruction: int, params: bytes = b"") -> bytes:
    """One instruction packet: FF FF ID LEN INSTR PARAM.. CHECKSUM."""
    length = len(params) + 2               # params + instruction + checksum
    body = bytes([servo_id, length, instruction]) + params
    return HEADER + body + bytes([checksum(servo_id, length, instruction, params)])


def build_ping(servo_id: int) -> bytes:
    return build_packet(servo_id, PING)


def build_read(servo_id: int, address: int, length: int) -> bytes:
    return build_packet(servo_id, READ, bytes([address, length]))


def build_write(servo_id: int, address: int, data: bytes) -> bytes:
    return build_packet(servo_id, WRITE, bytes([address]) + data)


def build_sync_write(address: int, per_servo_len: int,
                     items: List[Tuple[int, bytes]]) -> bytes:
    """One SYNC_WRITE to all servos: params = addr, len, id,data, id,data, ...

    A single bus transaction sets every servo -- batch writes are the primary
    operation in this stack, not a loop of individual writes.
    """
    params = bytearray([address, per_servo_len])
    for servo_id, data in items:
        if len(data) != per_servo_len:
            raise ValueError("sync-write data for id {} is {} bytes, expected {}"
                             .format(servo_id, len(data), per_servo_len))
        params.append(servo_id)
        params.extend(data)
    return build_packet(BROADCAST_ID, SYNC_WRITE, bytes(params))


def parse_status(packet: bytes) -> Optional[Tuple[int, int, bytes]]:
    """Parse a status reply into (id, error, data), or None if malformed.

    Returns None rather than raising on a bad checksum/length: a serial line
    drops and re-syncs bytes, and one corrupt reply is a dropped read, not a
    crash. The caller treats None as "no fresh state this cycle".
    """
    if len(packet) < 6 or packet[0:2] != HEADER:
        return None
    servo_id = packet[2]
    length = packet[3]
    if len(packet) != length + 4:
        return None
    error = packet[4]
    data = packet[5:-1]
    if checksum(servo_id, length, error, data) != packet[-1]:
        return None
    return servo_id, error, bytes(data)


def u16_be(value: int) -> bytes:
    """Two big-endian bytes (SCS series is high-byte-first)."""
    return bytes([(value >> 8) & 0xFF, value & 0xFF])


def from_u16_be(hi_lo: bytes) -> int:
    return (hi_lo[0] << 8) | hi_lo[1]


@dataclass
class ScsCalibration(Calibration):
    """SCS0009 calibration. Inherits the generic counts<->rad math; the only
    SCS-specific defaults are the 10-bit resolution and centre count."""

    counts_per_rev: float = COUNTS_PER_REV
    zero_offset_counts: float = float(SCS_CENTER_COUNTS)


@register_actuator("feetech_scs0009")
class Scs0009Actuator(Actuator):
    """One SCS0009 servo. POSITION control; reports position/speed/temperature.

    No torque constant exists for this servo, so CURRENT (N*m) is deliberately
    absent from ``capabilities`` and ``present_load`` surfaces only as a raw
    ``aux['load']`` -- never a faked N*m: a value this stack cannot measure is
    left absent rather than made up.
    """

    def __init__(self, actuator_id: int, calibration: ScsCalibration) -> None:
        super().__init__(actuator_id, calibration)

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            modes=(OperatingMode.POSITION,),
            has_position_sense=True,
            has_velocity_sense=True,
            has_current_sense=False,
            has_temperature_sense=True,
            multiturn=False,
        )

    @property
    def channels(self) -> Tuple[ChannelSpec, ...]:
        return (
            ChannelSpec("position", "rad", ChannelKind.COMMAND),
            ChannelSpec("position", "rad", ChannelKind.STATE),
            ChannelSpec("velocity", "rad/s", ChannelKind.STATE),
            ChannelSpec("temperature", "degC", ChannelKind.STATE),
        )

    # --- commands -----------------------------------------------------------
    def goal_position_counts(self, position_rad: float) -> int:
        """Clamp to the taught envelope, then convert to a 10-bit raw count."""
        if self.calibration.position_limit is not None:
            low, high = self.calibration.position_limit
            if position_rad < low or position_rad > high:
                raise ValueError(
                    "id {}: commanded {:.3f} rad outside limit [{:.3f}, {:.3f}]"
                    .format(self.id, position_rad, low, high))
        raw = int(round(self.calibration.rad_to_counts(position_rad)))
        return max(0, min(SCS_RESOLUTION - 1, raw))

    def goal_position_payload(self, position_rad: float) -> bytes:
        """The two goal_position bytes, for the bus to pack into a SYNC_WRITE."""
        return u16_be(self.goal_position_counts(position_rad))

    def encode_command(self, command: ActuatorCommand) -> Frame:
        self.capabilities.require(command.mode)
        if command.position is None:
            raise ValueError("POSITION mode needs command.position (rad)")
        payload = build_write(self.id, REG_GOAL_POSITION,
                              self.goal_position_payload(command.position))
        return Frame(payload=payload, address=self.id)

    def encode_mode(self, mode: OperatingMode) -> List[Frame]:
        self.capabilities.require(mode)
        return []                          # POSITION is the servo's only mode

    def encode_torque_enable(self, enabled: bool) -> Frame:
        payload = build_write(self.id, REG_TORQUE_ENABLE,
                              bytes([1 if enabled else 0]))
        return Frame(payload=payload, address=self.id)

    def encode_release(self) -> Frame:
        """Servo off: torque_enable = 0, so the joint is back-drivable."""
        return self.encode_torque_enable(False)

    # --- reads (request/response) -------------------------------------------
    def encode_query(self) -> List[Frame]:
        """READ the present-state block. This bus does not stream unsolicited."""
        payload = build_read(self.id, PRESENT_BLOCK_ADDR, PRESENT_BLOCK_LEN)
        return [Frame(payload=payload, address=self.id)]

    def matches(self, frame: Frame) -> bool:
        parsed = parse_status(frame.payload)
        return parsed is not None and parsed[0] == self.id

    def decode_state(self, frame: Frame) -> ActuatorState:
        parsed = parse_status(frame.payload)
        if parsed is None or parsed[0] != self.id:
            raise ValueError("id {}: frame is not this servo's status".format(self.id))
        _id, error, data = parsed
        if len(data) < PRESENT_BLOCK_LEN:
            raise ValueError("id {}: short status block ({} bytes)".format(
                self.id, len(data)))

        raw_pos = from_u16_be(data[0:2]) & 0x3FF
        position = self.calibration.counts_to_rad(raw_pos)

        raw_speed = from_u16_be(data[2:4])
        magnitude = (raw_speed & 0x3FF) * _SPEED_PER_COUNT
        velocity = -magnitude if raw_speed & (1 << 15) else magnitude
        velocity *= self.calibration.direction

        raw_load = from_u16_be(data[4:6])
        temperature = float(data[7])

        return ActuatorState(
            position=position,
            velocity=velocity,
            temperature=temperature,
            error=error or None,
            aux={"raw_position": float(raw_pos), "load": float(raw_load & 0x3FF)},
        )
