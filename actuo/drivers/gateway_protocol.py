"""Host side of the gateway protocol spoken by the servo gateway firmware.

The gateway firmware is not included in this package. This module is the
host's implementation of its wire protocol; the firmware's own C++ wire code is
the other. Two implementations of one protocol is a known way to fail: a C++
transport and a Python reimplementation can each pass their own tests and still
drift apart. The layout is therefore spelled out in the constants and builders
below, and ``tests/test_gateway_protocol.py`` pins the parts that must agree
byte for byte (the CRC-16 check value, framing, entry sizes and endianness)
with no hardware attached.

Nothing here knows what a radian is. The gateway sends raw counts plus a
descriptor (``counts_per_rev_x100``, centre, limits) and the existing
``KondoIcsActuator`` converts, so calibration stays in exactly one place.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple

SYNC0 = 0xA5
SYNC1 = 0x5A
PROTOCOL_VERSION = 1

HEADER_SIZE = 7          # sync(2) ver cmd len(2) seq
CRC_SIZE = 2
FRAME_OVERHEAD = HEADER_SIZE + CRC_SIZE
MAX_PAYLOAD = 1600
MAX_FRAME = FRAME_OVERHEAD + MAX_PAYLOAD

STATE_HEADER_SIZE = 10
STATE_ENTRY_SIZE = 8
TARGET_ENTRY_SIZE = 5
TORQUE_ENTRY_SIZE = 3
HELLO_PAYLOAD_SIZE = 44

#: `age_us` saturates here; the gateway sets `STALE` at the same time, so a
#: wrapped counter can never read as a fresh measurement.
MAX_AGE_US = 0xFFFF


class Command(IntEnum):
    HELLO = 0x01
    PING = 0x02
    ENUMERATE = 0x10
    GET_TOPOLOGY = 0x11
    WRITE_TARGETS = 0x20
    READ_STATE = 0x21
    SET_TORQUE = 0x22
    GET_PARAM = 0x30
    SET_PARAM = 0x31
    ACK = 0xF0
    NACK = 0xF1


class ErrorCode(IntEnum):
    NONE = 0x00
    BAD_VERSION = 0x01
    BAD_CRC = 0x02
    BAD_COMMAND = 0x03
    BAD_PAYLOAD = 0x04
    BAD_PORT = 0x05
    BAD_ID = 0x06
    PORT_BUSY = 0x07
    NOT_PRESENT = 0x08
    BUS_ERROR = 0x09
    UNSUPPORTED = 0x0A


class ProtocolError(RuntimeError):
    """A frame could not be built or parsed. Never silently repaired."""


# -- flags -------------------------------------------------------------------

TARGET_HOLD = 1 << 0
#: Bits this version defines. The gateway refuses anything else rather than
#: reinterpreting it, so the field stays extensible.
TARGET_KNOWN_MASK = TARGET_HOLD

STATE_PRESENT = 1 << 0
STATE_HOLDING = 1 << 1
STATE_STALE = 1 << 2
STATE_POSITION_VALID = 1 << 3

CAP_WRITE_RETURNS_POSITION = 1 << 0
CAP_CURRENT_SENSE = 1 << 1
CAP_TEMPERATURE_SENSE = 1 << 2
CAP_BATCH_WRITE = 1 << 3


# -- CRC ---------------------------------------------------------------------


def crc16(data: bytes) -> int:
    """CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection, no xorout.

    Pinned against the standard ``"123456789" -> 0x29B1`` check value, because
    "CRC-16" names a dozen mutually incompatible algorithms and picking the
    wrong one silently is easy.
    """
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


# -- framing -----------------------------------------------------------------


@dataclass(frozen=True)
class Frame:
    command: int
    seq: int
    payload: bytes
    version: int = PROTOCOL_VERSION


def encode_frame(command: int, seq: int, payload: bytes = b"") -> bytes:
    """Build one complete frame.

    The CRC covers ``ver``, ``cmd``, ``len``, ``seq`` and the payload -- not the
    sync bytes, which carry no information.
    """
    if len(payload) > MAX_PAYLOAD:
        raise ProtocolError(
            "payload of {} bytes exceeds the protocol's {}".format(
                len(payload), MAX_PAYLOAD))
    body = struct.pack("<BBHB", PROTOCOL_VERSION, int(command), len(payload),
                       seq) + payload
    return bytes([SYNC0, SYNC1]) + body + struct.pack("<H", crc16(body))


def decode_frame(data: bytes) -> Frame:
    """Parse one complete frame, or raise.

    Deliberately strict: this is for a buffer already known to hold exactly one
    frame (a test vector, a reply just read off the wire). Streaming lives in
    :class:`FrameReader`.
    """
    if len(data) < FRAME_OVERHEAD:
        raise ProtocolError("frame of {} bytes is too short".format(len(data)))
    if data[0] != SYNC0 or data[1] != SYNC1:
        raise ProtocolError("bad sync: {!r}".format(data[:2]))
    version, command, length, seq = struct.unpack("<BBHB", data[2:HEADER_SIZE])
    total = FRAME_OVERHEAD + length
    if len(data) != total:
        raise ProtocolError(
            "declared length {} implies {} bytes, got {}".format(
                length, total, len(data)))
    payload = data[HEADER_SIZE:HEADER_SIZE + length]
    expected = crc16(data[2:HEADER_SIZE + length])
    actual = struct.unpack("<H", data[HEADER_SIZE + length:])[0]
    if expected != actual:
        raise ProtocolError(
            "CRC mismatch: computed {:#06x}, frame carries {:#06x}".format(
                expected, actual))
    if version != PROTOCOL_VERSION:
        # The payload is deliberately not returned. A v2 body read with v1
        # offsets yields plausible numbers, and a plausible wrong servo angle is
        # the one output this stack must never produce.
        raise ProtocolError(
            "gateway speaks protocol version {}, this host speaks {}".format(
                version, PROTOCOL_VERSION))
    return Frame(command=command, seq=seq, payload=payload, version=version)


class FrameReader:
    """Reassembles frames from a byte stream, resynchronising on garbage."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self.resyncs = 0
        self.crc_errors = 0

    def feed(self, data: bytes) -> List[Frame]:
        """Adds bytes and returns whatever frames completed."""
        self._buffer.extend(data)
        frames: List[Frame] = []
        while True:
            start = self._buffer.find(bytes([SYNC0, SYNC1]))
            if start < 0:
                # Keep a trailing 0xA5 that might begin the next sync.
                keep = 1 if self._buffer[-1:] == bytes([SYNC0]) else 0
                self.resyncs += len(self._buffer) - keep
                del self._buffer[:len(self._buffer) - keep]
                return frames
            if start > 0:
                self.resyncs += start
                del self._buffer[:start]
            if len(self._buffer) < HEADER_SIZE:
                return frames
            length = struct.unpack("<H", self._buffer[4:6])[0]
            if length > MAX_PAYLOAD:
                del self._buffer[:2]
                continue
            total = FRAME_OVERHEAD + length
            if len(self._buffer) < total:
                return frames
            candidate = bytes(self._buffer[:total])
            del self._buffer[:total]
            try:
                frames.append(decode_frame(candidate))
            except ProtocolError:
                self.crc_errors += 1
        return frames


# -- payloads ----------------------------------------------------------------


@dataclass
class Hello:
    proto_version: int
    fw_version: Tuple[int, int, int]
    git_hash: str
    board_name: str
    port_count: int
    max_servos: int
    uptime_us: int
    rx_resyncs: int
    rx_crc_errors: int
    rx_version_errors: int
    rx_overlong: int
    rx_aborted: int


def decode_hello(payload: bytes) -> Hello:
    if len(payload) != HELLO_PAYLOAD_SIZE:
        raise ProtocolError(
            "HELLO is {} bytes, expected {}".format(len(payload),
                                                    HELLO_PAYLOAD_SIZE))
    proto, major, minor, patch = payload[0:4]
    git = payload[4:12].decode("ascii", "replace").rstrip()
    board = payload[12:28].decode("ascii", "replace").rstrip("\x00")
    port_count, max_servos = payload[28:30]
    uptime, resyncs, crc, version_errors, overlong, aborted = struct.unpack(
        "<IHHHHH", payload[30:44])
    return Hello(proto, (major, minor, patch), git, board, port_count,
                 max_servos, uptime, resyncs, crc, version_errors, overlong,
                 aborted)


@dataclass
class PortDescriptor:
    port: int
    protocol: int
    baudrate: int
    servo_count: int
    counts_per_rev_x100: int
    pos_min: int
    pos_max: int
    pos_center: int
    caps: int
    servos: List[Tuple[int, int]] = field(default_factory=list)  # (id, position)

    @property
    def batch_write(self) -> bool:
        return bool(self.caps & CAP_BATCH_WRITE)

    @property
    def write_returns_position(self) -> bool:
        return bool(self.caps & CAP_WRITE_RETURNS_POSITION)


@dataclass
class Topology:
    generation: int
    ports: List[PortDescriptor]


def decode_topology(payload: bytes) -> Topology:
    port_count, generation = payload[0], payload[1]
    offset = 2
    ports: List[PortDescriptor] = []
    for _ in range(port_count):
        (index, protocol, baudrate, servo_count, counts, pos_min, pos_max,
         pos_center, caps) = struct.unpack("<BBIBIHHHB", payload[offset:offset + 18])
        offset += 18
        descriptor = PortDescriptor(index, protocol, baudrate, servo_count,
                                    counts, pos_min, pos_max, pos_center, caps)
        for _ in range(servo_count):
            servo_id, position = struct.unpack("<BH", payload[offset:offset + 3])
            offset += 3
            descriptor.servos.append((servo_id, position))
        ports.append(descriptor)
    if offset != len(payload):
        raise ProtocolError(
            "TOPOLOGY has {} trailing bytes; host and gateway disagree about "
            "the layout".format(len(payload) - offset))
    return Topology(generation, ports)


@dataclass
class ServoState:
    port: int
    id: int
    position: int          # raw counts
    age_us: int
    flags: int
    error: int

    @property
    def present(self) -> bool:
        return bool(self.flags & STATE_PRESENT)

    @property
    def holding(self) -> bool:
        return bool(self.flags & STATE_HOLDING)

    @property
    def stale(self) -> bool:
        return bool(self.flags & STATE_STALE)

    @property
    def position_valid(self) -> bool:
        return bool(self.flags & STATE_POSITION_VALID)


@dataclass
class StateReply:
    uptime_us: int
    cycle_us: int
    dropped: int
    topology_generation: int
    servos: List[ServoState]


def decode_state(payload: bytes) -> StateReply:
    if len(payload) < STATE_HEADER_SIZE:
        raise ProtocolError("STATE payload is too short")
    uptime, cycle_us, dropped, generation, count = struct.unpack(
        "<IHHBB", payload[:STATE_HEADER_SIZE])
    expected = STATE_HEADER_SIZE + count * STATE_ENTRY_SIZE
    if len(payload) != expected:
        raise ProtocolError(
            "STATE claims {} servos ({} bytes) but carries {}".format(
                count, expected, len(payload)))
    servos = []
    for i in range(count):
        base = STATE_HEADER_SIZE + i * STATE_ENTRY_SIZE
        port, servo_id, position, age, flags, error = struct.unpack(
            "<BBHHBB", payload[base:base + STATE_ENTRY_SIZE])
        servos.append(ServoState(port, servo_id, position, age, flags, error))
    return StateReply(uptime, cycle_us, dropped, generation, servos)


def encode_port_mask(mask: int) -> bytes:
    return bytes([mask & 0xFF])


def encode_targets(targets: List[Tuple[int, int, int, bool]]) -> bytes:
    """``[(port, id, position, hold), ...]`` -> WRITE_TARGETS payload."""
    if len(targets) > 0xFF:
        raise ProtocolError("a batch of {} exceeds the count byte".format(
            len(targets)))
    out = bytearray([len(targets)])
    for port, servo_id, position, hold in targets:
        out += struct.pack("<BBHB", port, servo_id, position,
                           TARGET_HOLD if hold else 0)
    return bytes(out)


def encode_torque(entries: List[Tuple[int, int, bool]]) -> bytes:
    """``[(port, id, on), ...]`` -> SET_TORQUE payload."""
    out = bytearray([len(entries)])
    for port, servo_id, on in entries:
        out += struct.pack("<BBB", port, servo_id, 1 if on else 0)
    return bytes(out)


def encode_param_request(port: int, servo_id: int, param: int) -> bytes:
    return struct.pack("<BBH", port, servo_id, param)


def encode_param(port: int, servo_id: int, param: int, value: int) -> bytes:
    return struct.pack("<BBHi", port, servo_id, param, value)


@dataclass
class ParamReply:
    port: int
    id: int
    param: int
    value: int


def decode_param(payload: bytes) -> ParamReply:
    port, servo_id, param, value = struct.unpack("<BBHi", payload)
    return ParamReply(port, servo_id, param, value)


@dataclass
class Nack:
    code: ErrorCode
    detail: int


def decode_nack(payload: bytes) -> Nack:
    if len(payload) != 2:
        raise ProtocolError("NACK payload is {} bytes, expected 2".format(
            len(payload)))
    return Nack(ErrorCode(payload[0]), payload[1])


# -- ICS parameter ids ---------------------------------------------------------

ICS_PARAM_STRETCH = 0x0100
ICS_PARAM_SPEED = 0x0101
ICS_PARAM_CURRENT_LIMIT = 0x0102
ICS_PARAM_TEMPERATURE_LIMIT = 0x0103
ICS_PARAM_CURRENT = 0x0110
ICS_PARAM_TEMPERATURE = 0x0111
