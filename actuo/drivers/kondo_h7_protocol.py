"""Wire protocol of the Kondo ARM-H7 control board (RCB-4 derived).

This module is pure byte-shuffling: it builds and parses the packets the H7
board speaks, and holds the board's memory layout. It opens no port, owns no
actuator, and imports nothing from ``actuo.core`` -- so every constant in it is
testable against a frozen byte vector with no hardware present.

Layering note (this is the whole point of the Kondo port)
--------------------------------------------------------
The H7 board is a **gateway, not a servo**. The host speaks this packet
protocol to the board; the board runs the ICS servo loop on six UARTs. So the
knowledge splits three ways:

* framing (length byte + checksum) -> ``KondoH7Transport``
* board protocol: opcodes, memory layout, the multi-servo command -> **here**,
  used by ``KondoH7Bus``
* ICS servo semantics: pulse<->rad, current/temperature scaling, limits
  -> ``actuo.drivers.actuators.kondo_ics``

Nothing in this file knows what a radian is, and nothing in ``kondo_ics`` knows
what a packet looks like. Without that split, a design that assumes
memory-mapped registers ends up faking Kondo's command/sub-command protocol by
reinterpreting a register address as a sub-command.

Packet form
-----------
Every packet, in both directions, is::

    [len] [body ...] [checksum]

where ``len`` counts the whole packet (so ``len = len(body) + 2``) and
``checksum = (len + sum(body)) & 0xFF``. That framing, the ack and version
requests and the batch servo move are shared with the RCB-4mini and live in
``rcb4.protocol.framing`` (one copy shared by rcb4 and actuo); they are
re-exported here so H7 code finds the whole protocol in one module.

Provenance: see ``actuo/drivers/references.md``, section "Kondo ARM-H7". Every
byte layout below is pinned by ``test_kondo_h7_protocol.py`` against vectors
generated from the hardware-validated ``rcb4`` implementation.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

from rcb4.protocol.framing import (  # noqa: F401  (re-exported: shared with the RCB-4mini)
    MAX_PACKET_LENGTH,
    OP_ACK,
    OP_MULTI_SERVO_MULTI_VELOCITIES,
    OP_MULTI_SERVO_SINGLE_VELOCITY,
    OP_VERSION,
    SERVO_ID_BITMAP_BYTES,
    ChecksumError,
    ProtocolError,
    build_ack,
    build_multi_servo,
    build_version,
    checksum,
    clamp_velocity,
    encode_id_bitmap,
    frame,
    unframe,
)

# --------------------------------------------------------------------------
# Opcodes
# --------------------------------------------------------------------------

#: Read memory: strided vector read (address, count, element size, stride).
OP_MREADV = 0xFB
#: Write memory.
OP_MWRITEV = 0xFC
#: Call a function at an address in the board's firmware.
OP_CALL = 0xFA

#: A single ``MREADV`` reply is capped by the board's buffer; longer reads must
#: be split. ``rcb4`` uses 250 bytes per request, so we inherit that bound.
MAX_READ_BYTES = 250


# --------------------------------------------------------------------------
# H7-only packet builders (bodies -- the transport adds length + checksum).
# Framing, ack, version and the batch servo move are rcb4.protocol.framing.
# --------------------------------------------------------------------------


def build_read(address: int, element_size: int, count: int = 1,
               stride: int = 0) -> bytes:
    """Body of a strided memory read.

    One primitive covers both uses: a flat block read (``count=1``,
    ``element_size=n``) and the batch state read that makes this port viable --
    ``count=36, element_size=2, stride=120`` pulls one 2-byte slot out of all 36
    servo structs in a **single** round trip.

    Parameters
    ----------
    address : int
        Byte address in the board's memory.
    element_size : int
        Bytes to read at each stride step.
    count : int
        Number of elements.
    stride : int
        Byte distance between consecutive elements (0 for a flat read).

    Returns
    -------
    bytes
        The request body.

    Raises
    ------
    ProtocolError
        If a field does not fit the wire, or the reply would exceed the board's
        buffer -- caught here rather than producing a truncated read.
    """
    if not 0 <= count <= 0xFF:
        raise ProtocolError("count {} does not fit one byte".format(count))
    if not 0 <= element_size <= 0xFF:
        raise ProtocolError(
            "element_size {} does not fit one byte".format(element_size))
    if not 0 <= stride <= 0xFFFF:
        raise ProtocolError("stride {} does not fit two bytes".format(stride))
    total = count * element_size
    if total > MAX_READ_BYTES:
        raise ProtocolError(
            "read of {} bytes exceeds the board's {}-byte reply buffer; "
            "split the request".format(total, MAX_READ_BYTES))
    return (bytes([OP_MREADV])
            + int(address).to_bytes(4, "little")
            + bytes([count, element_size])
            + int(stride).to_bytes(2, "little"))


def build_write(address: int, data: bytes) -> bytes:
    """Body of a memory write of ``data`` at ``address``."""
    if not 0 <= len(data) <= 0xFF:
        raise ProtocolError("write of {} bytes does not fit".format(len(data)))
    return (bytes([OP_MWRITEV])
            + int(address).to_bytes(4, "little")
            + bytes([1, len(data)])
            + (0).to_bytes(2, "little")
            + bytes(data))


def build_call(address: int, args: Sequence[int] = ()) -> bytes:
    """Body of a call to the firmware function at ``address``."""
    body = bytearray([OP_CALL])
    body += int(address).to_bytes(4, "little")
    body.append(len(args))
    for arg in args:
        body += int(arg).to_bytes(4, "little")
    return bytes(body)



# --------------------------------------------------------------------------
# Board memory layout
# --------------------------------------------------------------------------

#: Struct-module format code per firmware C type, used to decode a read.
_TYPE_FORMAT = {
    "uint8": "B",
    "int8": "b",
    "uint16": "H",
    "int16": "h",
    "uint32": "I",
    "int32": "i",
    "float": "f",
}


@dataclass(frozen=True)
class Slot:
    """One member of a firmware struct: where it is and how to read it."""

    offset: int
    ctype: str
    length: int = 1

    @property
    def size(self) -> int:
        return struct.calcsize(_TYPE_FORMAT[self.ctype]) * self.length

    def decode(self, raw: bytes) -> Tuple:
        if len(raw) != self.size:
            raise ProtocolError(
                "slot expects {} bytes, got {}".format(self.size, len(raw)))
        return struct.unpack("<" + _TYPE_FORMAT[self.ctype] * self.length, raw)

    def encode(self, values: Sequence) -> bytes:
        return struct.pack("<" + _TYPE_FORMAT[self.ctype] * self.length, *values)


#: Size of one ``ServoStruct`` in the board's memory, i.e. the stride between
#: consecutive servos in ``servo_vector``.
SERVO_STRUCT_SIZE = 120
#: Number of servo slots the firmware allocates (ids 0..35).
SERVO_VECTOR_LENGTH = 36

#: ``ServoStruct`` members this driver reads or writes. Offsets follow from the
#: firmware's C declaration under natural alignment; they are pinned by
#: ``test_kondo_h7_protocol.py`` against the layout ``rcb4`` computes from the
#: same declaration, so a silent drift fails a test rather than a robot.
SERVO_SLOTS: Mapping[str, Slot] = {
    "flag": Slot(0, "uint8"),            # 1 = a servo is present at this id
    "mode": Slot(1, "uint8"),
    "rotation": Slot(2, "uint8"),        # 1 = continuous-rotation servo
    "port": Slot(3, "uint8"),            # which ICS UART it hangs off
    "ref_angle": Slot(4, "uint16"),      # commanded pulse (holds the on/off sentinel)
    "current_angle": Slot(6, "uint16"),  # measured pulse
    "trim": Slot(8, "int16"),
    "stretch": Slot(81, "uint8"),
    "speed": Slot(82, "uint8"),
    "current_limit": Slot(83, "uint8"),
    "temperature_limit": Slot(84, "uint8"),
    "timeout_count": Slot(86, "uint8"),
    "rxerror_count": Slot(87, "uint8"),
    "init_state": Slot(96, "uint8"),
    "init_count": Slot(97, "uint8"),
    "error_angle": Slot(98, "int16"),    # ref - measured, in pulses
    "feedback": Slot(116, "uint8"),
    "move_state": Slot(117, "uint8"),
    "current": Slot(118, "uint8"),       # only updated while the read flag is on
    "temperature": Slot(119, "uint8"),   # ditto
}

#: Size of ``SystemStruct`` (the board-wide state block, a single instance).
SYSTEM_STRUCT_SIZE = 76

#: ``SystemStruct`` members this driver touches.
SYSTEM_SLOTS: Mapping[str, Slot] = {
    "battery_voltage": Slot(28, "float"),
    # Reading servo current/temperature costs ICS bus bandwidth, so the
    # firmware only polls them while these flags are set.
    "servo_current_read_flag": Slot(49, "uint8"),
    "servo_temperature_read_flag": Slot(50, "uint8"),
    "ics_comm_stop": Slot(64, "uint8", 6),   # one per ICS UART
}

#: Size of ``ControlStruct`` (board control and comm-status state).
CONTROL_STRUCT_SIZE = 424

#: ``ControlStruct`` members this driver reads.
CONTROL_SLOTS: Mapping[str, Slot] = {
    "device_port": Slot(356, "uint8", 64),
    "device_timeout_th": Slot(420, "uint8"),
}


def servo_slot_read(base_address: int, slot_name: str,
                    count: int = SERVO_VECTOR_LENGTH) -> bytes:
    """Build the strided read that pulls one servo slot from every servo.

    Parameters
    ----------
    base_address : int
        Address of ``servo_vector`` in the board's memory (from the symbol map).
    slot_name : str
        A key of :data:`SERVO_SLOTS`.
    count : int
        How many servo entries to read.
    """
    slot = _require_slot(SERVO_SLOTS, slot_name, "ServoStruct")
    return build_read(base_address + slot.offset, slot.size, count,
                      SERVO_STRUCT_SIZE)


def decode_servo_slot(slot_name: str, payload: bytes,
                      count: int = SERVO_VECTOR_LENGTH) -> List[int]:
    """Split a strided read's reply into one raw value per servo."""
    slot = _require_slot(SERVO_SLOTS, slot_name, "ServoStruct")
    expected = slot.size * count
    if len(payload) != expected:
        raise ProtocolError(
            "slot {!r}: expected {} bytes for {} servos, got {}"
            .format(slot_name, expected, count, len(payload)))
    return [slot.decode(payload[i * slot.size:(i + 1) * slot.size])[0]
            for i in range(count)]


def servo_slot_write(base_address: int, slot_name: str, servo_id: int,
                     value: int) -> bytes:
    """Write ONE servo's slot: ``servo_vector[servo_id].<slot> = value``.

    The strided read has no write counterpart on this board -- a write covers a
    contiguous range -- so this addresses a single entry, which is all any
    caller has needed. ``servo_id`` indexes the servo vector, and the address is
    ``base + servo_id * SERVO_STRUCT_SIZE + slot.offset``.
    """
    slot = _require_slot(SERVO_SLOTS, slot_name, "ServoStruct")
    if not 0 <= servo_id < SERVO_VECTOR_LENGTH:
        raise ProtocolError(
            "servo id {} outside 0..{}".format(servo_id,
                                               SERVO_VECTOR_LENGTH - 1))
    address = base_address + servo_id * SERVO_STRUCT_SIZE + slot.offset
    return build_write(address, slot.encode([value]))


def system_slot_write(base_address: int, slot_name: str,
                      values: Sequence) -> bytes:
    """Build the write that sets one ``SystemStruct`` member."""
    slot = _require_slot(SYSTEM_SLOTS, slot_name, "SystemStruct")
    return build_write(base_address + slot.offset, slot.encode(values))


def system_slot_read(base_address: int, slot_name: str) -> bytes:
    """Build the read of one ``SystemStruct`` member."""
    slot = _require_slot(SYSTEM_SLOTS, slot_name, "SystemStruct")
    return build_read(base_address + slot.offset, slot.size)


def control_slot_read(base_address: int, slot_name: str) -> bytes:
    """Build the read of one ``ControlStruct`` member."""
    slot = _require_slot(CONTROL_SLOTS, slot_name, "ControlStruct")
    return build_read(base_address + slot.offset, slot.size)


def _require_slot(slots: Mapping[str, Slot], name: str, struct_name: str) -> Slot:
    if name not in slots:
        raise ProtocolError(
            "{} has no slot {!r} in this driver's layout; known: {}"
            .format(struct_name, name, sorted(slots)))
    return slots[name]


# --------------------------------------------------------------------------
# Firmware symbol map
# --------------------------------------------------------------------------

#: Firmware symbols this driver needs addresses for.
REQUIRED_SYMBOLS = ("servo_vector", "SysB")


@dataclass(frozen=True)
class SymbolMap:
    """Addresses of the firmware globals this driver reads and writes.

    Why this is a *file* and not something we compute at run time
    ------------------------------------------------------------
    ``rcb4`` opens the firmware ELF at run time and resolves these symbols with
    pyelftools -- a host library lock-stepped to the firmware ELF's symbol
    layout, so a different ELF on the machine silently changes what the host
    reads. The lock-step
    itself is unavoidable while the firmware exposes state only as memory; what
    is avoidable is doing it implicitly, at run time, with a toolchain
    dependency, against whatever ELF happens to be on the machine.

    So the addresses are extracted **once, offline** (``actuo-kondo-symbols``),
    committed as data, and checked against the board's reported firmware
    version at connect time. A mismatch refuses to run: reading a struct at the
    wrong address would return plausible-looking numbers, and a plausible wrong
    angle is the one failure mode this stack must never produce.
    """

    version: str
    addresses: Mapping[str, int]

    def has(self, symbol: str) -> bool:
        """Whether this map provides an address for ``symbol``."""
        return symbol in self.addresses

    def address(self, symbol: str) -> int:
        if symbol not in self.addresses:
            raise ProtocolError(
                "symbol map for firmware {!r} has no address for {!r}; known: {}"
                .format(self.version, symbol, sorted(self.addresses)))
        return int(self.addresses[symbol])

    def require_version(self, reported: str) -> None:
        """Refuse to proceed if the board is not running the mapped firmware."""
        if reported.strip() != self.version.strip():
            raise ProtocolError(
                "firmware mismatch: board reports {!r} but the symbol map is "
                "for {!r}. Addresses would silently point at the wrong memory. "
                "Regenerate with: actuo-kondo-symbols <firmware.elf>"
                .format(reported.strip(), self.version.strip()))


def load_symbol_map(path: str) -> SymbolMap:
    """Load a symbol map YAML emitted by ``actuo-kondo-symbols``."""
    with open(path, "r") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise ProtocolError("{}: expected a YAML mapping".format(path))
    version = document.get("firmware_version")
    addresses = document.get("addresses")
    if not version or not isinstance(addresses, dict):
        raise ProtocolError(
            "{}: needs 'firmware_version' and an 'addresses' mapping".format(path))
    missing = [s for s in REQUIRED_SYMBOLS if s not in addresses]
    if missing:
        raise ProtocolError(
            "{}: symbol map is missing required symbols {}".format(path, missing))
    return SymbolMap(str(version), {str(k): int(v) for k, v in addresses.items()})


def dump_symbol_map(version: str, addresses: Mapping[str, int],
                    source: Optional[str] = None) -> str:
    """Render a symbol map as YAML (used by the offline extraction tool)."""
    document: Dict[str, object] = {
        "firmware_version": str(version),
        "addresses": {str(k): int(v) for k, v in sorted(addresses.items())},
    }
    if source is not None:
        document["source_elf"] = str(source)
    return yaml.safe_dump(document, default_flow_style=False, sort_keys=False)
