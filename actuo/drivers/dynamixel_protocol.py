"""DYNAMIXEL wire protocol, versions 1.0 and 2.0. Packet bytes only.

This module knows how to frame, checksum and parse a DYNAMIXEL packet. It knows
nothing about which register holds a goal position, what a count means in
radians, or which models exist -- those live in ``dynamixel_models`` (data) and
``actuators/dynamixel`` (units). Keeping the split means a new servo model is a
row of YAML, never a change here.

Why both versions live in one module
------------------------------------
They are not alternatives a robot picks once and forgets: a bus can be either,
and on an unlabelled robot the only honest way to find out is to *try both*
(``actuo-dxl-probe``). A module that implemented one version would push that
choice up into a caller-side ``if`` -- a per-protocol ``switch`` in shared
code, which this package avoids so that a new protocol never means editing the
core. Here the version is a parameter and every builder takes it.

The two wires differ more than a version byte suggests:

===============  =========================  ==============================
                 1.0 (AX / RX / MX-FW1)     2.0 (X / P / MX-FW2)
===============  =========================  ==============================
header           ``FF FF``                  ``FF FF FD 00``
length counts    params + 2                 params + 3
integrity        1-byte inverted sum        CRC-16 over the whole packet
escaping         none                       ``FF FF FD`` -> ``FF FF FD FD``
batch read       **none**                   ``SYNC_READ`` / ``FAST_SYNC_READ``
===============  =========================  ==============================

That last row has a real consequence and is not papered over here:
:func:`build_sync_read` raises on 1.0 rather than emulating a batch read with a
loop of singles behind the caller's back: an unsupported capability fails
loudly instead of being silently degraded. The bus decides
what to do about it, in the open.

Provenance
----------
Every constant and layout below comes from the ROBOTIS e-Manual; see
``actuo/drivers/references.md`` entries [14]-[16]. The example packets printed in
[15] are frozen as test vectors in ``tests/test_dynamixel_protocol.py`` -- an
implementation that agrees with a spec it also authored proves nothing, so the
vendor's own bytes are the fixture.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from actuo.core.contracts import CapabilityError

#: Protocol versions. Plain ints, because they are what the e-Manual and every
#: vendor tool call them, and a config file should read ``protocol: 2``.
PROTOCOL_1 = 1
PROTOCOL_2 = 2
PROTOCOLS = (PROTOCOL_1, PROTOCOL_2)

HEADER_1 = b"\xff\xff"
HEADER_2 = b"\xff\xff\xfd\x00"

#: Written to as a device id, answered by nobody (except PING / SYNC_READ).
BROADCAST_ID = 0xFE
#: 0..252 are assignable; 253 is reserved, 254 broadcast, 255 unusable.
MAX_ID = 252

# --- instructions (references.md [15]) --------------------------------------
PING = 0x01
READ = 0x02
WRITE = 0x03
REG_WRITE = 0x04
ACTION = 0x05
FACTORY_RESET = 0x06
REBOOT = 0x08          # 2.0 only
CLEAR = 0x10           # 2.0 only
STATUS = 0x55          # 2.0 only: the instruction byte of a status packet
SYNC_READ = 0x82       # 2.0 only
SYNC_WRITE = 0x83
FAST_SYNC_READ = 0x8A  # 2.0 only

#: 2.0 status-packet error field: bit 7 is an alert flag, bits 6-0 a code.
ERROR_ALERT_BIT = 0x80


class ProtocolError(ValueError):
    """A packet could not be built -- a programming error, not a bus glitch.

    Deliberately distinct from "no reply / bad checksum", which is a dropped
    read and is signalled by ``parse_status`` returning ``None``. Conflating the
    two would make a genuine encoding bug look like line noise and get retried
    forever.
    """


def _check_protocol(protocol: int) -> int:
    if protocol not in PROTOCOLS:
        raise ProtocolError(
            "unknown DYNAMIXEL protocol {!r}; supported: {}".format(
                protocol, list(PROTOCOLS)))
    return protocol


def header(protocol: int) -> bytes:
    """The sync bytes that open a packet of this protocol version."""
    return HEADER_1 if _check_protocol(protocol) == PROTOCOL_1 else HEADER_2


# --- integrity ---------------------------------------------------------------
def checksum_1(dxl_id: int, length: int, instruction: int, params: bytes) -> int:
    """Protocol 1.0: ``~(id + len + instr + sum(params)) & 0xFF``."""
    return (~(dxl_id + length + instruction + sum(params))) & 0xFF


def crc16(data: bytes) -> int:
    """CRC-16/IBM as used by protocol 2.0: poly 0x8005, init 0, not reflected.

    Computed bit-wise rather than via the e-Manual's 256-entry lookup table.
    The table is an optimisation of exactly this loop, and at the packet rates
    this driver runs (tens of bytes per millisecond, in a process that is
    explicitly *not* the hard real-time loop, which lives in the servos' own
    MCUs) the table would buy
    nothing and cost 256 transcribed constants that no reader can verify by eye.
    The vendor's own example packets pin the result either way.
    """
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x8005) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


# --- 2.0 byte stuffing -------------------------------------------------------
def add_stuffing(body: bytes) -> bytes:
    """Escape ``FF FF FD`` as ``FF FF FD FD`` inside a 2.0 packet body.

    ``body`` is the instruction byte plus its parameters -- the only region that
    can contain arbitrary data. The header's reserved ``00`` and the id/length
    fields cannot produce the pattern (an id never exceeds 0xFE), so escaping
    them would only make the length arithmetic harder to check.

    Without this, a goal position whose bytes happen to read ``FF FF FD`` would
    be resynchronised on as a header by the receiver, and the servo would act on
    a truncated packet whose CRC nonetheless passes on the fragment. That is the
    "plausible wrong angle" failure this stack refuses to produce.
    """
    out = bytearray()
    ff_run = 0  # trailing consecutive 0xFF bytes, saturating at two
    for byte in body:
        out.append(byte)
        if byte == 0xFF:
            # Saturate rather than reset: in ``FF FF FF FD`` the *last two* FFs
            # still form the pattern, so a third FF must not clear the run. That
            # asymmetry is easy to miss and produces a packet the servo silently
            # re-syncs on rather than one that visibly fails.
            ff_run = min(ff_run + 1, 2)
        elif byte == 0xFD and ff_run >= 2:
            out.append(0xFD)
            ff_run = 0
        else:
            ff_run = 0
    return bytes(out)


def remove_stuffing(body: bytes) -> bytes:
    """Inverse of :func:`add_stuffing`."""
    out = bytearray()
    index = 0
    while index < len(body):
        out.append(body[index])
        if (body[index] == 0xFD and len(out) >= 3
                and out[-3] == 0xFF and out[-2] == 0xFF):
            index += 2  # skip the escape byte that follows the pattern
            continue
        index += 1
    return bytes(out)


# --- building ----------------------------------------------------------------
def build_packet(protocol: int, dxl_id: int, instruction: int,
                 params: bytes = b"") -> bytes:
    """One instruction packet, ready for the wire."""
    _check_protocol(protocol)
    if not 0 <= dxl_id <= 0xFE:
        raise ProtocolError("device id {} out of range 0..254".format(dxl_id))

    if protocol == PROTOCOL_1:
        length = len(params) + 2  # instruction + checksum
        if length > 0xFF:
            raise ProtocolError(
                "protocol 1.0 packet too long: {} bytes of parameters "
                "(the length field is one byte)".format(len(params)))
        body = bytes([dxl_id, length, instruction]) + params
        return HEADER_1 + body + bytes(
            [checksum_1(dxl_id, length, instruction, params)])

    stuffed = add_stuffing(bytes([instruction]) + params)
    length = len(stuffed) + 2  # + CRC low/high
    packet = bytearray(HEADER_2)
    packet.append(dxl_id)
    packet.append(length & 0xFF)
    packet.append((length >> 8) & 0xFF)
    packet.extend(stuffed)
    crc = crc16(bytes(packet))
    packet.append(crc & 0xFF)
    packet.append((crc >> 8) & 0xFF)
    return bytes(packet)


def build_ping(protocol: int, dxl_id: int) -> bytes:
    return build_packet(protocol, dxl_id, PING)


def build_read(protocol: int, dxl_id: int, address: int, length: int) -> bytes:
    if protocol == PROTOCOL_1:
        params = bytes([address & 0xFF, length & 0xFF])
    else:
        params = bytes([address & 0xFF, (address >> 8) & 0xFF,
                        length & 0xFF, (length >> 8) & 0xFF])
    return build_packet(protocol, dxl_id, READ, params)


def build_write(protocol: int, dxl_id: int, address: int, data: bytes) -> bytes:
    if protocol == PROTOCOL_1:
        params = bytes([address & 0xFF]) + data
    else:
        params = bytes([address & 0xFF, (address >> 8) & 0xFF]) + data
    return build_packet(protocol, dxl_id, WRITE, params)


def build_sync_write(protocol: int, address: int, data_length: int,
                     items: Sequence[Tuple[int, bytes]]) -> bytes:
    """One SYNC_WRITE setting the same register on many servos.

    This is the batch primitive the buses are built on: every joint on
    the bus is commanded in a single transaction, not N round trips. Both
    protocol versions carry it; only the address/length field widths differ.
    """
    if not items:
        raise ProtocolError("sync-write with no items")
    if protocol == PROTOCOL_1:
        params = bytearray([address & 0xFF, data_length & 0xFF])
    else:
        params = bytearray([address & 0xFF, (address >> 8) & 0xFF,
                            data_length & 0xFF, (data_length >> 8) & 0xFF])
    for dxl_id, data in items:
        if len(data) != data_length:
            raise ProtocolError(
                "sync-write data for id {} is {} bytes, expected {}".format(
                    dxl_id, len(data), data_length))
        params.append(dxl_id)
        params.extend(data)
    return build_packet(protocol, BROADCAST_ID, SYNC_WRITE, bytes(params))


def build_sync_read(protocol: int, address: int, data_length: int,
                    ids: Sequence[int]) -> bytes:
    """One SYNC_READ asking many servos for the same register block (2.0 only).

    Protocol 1.0 has no such instruction. Rather than quietly emitting N READs
    and calling it a batch, this raises: the caller's cycle budget changes by a
    factor of N, and hiding that is exactly the kind of silent degradation that
    makes a performance claim unfalsifiable: batch I/O is the primary
    operation, and bus cost is something to measure, not assume.
    """
    if _check_protocol(protocol) == PROTOCOL_1:
        raise CapabilityError(
            "protocol 1.0 has no SYNC_READ instruction; read each servo with "
            "build_read() and account for the N round trips explicitly")
    if not ids:
        raise ProtocolError("sync-read with no ids")
    params = bytearray([address & 0xFF, (address >> 8) & 0xFF,
                        data_length & 0xFF, (data_length >> 8) & 0xFF])
    params.extend(ids)
    return build_packet(protocol, BROADCAST_ID, SYNC_READ, bytes(params))


# --- framing (used by the transport) -----------------------------------------
def total_length(protocol: int, buffer: bytes) -> Optional[int]:
    """How many bytes the packet starting at ``buffer[0]`` occupies.

    ``None`` means "not enough bytes yet to tell". The transport calls this so
    that framing has exactly one definition per protocol version; a transport
    that re-derived it would be a second source of truth for where a packet
    ends, and two sources of truth for one protocol drift apart.
    """
    _check_protocol(protocol)
    if protocol == PROTOCOL_1:
        if len(buffer) < 4:
            return None
        return buffer[3] + 4          # header(2) + id + len + body
    if len(buffer) < 7:
        return None
    return (buffer[5] | (buffer[6] << 8)) + 7  # header(4) + id + len(2) + body


def parse_status(protocol: int, packet: bytes) -> Optional[Tuple[int, int, bytes]]:
    """Parse a status packet into ``(id, error, params)``.

    Returns ``None`` for anything malformed -- bad header, wrong length, failed
    checksum/CRC. A serial line drops and re-syncs bytes, so one corrupt reply
    is a missed read, not a fault; the caller treats ``None`` as "no fresh state
    this cycle" and the bus counts it. Raising here would turn ordinary line
    noise into a crash.
    """
    _check_protocol(protocol)
    if protocol == PROTOCOL_1:
        if len(packet) < 6 or packet[0:2] != HEADER_1:
            return None
        dxl_id, length = packet[2], packet[3]
        if len(packet) != length + 4:
            return None
        error, data = packet[4], packet[5:-1]
        if checksum_1(dxl_id, length, error, data) != packet[-1]:
            return None
        return dxl_id, error, bytes(data)

    if len(packet) < 11 or packet[0:4] != HEADER_2:
        return None
    dxl_id = packet[4]
    length = packet[5] | (packet[6] << 8)
    if len(packet) != length + 7:
        return None
    expected = packet[-2] | (packet[-1] << 8)
    if crc16(packet[:-2]) != expected:
        return None
    body = remove_stuffing(bytes(packet[7:-2]))
    if len(body) < 2 or body[0] != STATUS:
        return None  # an echoed instruction is not a status packet
    return dxl_id, body[1], bytes(body[2:])


def parse_ping_status(protocol: int, packet: bytes
                      ) -> Optional[Tuple[int, int, int]]:
    """Parse a PING reply into ``(id, model_number, firmware_version)``.

    Protocol 2.0 answers a PING with the model number and firmware version in
    its parameters, which is what makes model identification possible without
    the user telling us what is plugged in. Protocol 1.0's PING carries no
    parameters, so the model number must be READ from address 0 afterwards --
    signalled here by returning ``None`` for the model, never a guess.
    """
    parsed = parse_status(protocol, packet)
    if parsed is None:
        return None
    dxl_id, _error, params = parsed
    if protocol == PROTOCOL_1 or len(params) < 3:
        return None
    model = params[0] | (params[1] << 8)
    return dxl_id, model, params[2]


# --- small integer helpers ---------------------------------------------------
def to_le(value: int, size: int) -> bytes:
    """Little-endian bytes, two's complement for negatives.

    Every multi-byte control-table field is little-endian in both protocol
    versions -- unlike the Feetech SCS series on the neighbouring plugin, which
    is big-endian. Getting this backwards yields plausible but wrong angles, so
    it is one helper used everywhere rather than inline shifts.
    """
    if value < 0:
        value += 1 << (8 * size)
    return int(value).to_bytes(size, "little")


def from_le(data: bytes, signed: bool = False) -> int:
    return int.from_bytes(data, "little", signed=signed)
