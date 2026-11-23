"""FEETECH packet framing: building instruction packets, parsing status packets.

Every FEETECH series (SCS, STS, SMS, HLS) frames its packets the DYNAMIXEL 1.0
way::

    instruction : FF FF ID LEN INSTR PARAM.. CHECKSUM
    status      : FF FF ID LEN ERROR DATA..  CHECKSUM

``LEN = len(params) + 2`` and ``CHECKSUM = ~(ID + LEN + INSTR + sum(params))``.
Unlike DYNAMIXEL 1.0, FEETECH servos answer a SYNC_READ (0x82) whose parameters
are ``address, length, id, id, ...``: one broadcast, then one status packet per
listed id, in the listed order (feetech-cli's ``PacketHandler.sync_read`` does
the same; it was checked on STS3215 servos).

Only framing lives here. Register addresses, byte order of multi-byte values
and their units belong to the actuator plugin: the SCS series is big-endian on
the wire, the STS/SMS/HLS series little-endian, and the framing is the same.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

HEADER = b"\xff\xff"
BROADCAST_ID = 0xFE

PING = 0x01
READ = 0x02
WRITE = 0x03
SYNC_READ = 0x82
SYNC_WRITE = 0x83


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
    operation in this stack, not a loop of individual writes. A broadcast is
    never answered, so nothing is left in the receive buffer.
    """
    params = bytearray([address, per_servo_len])
    for servo_id, data in items:
        if len(data) != per_servo_len:
            raise ValueError("sync-write data for id {} is {} bytes, expected {}"
                             .format(servo_id, len(data), per_servo_len))
        params.append(servo_id)
        params.extend(data)
    return build_packet(BROADCAST_ID, SYNC_WRITE, bytes(params))


def build_sync_read(address: int, length: int, ids: Sequence[int]) -> bytes:
    """One SYNC_READ asking every listed id for the same register block."""
    ids = [int(i) for i in ids]
    if not ids:
        raise ValueError("sync-read with no ids")
    if len(set(ids)) != len(ids):
        raise ValueError("sync-read ids must be distinct, got {}".format(ids))
    return build_packet(BROADCAST_ID, SYNC_READ,
                        bytes([address & 0xFF, length & 0xFF] + ids))


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
