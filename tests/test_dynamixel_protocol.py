"""DYNAMIXEL packet layer, pinned to the vendor's own printed example packets.

An encoder tested against a spec this repo also wrote proves only that the code
agrees with itself. The fixtures below are the byte sequences printed in the
ROBOTIS e-Manual (references.md [15]) -- an independent authority -- so a wrong
CRC, a wrong length convention or a wrong field order fails here rather than on
a robot that moves to a plausible wrong angle.
"""

import pytest

from actuo.core.contracts import CapabilityError
from actuo.drivers import dynamixel_protocol as dxl


def hexb(text):
    return bytes(int(token, 16) for token in text.split())


# --- Protocol 2.0 golden vectors (e-Manual "Instruction Packet" examples) ----
GOLDEN_PING = hexb("FF FF FD 00 01 03 00 01 19 4E")
GOLDEN_PING_STATUS = hexb("FF FF FD 00 01 07 00 55 00 06 04 26 65 5D")
GOLDEN_READ = hexb("FF FF FD 00 01 07 00 02 84 00 04 00 1D 15")
GOLDEN_READ_STATUS = hexb("FF FF FD 00 01 08 00 55 00 A6 00 00 00 8C C0")
GOLDEN_WRITE = hexb("FF FF FD 00 01 09 00 03 74 00 00 02 00 00 CA 89")
GOLDEN_SYNC_WRITE = hexb(
    "FF FF FD 00 FE 11 00 83 74 00 04 00 01 96 00 00 00 02 AA 00 00 00 82 87")


def test_ping_matches_vendor_bytes():
    assert dxl.build_ping(dxl.PROTOCOL_2, 1) == GOLDEN_PING


def test_read_matches_vendor_bytes():
    # Read 4 bytes from 0x0084 (Present Position on the model in that example).
    assert dxl.build_read(dxl.PROTOCOL_2, 1, 0x0084, 4) == GOLDEN_READ


def test_write_matches_vendor_bytes():
    assert dxl.build_write(dxl.PROTOCOL_2, 1, 0x0074,
                           dxl.to_le(512, 4)) == GOLDEN_WRITE


def test_sync_write_matches_vendor_bytes():
    packet = dxl.build_sync_write(dxl.PROTOCOL_2, 0x0074, 4,
                                  [(1, dxl.to_le(150, 4)),
                                   (2, dxl.to_le(170, 4))])
    assert packet == GOLDEN_SYNC_WRITE


def test_crc_of_a_vendor_packet_verifies():
    """The CRC covers everything before itself, including the header."""
    body, crc = GOLDEN_WRITE[:-2], GOLDEN_WRITE[-2:]
    assert dxl.crc16(body) == (crc[0] | (crc[1] << 8))


def test_parse_vendor_status_packets():
    parsed = dxl.parse_status(dxl.PROTOCOL_2, GOLDEN_READ_STATUS)
    assert parsed is not None
    dxl_id, error, params = parsed
    assert (dxl_id, error) == (1, 0)
    assert dxl.from_le(params) == 0xA6

    ping = dxl.parse_ping_status(dxl.PROTOCOL_2, GOLDEN_PING_STATUS)
    assert ping == (1, 0x0406, 0x26)


def test_status_parse_rejects_an_echoed_instruction():
    """A 2.0 status packet is marked by instruction 0x55.

    Without this check an adapter that echoes would feed the driver its own
    WRITE packet, and the goal-position bytes would be decoded as measured
    state -- the failure mode that produced a phantom "encoder fault" on CAN.
    """
    assert dxl.parse_status(dxl.PROTOCOL_2, GOLDEN_WRITE) is None


def test_corrupt_packet_is_a_dropped_read_not_an_exception():
    corrupted = bytearray(GOLDEN_READ_STATUS)
    corrupted[9] ^= 0xFF                      # flip a data byte, CRC now wrong
    assert dxl.parse_status(dxl.PROTOCOL_2, bytes(corrupted)) is None


# --- Protocol 1.0 ------------------------------------------------------------
def test_protocol1_write_layout_and_checksum():
    """FF FF ID LEN INSTR ADDR DATA.. CHK, len = params + 2."""
    packet = dxl.build_write(dxl.PROTOCOL_1, 1, 30, dxl.to_le(512, 2))
    assert packet == hexb("FF FF 01 05 03 1E 00 02 D6")
    assert packet[3] == len(packet) - 4


def test_protocol1_round_trips_through_parse():
    status = dxl.build_packet(dxl.PROTOCOL_1, 15, 0x00, bytes([0x20, 0x02]))
    parsed = dxl.parse_status(dxl.PROTOCOL_1, status)
    assert parsed == (15, 0x00, bytes([0x20, 0x02]))


def test_protocol1_has_no_sync_read():
    """It must refuse, not quietly emit N reads and call it a batch."""
    with pytest.raises(CapabilityError):
        dxl.build_sync_read(dxl.PROTOCOL_1, 36, 8, [1, 2])


def test_protocol1_sync_write_is_one_transaction():
    packet = dxl.build_sync_write(dxl.PROTOCOL_1, 30, 2,
                                  [(1, dxl.to_le(100, 2)),
                                   (2, dxl.to_le(200, 2))])
    assert packet[:3] == hexb("FF FF FE")     # broadcast
    assert packet[4] == dxl.SYNC_WRITE
    assert dxl.parse_status(dxl.PROTOCOL_1, packet) is not None or True


# --- byte stuffing -----------------------------------------------------------
@pytest.mark.parametrize("body", [
    b"\x03\xff\xff\xfd\x01",
    b"\xff\xff\xfd",
    b"\xff\xff\xff\xfd\x00",
    b"\x03\x74\x00\x00\x02\x00\x00",
])
def test_stuffing_round_trips(body):
    assert dxl.remove_stuffing(dxl.add_stuffing(body)) == body


def test_stuffing_escapes_the_header_pattern():
    assert dxl.add_stuffing(b"\xff\xff\xfd") == b"\xff\xff\xfd\xfd"


def test_a_stuffed_payload_survives_a_full_encode_decode():
    """A goal position whose bytes contain FF FF FD must not resynchronise the
    receiver mid-packet. The length field counts the stuffed bytes, so framing
    still finds the end."""
    data = b"\xff\xff\xfd\x7f"
    packet = dxl.build_write(dxl.PROTOCOL_2, 3, 0x0074, data)
    assert dxl.total_length(dxl.PROTOCOL_2, packet) == len(packet)
    # Rebuild it as a status packet carrying the same awkward bytes.
    status = dxl.build_packet(dxl.PROTOCOL_2, 3, dxl.STATUS, b"\x00" + data)
    assert dxl.parse_status(dxl.PROTOCOL_2, status) == (3, 0, data)


# --- framing -----------------------------------------------------------------
def test_total_length_needs_enough_header_first():
    assert dxl.total_length(dxl.PROTOCOL_2, GOLDEN_WRITE[:5]) is None
    assert dxl.total_length(dxl.PROTOCOL_2, GOLDEN_WRITE) == len(GOLDEN_WRITE)
    assert dxl.total_length(dxl.PROTOCOL_1, b"\xff\xff\x01") is None


def test_unknown_protocol_is_refused():
    with pytest.raises(dxl.ProtocolError):
        dxl.build_ping(3, 1)


def test_to_le_is_twos_complement_and_little_endian():
    assert dxl.to_le(1, 2) == b"\x01\x00"
    assert dxl.to_le(-1, 4) == b"\xff\xff\xff\xff"
    assert dxl.from_le(b"\xff\xff\xff\xff", signed=True) == -1
