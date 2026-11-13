"""Host-side gateway protocol: CRC, framing and payload builders."""

from __future__ import annotations

import pytest

from actuo.drivers import gateway_protocol as gw


# --------------------------------------------------------------------------
# CRC and framing
# --------------------------------------------------------------------------


def test_crc_matches_the_standard_check_value():
    # "CRC-16" names a dozen mutually incompatible algorithms. Without this,
    # a host built from a different table would agree with itself and fail only
    # against real hardware.
    assert gw.crc16(b"123456789") == 0x29B1


def test_frame_round_trips():
    payload = bytes([1, 2, 3])
    frame = gw.decode_frame(gw.encode_frame(gw.Command.WRITE_TARGETS, 7, payload))
    assert frame.command == gw.Command.WRITE_TARGETS
    assert frame.seq == 7
    assert frame.payload == payload


def test_a_corrupted_payload_is_caught():
    raw = bytearray(gw.encode_frame(gw.Command.READ_STATE, 1, bytes(4)))
    raw[gw.HEADER_SIZE] ^= 0xFF
    with pytest.raises(gw.ProtocolError, match="CRC"):
        gw.decode_frame(bytes(raw))


def test_a_version_mismatch_refuses_the_payload():
    """A v2 body read with v1 offsets yields plausible numbers, and a plausible
    wrong servo angle is the one output this stack must never produce."""
    raw = bytearray(gw.encode_frame(gw.Command.READ_STATE, 1, bytes(4)))
    raw[2] = gw.PROTOCOL_VERSION + 1
    body = bytes(raw[2:gw.HEADER_SIZE + 4])
    raw[-2:] = gw.crc16(body).to_bytes(2, "little")
    with pytest.raises(gw.ProtocolError, match="version"):
        gw.decode_frame(bytes(raw))


def test_the_reader_resynchronises_after_garbage():
    reader = gw.FrameReader()
    good = gw.encode_frame(gw.Command.PING, 3, b"hi")
    frames = reader.feed(b"\x00\xff\x5a\x12" + good)
    assert len(frames) == 1
    assert frames[0].seq == 3
    assert reader.resyncs > 0


def test_the_reader_handles_a_split_frame():
    reader = gw.FrameReader()
    good = gw.encode_frame(gw.Command.PING, 4, b"abc")
    assert reader.feed(good[:5]) == []
    frames = reader.feed(good[5:])
    assert len(frames) == 1 and frames[0].seq == 4


# --------------------------------------------------------------------------
# Payload builders
# --------------------------------------------------------------------------


def test_target_entries_are_five_bytes_little_endian():
    payload = gw.encode_targets([(3, 17, 8000, True)])
    assert len(payload) == 1 + gw.TARGET_ENTRY_SIZE
    assert payload[1:] == bytes([3, 17, 8000 & 0xFF, 8000 >> 8, gw.TARGET_HOLD])


def test_a_released_target_clears_the_hold_bit():
    payload = gw.encode_targets([(0, 1, 0, False)])
    assert payload[-1] == 0


def test_state_with_a_wrong_count_is_refused():
    """A payload that disagrees with its own header means the two sides do not
    agree about the layout; parsing it anyway hides that."""
    payload = bytes([0] * gw.STATE_HEADER_SIZE)
    broken = bytearray(payload)
    broken[9] = 3                     # claims three servos, carries none
    with pytest.raises(gw.ProtocolError):
        gw.decode_state(bytes(broken))
