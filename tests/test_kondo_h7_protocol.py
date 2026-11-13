"""Byte-level tests for the Kondo ARM-H7 packet protocol.

Every expected byte string below was **generated from ``rcb4``**, the
implementation that has been driving this board on real robots -- not written by
hand from a manual and not derived from our own encoder. That makes these tests
an equivalence check against a known-good implementation, the same technique
that pinned the CubeMars port against an earlier, hardware-proven script
(``tests/test_cubemars.py``).

Generated with ``rcb4`` at commit ``dc1ccf559ea90e0334a652711f3d2fec9aaab6a3``
(github.com/iory/rcb4) by capturing the argument of ``serial_write``::

    itf = ARMH7Interface.__new__(ARMH7Interface)
    itf.serial_write = capture
    itf.servo_on_states_dict = {i: True for i in (0, 1, 5, 23, 30)}
    itf._armh7_address = {'servo_vector': 0x24000100}
    itf.servo_angle_vector([0, 1, 5, 23, 30],
                           [7500, 8000, 7000, 32767, 32768], velocity=127)

If one of these fails, the driver has diverged from the board, not the other way
round.

The framing, ack/version and batch servo move tests moved to the rcb4
repository (``tests/rcb4_tests/test_protocol_framing.py`` there) with the code, 2026-11-09.
"""

import struct

import pytest

from actuo.drivers import kondo_h7_protocol as protocol


#: Address ``rcb4`` was told ``servo_vector`` lived at when generating the
#: reference vectors. Arbitrary but fixed -- the packets encode it verbatim.
SERVO_VECTOR = 0x24000100


def framed(body):
    """Helper: the complete on-wire packet for a body."""
    return protocol.frame(body)


# --------------------------------------------------------------------------
# Servo command packets
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Memory access packets
# --------------------------------------------------------------------------


def test_flat_read_matches_rcb4_memory_read_aux():
    """A flat block read: count 1, element size = the whole block, no stride."""
    body = protocol.build_read(SERVO_VECTOR, element_size=120)
    assert framed(body).hex() == "0bfb0001002401780000a4"


def test_write_matches_rcb4_memory_write():
    """Writing the two bytes of a servo's ref_angle."""
    body = protocol.build_write(SERVO_VECTOR + 4, bytes([0x4C, 0x1D]))
    assert framed(body).hex() == "0dfc04010024010200004c1d9e"


def test_strided_servo_slot_read_matches_rcb4():
    """The batch state read: every servo's current_angle in one round trip.

    This is the packet that makes the port viable -- 36 servos, one request.
    """
    body = protocol.servo_slot_read(SERVO_VECTOR, "current_angle")
    assert framed(body).hex() == "0bfb0601002424027800cf"


def test_call_matches_rcb4_cfunc_call():
    body = protocol.build_call(0x08001234)
    assert framed(body).hex() == "08fa341200080050"


def test_read_larger_than_the_reply_buffer_is_refused():
    """A 300-byte read would come back truncated; refuse instead of truncating."""
    with pytest.raises(protocol.ProtocolError):
        protocol.build_read(SERVO_VECTOR, element_size=250, count=2)


def test_reading_an_unknown_slot_names_the_known_ones():
    with pytest.raises(protocol.ProtocolError) as excinfo:
        protocol.servo_slot_read(SERVO_VECTOR, "torque")
    assert "current_angle" in str(excinfo.value)


# --------------------------------------------------------------------------
# Framing
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Board memory layout
# --------------------------------------------------------------------------


def test_servo_struct_layout_matches_the_firmware_declaration():
    """Offsets and stride, as computed from the firmware's C declaration.

    These numbers come from ``rcb4/struct_header.py``'s ``ServoStruct`` under
    natural alignment. They are the difference between reading a joint angle and
    reading a frame counter, so they are pinned rather than trusted.
    """
    assert protocol.SERVO_STRUCT_SIZE == 120
    assert protocol.SERVO_VECTOR_LENGTH == 36
    expected = {
        "flag": (0, "uint8"),
        "mode": (1, "uint8"),
        "rotation": (2, "uint8"),
        "port": (3, "uint8"),
        "ref_angle": (4, "uint16"),
        "current_angle": (6, "uint16"),
        "trim": (8, "int16"),
        "stretch": (81, "uint8"),
        "speed": (82, "uint8"),
        "current_limit": (83, "uint8"),
        "temperature_limit": (84, "uint8"),
        "timeout_count": (86, "uint8"),
        "rxerror_count": (87, "uint8"),
        "init_state": (96, "uint8"),
        "init_count": (97, "uint8"),
        "error_angle": (98, "int16"),
        "feedback": (116, "uint8"),
        "move_state": (117, "uint8"),
        "current": (118, "uint8"),
        "temperature": (119, "uint8"),
    }
    for name, (offset, ctype) in expected.items():
        slot = protocol.SERVO_SLOTS[name]
        assert (slot.offset, slot.ctype) == (offset, ctype), name


def test_system_struct_layout_matches_the_firmware_declaration():
    assert protocol.SYSTEM_STRUCT_SIZE == 76
    assert protocol.SYSTEM_SLOTS["battery_voltage"].offset == 28
    assert protocol.SYSTEM_SLOTS["servo_current_read_flag"].offset == 49
    assert protocol.SYSTEM_SLOTS["servo_temperature_read_flag"].offset == 50
    assert protocol.SYSTEM_SLOTS["ics_comm_stop"].offset == 64
    assert protocol.SYSTEM_SLOTS["ics_comm_stop"].size == 6


def test_control_struct_layout_matches_the_firmware_declaration():
    assert protocol.CONTROL_STRUCT_SIZE == 424
    assert protocol.CONTROL_SLOTS["device_port"].offset == 356
    assert protocol.CONTROL_SLOTS["device_port"].size == 64
    assert protocol.CONTROL_SLOTS["device_timeout_th"].offset == 420


def test_decode_servo_slot_splits_the_strided_reply():
    payload = struct.pack("<3H", 7500, 8000, 3500)
    assert protocol.decode_servo_slot("current_angle", payload, count=3) == \
        [7500, 8000, 3500]


def test_decode_servo_slot_applies_the_slots_signedness():
    """error_angle is int16 on the board: 0xFFFF is -1, not 65535."""
    payload = struct.pack("<2h", -1, 25)
    assert protocol.decode_servo_slot("error_angle", payload, count=2) == [-1, 25]


def test_decode_servo_slot_refuses_a_short_reply():
    """A short reply means the stream is out of step; decoding it anyway would
    attribute one servo's angle to another."""
    with pytest.raises(protocol.ProtocolError):
        protocol.decode_servo_slot("current_angle", struct.pack("<2H", 1, 2),
                                   count=3)


def test_system_slot_write_targets_the_right_address():
    body = protocol.system_slot_write(0x24000000, "servo_current_read_flag", [1])
    assert body[1:5] == (0x24000000 + 49).to_bytes(4, "little")
    assert body[-1:] == bytes([1])


def test_control_slot_read_targets_the_right_address():
    body = protocol.control_slot_read(0x24000000, "device_port")
    assert body[1:5] == (0x24000000 + 356).to_bytes(4, "little")
    assert body[5] == 1
    assert body[6] == 64


# --------------------------------------------------------------------------
# Symbol map
# --------------------------------------------------------------------------


def test_symbol_map_refuses_a_firmware_mismatch():
    """The whole point of the map: wrong firmware means wrong addresses, which
    would read plausible-looking numbers out of the wrong memory."""
    symbols = protocol.SymbolMap("v0.0.6", {"servo_vector": 1, "SysB": 2})
    symbols.require_version(" v0.0.6 ")  # whitespace only
    with pytest.raises(protocol.ProtocolError) as excinfo:
        symbols.require_version("v0.0.7")
    assert "actuo-kondo-symbols" in str(excinfo.value)


def test_symbol_map_has_reports_optional_symbols():
    symbols = protocol.SymbolMap("v0.0.6", {"servo_vector": 1, "SysB": 2})
    assert symbols.has("SysB") is True
    assert symbols.has("Control") is False


def test_symbol_map_round_trips_through_yaml(tmp_path):
    path = tmp_path / "symbols.yaml"
    path.write_text(protocol.dump_symbol_map(
        "v0.0.6", {"servo_vector": 0x24000100, "SysB": 0x24000000}))
    loaded = protocol.load_symbol_map(str(path))
    assert loaded.version == "v0.0.6"
    assert loaded.address("servo_vector") == 0x24000100


def test_symbol_map_requires_the_symbols_the_driver_uses(tmp_path):
    path = tmp_path / "symbols.yaml"
    path.write_text(protocol.dump_symbol_map("v0.0.6", {"servo_vector": 1}))
    with pytest.raises(protocol.ProtocolError):
        protocol.load_symbol_map(str(path))
