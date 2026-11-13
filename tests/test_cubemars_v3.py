"""3.0-generation force-control frame tests, checked against the vendor's code.

The reference implementation here is a transcription of ``pack_cmd`` from the
AK Series Module Product Manual V3, section 4.2. Comparing against it rather
than against hand-written expected bytes is the point: five fields straddle
byte boundaries, and a test written from the same reading as the code would
agree with it while both were wrong.

The one place the transcription deliberately departs from the manual is the
scaling denominator. The manual writes ``(1 << bits) / span``, which maps the
top of the range to ``1 << bits`` -- 4096 in a 12-bit field, one count past
what fits. ``(1 << bits) - 1`` is the value that round-trips, and is what
:mod:`cubemars_mit` already uses for the same reason.
"""

import math

import pytest

from actuo.core.contracts import ActuatorCommand, Frame, OperatingMode
from actuo.drivers.actuators.cubemars_v3 import (
    FORCE_CONTROL_PACKET,
    CubeMarsV3Actuator,
    CubeMarsV3Calibration,
)

#: AK70-9's published ranges, manual V3 section 4.2. On a 12-joint CubeMars
#: biped, four joints are AK70-9, and all four answered packet 8 on 2026-09-12.
P_MAX, V_MAX, T_MAX = 12.56, 30.0, 32.0

NODE_ID = 5  # right_hip_yaw, the joint the generation probe was run on


def reference_pack(p_des, v_des, kp, kd, t_ff):
    """The V3 manual's pack_cmd, transcribed."""
    def f2u(x, lo, hi, bits):
        return int((min(max(x, lo), hi) - lo) * ((1 << bits) - 1) / (hi - lo))

    kp_i = f2u(kp, 0.0, 500.0, 12)
    kd_i = f2u(kd, 0.0, 5.0, 12)
    p = f2u(p_des, -P_MAX, P_MAX, 16)
    v = f2u(v_des, -V_MAX, V_MAX, 12)
    t = f2u(t_ff, -T_MAX, T_MAX, 12)
    return bytes([
        kp_i >> 4,
        ((kp_i & 0xF) << 4) | (kd_i >> 8),
        kd_i & 0xFF,
        p >> 8,
        p & 0xFF,
        v >> 4,
        ((v & 0xF) << 4) | (t >> 8),
        t & 0xFF,
    ])


def make(**overrides):
    spec = dict(gear_ratio=9.0, direction=1, zero_offset_counts=0.0,
                pole_pairs=21, p_max=P_MAX, v_max=V_MAX, t_max=T_MAX,
                kp=60.0, kd=2.0)
    spec.update(overrides)
    return CubeMarsV3Actuator(NODE_ID, CubeMarsV3Calibration(**spec))


def position_command(rad, **kw):
    return ActuatorCommand(mode=OperatingMode.POSITION, position=rad, **kw)


# --- the wire -------------------------------------------------------------

def test_payload_matches_the_manuals_pack_cmd():
    actuator = make()
    frame = actuator.encode_command(position_command(0.3, velocity_ff=1.5,
                                                     current_ff=4.0))
    assert frame.payload == reference_pack(0.3, 1.5, 60.0, 2.0, 4.0)


def test_arbitration_id_is_packet_8_extended():
    """The control mode lives in the arbitration id, as it does in servo mode."""
    frame = make().encode_command(position_command(0.0))
    assert frame.address == NODE_ID | (FORCE_CONTROL_PACKET << 8)
    assert frame.meta["extended"] is True


def test_byte_order_is_not_the_2_0_mit_order():
    """kp leads here; position leads in 2.0 MIT. Same fields, different bytes.

    Pinned because the two generations can share one CAN bus (they do on a
    CubeMars biped), so reaching
    for the wrong packer is a live possibility rather than a hypothetical one.
    Checked against the other module's real encoder, not a transcription of it.
    """
    from actuo.drivers.actuators.cubemars_mit import (
        CubeMarsMitActuator, MitCalibration)

    mit = CubeMarsMitActuator(NODE_ID, MitCalibration(
        gear_ratio=9.0, p_max=P_MAX, v_max=V_MAX, t_max=T_MAX,
        kp=60.0, kd=2.0))
    command = position_command(0.3, velocity_ff=1.5, current_ff=4.0)
    assert make().encode_command(command).payload != \
        mit.encode_command(command).payload

    # Field by field: kp at full scale must fill byte 0 and byte 1's high
    # nibble, and position must be mid-scale in bytes 3 and 4 rather than 0
    # and 1. Under the 2.0 layout both claims are false.
    payload = make(kp=500.0, kd=0.0).encode_command(position_command(0.0)).payload
    assert payload[0] == 0xFF
    assert payload[1] & 0xF0 == 0xF0
    assert payload[3] == 0x7F and payload[4] == 0xFF


def test_release_zeroes_every_term_of_the_law():
    """tau = kp*(..) + kd*(..) + t_ff is zero whatever the targets say."""
    payload = make().encode_release().payload
    kp_int = (payload[0] << 4) | (payload[1] >> 4)
    kd_int = ((payload[1] & 0x0F) << 8) | payload[2]
    t_int = ((payload[6] & 0x0F) << 8) | payload[7]
    assert kp_int == 0
    assert kd_int == 0
    # Zero torque is mid-scale in a signed field, not a zero code.
    assert t_int == pytest.approx((1 << 12) // 2, abs=1)


def test_no_arming_frames():
    """Unlike 2.0 MIT, nothing has to be sent before commands are obeyed."""
    assert make().encode_mode(OperatingMode.POSITION) == []


# --- calibration ----------------------------------------------------------

def test_position_round_trips_through_the_servo_datum():
    """The command and the status frame must agree about where zero is."""
    cal = CubeMarsV3Calibration(gear_ratio=9.0, direction=-1,
                                zero_offset_counts=154.0, window_offset=2,
                                p_max=P_MAX, v_max=V_MAX, t_max=T_MAX,
                                kp=1.0, kd=1.0)
    for rad in (-1.2, 0.0, 0.35):
        assert cal.wire_rad_to_rad(cal.rad_to_wire_rad(rad)) == pytest.approx(rad)


def test_zero_offset_and_direction_reach_the_position_field():
    """A datum that the packer ignored would move the joint by the offset."""
    plain = make().encode_command(position_command(0.0)).payload
    shifted = make(zero_offset_counts=200.0).encode_command(
        position_command(0.0)).payload
    assert plain[3:5] != shifted[3:5]
    flipped = make(direction=-1).encode_command(position_command(0.4)).payload
    forward = make(direction=1).encode_command(position_command(0.4)).payload
    assert flipped[3:5] != forward[3:5]


def test_direction_flips_the_feedforwards():
    forward = make(direction=1).encode_command(
        position_command(0.0, velocity_ff=2.0, current_ff=3.0)).payload
    flipped = make(direction=-1).encode_command(
        position_command(0.0, velocity_ff=2.0, current_ff=3.0)).payload
    assert forward[5:] != flipped[5:]


def test_torque_limit_clamps_the_feedforward():
    limited = make(torque_limit_nm=1.0).encode_command(
        position_command(0.0, current_ff=30.0)).payload
    clamped = make(torque_limit_nm=1.0).encode_command(
        position_command(0.0, current_ff=1.0)).payload
    assert limited[6:] == clamped[6:]


# --- the ranges are required ---------------------------------------------

@pytest.mark.parametrize("field", ["p_max", "v_max", "t_max"])
def test_absent_range_raises_rather_than_defaulting(field):
    """A guessed range is a silently wrong command, not a smaller error."""
    with pytest.raises(ValueError, match=field):
        make(**{field: 0.0})


def test_kd_above_the_wires_range_is_refused():
    """The field is 12 bits over 0..5; a bigger gain arrives clamped, silently."""
    with pytest.raises(ValueError, match="kd"):
        make(kd=20.0)


def test_kp_above_the_wires_range_is_refused():
    with pytest.raises(ValueError, match="kp"):
        make(kp=900.0)


# --- feedback is the servo status frame ----------------------------------

def status_frame(pos_deg10, spd, cur, temp, error):
    import struct
    return Frame(payload=struct.pack(">hhhbB", pos_deg10, spd, cur, temp, error),
                 address=NODE_ID | (41 << 8), meta={})


def test_status_frame_decodes_through_the_inherited_servo_path():
    state = make().decode_state(status_frame(900, 0, 25, 37, 0))
    assert state.position == pytest.approx(math.radians(90.0))
    assert state.temperature == 37.0
    assert state.error == 0


def test_status_frame_is_matched():
    assert make().matches(status_frame(0, 0, 0, 30, 0))


def test_an_echoed_force_control_frame_is_not_mistaken_for_status():
    """gs_usb echoes what it transmits, and this frame passes every other test.

    Same node id in the low byte, same eight bytes. Decoding one would read
    the kp high byte as the position high byte and fabricate a joint angle.
    """
    actuator = make()
    echo = actuator.encode_command(position_command(0.5))
    assert not actuator.matches(echo)


def test_a_servo_command_echo_is_still_not_mistaken_for_status():
    """The inherited guard has to survive being wrapped."""
    actuator = make()
    servo_echo = Frame(payload=b"\x00" * 8, address=NODE_ID | (4 << 8), meta={})
    assert not actuator.matches(servo_echo)


def test_decode_command_inverts_the_encoder():
    actuator = make(direction=-1, zero_offset_counts=-600.0, kp=300.0, kd=5.0,
                    torque_limit_nm=20.0)
    frame = actuator.encode_command(position_command(
        -1.0, velocity_ff=0.4, current_ff=3.0))
    sent = actuator.decode_command(frame)
    assert sent["position"] == pytest.approx(-1.0, abs=2 * P_MAX / 65535)
    assert sent["velocity"] == pytest.approx(0.4, abs=2 * V_MAX / 4095)
    assert sent["torque"] == pytest.approx(3.0, abs=2 * T_MAX / 4095)
    assert sent["kp"] == pytest.approx(300.0, abs=500.0 / 4095)
    assert sent["kd"] == pytest.approx(5.0, abs=5.0 / 4095)
