"""MIT-mode frame tests, checked against the vendor's own sample code.

The reference implementations in here are transcriptions of pack_cmd and
unpack_reply from the AK Series Module Driver User Manual V1.0.14, section 5.3.
Comparing against them rather than against hand-written expected bytes is the
point: the bit layout is four fields straddling byte boundaries, and a test
written from the same reading as the code would agree with it while both were
wrong.
"""

import math

import pytest

from actuo.core.contracts import ActuatorCommand, Frame, OperatingMode
from actuo.drivers.actuators.cubemars_mit import (
    ENTER_MOTOR_MODE,
    EXIT_MOTOR_MODE,
    SET_ZERO_POSITION,
    CubeMarsMitActuator,
    MitCalibration,
    float_to_uint,
    uint_to_float,
)

#: AK80-80's published ranges, used here only because they are in the vendor
#: table. The AK80-64 these tests were written for is NOT in that table.
P_MAX, V_MAX, T_MAX = 12.5, 8.0, 144.0


def reference_pack(p_des, v_des, kp, kd, t_ff):
    """The manual's pack_cmd, transcribed."""
    def f2u(x, lo, hi, bits):
        return int((min(max(x, lo), hi) - lo) * ((1 << bits) - 1) / (hi - lo))

    p = f2u(p_des, -P_MAX, P_MAX, 16)
    v = f2u(v_des, -V_MAX, V_MAX, 12)
    kp_i = f2u(kp, 0.0, 500.0, 12)
    kd_i = f2u(kd, 0.0, 5.0, 12)
    t = f2u(t_ff, -T_MAX, T_MAX, 12)
    return bytes([
        p >> 8, p & 0xFF,
        v >> 4, ((v & 0xF) << 4) | (kp_i >> 8), kp_i & 0xFF,
        kd_i >> 4, ((kd_i & 0xF) << 4) | (t >> 8), t & 0xFF,
    ])


def make(kp=500.0, kd=5.0, direction=1, zero_offset_deg=0.0):
    return CubeMarsMitActuator(3, MitCalibration(
        gear_ratio=64.0, direction=direction,
        zero_offset_rad=math.radians(zero_offset_deg),
        p_max=P_MAX, v_max=V_MAX, t_max=T_MAX, kp=kp, kd=kd))


@pytest.mark.parametrize("position", [0.0, 0.5, -1.25, 12.5, -12.5, 100.0])
def test_packing_matches_the_manual(position):
    frame = make().encode_command(
        ActuatorCommand(mode=OperatingMode.POSITION, position=position))
    assert frame.payload == reference_pack(position, 0.0, 500.0, 5.0, 0.0)


def test_command_is_a_standard_frame_addressed_by_node_id():
    # Servo mode's extended id with the packet type in the upper bits is the
    # other protocol entirely; sending it here is what the manual warns about.
    frame = make().encode_command(
        ActuatorCommand(mode=OperatingMode.POSITION, position=0.0))
    assert frame.address == 3
    assert frame.meta["extended"] is False
    assert len(frame.payload) == 8


def test_velocity_target_defaults_to_zero_not_the_target_derivative():
    # The damping term pulls the MEASURED velocity to zero. Anything else is a
    # different control law.
    frame = make().encode_command(
        ActuatorCommand(mode=OperatingMode.POSITION, position=1.0))
    assert frame.payload == reference_pack(1.0, 0.0, 500.0, 5.0, 0.0)


def test_special_commands():
    actuator = make()
    assert actuator.enter_motor_mode().payload == ENTER_MOTOR_MODE
    assert actuator.exit_motor_mode().payload == EXIT_MOTOR_MODE
    assert actuator.set_zero_position().payload == SET_ZERO_POSITION
    # Entering is not optional: the motor ignores commands until it lands.
    assert actuator.encode_mode(OperatingMode.POSITION) == [
        actuator.enter_motor_mode()]
    assert actuator.encode_release().payload == EXIT_MOTOR_MODE


def reference_reply(motor_id, position, temperature_raw=65, error=0):
    """The manual's "MIT Mode Driver Board Send Data Definition", DLC 8.

    Eight bytes, not seven. This file and the decoder both used to say seven,
    which is what made the driver reject the first real motor's answer.
    """
    raw = float_to_uint(position, -P_MAX, P_MAX, 16)
    return bytes([motor_id, raw >> 8, raw & 0xFF, 0x80, 0x00, 0x00,
                  temperature_raw, error])


@pytest.mark.parametrize("position", [0.0, 1.234, -2.5, 12.4])
def test_reply_round_trips_within_quantisation(position):
    frame = Frame(payload=reference_reply(3, position), address=3)
    state = make().decode_state(frame)
    # One count of a 16-bit field over +/-12.5 rad.
    assert abs(state.position - position) < 2.0 * P_MAX / (1 << 16)
    assert state.temperature == pytest.approx(25.0)


def test_a_real_motors_reply_is_eight_bytes_and_is_accepted():
    """Verbatim from the first AK80-64 switched into MIT mode, 2026-09-12.

    The driver believed a reply was seven bytes, so `matches` refused this and
    a motor that was answering correctly looked like one that was not there.
    """
    frame = Frame(payload=bytes.fromhex("01 80 80 7f d7 ff 42 00"), address=1)
    actuator = CubeMarsMitActuator(1, MitCalibration(
        gear_ratio=64.0, p_max=12.56, v_max=8.0, t_max=144.0, kp=0.0, kd=0.0))
    assert actuator.matches(frame)
    state = actuator.decode_state(frame)
    assert state.position == pytest.approx(0.05, abs=0.01)
    assert state.velocity == pytest.approx(0.0, abs=0.05)
    assert state.current == pytest.approx(0.0, abs=0.5)
    assert state.temperature == pytest.approx(26.0)
    assert state.error == 0


def test_a_neighbours_servo_status_frame_is_not_decoded_as_a_reply():
    """Observed on the real bus, 2026-09-12, and it read as a 144 N*m fault.

    On a 12-joint CubeMars biped: one motor in MIT mode, eleven still in
    servo mode, one CAN bus. A servo
    status frame is also eight bytes and its byte 0 is a position high byte,
    so right hip roll's `01 0c 00 00 00 00 23 00` satisfied every test a MIT
    reply had to pass for node id 1. A reply is a standard frame and a servo
    frame is extended; nothing else separates them.
    """
    actuator = CubeMarsMitActuator(1, MitCalibration(
        gear_ratio=64.0, p_max=12.56, v_max=8.0, t_max=144.0, kp=0.0, kd=0.0))
    neighbour = Frame(payload=bytes.fromhex("01 0c 00 00 00 00 23 00"),
                      address=(41 << 8) | 7, meta={"extended": True})
    assert actuator.matches(neighbour) is False


def test_the_error_flag_is_decoded_rather_than_dropped():
    """Byte 7, which fell off the end while the reply was thought to be seven."""
    frame = Frame(payload=reference_reply(3, 0.0, error=4), address=3)
    assert make().decode_state(frame).error == 4


def test_our_own_command_is_not_decoded_as_state():
    # python-can's gs_usb backend echoes transmitted frames, and a command is
    # eight bytes on the same standard id a reply comes back on -- so length
    # cannot separate them, which is what this used to rely on. Direction can:
    # an echo carries is_rx False.
    actuator = make()
    command = actuator.encode_command(
        ActuatorCommand(mode=OperatingMode.POSITION, position=0.0))
    assert actuator.matches(command) is False

    # And explicitly, for a command whose first byte DOES equal the node id.
    echo = Frame(payload=bytes([3, 0, 0, 0, 0, 0, 0, 0]), address=3,
                 meta={"is_rx": False})
    assert actuator.matches(echo) is False
    assert actuator.matches(Frame(payload=echo.payload, address=3,
                                  meta={"is_rx": True})) is True


def test_reply_from_another_motor_is_rejected():
    actuator = make()
    frame = Frame(payload=reference_reply(9, 0.0), address=3)
    assert actuator.matches(frame) is False
    with pytest.raises(ValueError):
        actuator.decode_state(frame)


def test_direction_and_offset_apply_to_both_directions():
    actuator = make(direction=-1, zero_offset_deg=90.0)
    cal = actuator.calibration
    for model in (0.0, 0.5, -1.5):
        assert cal.to_model(cal.to_wire(model)) == pytest.approx(model)


def test_kd_above_the_wire_range_is_refused():
    # The field is 12 bits over 0..5 on every AK model. A larger gain would
    # arrive clamped with no error anywhere, which is the failure this guards.
    with pytest.raises(ValueError, match="kd"):
        MitCalibration(gear_ratio=9.0, p_max=P_MAX, v_max=V_MAX, t_max=T_MAX,
                       kp=300.0, kd=12.0)


def test_missing_ranges_are_refused():
    with pytest.raises(ValueError, match="p_max"):
        MitCalibration(gear_ratio=9.0, kp=300.0, kd=5.0)


def test_scaling_helpers_are_inverses():
    for value in (-12.5, -3.0, 0.0, 7.5, 12.5):
        raw = float_to_uint(value, -P_MAX, P_MAX, 16)
        assert abs(uint_to_float(raw, -P_MAX, P_MAX, 16) - value) < 4e-4


def test_the_position_fence_is_enforced_at_encode_time():
    """On a CubeMars biped, this model drives the hip pitch, hip roll and knee.

    It once had no fence at all: the config carried position_limit_deg for
    those joints and MitCalibration never read it, so every target was
    accepted. The servo model has refused out-of-range targets from
    the start.
    """
    actuator = CubeMarsMitActuator(3, MitCalibration(
        gear_ratio=64.0, p_max=P_MAX, v_max=V_MAX, t_max=T_MAX,
        kp=100.0, kd=2.0,
        position_limit=(math.radians(-30.0), math.radians(30.0))))

    inside = ActuatorCommand(mode=OperatingMode.POSITION,
                             position=math.radians(20.0))
    actuator.encode_command(inside)          # does not raise
    assert actuator.position_refusal(math.radians(20.0)) is None

    outside = math.radians(45.0)
    assert "outside this joint's limit" in actuator.position_refusal(outside)
    with pytest.raises(ValueError, match="outside this joint's limit"):
        actuator.encode_command(
            ActuatorCommand(mode=OperatingMode.POSITION, position=outside))


def test_no_fence_configured_means_no_refusal():
    assert make().position_refusal(math.radians(3000.0)) is None


def test_decode_command_inverts_the_encoder():
    actuator = make(kp=400.0, kd=5.0, direction=-1, zero_offset_deg=5.0)
    frame = actuator.encode_command(ActuatorCommand(
        mode=OperatingMode.POSITION, position=-0.3, velocity_ff=0.2,
        current_ff=-7.0))
    sent = actuator.decode_command(frame)
    assert sent["position"] == pytest.approx(-0.3, abs=2 * P_MAX / 65535)
    assert sent["velocity"] == pytest.approx(0.2, abs=2 * V_MAX / 4095)
    assert sent["torque"] == pytest.approx(-7.0, abs=2 * T_MAX / 4095)
    assert sent["kp"] == pytest.approx(400.0, abs=500.0 / 4095)
    assert sent["kd"] == pytest.approx(5.0, abs=5.0 / 4095)


WINDOW = math.radians(360.0 / 64.0)


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("offset", [0, 1, -3])
def test_window_offset_round_trips(direction, offset):
    """Reporting one window and commanding another is a whole-window lurch."""
    cal = make(direction=direction, zero_offset_deg=12.0).calibration
    cal.window_offset = offset
    for model in (-1.2, -0.087, 0.0, 0.4):
        assert cal.to_model(cal.to_wire(model)) == pytest.approx(model, abs=1e-12)


@pytest.mark.parametrize("direction", [1, -1])
def test_a_knee_back_one_window_is_snapped_home(direction):
    """A real case (2026-09-30): after a restart a biped's right knee read
    ~5.4 deg from where it was, i.e. one rotor turn off. With the last limp position as the reference, the snap puts
    it back exactly, and commands go out in the corrected frame."""
    actuator = make(direction=direction)
    truth = math.radians(-5.0)
    reported = truth - WINDOW + math.radians(0.3)     # lost a rotor turn, plus sag
    offset = actuator.snap_to_window(reported, truth)
    assert offset == direction * 1
    wire = actuator.calibration.to_wire(truth)
    assert actuator.calibration.to_model(wire) == pytest.approx(truth, abs=1e-12)
    # idempotent: the same reference again changes nothing
    now = reported + WINDOW
    assert actuator.snap_to_window(now, truth) == offset


def test_an_ambiguous_reference_is_refused_not_guessed():
    """A hand-placed pose 2 deg off on a 5.625 deg window cannot pick one."""
    actuator = make()
    with pytest.raises(ValueError):
        actuator.snap_to_window(math.radians(-5.0), math.radians(-5.0 + 2.0))
    assert actuator.calibration.window_offset == 0
