"""Unit tests for the Kondo ICS servo model.

Pure unit conversion, capability, and decoding tests -- no board, no serial
port. The reference numbers are the ICS servo's own: neutral pulse 7500 and
4000 pulses per 135 degrees.
"""

import math
import struct

import pytest

from actuo.core.contracts import (
    ActuatorCommand, CapabilityError, Frame, OperatingMode)

from actuo.drivers.actuators.kondo_ics import (
    CENTER_PULSE, KondoIcsActuator, KondoIcsCalibration, PULSE_PER_DEG,
    SERVO_OFF_PULSE, SERVO_ON_PULSE)
from actuo.drivers.kondo_h7_protocol import unframe


def make_actuator(servo_id=3, **kwargs):
    calibration = KondoIcsCalibration(**kwargs)
    return KondoIcsActuator(servo_id, calibration)


def state_frame(servo_id, **slots):
    """Build the per-servo state slice the bus hands to ``decode_state``."""
    names = tuple(slots)
    payload = struct.pack("<" + "i" * len(names), *(slots[n] for n in names))
    return Frame(payload=payload, address=servo_id,
                 meta={"kind": "servo_state", "slots": names})


# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------


def test_neutral_pulse_is_zero_radians():
    actuator = make_actuator()
    assert actuator.pulse_to_rad(CENTER_PULSE) == pytest.approx(0.0)
    assert actuator.rad_to_pulse(0.0) == pytest.approx(CENTER_PULSE)


def test_ninety_degrees_is_the_servos_own_scale():
    """90 deg must be 90 * 4000/135 pulses off neutral, not a rounded guess."""
    actuator = make_actuator()
    pulse = actuator.rad_to_pulse(math.radians(90.0))
    assert pulse == pytest.approx(CENTER_PULSE + 90.0 * PULSE_PER_DEG)


def test_pulse_round_trips_through_radians():
    actuator = make_actuator()
    for pulse in (3500, 5000, 7500, 9000, 11500):
        assert actuator.rad_to_pulse(actuator.pulse_to_rad(pulse)) == \
            pytest.approx(pulse)


def test_direction_flips_the_sign_but_not_the_neutral():
    reversed_actuator = make_actuator(direction=-1)
    assert reversed_actuator.pulse_to_rad(CENTER_PULSE) == pytest.approx(0.0)
    forward = make_actuator()
    angle = math.radians(30.0)
    assert (reversed_actuator.rad_to_pulse(angle) - CENTER_PULSE) == \
        pytest.approx(-(forward.rad_to_pulse(angle) - CENTER_PULSE))


def test_offset_moves_the_joint_zero_off_the_servo_centre():
    """A joint whose zero sits 10 deg off the servo's mechanical centre."""
    actuator = make_actuator(
        zero_offset_counts=CENTER_PULSE + 10.0 * PULSE_PER_DEG)
    assert actuator.rad_to_pulse(0.0) == \
        pytest.approx(CENTER_PULSE + 10.0 * PULSE_PER_DEG)
    assert actuator.pulse_to_rad(CENTER_PULSE) == \
        pytest.approx(math.radians(-10.0))


# --------------------------------------------------------------------------
# Limits -- refused, never clamped
# --------------------------------------------------------------------------


def test_command_outside_the_taught_limit_is_refused():
    actuator = make_actuator(
        position_limit=(math.radians(-30.0), math.radians(30.0)))
    actuator.goal_pulse(math.radians(29.0))
    with pytest.raises(ValueError) as excinfo:
        actuator.goal_pulse(math.radians(31.0))
    assert "taught limit" in str(excinfo.value)


def test_command_outside_the_mechanical_envelope_is_refused():
    """No configured limit still leaves the servo's own +/-135 deg envelope."""
    actuator = make_actuator()
    with pytest.raises(ValueError) as excinfo:
        actuator.goal_pulse(math.radians(140.0))
    assert "mechanical envelope" in str(excinfo.value)


def test_the_torque_layer_can_ask_before_it_enables():
    """The same two envelopes, as a REASON rather than an exception.

    Without this, enabling a joint parked outside its limit passed every guard
    -- the controller adopts the measured pose, so the jump is exactly zero --
    and one cycle later goal_pulse raised inside service() and killed the whole
    servicer, every other joint's telemetry with it. Every other guard on this
    bus declines with a reason; this one aborted the process.
    """
    actuator = make_actuator(
        position_limit=(math.radians(-30.0), math.radians(30.0)))

    assert actuator.position_refusal(math.radians(29.0)) is None
    assert "taught limit" in actuator.position_refusal(math.radians(31.0))
    # No configured limit still leaves the servo's own mechanical envelope.
    assert "mechanical envelope" in \
        make_actuator().position_refusal(math.radians(140.0))


def test_refusing_and_raising_cannot_disagree():
    """One implementation, two entry points.

    If the encoder rejected an angle the torque layer had already accepted, the
    servicer would still die at exactly the moment torque went on -- which is
    the bug, not a variation of it.
    """
    actuator = make_actuator(
        position_limit=(math.radians(-30.0), math.radians(30.0)))
    for degrees in (-200, -140, -31, -30, 0, 29.9, 30, 31, 140, 200):
        refused = actuator.position_refusal(math.radians(degrees)) is not None
        try:
            actuator.goal_pulse(math.radians(degrees))
            raised = False
        except ValueError:
            raised = True
        assert refused == raised, "{} deg: refusal={} raise={}".format(
            degrees, refused, raised)


def test_an_out_of_range_command_is_not_silently_clamped():
    """Clamping would deliver a different pose than the controller asked for."""
    actuator = make_actuator()
    with pytest.raises(ValueError):
        actuator.encode_command(ActuatorCommand(
            mode=OperatingMode.POSITION, position=math.radians(200.0)))


# --------------------------------------------------------------------------
# Capabilities
# --------------------------------------------------------------------------


def test_only_position_mode_is_supported():
    actuator = make_actuator()
    assert actuator.capabilities.modes == (OperatingMode.POSITION,)
    for mode in (OperatingMode.VELOCITY, OperatingMode.CURRENT,
                 OperatingMode.PWM):
        with pytest.raises(CapabilityError):
            actuator.encode_mode(mode)


def test_position_mode_needs_no_wire_switch():
    actuator = make_actuator()
    assert actuator.encode_mode(OperatingMode.POSITION) == []


def test_velocity_is_never_sensed():
    """The servo does not report speed. Differentiating position behind the
    caller's back is exactly the fabrication this project bans."""
    actuator = make_actuator()
    assert actuator.capabilities.has_velocity_sense is False
    state = actuator.decode_state(state_frame(3, current_angle=8000))
    assert state.velocity is None


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------


def test_single_command_uses_the_validated_multi_servo_form():
    actuator = make_actuator(servo_id=5)
    frame = actuator.encode_command(
        ActuatorCommand(mode=OperatingMode.POSITION, position=0.0))
    assert frame.address == 5
    # opcode + 5-byte bitmap + velocity + one 16-bit pulse
    assert len(frame.payload) == 1 + 5 + 1 + 2
    assert frame.payload[0] == 0x10
    assert frame.payload[1] == 0b00100000       # bit 5 set
    assert struct.unpack("<H", frame.payload[-2:])[0] == CENTER_PULSE


def test_release_and_hold_carry_the_boards_sentinels():
    """Servo off has no packet of its own -- it is a move to a sentinel."""
    actuator = make_actuator(servo_id=1)
    off = struct.unpack("<H", actuator.encode_release().payload[-2:])[0]
    on = struct.unpack("<H", actuator.encode_hold().payload[-2:])[0]
    assert off == SERVO_OFF_PULSE
    assert on == SERVO_ON_PULSE


def test_emitted_command_is_a_well_formed_packet_body():
    """The actuator emits a body; framing belongs to the transport."""
    actuator = make_actuator()
    payload = actuator.encode_command(
        ActuatorCommand(mode=OperatingMode.POSITION, position=0.0)).payload
    from actuo.drivers.kondo_h7_protocol import frame as add_framing
    body = unframe(add_framing(payload))
    assert body == payload


# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------


def test_state_decodes_position_from_the_measured_pulse():
    actuator = make_actuator()
    state = actuator.decode_state(state_frame(
        3, current_angle=int(CENTER_PULSE + 45.0 * PULSE_PER_DEG)))
    assert state.position == pytest.approx(math.radians(45.0), abs=1e-3)
    assert state.aux["pulse"] == pytest.approx(CENTER_PULSE + 45.0 * PULSE_PER_DEG,
                                               abs=1.0)


def test_an_unheard_servo_reports_no_position_rather_than_neutral():
    """Pulse 0 means the firmware has never heard from the servo. Decoding it
    would report a confident angle far outside the mechanical range."""
    actuator = make_actuator()
    state = actuator.decode_state(state_frame(3, current_angle=0))
    assert state.position is None


def test_current_byte_sign_convention():
    """0..63 is one direction of load, 64..127 the other, in 0.1 A steps."""
    actuator = make_actuator()
    assert actuator.decode_state(
        state_frame(3, current=25)).aux["current_a"] == pytest.approx(2.5)
    assert actuator.decode_state(
        state_frame(3, current=64 + 25)).aux["current_a"] == pytest.approx(-2.5)


def test_unpolled_current_is_absent_not_zero():
    """While the board's read flag is off the byte stays 0; reporting 0 A would
    be a measurement the robot never took."""
    actuator = make_actuator()
    state = actuator.decode_state(state_frame(3, current=0))
    assert "current_a" not in state.aux
    assert state.current is None


def test_torque_is_only_reported_with_a_torque_constant():
    without = make_actuator()
    assert without.decode_state(state_frame(3, current=30)).current is None
    with_kt = make_actuator(torque_constant=0.5)
    assert with_kt.decode_state(state_frame(3, current=30)).current == \
        pytest.approx(1.5)


def test_temperature_uses_the_servos_table_and_skips_zero():
    actuator = make_actuator()
    assert actuator.decode_state(
        state_frame(3, temperature=0)).temperature is None
    assert actuator.decode_state(
        state_frame(3, temperature=99)).temperature == pytest.approx(50.0)


def test_error_angle_is_a_difference_so_neutral_is_not_subtracted():
    """error_angle is ref-minus-measured in pulses. Running it through the
    position conversion would subtract the 7500 neutral and report a huge
    error for a joint that is tracking perfectly."""
    actuator = make_actuator()
    state = actuator.decode_state(state_frame(3, error_angle=0))
    assert state.aux["error_angle_rad"] == pytest.approx(0.0)
    state = actuator.decode_state(state_frame(3, error_angle=30))
    assert state.aux["error_angle_rad"] == pytest.approx(
        math.radians(30.0 / PULSE_PER_DEG), abs=1e-9)


def test_ref_angle_reveals_the_servo_on_state():
    actuator = make_actuator()
    off = actuator.decode_state(state_frame(3, ref_angle=SERVO_OFF_PULSE))
    on = actuator.decode_state(state_frame(3, ref_angle=7600))
    assert off.aux["servo_on"] == 0.0
    assert on.aux["servo_on"] == 1.0


def test_an_undecodable_slot_fails_loudly():
    actuator = make_actuator()
    with pytest.raises(ValueError) as excinfo:
        actuator.decode_state(state_frame(3, trim=5))
    assert "trim" in str(excinfo.value)


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------


def test_matches_requires_both_the_id_and_the_frame_kind():
    """A servo id alone is not identity. The CubeMars port learned this the
    expensive way: a command frame carrying the same id decoded as a fault."""
    actuator = make_actuator(servo_id=3)
    assert actuator.matches(state_frame(3, current_angle=7500))
    assert not actuator.matches(state_frame(4, current_angle=7500))
    assert not actuator.matches(
        Frame(payload=b"\x00\x00\x00\x00", address=3))
