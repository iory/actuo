"""A continuous-rotation ICS servo is a wheel: it takes a speed, not a pose.

The numbers here are the ones measured on wheel servo 20 of a mecanum-base humanoid (deadband 6 deg,
saturation 20 deg, 0.0528 rev/s per deg). What is pinned is not the fit but the
three things a mecanum base breaks on if they are wrong: the deadband is a
refusal and not a crawl, saturation is reported and not silently swallowed, and
a wheel never reports a velocity it did not measure.
"""

import math

import pytest

from actuo.core.contracts import (
    ActuatorCommand, CapabilityError, Frame, OperatingMode)
from actuo.drivers.actuators.kondo_ics import (
    CENTER_PULSE, KondoIcsActuator, KondoIcsCalibration, PULSE_PER_DEG,
    SERVO_OFF_PULSE)
from actuo.drivers.actuators.kondo_ics_wheel import (
    KondoIcsWheelActuator, KondoIcsWheelCalibration, WHEEL_MOVE_BYTE)

MEASURED = dict(deadband_deg=6.0, saturate_deg=20.0, rev_s_per_deg=0.0528)


def wheel(servo_id=20, **overrides):
    spec = dict(MEASURED)
    spec.update(overrides)
    return KondoIcsWheelActuator(servo_id, KondoIcsWheelCalibration(**spec))


def test_the_three_measured_numbers_are_required():
    """No defaults: they differ by up to 2x between wheels of one model, and a
    shared guess does not slow a mecanum base, it aims it somewhere else."""
    for missing in ("deadband_deg", "saturate_deg", "rev_s_per_deg"):
        spec = dict(MEASURED)
        spec[missing] = 0.0
        with pytest.raises(ValueError, match=missing):
            KondoIcsWheelCalibration(**spec)


def test_a_saturation_below_the_deadband_is_refused():
    with pytest.raises(ValueError, match="must exceed"):
        KondoIcsWheelCalibration(deadband_deg=20.0, saturate_deg=6.0,
                                 rev_s_per_deg=0.05)


def test_stop_is_the_centre_pulse():
    assert wheel().pulse_for(0.0) == CENTER_PULSE


def test_below_the_deadband_it_stops_and_says_why():
    """Not a crawl. Commanding into the deadband produces nothing on one servo
    and a lurch on the next, so the honest answer is a stop plus a reason."""
    actuator = wheel()
    assert actuator.pulse_for(0.5) == CENTER_PULSE
    assert "below" in actuator.last_refusal
    assert "deadband" in actuator.last_refusal


def test_inside_the_band_the_measured_slope_is_what_comes_out():
    actuator = wheel()
    # 10 command deg is 0.528 rev/s = 3.317 rad/s by the measured fit.
    speed = 2.0 * math.pi * MEASURED["rev_s_per_deg"] * 10.0
    assert actuator.pulse_for(speed) == pytest.approx(
        CENTER_PULSE + 10.0 * PULSE_PER_DEG, abs=1)
    assert actuator.last_refusal is None


def test_over_the_top_speed_is_clamped_AND_reported():
    """Clamping alone would be a silent degradation -- and worse than
    usual here, because between wheels the ratio IS the heading."""
    actuator = wheel()
    top = actuator.calibration.max_rad_s
    pulse = actuator.pulse_for(top * 2.0)
    assert pulse == pytest.approx(
        CENTER_PULSE + MEASURED["saturate_deg"] * PULSE_PER_DEG, abs=1)
    assert "saturates" in actuator.last_refusal


def test_direction_flips_the_pulse_not_the_refusal():
    forward = wheel(direction=1)
    reverse = wheel(direction=-1)
    speed = 3.0
    assert forward.pulse_for(speed) > CENTER_PULSE
    assert reverse.pulse_for(speed) < CENTER_PULSE
    assert forward.pulse_for(speed) - CENTER_PULSE == pytest.approx(
        CENTER_PULSE - reverse.pulse_for(speed), abs=1)


def test_a_wheel_takes_velocity_and_refuses_position():
    actuator = wheel()
    assert actuator.capabilities.modes == (OperatingMode.VELOCITY,)
    with pytest.raises(CapabilityError):
        actuator.batch_pulse(ActuatorCommand(
            mode=OperatingMode.POSITION, position=0.0))


def test_velocity_mode_without_a_velocity_is_an_error_not_a_stop():
    with pytest.raises(ValueError, match="needs command.velocity"):
        wheel().batch_pulse(ActuatorCommand(mode=OperatingMode.VELOCITY))


def test_it_reports_no_velocity_because_it_cannot_measure_one():
    """The reading is a pot that wraps once per revolution with a blind sector.
    A per-cycle velocity from it would be a differentiated staircase, so the
    field stays absent and the raw count rides in aux, never a made-up value."""
    actuator = wheel()
    frame = Frame(payload=(4200).to_bytes(4, "little"), address=20,
                  meta={"kind": "servo_state", "slots": ("current_angle",)})
    state = actuator.decode_state(frame)
    assert state.velocity is None
    assert state.position is None
    assert state.aux["pulse"] == 4200.0


def test_a_silent_servo_reports_nothing_rather_than_a_plausible_zero():
    actuator = wheel()
    frame = Frame(payload=(0).to_bytes(4, "little"), address=20,
                  meta={"kind": "servo_state", "slots": ("current_angle",)})
    assert actuator.decode_state(frame).aux == {}


def test_the_reference_slot_is_labelled_a_command_not_a_measurement():
    actuator = wheel()
    reference = CENTER_PULSE + int(10.0 * PULSE_PER_DEG)
    frame = Frame(payload=reference.to_bytes(4, "little"), address=20,
                  meta={"kind": "servo_state", "slots": ("ref_angle",)})
    state = actuator.decode_state(frame)
    assert state.velocity is None
    assert state.aux["commanded_offset_deg"] == pytest.approx(10.0, abs=0.05)
    assert state.aux["servo_on"] == 1.0


def test_an_off_servo_reports_off_and_no_commanded_speed():
    frame = Frame(payload=SERVO_OFF_PULSE.to_bytes(4, "little"), address=20,
                  meta={"kind": "servo_state", "slots": ("ref_angle",)})
    state = wheel().decode_state(frame)
    assert state.aux["servo_on"] == 0.0
    assert "commanded_offset_deg" not in state.aux


def test_a_command_frame_is_not_mistaken_for_state():
    """An id alone is not identity: a frame we sent to id 20 must not be read
    back as id 20's state. On a CAN bus the adapter echoes our own commands,
    and the CubeMars driver once read its own echoed command as a status
    frame reporting an encoder fault on a healthy motor."""
    actuator = wheel()
    assert not actuator.matches(Frame(payload=b"\x00" * 4, address=20, meta={}))
    assert actuator.matches(Frame(payload=b"\x00" * 4, address=20,
                                  meta={"kind": "servo_state",
                                        "slots": ("current_angle",)}))


def test_rotation_mode_cannot_be_commanded_into_existence():
    """Whether a servo is a wheel is a property of the SERVO
    (``ServoStruct.rotation``), so there is no frame to emit for it -- and a
    config cannot turn a position servo into a wheel by asking."""
    assert wheel().encode_mode(OperatingMode.VELOCITY) == []


def test_the_move_byte_is_1_whatever_the_bus_wants():
    """The field is a speed, so there is nothing to interpolate towards and a
    long duration only delays the board acting on it."""
    actuator = wheel()
    assert actuator.move_byte(None) == WHEEL_MOVE_BYTE
    assert actuator.move_byte(200) == WHEEL_MOVE_BYTE


def test_a_position_servo_still_honours_the_bus_move_time():
    """The same contract, the other answer -- this is what lets an arm joint
    and a wheel ride in one packet without the bus knowing which is which."""
    joint = KondoIcsActuator(3, KondoIcsCalibration(), velocity=127)
    assert joint.move_byte(None) == 127
    assert joint.move_byte(2) == 2
