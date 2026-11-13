"""The DYNAMIXEL actuator and its model table.

The tests that matter here are the ones pinning conversions that would
otherwise fail *plausibly*: an angle scaled by 300/360, a velocity whose sign
flips in the wrong place, a mode offered without the number needed to honour it.
Each of those produces a robot that moves smoothly to the wrong place, which no
amount of "it ran" testing catches.
"""

import math

import pytest

from actuo.core.contracts import (
    ActuatorCommand, CapabilityError, OperatingMode)

from actuo.drivers import dynamixel_protocol as dxl
from actuo.drivers.actuators.dynamixel import (
    DynamixelActuator, DynamixelCalibration)
from actuo.drivers.dynamixel_models import (
    UnknownModelError, load_model_table)


@pytest.fixture(scope="module")
def table():
    return load_model_table()


@pytest.fixture
def ax12(table):
    model = table.by_name("AX-12A")
    calibration = DynamixelCalibration(
        counts_per_rev=model.counts_per_rev,
        zero_offset_counts=model.center_counts)
    return DynamixelActuator(15, calibration, model)


@pytest.fixture
def xl430(table):
    model = table.by_name("XL430-W250-T")
    calibration = DynamixelCalibration(
        counts_per_rev=model.counts_per_rev,
        zero_offset_counts=model.center_counts)
    return DynamixelActuator(1, calibration, model)


# --- the model table ---------------------------------------------------------
def test_unknown_model_raises_with_a_fillable_template(table):
    """The whole "supports every model" design rests on this failing loudly."""
    with pytest.raises(UnknownModelError) as excinfo:
        table.by_number(65535)
    message = str(excinfo.value)
    assert "emanual.robotis.com" in message      # where to get the truth
    assert "65535:" in message                   # a pasteable YAML key
    assert "source:" in message                  # the citation is not optional


def test_every_shipped_row_carries_a_source_url(table):
    for number in table.numbers:
        model = table.by_number(number)
        assert model.source.startswith("https://emanual.robotis.com/"), number
        assert model.table.source.startswith("https://emanual.robotis.com/")


def test_travel_is_not_a_revolution(ax12, xl430):
    """An AX spans 1024 counts over 300 deg, so counts-per-REV is 1228.8.

    Using 1024 would scale every angle by 0.83 -- still a plausible pose, which
    is exactly why it is pinned.
    """
    assert ax12.model.counts_per_rev == pytest.approx(1228.8)
    assert xl430.model.counts_per_rev == pytest.approx(4096.0)


# --- position conversion -----------------------------------------------------
def test_centre_count_is_zero_radians(ax12, xl430):
    assert ax12.position_counts(0.0) == 512
    assert xl430.position_counts(0.0) == 2048


def test_position_round_trips(ax12):
    for angle in (-1.0, -0.25, 0.0, 0.3, 1.2):
        counts = ax12.position_counts(angle)
        assert ax12.calibration.counts_to_rad(counts) == pytest.approx(
            angle, abs=math.radians(0.3))     # one AX count is 0.29 deg


def test_a_pose_that_cannot_be_expressed_as_a_goal_is_refused():
    """The 2026-07-26 fault, pinned.

    A DYNAMIXEL arm's head joint sat at -6.7058 rad. On an XL430 that is goal
    count 4392, outside 0..4095, so the servo took it modulo its range and
    drove ~340 deg into an overload. In radians the command matched the
    measurement exactly, so no jump guard could see it -- only the actuator
    knows what its wire format can say.
    """
    model = load_model_table().by_name("XL430-W250-T")
    actuator = DynamixelActuator(
        5, DynamixelCalibration(counts_per_rev=model.counts_per_rev,
                                zero_offset_counts=model.center_counts), model)
    inside = 1.0
    assert actuator.position_refusal(inside) is None
    assert 0 <= actuator.position_counts(inside) <= model.resolution - 1

    outside = 2.0 * math.pi + 0.5     # more than a full turn from zero
    reason = actuator.position_refusal(outside)
    assert reason is not None and "modulo" in reason
    with pytest.raises(ValueError):
        actuator.position_counts(outside)


def test_homing_offset_shifts_the_valid_goal_window():
    """The 2026-07-27 fault, pinned.

    id 5's Homing Offset register was written to +512 to fix the 2026-07-26
    fault above -- but position_refusal, checking the un-shifted 0..4095,
    then refused a perfectly reachable pose the servo itself was reporting
    (raw goal count 4244, comfortably inside the correctly-shifted
    512..4607). Same servo, same underlying formula, opposite failure
    direction: the fix for one fault silently re-broke the guard that
    prevented the other until homing_offset_counts existed to describe it.
    """
    model = load_model_table().by_name("XL430-W250-T")
    unshifted = DynamixelActuator(
        5, DynamixelCalibration(counts_per_rev=model.counts_per_rev,
                                zero_offset_counts=model.center_counts), model)
    shifted = DynamixelActuator(
        5, DynamixelCalibration(counts_per_rev=model.counts_per_rev,
                                zero_offset_counts=model.center_counts,
                                homing_offset_counts=512.0), model)

    # raw 4244 (the pose from the fault above) is refused with the un-shifted
    # window (0..4095) and accepted with the shifted one (512..4607).
    raw_4244_rad = (4244 - model.center_counts) * (2.0 * math.pi / model.counts_per_rev)

    assert unshifted.position_refusal(raw_4244_rad) is not None
    assert shifted.position_refusal(raw_4244_rad) is None

    # And the reverse holds too: a pose safely inside the UN-shifted window
    # (raw ~100, near the old seam) is now outside the shifted one.
    raw_100_rad = (100 - model.center_counts) * (2.0 * math.pi / model.counts_per_rev)
    assert unshifted.position_refusal(raw_100_rad) is None
    reason = shifted.position_refusal(raw_100_rad)
    assert reason is not None and "homing_offset_counts=512" in reason


def test_ax_goal_position_stays_inside_its_10_bit_range(ax12):
    """An out-of-range count wraps into a wildly different pose on the wire."""
    assert ax12.position_counts(math.radians(200.0)) == 1023
    assert ax12.position_counts(math.radians(-200.0)) == 0


def test_direction_and_offset_come_from_config(table):
    model = table.by_name("AX-12A")
    counts_per_deg = model.counts_per_rev / 360.0
    calibration = DynamixelCalibration(
        counts_per_rev=model.counts_per_rev,
        zero_offset_counts=model.center_counts + 30.0 * counts_per_deg,
        direction=-1)
    actuator = DynamixelActuator(2, calibration, model)
    # 0 rad now reports at the shifted centre, and the axis is reversed.
    assert actuator.position_counts(0.0) == pytest.approx(614, abs=1)
    assert actuator.position_counts(math.radians(10.0)) < 614


def test_position_outside_the_taught_limit_is_refused_not_clamped(table):
    model = table.by_name("AX-12A")
    calibration = DynamixelCalibration(
        counts_per_rev=model.counts_per_rev,
        zero_offset_counts=model.center_counts,
        position_limit=(math.radians(-30.0), math.radians(30.0)))
    actuator = DynamixelActuator(3, calibration, model)
    actuator.position_counts(math.radians(29.0))
    with pytest.raises(ValueError):
        actuator.position_counts(math.radians(31.0))


# --- capabilities are computed, never merely declared ------------------------
def test_ax12_offers_position_only(ax12):
    assert ax12.capabilities.modes == (OperatingMode.POSITION,)


def test_requesting_wheel_mode_on_an_ax_fails_loudly(ax12):
    """AX wheel mode is entered by zeroing the angle limits -- a mode switch
    disguised as a limit write. It is not implemented, so it must raise rather
    than move the joint in an unexpected way."""
    with pytest.raises(CapabilityError):
        ax12.require_mode(OperatingMode.VELOCITY)


def test_xl430_current_mode_is_absent_because_the_servo_has_none(xl430):
    assert OperatingMode.CURRENT not in xl430.capabilities.modes
    assert not xl430.capabilities.has_current_sense


def test_a_withheld_mode_explains_which_datum_is_missing(xl430):
    reason = xl430.unsupported_reason(OperatingMode.PWM)
    assert "full scale" in reason


def test_velocity_mode_needs_a_documented_unit(table):
    """A model whose velocity unit is unknown must not offer VELOCITY -- the
    same rule CubeMars applies to pole_pairs."""
    model = table.by_name("XL430-W250-T")
    stripped = type(model)(**{**model.__dict__, "velocity_unit_rpm": None})
    actuator = DynamixelActuator(
        1, DynamixelCalibration(counts_per_rev=model.counts_per_rev), stripped)
    assert OperatingMode.VELOCITY not in actuator.capabilities.modes
    assert "velocity_unit_rpm" in actuator.unsupported_reason(
        OperatingMode.VELOCITY)


# --- command encoding --------------------------------------------------------
def test_ax_position_command_targets_register_30(ax12):
    frame = ax12.encode_command(
        ActuatorCommand(mode=OperatingMode.POSITION, position=0.0))
    parsed_addr = frame.payload[5]
    assert frame.payload[4] == dxl.WRITE
    assert parsed_addr == 30
    assert frame.payload[6:8] == dxl.to_le(512, 2)


def test_xl430_position_command_targets_register_116_as_4_bytes(xl430):
    register, data = xl430.goal_payload(
        ActuatorCommand(mode=OperatingMode.POSITION, position=0.0))
    assert register == "goal_position"
    assert xl430.model.table[register].address == 116
    assert data == dxl.to_le(2048, 4)


def test_velocity_command_uses_the_models_rpm_unit(xl430):
    """1 rad/s = 9.5493 rev/min; at 0.229 rev/min per LSB that is 42 counts."""
    counts = xl430.velocity_counts(1.0)
    assert counts == pytest.approx(9.5493 / 0.229, abs=1)


def test_release_is_a_register_write_not_silence(ax12):
    """A DYNAMIXEL holds its last goal forever, so "stop commanding" is not off."""
    frame = ax12.encode_release()
    assert frame.payload[4] == dxl.WRITE
    assert frame.payload[5] == 24            # torque_enable on the AX table
    assert frame.payload[6] == 0


def test_mode_switch_disables_torque_first(xl430):
    """operating_mode only accepts a write while torque is off."""
    frames = xl430.encode_mode(OperatingMode.VELOCITY)
    assert len(frames) == 2
    torque, mode = frames
    assert dxl.remove_stuffing(torque.payload[7:-2])[1:] == bytes([64, 0, 0])
    assert dxl.remove_stuffing(mode.payload[7:-2])[1:] == bytes([11, 0, 1])


def test_ax_mode_switch_needs_no_frames(ax12):
    assert ax12.encode_mode(OperatingMode.POSITION) == []


# --- state decoding ----------------------------------------------------------
def test_ax_status_block_decodes_every_channel(ax12):
    """Block 36..43: position, velocity, load, voltage, temperature."""
    block = (dxl.to_le(512, 2)          # present_position = centre
             + dxl.to_le(100, 2)        # present_velocity, CCW (positive)
             + dxl.to_le(1024 + 50, 2)  # present_load, CW (negative)
             + bytes([120])             # present_voltage = 12.0 V
             + bytes([41]))             # present_temperature = 41 C
    state = ax12.decode_block(block)
    assert state.position == pytest.approx(0.0)
    assert state.velocity == pytest.approx(100 * 0.111 / (60 / (2 * math.pi)))
    assert state.voltage == pytest.approx(12.0)
    assert state.temperature == pytest.approx(41.0)
    assert state.aux["load_percent"] == pytest.approx(-5.0)
    # No torque constant exists for an AX, so N*m stays None rather than faked.
    assert state.current is None


def test_protocol1_direction_bit_is_not_twos_complement(ax12):
    """Bit 10 is a direction flag. Decoding the word as signed would make 1074
    read as +1074 (or -1074 at 16 bits), not -50."""
    block = (dxl.to_le(512, 2) + dxl.to_le(1024 + 50, 2)
             + dxl.to_le(0, 2) + bytes([120, 41]))
    state = ax12.decode_block(block)
    assert state.velocity < 0
    assert state.velocity == pytest.approx(-50 * 0.111 / (60 / (2 * math.pi)))


def test_xl430_block_decodes_signed_position(xl430):
    block = (dxl.to_le(0, 2)            # present_load
             + dxl.to_le(-42, 4)        # present_velocity, two's complement
             + dxl.to_le(1024, 4))      # present_position
    state = xl430.decode_block(block)
    assert state.velocity == pytest.approx(-1.0, abs=0.05)
    assert state.position == pytest.approx(-math.pi / 2, abs=1e-3)
    # temperature is not in this model's cyclic block: None, never a fake 0.
    assert state.temperature is None


def test_matches_rejects_another_servos_status(ax12):
    from actuo.core.contracts import Frame
    other = dxl.build_packet(dxl.PROTOCOL_1, 16, 0x00, bytes(8))
    assert not ax12.matches(Frame(payload=other, address=16))


def test_matches_rejects_an_echoed_command_on_protocol1(ax12):
    """A 1.0 instruction packet has the same shape as a status packet, so the
    block length is checked too. This is the guard against decoding a frame we
    transmitted ourselves."""
    from actuo.core.contracts import Frame
    echo = ax12.encode_command(
        ActuatorCommand(mode=OperatingMode.POSITION, position=0.0))
    assert not ax12.matches(Frame(payload=echo.payload, address=ax12.id))


def test_short_block_raises_rather_than_decoding_garbage(ax12):
    with pytest.raises(ValueError):
        ax12.decode_block(bytes(4))


# --- hardware fault diagnosis -------------------------------------------------
def test_xl430_has_hardware_error_status_ax12_does_not(xl430, ax12):
    """AX-12A's equivalent (Alarm LED / Shutdown, addr 17/18) is a different
    scheme and out of scope here -- this must degrade to "no diagnosis
    available", never guess an X-series address on a servo that has none."""
    assert xl430.has_hardware_error_status
    assert not ax12.has_hardware_error_status


def test_asking_an_ax12_for_hardware_error_status_is_refused_not_guessed(ax12):
    with pytest.raises(CapabilityError):
        ax12.encode_hardware_error_query()


def test_hardware_error_query_targets_register_70(xl430):
    frame = xl430.encode_hardware_error_query()[0]
    body = dxl.remove_stuffing(frame.payload[7:-2])
    assert body[0] == dxl.READ
    assert dxl.from_le(body[1:3]) == 70
    assert dxl.from_le(body[3:5]) == 1     # one byte


def test_decode_hardware_error_status_round_trips(xl430):
    from actuo.core.contracts import Frame
    reply = Frame(payload=dxl.build_packet(
        dxl.PROTOCOL_2, xl430.id, dxl.STATUS, bytes([0x00, 0x20])),
        address=xl430.id)
    assert xl430.decode_hardware_error_status(reply) == 0x20


@pytest.mark.parametrize("byte,expected", [
    (0x00, []),
    (0x20, ["overload"]),
    (0x01, ["input_voltage"]),
    (0x24, ["overheating", "overload"]),   # two real bits at once
])
def test_hardware_error_names_decodes_known_bits(byte, expected):
    assert DynamixelActuator.hardware_error_names(byte) == expected


def test_hardware_error_names_does_not_drop_an_unrecognised_bit():
    """The whole point: a fault the table does not happen to list must still
    be SEEN, even if it cannot be named yet."""
    names = DynamixelActuator.hardware_error_names(0x40)   # bit 6, unlisted
    assert names == ["bit6"]


def test_error_names_table_never_collides_with_the_alert_only_case():
    """0x80 (Alert, no detail) must mean one thing: it is not a value
    Hardware Error Status can itself take (bits 6/7 are unused), so the two
    meanings of "state.error == 0x80" -- undiagnosed vs. somehow-diagnosed
    -- can never be confused."""
    from actuo.drivers.actuators.dynamixel import ERROR_NAMES
    assert ERROR_NAMES[dxl.ERROR_ALERT_BIT] == "hardware fault (detail unavailable)"
    assert ERROR_NAMES[0] == "none"
    assert ERROR_NAMES[0x20] == "overload"
