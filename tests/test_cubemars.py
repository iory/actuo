"""Unit tests for the CubeMars AK servo-mode actuator model.

Pure unit conversion + frame building: no CAN hardware, no python-can, no ROS,
so this runs anywhere.

    pytest tests/test_cubemars.py -q

The anchor test is ``test_matches_reference_script_wire_bytes``: it reproduces the exact
frame that an earlier standalone servo-mode script sends (the "reference
script" in the test names; it is not part of this repository) -- a script proven
against the real motors -- so a regression in the port shows up as a byte
diff rather than as unexpected hardware motion.
"""

from __future__ import annotations

import math
import os
import struct
import sys

import pytest

_HERE = os.path.dirname(__file__)

from actuo.core.contracts import (  # noqa: E402
    ActuatorCommand,
    CapabilityError,
    Frame,
    OperatingMode,
)
from actuo.drivers.actuators.cubemars import (  # noqa: E402
    CubeMarsActuator,
    CubeMarsCalibration,
    CubeMarsLimits,
    ServoPacket,
)

# Rotor pole pairs are a datasheet number, NOT something to infer. It is left
# unset (0) wherever a test does not need it, which keeps the position-mode
# tests honest: they must pass without it, because position never uses it.
SOME_POLE_PAIRS = 21


def make_actuator(
    node_id=105,
    gear_ratio=9.0,
    direction=1,
    zero_offset_counts=0.0,
    speed_erpm=50.0,
    accel_erpm_s=50.0,
    pole_pairs=0,
    torque_constant=None,
):
    calib = CubeMarsCalibration(
        gear_ratio=gear_ratio,
        direction=direction,
        zero_offset_counts=zero_offset_counts,
        pole_pairs=pole_pairs,
        torque_constant=torque_constant,
    )
    limits = CubeMarsLimits(
        max_velocity_erpm=speed_erpm,
        max_acceleration_erpm_s=accel_erpm_s,
    )
    return CubeMarsActuator(node_id, calib, limits=limits)


def reference_script_frame(motor_id, target_deg, speed_erpm, accel_erpm_s):
    """Byte-for-byte reimplementation of the reference script's position-speed frame."""
    position_raw = round(target_deg * 10000.0)
    speed_raw = round(speed_erpm / 10.0)
    accel_raw = round(accel_erpm_s / 10.0)
    return (
        (6 << 8) | motor_id,
        struct.pack(">ihh", position_raw, speed_raw, accel_raw),
    )


def test_matches_reference_script_wire_bytes():
    """The actuo path must emit exactly what the proven script emits.

    Note pole_pairs is deliberately unset here: position mode must not depend on
    it, so this equivalence holds with nothing guessed.
    """
    actuator = make_actuator(node_id=105, gear_ratio=9.0, speed_erpm=50.0,
                             accel_erpm_s=50.0)
    frame = actuator.encode_command(
        ActuatorCommand(mode=OperatingMode.POSITION, position=math.radians(30.0))
    )

    want_id, want_payload = reference_script_frame(105, 30.0, 50.0, 50.0)
    assert frame.address == want_id
    assert frame.payload == want_payload
    assert frame.meta["extended"] is True


def test_arbitration_id_encodes_control_mode_and_node():
    actuator = make_actuator(node_id=104)
    frame = actuator.encode_command(
        ActuatorCommand(mode=OperatingMode.POSITION, position=0.0)
    )
    # ref: manual 5.2 -- controller_id | (packet << 8)
    assert frame.address == (int(ServoPacket.SET_POS_SPD) << 8) | 104
    assert frame.address & 0xFF == 104


def test_position_is_gear_ratio_independent():
    """The firmware de-gears position itself, so a 39:1 and a 6:1 motor commanded
    to the same joint angle must put the SAME position on the wire.

    Applying gear_ratio to position would command the AK60-39 to 39x the angle --
    this test is the guard against that specific, dangerous mistake.
    """
    target = math.radians(20.0)
    payloads = []
    for ratio in (39.0, 9.0, 6.0):
        actuator = make_actuator(gear_ratio=ratio)
        frame = actuator.encode_command(
            ActuatorCommand(mode=OperatingMode.POSITION, position=target)
        )
        payloads.append(frame.payload[:4])  # int32 position field only

    assert payloads[0] == payloads[1] == payloads[2]
    assert struct.unpack(">i", payloads[0])[0] == 200000  # 20 deg * 10000


def test_erpm_scales_with_gear_ratio_like_reference_script():
    """The reference script hand-tuned 220/50/35 ERPM for 39:1/9:1/6:1 at one output speed.

    That is exactly ERPM ~ gear_ratio, so converting one common output velocity
    through each motor's calibration must reproduce those numbers.
    """
    reference = CubeMarsCalibration(gear_ratio=9.0, pole_pairs=SOME_POLE_PAIRS)
    output_rad_s = reference.erpm_to_rad_s(50.0)  # the speed the reference script normalises to

    got = {}
    for name, ratio in (("104", 39.0), ("105", 9.0), ("106", 6.0)):
        calib = CubeMarsCalibration(gear_ratio=ratio, pole_pairs=SOME_POLE_PAIRS)
        got[name] = calib.rad_s_to_erpm(output_rad_s)

    assert got["105"] == pytest.approx(50.0)
    assert got["104"] == pytest.approx(216.67, abs=0.1)  # the reference script uses 220
    assert got["106"] == pytest.approx(33.33, abs=0.1)   # the reference script uses 35


def test_position_round_trip_through_status_frame():
    actuator = make_actuator(gear_ratio=39.0)
    # A status frame reporting 123.4 deg: int16 deg*10 == 1234 (manual 5.2.1).
    frame = Frame(
        payload=struct.pack(">hhhbB", 1234, 0, 0, 30, 0),
        address=105,
        meta={"extended": True},
    )
    state = actuator.decode_state(frame)
    assert state.position == pytest.approx(math.radians(123.4))
    assert state.temperature == pytest.approx(30.0)
    assert state.error == 0


def test_decode_matches_reference_script_unpack():
    """The reference script unpacks '>hhh b B' with pos*0.1, speed*10, current*0.01."""
    actuator = make_actuator(gear_ratio=9.0)
    payload = struct.pack(">hhhbB", -456, 12, 250, 41, 0)
    state = actuator.decode_state(Frame(payload=payload, address=105))

    assert state.position == pytest.approx(math.radians(-456 * 0.1))
    assert state.aux["erpm"] == pytest.approx(12 * 10.0)
    assert state.aux["current_a"] == pytest.approx(250 * 0.01)
    assert state.temperature == pytest.approx(41.0)


def test_absent_datasheet_constants_leave_channels_none_not_zero():
    """Without pole_pairs/kt we cannot honestly report rad/s or N*m.

    A 0.0 here would look like "motor stopped" / "no torque" to a controller,
    which is exactly the silent fake-a-zero failure this stack never allows. The raw
    ERPM and amps must still come through as named aux channels.
    """
    actuator = make_actuator(pole_pairs=0, torque_constant=None)
    state = actuator.decode_state(
        Frame(payload=struct.pack(">hhhbB", 100, 12, 250, 40, 0), address=105)
    )
    assert state.velocity is None
    assert state.current is None
    assert state.aux["erpm"] == pytest.approx(120.0)
    assert state.aux["current_a"] == pytest.approx(2.5)
    assert state.position is not None  # position never needed a datasheet number


def test_velocity_reported_in_si_once_pole_pairs_known():
    actuator = make_actuator(gear_ratio=9.0, pole_pairs=SOME_POLE_PAIRS)
    state = actuator.decode_state(
        Frame(payload=struct.pack(">hhhbB", 0, 12, 0, 40, 0), address=105)
    )
    expected = actuator.calibration.erpm_to_rad_s(120.0)
    assert state.velocity == pytest.approx(expected)


def test_direction_and_offset_are_per_motor():
    flipped = make_actuator(direction=-1, zero_offset_counts=100.0)  # 10.0 deg
    frame = flipped.encode_command(
        ActuatorCommand(mode=OperatingMode.POSITION, position=math.radians(30.0))
    )
    # -1 * 30 deg = -30 deg, then +10 deg offset => -20 deg => -200000
    assert struct.unpack(">i", frame.payload[:4])[0] == -200000


def test_matches_routes_on_node_id_low_byte():
    """Status is identified by the low byte, for any NON-command packet id.

    The upload's packet id is undocumented and firmware-dependent (41 here, 9 on
    VESC-derived builds), so it must not be hardcoded. A bare id equal to the
    node id is deliberately NOT accepted: that is packet 0, which the vendor
    defines as SET_DUTY -- indistinguishable from a duty command.
    """
    actuator = make_actuator(node_id=106)
    # the id this firmware actually stamps on its periodic upload
    assert actuator.matches(Frame(payload=b"", address=(41 << 8) | 106))
    assert actuator.matches(Frame(payload=b"", address=(9 << 8) | 106))
    assert not actuator.matches(Frame(payload=b"", address=(41 << 8) | 105))
    assert not actuator.matches(Frame(payload=b"", address=None))


def test_own_command_frame_is_never_decoded_as_status():
    """Regression: this reported a healthy motor as 'encoder fault'.

    A SET_POS_SPD frame carries the node id in the same low byte as a status
    frame and is also 8 bytes, so matching on the low byte alone accepted it.
    Its payload's last byte is the speed's low byte -- 0x05 for a 50 ERPM
    envelope -- which decodes as error code 5, "encoder fault". Backends that
    echo transmitted frames (python-can's gs_usb) then made the driver diagnose
    a fault from a frame it had sent itself.
    """
    actuator = make_actuator(node_id=105, gear_ratio=9.0, speed_erpm=50.0,
                             accel_erpm_s=50.0)
    own = actuator.encode_command(
        ActuatorCommand(mode=OperatingMode.POSITION, position=math.radians(80.0))
    )
    # the trap: it really does look like status on every other axis
    assert own.address & 0xFF == 105
    assert len(own.payload) == 8
    assert struct.unpack(">hhhbB", own.payload)[4] == 5   # would read as "encoder fault"

    assert not actuator.matches(own)


def test_no_command_packet_is_ever_treated_as_status():
    actuator = make_actuator(node_id=105)
    for packet in ServoPacket:
        frame = Frame(payload=bytes(8), address=(int(packet) << 8) | 105,
                      meta={"extended": True})
        assert not actuator.matches(frame), "packet {} accepted".format(packet)


def test_error_and_remote_frames_are_not_status():
    actuator = make_actuator(node_id=105)
    status_id = (41 << 8) | 105
    assert actuator.matches(Frame(payload=bytes(8), address=status_id))
    assert not actuator.matches(
        Frame(payload=bytes(8), address=status_id, meta={"error_frame": True}))
    assert not actuator.matches(
        Frame(payload=bytes(8), address=status_id, meta={"remote_frame": True}))


def test_current_mode_unsupported_without_kt():
    actuator = make_actuator(torque_constant=None)
    assert OperatingMode.CURRENT not in actuator.capabilities.modes
    with pytest.raises(CapabilityError):
        actuator.encode_command(
            ActuatorCommand(mode=OperatingMode.CURRENT, current=1.0)
        )


def test_current_mode_applies_gear_ratio_to_torque():
    actuator = make_actuator(gear_ratio=9.0, torque_constant=0.1)
    assert OperatingMode.CURRENT in actuator.capabilities.modes
    # 1.8 N*m at the output = 1.8 / (0.1 * 9) = 2.0 A at the motor
    frame = actuator.encode_command(
        ActuatorCommand(mode=OperatingMode.CURRENT, current=1.8)
    )
    assert frame.address == (int(ServoPacket.SET_CURRENT) << 8) | 105
    assert struct.unpack(">i", frame.payload)[0] == 2000  # amps * 1000


def test_pole_pairs_required_rather_than_guessed():
    calib = CubeMarsCalibration(gear_ratio=9.0)  # pole_pairs left unset
    with pytest.raises(ValueError, match="pole_pairs"):
        calib.rad_s_to_erpm(1.0)
    # ...and the SI convenience constructor must refuse too, rather than
    # silently picking a pole-pair count and scaling real motion by it.
    with pytest.raises(ValueError, match="pole_pairs"):
        CubeMarsLimits.from_rad_s(calib, max_velocity=1.0, max_acceleration=1.0)


def test_limits_from_rad_s_round_trips_with_pole_pairs():
    calib = CubeMarsCalibration(gear_ratio=9.0, pole_pairs=SOME_POLE_PAIRS)
    limits = CubeMarsLimits.from_rad_s(
        calib,
        max_velocity=calib.erpm_to_rad_s(50.0),
        max_acceleration=calib.erpm_to_rad_s(120.0),
    )
    assert limits.max_velocity_erpm == pytest.approx(50.0)
    assert limits.max_acceleration_erpm_s == pytest.approx(120.0)


def test_zero_speed_envelope_is_refused():
    """A speed of 0 would be sent as int16 0 -- a silent never-arrives."""
    calib = CubeMarsCalibration(gear_ratio=9.0)
    actuator = CubeMarsActuator(
        105,
        calib,
        limits=CubeMarsLimits(max_velocity_erpm=0.0, max_acceleration_erpm_s=50.0),
    )
    with pytest.raises(ValueError, match="max_velocity"):
        actuator.encode_command(
            ActuatorCommand(mode=OperatingMode.POSITION, position=0.0)
        )


def test_position_limit_is_enforced():
    calib = CubeMarsCalibration(
        gear_ratio=9.0,
        position_limit=(-math.radians(45.0), math.radians(45.0)),
    )
    actuator = CubeMarsActuator(
        105,
        calib,
        limits=CubeMarsLimits(max_velocity_erpm=50.0, max_acceleration_erpm_s=50.0),
    )
    actuator.encode_command(
        ActuatorCommand(mode=OperatingMode.POSITION, position=math.radians(44.0))
    )
    with pytest.raises(ValueError, match="outside this joint's limit"):
        actuator.encode_command(
            ActuatorCommand(mode=OperatingMode.POSITION, position=math.radians(46.0))
        )


def test_the_torque_layer_can_ask_before_it_enables():
    """The taught envelope, as a REASON rather than an exception.

    A CubeMars arm hit the alternative twice in one day: a joint parked
    just outside its limit satisfied every guard -- the controller adopts the measured pose,
    so the jump is zero -- and one cycle later encode_command raised inside
    service(), killing the servicer with `bus write failed on cycle N`.
    """
    calib = CubeMarsCalibration(
        gear_ratio=9.0,
        position_limit=(-math.radians(45.0), math.radians(45.0)),
    )
    actuator = CubeMarsActuator(
        105, calib,
        limits=CubeMarsLimits(max_velocity_erpm=50.0, max_acceleration_erpm_s=50.0),
    )

    assert actuator.position_refusal(math.radians(44.0)) is None
    assert "outside this joint's limit" in \
        actuator.position_refusal(math.radians(46.0))
    # A joint with no taught limit has nothing to object to, and must say so
    # rather than inventing a bound.
    assert CubeMarsActuator(
        105, CubeMarsCalibration(gear_ratio=9.0)).position_refusal(99.0) is None


def test_bad_status_frame_length_is_refused():
    actuator = make_actuator()
    with pytest.raises(ValueError, match="8 bytes"):
        actuator.decode_state(Frame(payload=b"\x00\x01", address=105))


def test_node_id_must_fit_arbitration_low_byte():
    calib = CubeMarsCalibration(gear_ratio=9.0)
    with pytest.raises(ValueError, match="1-255"):
        CubeMarsActuator(300, calib)


def test_velocity_mode_needs_pole_pairs_but_position_mode_does_not():
    """Pins the boundary: the datasheet number is only load-bearing for rad/s."""
    actuator = make_actuator(pole_pairs=0)
    # position works with nothing guessed
    actuator.encode_command(
        ActuatorCommand(mode=OperatingMode.POSITION, position=0.1)
    )
    # velocity is not merely an error -- it is not advertised as a capability
    assert OperatingMode.VELOCITY not in actuator.capabilities.modes
    assert actuator.capabilities.has_velocity_sense is False
    with pytest.raises(CapabilityError):
        actuator.encode_command(
            ActuatorCommand(mode=OperatingMode.VELOCITY, velocity=0.1)
        )

    # ...and it appears as soon as the datasheet number is supplied
    informed = make_actuator(pole_pairs=SOME_POLE_PAIRS)
    assert OperatingMode.VELOCITY in informed.capabilities.modes
    informed.encode_command(
        ActuatorCommand(mode=OperatingMode.VELOCITY, velocity=0.1)
    )


def test_release_is_zero_current_and_needs_no_torque_constant():
    """Servo OFF is synthesised: the protocol has no disable packet.

    It must work on a motor whose kt we were never told, because zero torque is
    zero amps in any calibration -- nothing is inferred.
    """
    actuator = make_actuator(node_id=105, torque_constant=None)
    assert OperatingMode.CURRENT not in actuator.capabilities.modes  # no kt
    frame = actuator.encode_release()
    assert frame.address == (int(ServoPacket.SET_CURRENT) << 8) | 105
    assert struct.unpack(">i", frame.payload)[0] == 0
    assert frame.meta["extended"] is True


def test_release_frame_is_not_mistaken_for_status():
    actuator = make_actuator(node_id=105)
    assert not actuator.matches(actuator.encode_release())


# --------------------------------------------------------------------------
# Gear-window origin recovery.
#
# The AK series' rotor encoder is single-turn absolute, so a power cycle can
# lose which ROTOR TURN the output shaft is on -- an integer number of
# 360/gear_ratio output degrees, and nothing else.
#
# PROVEN ON HARDWARE 2026-07-30: joint3 (AK80-6, gear_ratio 6.0), arm untouched
# across a power cycle, reported 160.40 deg -> 100.40 deg. Exactly -60.00, i.e.
# one window of 360/6, with residual 0.00. The other three joints moved <=0.3.
# --------------------------------------------------------------------------

def _ak80_6(direction=1, zero_offset_deg=-12.10):
    return CubeMarsActuator(
        106,
        CubeMarsCalibration(
            gear_ratio=6.0,
            direction=direction,
            zero_offset_counts=zero_offset_deg * 10.0,
            pole_pairs=21,
        ),
    )


def test_gear_window_is_one_rotor_revolution_at_the_output():
    assert _ak80_6().calibration.gear_window_deg == pytest.approx(60.0)
    # The whole arm, for the record -- these set each joint's required coarse
    # reference accuracy (half a window).
    assert CubeMarsCalibration(gear_ratio=9.0).gear_window_deg == pytest.approx(40.0)
    assert CubeMarsCalibration(gear_ratio=39.0).gear_window_deg == pytest.approx(
        9.23076923, rel=1e-6)


def test_the_measured_power_cycle_jump_is_recovered_exactly():
    """The real numbers off the robot, end to end.

    After the power cycle the joint reported 100.40 deg for a pose it had been
    reporting as 160.40 deg. Given a coarse reference anywhere near the truth,
    the window snap must restore 160.40 -- not 160.4 +- a bit, exactly.
    """
    actuator = _ak80_6()
    period = actuator.calibration.gear_window_deg

    # what the wire carried after the power cycle, in this joint's counts
    counts_after = actuator.calibration.rad_to_counts(math.radians(100.40))
    assert actuator.calibration.counts_to_rad(counts_after) == pytest.approx(
        math.radians(100.40))

    # a deliberately sloppy reference: 11.6 deg out, the size of the
    # hand-positioning error this joint's config actually records
    actuator.snap_to_window(
        measured_rad=math.radians(100.40), coarse_rad=math.radians(172.0))

    assert actuator.calibration.window_offset == 1
    assert math.degrees(
        actuator.calibration.counts_to_rad(counts_after)) == pytest.approx(
            160.40, abs=1e-9)
    assert actuator.calibration.gear_window_deg == pytest.approx(period)


def test_window_offset_round_trips():
    """Command and feedback must agree, or the driver chases a 60 deg lurch."""
    actuator = _ak80_6()
    actuator.calibration.window_offset = 2
    for deg in (-91.3, 0.0, 40.4, 119.8, 160.4):
        rad = math.radians(deg)
        counts = actuator.calibration.rad_to_counts(rad)
        assert actuator.calibration.counts_to_rad(counts) == pytest.approx(
            rad, abs=1e-12), "window offset broke the round trip at {} deg".format(deg)


def test_a_reference_near_the_window_boundary_is_REFUSED():
    """Half a window away, the choice is a coin flip. Refuse, never guess.

    A wrong window is a position exactly 60 deg from the truth that looks
    entirely normal downstream -- the one output this stack must never emit.
    """
    actuator = _ak80_6()
    # 29 deg out of a 60 deg window: 96% of the way to the boundary.
    with pytest.raises(ValueError, match="cannot pick a gear window"):
        actuator.snap_to_window(
            measured_rad=math.radians(100.40), coarse_rad=math.radians(129.4))
    assert actuator.calibration.window_offset == 0, "refused but still mutated"


def test_the_accepted_band_matches_the_hand_error_the_config_records():
    """+-15 deg on a 60 deg window: every error this joint has ever shown.

    Its config history records 7.7, 11.2 and 13.9 deg of hand-positioning
    error. A margin that refused those would refuse the ordinary case and get
    widened until it was useless.
    """
    for off in (7.7, 11.2, 13.9, 14.9):
        actuator = _ak80_6()
        actuator.snap_to_window(
            measured_rad=math.radians(100.40),
            coarse_rad=math.radians(160.40 + off))
        assert actuator.calibration.window_offset == 1, off
    # ...and just past a quarter window it declines rather than guessing.
    actuator = _ak80_6()
    with pytest.raises(ValueError):
        actuator.snap_to_window(
            measured_rad=math.radians(100.40),
            coarse_rad=math.radians(160.40 + 15.2))


def test_the_refusal_names_the_accuracy_THIS_joint_needs():
    """joint1's window is 9.2 deg, so its reference must be far better."""
    tight = CubeMarsActuator(
        104, CubeMarsCalibration(gear_ratio=39.0, pole_pairs=21))
    with pytest.raises(ValueError) as caught:
            tight.snap_to_window(measured_rad=0.0, coarse_rad=math.radians(4.0))
    text = str(caught.value)
    assert "9.2" in text            # the window
    assert "4.6" in text            # the accuracy required


def test_snapping_twice_with_the_same_reference_is_a_no_op():
    """It accumulates against the CURRENT decode, so it must be idempotent."""
    actuator = _ak80_6()
    counts = actuator.calibration.rad_to_counts(math.radians(100.40))

    actuator.snap_to_window(math.radians(100.40), math.radians(160.40))
    assert actuator.calibration.window_offset == 1
    settled = actuator.calibration.counts_to_rad(counts)

    # second call, now passing what the actuator ACTUALLY reports
    actuator.snap_to_window(settled, math.radians(160.40))
    assert actuator.calibration.window_offset == 1, "second snap moved it again"
    assert actuator.calibration.counts_to_rad(counts) == pytest.approx(settled)


def test_an_exact_reference_needs_no_window():
    actuator = _ak80_6()
    actuator.snap_to_window(math.radians(160.40), math.radians(160.40))
    assert actuator.calibration.window_offset == 0


def test_a_reversed_joint_snaps_in_the_right_direction():
    """direction flips the calibrated frame; the window lives in the raw one."""
    actuator = _ak80_6(direction=-1)
    counts = actuator.calibration.rad_to_counts(math.radians(100.40))
    actuator.snap_to_window(math.radians(100.40), math.radians(160.40))
    assert math.degrees(
        actuator.calibration.counts_to_rad(counts)) == pytest.approx(
            160.40, abs=1e-9)
