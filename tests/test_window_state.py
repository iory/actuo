"""The remembered gear-window reference, and every way it is allowed to fail.

This file exists to hold one joint out of a 60 deg error, so its failure
behaviour matters more than its happy path: a reference it cannot vouch for must
read as ABSENT (fall back to the config constant), never as a number.

    pytest tests/test_window_state.py -q
"""

from __future__ import annotations

import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))

from actuo.drivers import window_state  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Never touch the real ~/.local/state while testing."""
    monkeypatch.delenv(window_state.STATE_DIR_ENV, raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    os.makedirs(window_state.state_dir(), exist_ok=True)
    return tmp_path


def test_state_dir_is_actuo_under_xdg_unless_overridden(tmp_path, monkeypatch):
    assert window_state.state_dir() == os.path.join(str(tmp_path), "actuo")
    elsewhere = str(tmp_path / "elsewhere")
    monkeypatch.setenv(window_state.STATE_DIR_ENV, elsewhere)
    assert window_state.state_dir() == elsewhere
    assert window_state.state_path("arm") == os.path.join(
        elsewhere, "window_arm.json")


def test_a_saved_position_comes_back():
    assert window_state.save_positions("arm", {"joint3": -0.41})
    assert window_state.load_positions("arm") == {"joint3": -0.41}


def test_state_survives_in_a_place_that_survives_a_reboot():
    """/tmp would be cleared exactly when the reference is still valid.

    The arm's physical pose survives a reboot, so the reference must too.
    """
    window_state.save_positions("arm", {"joint3": 1.0})
    path = window_state.state_path("arm")
    # Under XDG_STATE_HOME, not a per-boot scratch dir. The test pins the rule
    # by construction: state_dir() must honour XDG_STATE_HOME, which on a real
    # machine defaults to ~/.local/state and survives a reboot.
    assert path.startswith(os.environ["XDG_STATE_HOME"]), path
    assert os.path.exists(path)


def test_two_robots_do_not_share_a_reference():
    window_state.save_positions("left", {"joint3": 1.0})
    window_state.save_positions("right", {"joint3": -1.0})
    assert window_state.load_positions("left") == {"joint3": 1.0}
    assert window_state.load_positions("right") == {"joint3": -1.0}


def test_no_file_means_no_reference_not_an_error():
    assert window_state.load_positions("never-run") == {}
    assert window_state.load_age("never-run") is None


def test_a_truncated_file_reads_as_absent():
    """A crash mid-write must not break the next start-up."""
    window_state.save_positions("arm", {"joint3": 1.0})
    with open(window_state.state_path("arm"), "w") as handle:
        handle.write('{"version": 1, "positio')
    assert window_state.load_positions("arm") == {}


def test_garbage_and_wrong_shapes_read_as_absent():
    for content in ("", "null", "[]", '"hello"', '{"positions": 3}',
                    '{"positions": []}', '{}'):
        with open(window_state.state_path("arm"), "w") as handle:
            handle.write(content)
        assert window_state.load_positions("arm") == {}, content


def test_NaN_and_infinity_are_rejected():
    """json.load happily returns NaN, and it would poison the snap.

    int(round(nan)) raises far from here; inf silently picks an absurd window.
    Neither may reach snap_to_window.
    """
    for bad in ("NaN", "Infinity", "-Infinity"):
        with open(window_state.state_path("arm"), "w") as handle:
            handle.write('{"stamp": 1.0, "positions": {"joint3": %s}}' % bad)
        assert window_state.load_positions("arm") == {}, bad


def test_a_bad_entry_does_not_discard_the_good_ones():
    with open(window_state.state_path("arm"), "w") as handle:
        json.dump({"stamp": 1.0,
                   "positions": {"joint3": 0.5, "joint0": "elbow"}}, handle)
    assert window_state.load_positions("arm") == {"joint3": 0.5}


def test_saving_nothing_writes_nothing():
    """An empty dict is not a reference saying 'everything is at zero'."""
    assert window_state.save_positions("arm", {}) is False
    assert window_state.load_positions("arm") == {}


def test_a_save_failure_is_reported_not_raised():
    """Failing to save must never take down a running robot."""
    window_state.save_positions("arm", {"joint3": 1.0})
    path = window_state.state_path("arm")
    os.chmod(os.path.dirname(path), 0o500)          # read+execute, no write
    try:
        assert window_state.save_positions("arm", {"joint3": 2.0}) is False
    finally:
        os.chmod(os.path.dirname(path), 0o700)
    # ...and the previous reference is still intact.
    assert window_state.load_positions("arm") == {"joint3": 1.0}


def test_age_is_reported_but_does_not_invalidate():
    """Age is for the operator to judge; the code must not decide on it.

    What matters is whether the arm MOVED, which no timestamp can tell you. A
    week-old reference for an arm nobody touched is perfectly good.
    """
    window_state.save_positions("arm", {"joint3": 1.0})
    age = window_state.load_age("arm")
    assert age is not None and 0.0 <= age < 60.0

    with open(window_state.state_path("arm")) as handle:
        document = json.load(handle)
    document["stamp"] = 0.0                         # 1970
    with open(window_state.state_path("arm"), "w") as handle:
        json.dump(document, handle)

    assert window_state.load_age("arm") > 1e9
    assert window_state.load_positions("arm") == {"joint3": 1.0}, (
        "an old reference was discarded; age must not invalidate")


def test_no_temp_files_are_left_behind():
    for i in range(5):
        window_state.save_positions("arm", {"joint3": float(i)})
    leftovers = [f for f in os.listdir(os.path.dirname(
        window_state.state_path("arm"))) if f.endswith(".tmp")]
    assert leftovers == [], leftovers


def test_the_reference_round_trips_through_a_realistic_snap():
    """End to end with the actuator this exists for."""
    import math
    from actuo.drivers.actuators.cubemars import (
        CubeMarsActuator, CubeMarsCalibration)

    # Last run ended with joint3 correctly at -0.41 rad.
    window_state.save_positions("biped", {"joint3": -0.41})

    # Next start-up: the driver powered up two windows out, and the arm sagged
    # 3 deg while off.
    actuator = CubeMarsActuator(
        106, CubeMarsCalibration(gear_ratio=6.0, pole_pairs=21))
    period = math.radians(actuator.calibration.gear_window_deg)
    sagged = -0.41 + math.radians(3.0)
    measured = sagged - 2 * period

    reference = window_state.load_positions("biped")["joint3"]
    actuator.snap_to_window(measured, reference)
    recovered = measured + actuator.calibration.window_offset * period
    assert recovered == pytest.approx(sagged, abs=1e-9)


def test_a_held_joint_keeps_the_position_it_had_limp():
    """Held arms fall when the servicer dies; the file must say where they
    rest, not where they were held. 2026-09-29/30: held 30 deg forward, the
    abort left -29 deg in the file, the fallen arm was at -8, and the next
    start-up refused until the file was moved aside by hand."""
    rest, held_forward = -0.0035, -0.5078
    window_state.update_positions("upper", {"right_shoulder_pitch": rest})      # limp at rest
    window_state.update_positions("upper", {}, held={"right_shoulder_pitch"})   # held: nothing new
    assert window_state.load_positions("upper") == {"right_shoulder_pitch": rest}
    # Other joints still update while this one is held.
    window_state.update_positions("upper", {"neck_yaw": 0.0}, held={"right_shoulder_pitch"})
    assert window_state.load_positions("upper") == {"right_shoulder_pitch": rest, "neck_yaw": 0.0}
    # Limp again, wherever it came to rest: recorded.
    window_state.update_positions("upper", {"right_shoulder_pitch": held_forward})
    assert window_state.load_positions("upper")["right_shoulder_pitch"] == held_forward


def test_a_joint_released_is_forgotten_only_when_it_is_recorded_limp_again():
    window_state.update_positions("upper", {"a": 1.0, "b": 2.0})
    window_state.update_positions("upper", {"b": 2.5}, held={"a"})
    assert window_state.load_positions("upper") == {"a": 1.0, "b": 2.5}
    # A held joint with no previous record is not invented.
    window_state.update_positions("upper", {"b": 2.5}, held={"c"})
    assert "c" not in window_state.load_positions("upper")


# --- the datum a position was recorded under -------------------------------

def _datum(zero, direction):
    return {"zero_offset_deg": zero, "direction": direction}


def test_a_re_zero_converts_the_remembered_position():
    import math
    # Recorded at 10 deg with zero 0, direction -1: raw = -10 deg. The zero
    # then moves to -3 (the panel re-zero): the same raw now reads
    # -1 * (-10 - -3) = 7 deg.
    window_state.save_positions("r", {"j": math.radians(10.0)},
                                {"j": _datum(0.0, -1)})
    got = window_state.load_positions("r", {"j": _datum(-3.0, -1)})
    assert math.degrees(got["j"]) == pytest.approx(7.0)


def test_an_unchanged_datum_is_a_no_op():
    window_state.save_positions("r", {"j": 0.25}, {"j": _datum(5.0, 1)})
    assert window_state.load_positions("r", {"j": _datum(5.0, 1)})["j"] == \
        pytest.approx(0.25)


def test_a_file_without_datums_is_read_as_it_was():
    window_state.save_positions("r", {"j": 0.25})
    assert window_state.load_positions("r", {"j": _datum(-3.0, -1)})["j"] == \
        pytest.approx(0.25)


def test_update_rewrites_kept_held_values_in_the_current_frame():
    import math
    window_state.save_positions("r", {"held": math.radians(10.0)},
                                {"held": _datum(0.0, 1)})
    now = {"held": _datum(2.0, 1), "limp": _datum(0.0, 1)}
    window_state.update_positions("r", {"limp": 0.1}, {"held"}, now)
    with open(window_state.state_path("r")) as handle:
        document = json.load(handle)
    assert math.degrees(document["positions"]["held"]) == pytest.approx(8.0)
    assert document["datums"]["held"] == _datum(2.0, 1)
    # And read back under the same datum, it is not converted twice.
    assert math.degrees(window_state.load_positions("r", now)["held"]) == \
        pytest.approx(8.0)
