"""Config-loading tests: the joint map lives in YAML, not in code.

Also guards the shipped robot config, so a typo there fails in CI rather than on
a powered bus.
"""

from __future__ import annotations

import math
import os
import sys

import pytest

pytest.importorskip("can", reason="python-can not installed")
pytest.importorskip("yaml")

_HERE = os.path.dirname(__file__)

from actuo.core.contracts import OperatingMode  # noqa: E402
from actuo.drivers.config import build_bus  # noqa: E402

VIRTUAL_CONFIG = """
bus:
  stale_after: 1.0
  transport:
    plugin: python_can
    interface: virtual
    channel: cfg-test
joints:
  - name: shoulder
    actuator:
      plugin: cubemars
      id: 104
      gear_ratio: 39.0
      direction: -1
      zero_offset_deg: 5.0
      position_limit_deg: [-45.0, 45.0]
      limits:
        max_velocity_erpm: 220.0
        max_acceleration_erpm_s: 220.0
  - name: elbow
    actuator:
      plugin: cubemars
      id: 105
      gear_ratio: 9.0
      pole_pairs: 21
      torque_constant: 0.1
      limits:
        max_velocity_erpm: 50.0
        max_acceleration_erpm_s: 50.0
"""


def write(tmp_path, text, name="robot.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_config_drives_calibration_and_capabilities(tmp_path):
    bus, names = build_bus(write(tmp_path, VIRTUAL_CONFIG))
    assert names == ["shoulder", "elbow"]

    shoulder, elbow = bus.actuators
    assert shoulder.calibration.direction == -1
    # 5.0 deg offset is stored as deg*10 counts
    assert shoulder.calibration.zero_offset_counts == pytest.approx(50.0)
    assert shoulder.calibration.position_limit[1] == pytest.approx(math.radians(45.0))
    assert OperatingMode.VELOCITY not in shoulder.capabilities.modes  # no pole_pairs

    # elbow supplied both datasheet numbers, so it gains both modes
    assert OperatingMode.VELOCITY in elbow.capabilities.modes
    assert OperatingMode.CURRENT in elbow.capabilities.modes
    assert bus.stale_after == 1.0


def test_duplicate_node_ids_are_rejected(tmp_path):
    text = VIRTUAL_CONFIG.replace("id: 105", "id: 104")
    with pytest.raises(ValueError, match="duplicate actuator ids"):
        build_bus(write(tmp_path, text))


def test_duplicate_joint_names_are_rejected(tmp_path):
    text = VIRTUAL_CONFIG.replace("name: elbow", "name: shoulder")
    with pytest.raises(ValueError, match="duplicate joint names"):
        build_bus(write(tmp_path, text))


def test_missing_required_key_names_the_joint(tmp_path):
    text = VIRTUAL_CONFIG.replace("      gear_ratio: 39.0\n", "")
    with pytest.raises(KeyError, match="gear_ratio"):
        build_bus(write(tmp_path, text))


def test_unknown_plugin_is_refused(tmp_path):
    text = VIRTUAL_CONFIG.replace("plugin: cubemars", "plugin: nonexistent_motor", 1)
    with pytest.raises(KeyError, match="nonexistent_motor"):
        build_bus(write(tmp_path, text))


# --------------------------------------------------------------------------
# Kondo ARM-H7: the same loader, a completely different bus family
# --------------------------------------------------------------------------

KONDO_CONFIG = """
bus:
  plugin: kondo_h7_bus
  symbols: symbols.yaml
  state_slots: [current_angle]
  move_time_ms: 20.0
  transport:
    plugin: kondo_h7
    channel: /dev/null
    bitrate: 1000000

joints:
  - name: head_neck_y
    actuator:
      plugin: kondo_ics
      id: 0
      ics_channel: J1
      direction: 1
      offset_deg: 0.0
      position_limit_deg: [-60.0, 60.0]
  - name: head_neck_p
    actuator:
      plugin: kondo_ics
      id: 1
      direction: -1
      offset_deg: 10.0
"""

KONDO_SYMBOLS = """
firmware_version: v-test
addresses:
  servo_vector: 604112640
  SysB: 604110848
"""


def write_kondo(tmp_path):
    """A Kondo robot config plus the firmware symbol map it names."""
    (tmp_path / "symbols.yaml").write_text(KONDO_SYMBOLS)
    path = tmp_path / "robot.yaml"
    path.write_text(KONDO_CONFIG)
    return str(path)


def test_kondo_config_builds_a_bus_without_touching_the_core(tmp_path):
    """A second protocol family, wired entirely from config, no core edit."""
    from actuo.drivers.actuators.kondo_ics import CENTER_PULSE, PULSE_PER_DEG

    bus, names = build_bus(write_kondo(tmp_path))
    assert names == ["head_neck_y", "head_neck_p"]
    assert [a.id for a in bus.actuators] == [0, 1]
    assert bus.state_slots == ("current_angle",)
    assert bus.move_time_ms == 20.0
    assert bus.symbols.version == "v-test"

    neck_y, neck_p = bus.actuators
    assert neck_y.ics_channel == "J1"
    assert neck_y.calibration.direction == 1
    assert neck_p.calibration.direction == -1
    # offset_deg shifts which angle the servo's neutral pulse reports
    assert neck_p.calibration.zero_offset_counts == pytest.approx(
        CENTER_PULSE + 10.0 * PULSE_PER_DEG)
    assert neck_y.calibration.position_limit == pytest.approx(
        (math.radians(-60.0), math.radians(60.0)))


def test_kondo_config_resolves_the_symbol_map_next_to_itself(tmp_path):
    """The map is named relative to the robot config, so a robot directory
    stays movable. A missing map is refused, never defaulted."""
    (tmp_path / "robot.yaml").write_text(KONDO_CONFIG)
    with pytest.raises(FileNotFoundError):
        build_bus(str(tmp_path / "robot.yaml"))


def test_kondo_config_requires_a_symbol_map(tmp_path):
    """Without addresses there is nothing to read; guessing one is not an
    option, so the loader refuses rather than starting blind."""
    path = tmp_path / "robot.yaml"
    path.write_text(KONDO_CONFIG.replace("  symbols: symbols.yaml\n", ""))
    with pytest.raises(KeyError, match="symbols"):
        build_bus(str(path))


def test_a_package_reference_finds_a_map_the_config_does_not_live_beside(
        tmp_path, monkeypatch):
    # package:// is resolved through ROS's ament index (a ROS 2 install).
    pytest.importorskip("ament_index_python")
    """`package://<pkg>/<path>` resolves through the ament index.

    A robot's config and the board firmware's symbol map do not belong to the
    same package: the map describes a firmware build shared by every Kondo
    robot, so a config names the one copy rather than carrying a duplicate.
    """
    share = tmp_path / "share" / "some_pkg"
    (share / "config").mkdir(parents=True)
    (share / "config" / "map.yaml").write_text(KONDO_SYMBOLS)
    monkeypatch.setattr(
        "ament_index_python.packages.get_package_share_directory",
        lambda name: str(share) if name == "some_pkg" else _raise_missing(name))

    path = tmp_path / "robot.yaml"
    path.write_text(KONDO_CONFIG.replace(
        "  symbols: symbols.yaml",
        "  symbols: package://some_pkg/config/map.yaml"))

    bus, _ = build_bus(str(path))
    assert bus.symbols.version == "v-test"


def _raise_missing(name):
    raise LookupError(name)


def test_an_unbuilt_package_is_named_in_the_error_not_silently_skipped(tmp_path):
    """"No such file: map.yaml" after a silent path join tells the operator
    nothing about where it was looked for."""
    path = tmp_path / "robot.yaml"
    path.write_text(KONDO_CONFIG.replace(
        "  symbols: symbols.yaml",
        "  symbols: package://no_such_pkg_here/config/map.yaml"))
    with pytest.raises(FileNotFoundError) as excinfo:
        build_bus(str(path))
    assert "no_such_pkg_here" in str(excinfo.value)
    assert "ACTUO_PACKAGE_PATH" in str(excinfo.value)


def test_a_package_reference_resolves_without_ros(tmp_path, monkeypatch):
    """ACTUO_PACKAGE_PATH finds the package with no ROS installed: actuo does
    not depend on ROS, and the ament index is only one place to look."""
    root = tmp_path / "packages"
    (root / "some_pkg" / "config").mkdir(parents=True)
    (root / "some_pkg" / "config" / "map.yaml").write_text(KONDO_SYMBOLS)
    monkeypatch.setenv("ACTUO_PACKAGE_PATH", str(tmp_path / "empty") + os.pathsep
                       + str(root))
    path = tmp_path / "robot.yaml"
    path.write_text(KONDO_CONFIG.replace(
        "  symbols: symbols.yaml",
        "  symbols: package://some_pkg/config/map.yaml"))
    bus, _ = build_bus(str(path))
    assert bus.symbols.version == "v-test"


def test_without_ros_the_error_says_so(tmp_path, monkeypatch):
    """With no ROS 2 to look in, say that, not "not on the ament index"."""
    import builtins
    real_import = builtins.__import__

    def no_ament(name, *args, **kwargs):
        if name.startswith("ament_index_python"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_ament)
    monkeypatch.delenv("ACTUO_PACKAGE_PATH", raising=False)
    path = tmp_path / "robot.yaml"
    path.write_text(KONDO_CONFIG.replace(
        "  symbols: symbols.yaml",
        "  symbols: package://some_pkg/config/map.yaml"))
    with pytest.raises(FileNotFoundError, match="ROS 2 is not installed"):
        build_bus(str(path))


def test_a_malformed_package_reference_is_refused(tmp_path):
    path = tmp_path / "robot.yaml"
    path.write_text(KONDO_CONFIG.replace(
        "  symbols: symbols.yaml", "  symbols: package://only_a_package"))
    with pytest.raises(ValueError, match="package://"):
        build_bus(str(path))


# --- sensors (non-joint devices on the bus) ----------------------------------
def test_build_sensors_instantiates_a_kjs03_from_config():
    """A `sensors:` entry becomes a plugin instance, data-driven and with no
    core edit -- the same open/closed contract the actuators use."""
    from actuo.drivers.config import build_sensors
    from actuo.drivers.sensors.kjs03 import Kjs03Sensor

    built = build_sensors([{"name": "air_board",
                            "sensor": {"plugin": "kjs03", "id": 19}}])
    assert len(built) == 1
    name, sensor = built[0]
    assert name == "air_board"
    assert isinstance(sensor, Kjs03Sensor)
    assert sensor.id == 19


def test_build_sensors_rejects_an_unknown_plugin():
    from actuo.drivers.config import build_sensors
    with pytest.raises(KeyError, match="no config builder for sensor plugin"):
        build_sensors([{"name": "x", "sensor": {"plugin": "nope", "id": 1}}])


def test_build_sensors_is_empty_when_absent():
    from actuo.drivers.config import build_sensors
    assert build_sensors(None) == []


def _cubemars_joint(plugin, **extra):
    spec = {"plugin": plugin, "id": 21, "model": "AK80-9", "gear_ratio": 9.0,
            "limits": {"max_velocity_erpm": 100.0,
                       "max_acceleration_erpm_s": 100.0}}
    if plugin == "cubemars_v3":
        spec["force_control"] = {"p_max": 12.56, "v_max": 65.0, "t_max": 18.0}
        spec["impedance"] = {"kp": 200.0, "kd": 5.0}
    spec.update(extra)
    return {"name": "neck_yaw", "actuator": spec}


@pytest.mark.parametrize("plugin", ["cubemars", "cubemars_v3"])
def test_startup_reference_reaches_every_servo_status_model(plugin):
    # cubemars_v3 decodes the servo status frame and has the same gear
    # windows, so it must get the same recovery reference. It did not: the
    # attribute was set by the servo builder only, and every reader asking
    # getattr(..., None) took its absence for "no recovery wanted".
    from actuo.drivers.config import build_actuators
    (actuator,), _ = build_actuators(
        [_cubemars_joint(plugin, startup_reference_deg=-86.0)])
    assert actuator.startup_reference_rad == pytest.approx(math.radians(-86.0))
    (bare,), _ = build_actuators([_cubemars_joint(plugin)])
    assert bare.startup_reference_rad is None


def test_kondo_ics_stretch_and_hold_relax_are_read():
    from actuo.drivers.config import _build_kondo_ics

    actuator = _build_kondo_ics(
        {"id": 14, "stretch": 40,
         "hold_relax": {"error_deg": 3.0, "seconds": 1.0, "squeeze_deg": 1.0}},
        "joints[0].actuator")

    assert actuator.stretch == 40
    assert actuator.hold_relax.error_rad == pytest.approx(math.radians(3.0))
    assert actuator.hold_relax.seconds == 1.0
    assert actuator.hold_relax.squeeze_rad == pytest.approx(math.radians(1.0))


def test_kondo_ics_hold_relax_needs_every_key():
    from actuo.drivers.config import _build_kondo_ics

    with pytest.raises(KeyError, match="squeeze_deg"):
        _build_kondo_ics({"id": 14, "hold_relax": {"error_deg": 3.0, "seconds": 1.0}},
                         "joints[0].actuator")


def test_the_arm_h7_bus_refuses_stretch_it_would_ignore(tmp_path):
    (tmp_path / "symbols.yaml").write_text(KONDO_SYMBOLS)
    path = tmp_path / "robot.yaml"
    path.write_text(KONDO_CONFIG.replace("      ics_channel: J1\n",
                                         "      ics_channel: J1\n      stretch: 40\n"))
    with pytest.raises(ValueError, match="only the RCB-4mini"):
        build_bus(str(path))
