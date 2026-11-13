"""Tests for the rcb4->actuo Kondo importer."""

from __future__ import annotations

import math
import os

import pytest

pytest.importorskip("yaml")

_HERE = os.path.dirname(__file__)

from actuo.core.contracts import CapabilityError, Frame, Transport  # noqa: E402
from actuo.core.registry import register_transport  # noqa: E402
from actuo.drivers.config import build_bus  # noqa: E402
from actuo.drivers.kondo_import_rcb4 import (  # noqa: E402
    build_import,
    format_summary,
    load_rcb4_entries,
    render_yaml,
)


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def symbols_yaml():
    return """\
firmware_version: v-test
addresses:
  servo_vector: 604112640
  SysB: 604110848
"""


def base_urdf():
    return """\
<robot name="t">
  <joint name="joint_a" type="revolute">
    <limit lower="-1.57079632679" upper="1.57079632679" effort="1" velocity="1"/>
  </joint>
  <joint name="joint_b" type="revolute">
    <limit lower="-1.0" upper="0.5" effort="1" velocity="1"/>
  </joint>
  <joint name="joint_c" type="revolute">
    <limit lower="-0.25" upper="0.75" effort="1" velocity="1"/>
  </joint>
  <ros2_control name="t" type="system">
    <hardware>
      <plugin>my_hardware/ShmSystem</plugin>
    </hardware>
    <joint name="joint_b"/>
    <joint name="joint_a"/>
    <joint name="joint_c"/>
  </ros2_control>
</robot>
"""


def test_load_rcb4_entries_accepts_int_and_mapping(tmp_path):
    path = write(
        tmp_path,
        "servo.yaml",
        """\
joint_name_to_servo_id:
  j_int: 12
  j_dict:
    id: 34
    direction: -1
    offset: 5
    type: continuous
""",
    )

    entries = load_rcb4_entries(path)
    assert [entry.name for entry in entries] == ["j_int", "j_dict"]

    j_int, j_dict = entries
    assert j_int.servo_id == 12
    assert j_int.direction == 1
    assert j_int.offset == pytest.approx(0.0)
    assert j_int.servo_type is None

    assert j_dict.servo_id == 34
    assert j_dict.direction == -1
    assert j_dict.offset == pytest.approx(5.0)
    assert j_dict.servo_type == "continuous"


def test_continuous_joint_is_skipped_not_emitted(tmp_path):
    entries = load_rcb4_entries(
        write(
            tmp_path,
            "servo.yaml",
            """\
joint_name_to_servo_id:
  arm_joint: {id: 1, direction: 1, offset: 0}
  wheel_joint: {id: 20, direction: -1, offset: 0, type: continuous}
""",
        )
    )

    result = build_import(
        entries=entries,
        source_config_path="servo.yaml",
        urdf_path=None,
        symbols_file="symbols.yaml",
        channel_placeholder='"__SET_REAL_DEVICE_PATH__"',
    )

    assert [joint.entry.name for joint in result.active_joints] == ["arm_joint"]
    assert [entry.name for entry in result.report.continuous_skipped] == ["wheel_joint"]

    rendered = render_yaml(result)
    assert "- name: arm_joint" in rendered
    assert "# - name: wheel_joint" in rendered
    assert "continuous-rotation" in rendered


def test_shared_servo_id_is_reported_and_emitted_once(tmp_path):
    entries = load_rcb4_entries(
        write(
            tmp_path,
            "servo.yaml",
            """\
joint_name_to_servo_id:
  joint_one: 19
  joint_two: 19
""",
        )
    )

    result = build_import(
        entries=entries,
        source_config_path="servo.yaml",
        urdf_path=None,
        symbols_file="symbols.yaml",
        channel_placeholder='"__SET_REAL_DEVICE_PATH__"',
    )

    assert [joint.entry.name for joint in result.active_joints] == ["joint_one"]
    assert result.report.shared_id_conflicts
    conflict = result.report.shared_id_conflicts[0]
    assert conflict.servo_id == 19
    assert conflict.kept_joint == "joint_one"
    assert conflict.skipped_joints == ("joint_two",)

    rendered = render_yaml(result)
    assert rendered.count("      id: 19") == 1
    assert "# - id 19: kept joint_one, skipped joint_two" in rendered


def test_urdf_order_and_joint_set_disagreement_are_reported(tmp_path):
    servo_path = write(
        tmp_path,
        "servo.yaml",
        """\
joint_name_to_servo_id:
  joint_a: {id: 10, direction: 1, offset: 1}
  joint_b: {id: 11, direction: -1, offset: 2}
  extra_joint: {id: 12, direction: 1, offset: 3}
""",
    )
    urdf_path = write(tmp_path, "robot.urdf", base_urdf())

    result = build_import(
        entries=load_rcb4_entries(servo_path),
        source_config_path=servo_path,
        urdf_path=urdf_path,
        symbols_file="symbols.yaml",
        channel_placeholder='"__SET_REAL_DEVICE_PATH__"',
    )

    assert [joint.entry.name for joint in result.active_joints] == ["joint_b", "joint_a"]
    assert [entry.name for entry in result.report.servo_not_in_urdf] == ["extra_joint"]
    assert result.report.urdf_not_in_servo == ["joint_c"]

    summary = format_summary(result)
    assert "servo-only joints vs URDF: 1" in summary
    assert "URDF-only joints vs servo config: 1" in summary


def test_urdf_limit_radian_to_degree_conversion(tmp_path):
    servo_path = write(
        tmp_path,
        "servo.yaml",
        """\
joint_name_to_servo_id:
  joint_a: {id: 10, direction: 1, offset: 0}
  joint_b: {id: 11, direction: 1, offset: 0}
  joint_c: {id: 12, direction: 1, offset: 0}
""",
    )
    urdf_path = write(tmp_path, "robot.urdf", base_urdf())

    result = build_import(
        entries=load_rcb4_entries(servo_path),
        source_config_path=servo_path,
        urdf_path=urdf_path,
        symbols_file="symbols.yaml",
        channel_placeholder='"__SET_REAL_DEVICE_PATH__"',
    )

    limits = {joint.entry.name: joint.limit_deg for joint in result.active_joints}
    assert limits["joint_b"][0] == pytest.approx(math.degrees(-1.0))
    assert limits["joint_b"][1] == pytest.approx(math.degrees(0.5))
    assert limits["joint_a"][0] == pytest.approx(-90.0)
    assert limits["joint_a"][1] == pytest.approx(90.0)


@register_transport("kondo_import_fake")
class _FakeTransport(Transport):
    def __init__(self, **_kwargs):
        self._open = False

    def open(self):
        self._open = True

    def close(self):
        self._open = False

    @property
    def is_open(self):
        return self._open

    def write(self, frames):
        del frames

    def recv(self, timeout):
        del timeout
        return None

    def transact(self, frames, timeout):
        del frames, timeout
        raise CapabilityError("fake transport has no transact in this test")


def test_emitted_document_loads_via_build_bus(tmp_path):
    servo_path = write(
        tmp_path,
        "servo.yaml",
        """\
joint_name_to_servo_id:
  joint_a: {id: 2, direction: 1, offset: 0}
  joint_b: {id: 1, direction: -1, offset: 5}
  wheel_joint: {id: 20, direction: 1, offset: 0, type: continuous}
""",
    )
    urdf_path = write(tmp_path, "robot.urdf", base_urdf())

    result = build_import(
        entries=load_rcb4_entries(servo_path),
        source_config_path=servo_path,
        urdf_path=urdf_path,
        symbols_file="symbols.yaml",
        channel_placeholder='"__SET_REAL_DEVICE_PATH__"',
    )
    rendered = render_yaml(result)

    config_path = write(tmp_path, "robot.yaml", rendered)
    write(tmp_path, "symbols.yaml", symbols_yaml())

    bus, names = build_bus(
        config_path,
        transport_overrides={"plugin": "kondo_import_fake"},
    )

    assert names == ["joint_b", "joint_a"]
    assert [actuator.id for actuator in bus.actuators] == [1, 2]
