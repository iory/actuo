"""Tests for the taught-limits URDF writer.

The generated file is a safety artefact -- it can carry the only real joint
limits a robot has -- so it must actually parse, and its numbers must survive the
round trip from taught degrees to URDF radians.
"""

from __future__ import annotations

import math
import os
import sys
import xml.etree.ElementTree as ET

import pytest

_HERE = os.path.dirname(__file__)

from actuo.cli import teach_limits  # noqa: E402

TAUGHT = [("shoulder", -38.0, 38.0), ("elbow", -88.0, 88.5), ("wrist", -28.0, 28.0)]
PLUGIN = "my_hardware/ShmSystem"


def write(tmp_path):
    path = str(tmp_path / "taught.urdf")
    teach_limits.write_urdf(
        path, robot="teachbot", shm="teachbot", taught=TAUGHT,
        effort="12.0", velocity="2.0", stamp="2026-07-17 00:00:00",
        plugin=PLUGIN,
    )
    return path


def test_generated_urdf_is_well_formed_xml(tmp_path):
    """Regression: the header used '--' inside an XML comment, which is illegal
    and made every generated URDF unparseable."""
    root = ET.parse(write(tmp_path)).getroot()
    assert root.get("name") == "teachbot"


def test_hardware_plugin_and_shm_name_come_from_the_caller(tmp_path):
    hardware = ET.parse(write(tmp_path)).getroot().find("ros2_control/hardware")
    assert hardware.find("plugin").text == PLUGIN
    assert hardware.find("param[@name='shm_name']").text == "teachbot"


def test_a_urdf_without_a_hardware_plugin_is_refused_before_the_bus(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        teach_limits.main(["--config", str(tmp_path / "unused.yaml"),
                           "--shm-name", "teachbot",
                           "--urdf", str(tmp_path / "t.urdf")])
    assert excinfo.value.code == 2


def test_limits_round_trip_from_taught_degrees_to_urdf_radians(tmp_path):
    root = ET.parse(write(tmp_path)).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    assert set(joints) == {"shoulder", "elbow", "wrist"}

    for name, low, high in TAUGHT:
        limit = joints[name].find("limit")
        assert math.degrees(float(limit.get("lower"))) == pytest.approx(low)
        assert math.degrees(float(limit.get("upper"))) == pytest.approx(high)
        assert joints[name].get("type") == "revolute"


def test_chain_is_serial_and_joint_order_matches_the_config(tmp_path):
    """ros2_control joint order is the contract with the shm layout."""
    root = ET.parse(write(tmp_path)).getroot()
    assert [j.get("name") for j in root.findall("joint")] == \
        [n for n, _, _ in TAUGHT]

    control = root.find("ros2_control")
    assert [j.get("name") for j in control.findall("joint")] == \
        [n for n, _, _ in TAUGHT]

    parents = [j.find("parent").get("link") for j in root.findall("joint")]
    assert parents == ["base_link", "shoulder_link", "elbow_link"]


def test_command_interface_limits_match_the_joint_limits(tmp_path):
    """A command_interface wider than the joint limit would defeat the point."""
    root = ET.parse(write(tmp_path)).getroot()
    control = root.find("ros2_control")
    for name, low, high in TAUGHT:
        joint = control.find("joint[@name='{}']".format(name))
        command = joint.find("command_interface[@name='position']")
        params = {p.get("name"): float(p.text) for p in command.findall("param")}
        assert math.degrees(params["min"]) == pytest.approx(low)
        assert math.degrees(params["max"]) == pytest.approx(high)


def test_placeholder_geometry_is_declared_loudly(tmp_path):
    """Nobody must mistake this for a measured model."""
    text = open(write(tmp_path)).read()
    assert "PLACEHOLDER" in text
    assert "DO NOT trust the geometry" in text
