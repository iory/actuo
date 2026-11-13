"""Robot modes: what is attached, and the models built from it (no ROS)."""

import os
import sys
import xml.etree.ElementTree as ET

import pytest


from actuo.drivers.robot_modes import RobotModes  # noqa: E402

BASE = """<robot name="r">
  <link name="base_link"/>
  <link name="j0_link"/>
  <joint name="j0" type="revolute"><parent link="base_link"/><child link="j0_link"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
  <ros2_control name="arm" type="system"><joint name="j0"/></ros2_control>
</robot>"""


def tool(name, joint):
    return """<robot name="{n}">
  <link name="{n}_base"/>
  <link name="{j}_link"/>
  <joint name="{j}" type="revolute"><parent link="{n}_base"/><child link="{j}_link"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
  <ros2_control name="{n}" type="system"><joint name="{j}"/></ros2_control>
</robot>""".format(n=name, j=joint)


def write(tmp_path, mode, slots, required=None):
    (tmp_path / "base.urdf").write_text(BASE)
    lines = ["mode: {}".format(mode), "base:", "  urdf: base.urdf", "  modules:",
             "    arm: {joints: [j0], controllers: [jtc_arm]}", "slots:"]
    for slot, candidates in slots.items():
        lines += ["  {}:".format(slot), "    parent_link: j0_link", "    xyz: [0, 0, 0.05]"]
        if required is not None:
            lines.append("    required: {}".format(str(required).lower()))
        lines.append("    candidates:")
        for name, joint in candidates:
            (tmp_path / (name + ".urdf")).write_text(tool(name, joint))
            lines.append("      {n}: {{urdf: {n}.urdf, joints: [{j}], "
                         "controllers: [jtc_{n}]}}".format(n=name, j=joint))
    path = tmp_path / "modes.yaml"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def joints_of(urdf):
    return sorted(j.get("name") for j in ET.fromstring(urdf).findall("joint"))


def test_tool_change_follows_what_answers(tmp_path):
    robot = RobotModes(write(tmp_path, "tool_change",
                             {"tool": [("gripper", "j8"), ("drill", "j2")]}))
    sel = robot.select({"j0", "j8"})
    assert sel.attached == {"tool": "gripper"} and not sel.problems
    assert [m.name for m in robot.running(sel)] == ["arm", "gripper"]
    assert joints_of(robot.compose_urdf(sel.attached)) == ["gripper_mount", "j0", "j8"]

    sel = robot.select({"j0", "j2"})
    assert sel.attached == {"tool": "drill"}
    sel = robot.select({"j0"})
    assert sel.attached == {"tool": None} and not sel.problems   # tool slots are optional


def test_two_tools_answering_at_once_is_a_conflict_not_a_guess(tmp_path):
    robot = RobotModes(write(tmp_path, "tool_change",
                             {"tool": [("gripper", "j8"), ("drill", "j2")]}))
    sel = robot.select({"j0", "j2", "j8"})
    assert sel.attached == {"tool": None}
    assert "only one can be attached" in sel.problems[0]
    assert [m.name for m in robot.running(sel)] == ["arm"]


def test_the_controller_manager_gets_every_candidate(tmp_path):
    robot = RobotModes(write(tmp_path, "tool_change",
                             {"tool": [("gripper", "j8"), ("drill", "j2")]}))
    root = ET.fromstring(robot.superset_urdf())
    assert sorted(c.get("name") for c in root.findall("ros2_control")) == [
        "arm", "drill", "gripper"]


def test_a_fixed_robot_reports_a_missing_module_and_keeps_its_model(tmp_path):
    robot = RobotModes(write(tmp_path, "fixed", {"tool": [("gripper", "j8")]}))
    sel = robot.select({"j0"})
    assert sel.problems == ["slot tool: gripper is missing"]
    assert robot.static_selection() == {"tool": "gripper"}
    sel = robot.select({"j8"})
    assert sel.missing_base == ["arm"] and "arm is missing" in sel.problems[0]


def test_a_fixed_robot_has_one_required_module_per_slot(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        RobotModes(write(tmp_path, "fixed", {"tool": [("gripper", "j8"), ("drill", "j2")]}))


def test_reconfigurable_requires_nothing(tmp_path):
    robot = RobotModes(write(tmp_path, "reconfigurable", {"tool": [("gripper", "j8")]}))
    sel = robot.select(set())
    assert sel.problems == [] and sel.missing_base == ["arm"]
    assert robot.running(sel) == []


def test_one_joint_in_two_modules_is_refused(tmp_path):
    with pytest.raises(ValueError, match="two hardware components"):
        RobotModes(write(tmp_path, "tool_change",
                         {"tool": [("gripper", "j0")]}))
