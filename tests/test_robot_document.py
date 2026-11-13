"""robot_document: a whole robot from one config and its URDF."""

import math
import xml.etree.ElementTree as ET

import pytest
import yaml

from actuo.drivers import robot_document as rd
from actuo.drivers.config import load_document

URDF = """<robot name="toy">
  <link name="base"/><link name="l1"/><link name="l2"/><link name="l3"/><link name="l4"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
    <limit lower="-1.0" upper="1.0" effort="1" velocity="1"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/>
    <limit lower="-0.5" upper="2.0" effort="1" velocity="1"/></joint>
  <joint name="grip" type="revolute"><parent link="l2"/><child link="l3"/>
    <limit lower="0.0" upper="0.8" effort="1" velocity="1"/></joint>
  <joint name="wheel" type="continuous"><parent link="base"/><child link="l4"/></joint>
  <ros2_control name="sim" type="system"><hardware><plugin>gz</plugin></hardware></ros2_control>
  <gazebo/>
</robot>"""

CONFIG = """
urdf: toy.urdf
bus: {plugin: kondo_rcb4_bus, transport: {plugin: kondo_rcb4, channel: /dev/null}}
modules:
  arm: {joints: [j1, j2], gripper: [grip]}
joint_defaults:
  max_velocity: 2.0
  actuator: {plugin: kondo_ics, velocity: 1}
joints:
  - {name: j1, actuator: {id: 3}}
  - {name: j2, actuator: {id: 5, position_limit_deg: [-10.0, 10.0]}}
  - {name: grip, actuator: {id: 7, stretch: 40}}
"""


def write(tmp_path, config=CONFIG):
    (tmp_path / "toy.urdf").write_text(URDF)
    path = tmp_path / "robot.yaml"
    path.write_text(config)
    return str(path)


def test_the_joints_are_the_modules_with_defaults_and_urdf_limits(tmp_path):
    document = load_document(write(tmp_path))

    joints = {j["name"]: j for j in document["joints"]}
    assert [j["name"] for j in document["joints"]] == ["j1", "j2", "grip"]
    assert joints["j1"]["max_velocity"] == 2.0
    assert joints["j1"]["actuator"] == {
        "plugin": "kondo_ics", "velocity": 1, "id": 3,
        "position_limit_deg": [pytest.approx(math.degrees(-1.0)),
                               pytest.approx(math.degrees(1.0))]}
    # A limit given in the config wins over the URDF's.
    assert joints["j2"]["actuator"]["position_limit_deg"] == [-10.0, 10.0]
    assert joints["grip"]["actuator"]["stretch"] == 40


def test_a_joints_entry_in_no_module_is_refused(tmp_path):
    config = CONFIG + "  - {name: wheel, actuator: {id: 9}}\n"
    with pytest.raises(ValueError, match="'wheel' is in no module"):
        load_document(write(tmp_path, config))


def test_a_module_joint_not_in_the_model_is_refused(tmp_path):
    config = CONFIG.replace("joints: [j1, j2]", "joints: [j1, j9]").replace(
        "  - {name: j2, actuator: {id: 5, position_limit_deg: [-10.0, 10.0]}}\n", "")
    with pytest.raises(ValueError, match="not in the model.*j9"):
        load_document(write(tmp_path, config))


def test_a_joint_in_two_modules_is_refused(tmp_path):
    config = CONFIG.replace("arm: {joints: [j1, j2], gripper: [grip]}",
                            "arm: {joints: [j1, j2]}\n  hand: {joints: [j2], gripper: [grip]}")
    with pytest.raises(ValueError, match="j2 is in modules arm and hand"):
        load_document(write(tmp_path, config))


def test_a_plain_bus_config_is_left_alone(tmp_path):
    path = tmp_path / "bus.yaml"
    text = "bus: {plugin: x}\njoints:\n  - {name: a, actuator: {plugin: kondo_ics, id: 1}}\n"
    path.write_text(text)
    assert load_document(str(path)) == yaml.safe_load(text)


def test_controllers_for_a_module_and_its_gripper(tmp_path):
    document = load_document(write(tmp_path))

    params = rd.controller_parameters(document)

    manager = params["controller_manager"]["ros__parameters"]
    assert manager["update_rate"] == rd.DEFAULT_UPDATE_RATE
    assert manager["arm_controller"] == {"type": rd.TRAJECTORY_CONTROLLER}
    assert manager["arm_gripper_controller"] == {"type": rd.TRAJECTORY_CONTROLLER}
    assert params["joint_state_broadcaster"]["ros__parameters"]["joints"] == [
        "j1", "j2", "grip"]
    arm = params["arm_controller"]["ros__parameters"]
    assert arm["joints"] == ["j1", "j2"]
    assert arm["constraints"]["j1"] == {"trajectory": 0.5, "goal": 0.1}
    grip = params["arm_gripper_controller"]["ros__parameters"]
    assert grip["joints"] == ["grip"]
    assert grip["constraints"]["grip"] == {"trajectory": 0.0, "goal": 0.0}


def test_controller_parameters_merge_over_what_is_generated(tmp_path):
    config = CONFIG + (
        "trajectory_tolerance: {goal: 0.05}\n"
        "controller_parameters:\n"
        "  arm_controller: {ros__parameters: {action_monitor_rate: 5.0}}\n")
    params = rd.controller_parameters(load_document(write(tmp_path, config)))

    arm = params["arm_controller"]["ros__parameters"]
    assert arm["action_monitor_rate"] == 5.0
    assert arm["constraints"]["j2"] == {"trajectory": 0.5, "goal": 0.05}


SHM_PLUGIN = "my_hardware/ShmSystem"


@pytest.mark.parametrize("backend, plugin", [
    ("shm", SHM_PLUGIN), ("mock", "mock_components/GenericSystem")])
def test_hardware_urdf_has_one_component_per_module(tmp_path, backend, plugin):
    document = load_document(write(tmp_path))

    robot = ET.fromstring(rd.hardware_urdf(URDF, document, "toy", backend,
                                           hardware_plugin=SHM_PLUGIN))

    assert robot.find("gazebo") is None
    blocks = robot.findall("ros2_control")
    assert [b.get("name") for b in blocks] == ["arm"]
    assert blocks[0].find("hardware/plugin").text == plugin
    joints = blocks[0].findall("joint")
    assert [j.get("name") for j in joints] == ["j1", "j2", "grip"]
    slots = [j.find("param[@name='slot']") for j in joints]
    if backend == "shm":
        assert blocks[0].find("hardware/param[@name='shm_name']").text == "toy"
        assert [s.text for s in slots] == ["0", "1", "2"]
    else:
        assert slots == [None, None, None]


def test_hardware_urdf_shm_backend_needs_a_plugin_name(tmp_path):
    document = load_document(write(tmp_path))
    with pytest.raises(ValueError, match="hardware_plugin"):
        rd.hardware_urdf(URDF, document, "toy", "shm")


def test_robot_modes_lists_each_module_with_its_controllers(tmp_path):
    document = load_document(write(tmp_path))

    modes = rd.robot_modes(document, "/tmp/toy.urdf")

    assert modes == {"mode": "fixed", "base": {"urdf": "/tmp/toy.urdf", "modules": {
        "arm": {"joints": ["j1", "j2", "grip"],
                "controllers": ["arm_controller", "arm_gripper_controller"]}}}}
