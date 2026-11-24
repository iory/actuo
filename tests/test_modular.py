"""Catalog of a modular robot: loading, and which configuration the answering ids make."""

import pytest

from actuo.modular import Catalog


def robot(tmp_path, name, ids, channel="/dev/ttyACM0"):
    joints = "".join(
        "  - {{name: {n}_j{i}, actuator: {{plugin: feetech_sts, id: {i}, model: STS3215}}}}\n"
        .format(n=name, i=i) for i in ids)
    (tmp_path / (name + ".yaml")).write_text(
        "bus: {{transport: {{plugin: serial, channel: {}, bitrate: 1000000}}}}\n"
        "joints:\n{}".format(channel, joints))
    (tmp_path / (name + ".urdf")).write_text("<robot name='{}'/>".format(name))


def catalog(tmp_path, configs, extra=""):
    lines = ["configurations:"]
    for name, ids in configs.items():
        robot(tmp_path, name, ids)
        lines.append("  {n}: {{robot: {n}.yaml, urdf: {n}.urdf}}".format(n=name))
    path = tmp_path / "catalog.yaml"
    path.write_text("\n".join(lines) + "\n" + extra)
    return Catalog(str(path))


def test_ids_and_joint_names_come_from_each_robot_config(tmp_path):
    c = catalog(tmp_path, {"gripper": [1, 2, 3], "drill": [1, 2, 4]})
    assert c.configurations["gripper"].ids == {1, 2, 3}
    assert c.configurations["drill"].joint_ids == {"drill_j1": 1, "drill_j2": 2, "drill_j4": 4}
    assert c.all_ids() == {1, 2, 3, 4}


def test_the_configuration_whose_servos_all_answer_is_detected(tmp_path):
    c = catalog(tmp_path, {"gripper": [1, 2, 3], "drill": [1, 2, 4]})
    assert c.detect({1, 2, 3}) == ("gripper", [])
    assert c.detect({1, 2, 4}) == ("drill", [])


def test_the_larger_complete_configuration_wins(tmp_path):
    c = catalog(tmp_path, {"arm": [1, 2], "arm_gripper": [1, 2, 3]})
    assert c.detect({1, 2, 3}) == ("arm_gripper", [])
    assert c.detect({1, 2}) == ("arm", [])


def test_two_complete_configurations_of_one_size_are_a_conflict_not_a_guess(tmp_path):
    c = catalog(tmp_path, {"gripper": [1, 2, 3], "drill": [1, 2, 4]})
    name, problems = c.detect({1, 2, 3, 4})
    assert name is None and "only one can be the robot" in problems[0]


def test_nothing_complete_is_reported(tmp_path):
    c = catalog(tmp_path, {"gripper": [1, 2, 3]})
    name, problems = c.detect({1})
    assert name is None and "no configuration" in problems[0]


def test_configurations_on_two_serial_lines_are_refused(tmp_path):
    robot(tmp_path, "a", [1])
    robot(tmp_path, "b", [2], channel="/dev/ttyACM9")
    (tmp_path / "catalog.yaml").write_text(
        "configurations:\n  a: {robot: a.yaml, urdf: a.urdf}\n"
        "  b: {robot: b.yaml, urdf: b.urdf}\n")
    with pytest.raises(ValueError, match="same serial line"):
        Catalog(str(tmp_path / "catalog.yaml"))


def test_a_declared_signature_overrides_the_config_ids(tmp_path):
    robot(tmp_path, "a", [1, 2])
    (tmp_path / "catalog.yaml").write_text(
        "configurations:\n  a: {robot: a.yaml, urdf: a.urdf, ids: [1]}\n"
        "default: a\nselect: declare\nhold: none\n")
    c = Catalog(str(tmp_path / "catalog.yaml"))
    assert c.configurations["a"].ids == {1}
    assert (c.default, c.select, c.hold) == ("a", "declare", "none")


# --- Switching (inside a bus servicer) -------------------------------------------

class FakeBus:
    def __init__(self, ids, answering):
        self.actuators = [type("A", (), {"id": i})() for i in ids]
        self._answering = answering

    def answering_ids(self):
        return sorted(self._answering)


class FakeSupervisor:
    def __init__(self, enabled):
        self.enabled_joints = list(enabled)
        self.disabled = []

    def disable(self, joints):
        self.disabled += list(joints)
        self.enabled_joints = [j for j in self.enabled_joints if j not in joints]
        return True, ""


def gripper_drill(tmp_path, extra=""):
    from actuo.modular import Switching
    c = catalog(tmp_path, {"gripper": [1, 2, 3], "drill": [1, 2, 4]}, extra)
    return Switching(c, "gripper")


def test_detection_asks_to_leave_and_carries_the_shared_servos(tmp_path):
    s = gripper_drill(tmp_path)
    names = ["gripper_j1", "gripper_j2", "gripper_j3"]
    sup = FakeSupervisor(["gripper_j1", "gripper_j3"])
    assert s.poll(FakeBus([1, 2, 3], {1, 2, 3}), sup, names) is None
    request = s.poll(FakeBus([1, 2, 3], {1, 2, 4}), sup, names)
    assert request.target == "drill" and request.hold_ids == {1}
    assert sup.disabled == ["gripper_j3"]          # id 3 is not in drill: released
    s.started(request)
    assert s.current == "drill"
    assert s.hold_joints(["drill_j1", "drill_j2", "drill_j4"],
                         FakeBus([1, 2, 4], set()).actuators) == ["drill_j1"]


def test_a_declaration_wins_over_detection_until_resumed(tmp_path):
    s = gripper_drill(tmp_path)
    names = ["gripper_j1", "gripper_j2", "gripper_j3"]
    ok, _ = s.declare("drill")
    assert ok and s.select == "declare"
    request = s.poll(FakeBus([1, 2, 3], {1, 2, 3}), FakeSupervisor([]), names)
    assert request.target == "drill"
    assert "do not answer" in s.problems[0]         # id 4 is not there: said, not refused
    s.started(request)
    assert s.poll(FakeBus([1, 2, 4], {1, 2, 3}), FakeSupervisor([]), names) is None
    s.resume_detection()
    assert s.poll(FakeBus([1, 2, 4], {1, 2, 3}), FakeSupervisor([]), names).target == "gripper"


def test_an_unknown_declaration_is_refused(tmp_path):
    s = gripper_drill(tmp_path)
    ok, message = s.declare("hammer")
    assert not ok and "hammer" in message and s.select == "detect"


def test_hold_none_releases_everything(tmp_path):
    s = gripper_drill(tmp_path, "hold: none\n")
    sup = FakeSupervisor(["gripper_j1"])
    request = s.poll(FakeBus([1, 2, 3], {1, 2, 4}), sup,
                     ["gripper_j1", "gripper_j2", "gripper_j3"])
    assert request.hold_ids == frozenset() and sup.disabled == ["gripper_j1"]


# --- ros2_control files ----------------------------------------------------------

def modular_robot(tmp_path, name, joints, base="base_link"):
    links = "".join("<link name='{}_link'/>".format(j) for j in joints)
    chain, parent = "", base
    for j in joints:
        chain += ("<joint name='{j}' type='revolute'><parent link='{p}'/><child link='{j}_link'/>"
                  "<limit lower='-1' upper='1' effort='1' velocity='1'/></joint>").format(j=j, p=parent)
        parent = j + "_link"
    (tmp_path / (name + ".urdf")).write_text(
        "<robot name='{}'><link name='{}'/>{}{}</robot>".format(name, base, links, chain))
    actuators = "".join(
        "  - {{name: {j}, actuator: {{plugin: feetech_sts, id: {i}, model: STS3215}}}}\n"
        .format(j=j, i=i) for i, j in enumerate(joints, 1))
    (tmp_path / (name + ".yaml")).write_text(
        "bus: {{transport: {{plugin: serial, channel: /dev/null, bitrate: 1000000}}}}\n"
        "urdf: {n}.urdf\nmodules:\n  arm: {{joints: [{js}]}}\njoints:\n{a}".format(
            n=name, js=", ".join(joints), a=actuators))


def test_every_configuration_goes_into_one_controller_manager(tmp_path):
    import xml.etree.ElementTree as ET

    import yaml as y

    from actuo.modular import ros2_control_files

    modular_robot(tmp_path, "gripper", ["g1", "g2"])
    modular_robot(tmp_path, "drill", ["d1"])
    (tmp_path / "catalog.yaml").write_text(
        "configurations:\n  gripper: {robot: gripper.yaml, urdf: gripper.urdf}\n"
        "  drill: {robot: drill.yaml, urdf: drill.urdf}\ndefault: gripper\n")
    m = ros2_control_files(str(tmp_path / "catalog.yaml"), str(tmp_path / "out"),
                           "mod", "pkg/Plugin")
    root = ET.parse(m["urdf"]).getroot()
    assert [l.get("name") for l in root.findall("link")].count("base_link") == 1
    blocks = {b.get("name"): b for b in root.findall("ros2_control")}
    assert sorted(blocks) == ["drill_arm", "gripper_arm"]
    assert blocks["drill_arm"].find("hardware/param[@name='shm_name']").text == "mod_drill"
    assert m["configurations"]["drill"]["controllers"] == [
        "drill_arm_joint_state_broadcaster", "drill_arm_controller"]
    assert m["configurations"]["drill"]["modules"]["drill_arm"]["joints"] == ["d1"]
    params = y.safe_load(open(m["controllers_yaml"]))
    cm = params["controller_manager"]["ros__parameters"]
    assert cm["hardware_components_initial_state"]["inactive"] == ["gripper_arm", "drill_arm"]
    assert params["drill_arm_joint_state_broadcaster"]["ros__parameters"]["joints"] == ["d1"]


def test_one_joint_name_in_two_configurations_is_refused(tmp_path):
    from actuo.modular import ros2_control_files

    modular_robot(tmp_path, "a", ["j1"])
    modular_robot(tmp_path, "b", ["j1"])
    (tmp_path / "catalog.yaml").write_text(
        "configurations:\n  a: {robot: a.yaml, urdf: a.urdf}\n  b: {robot: b.yaml, urdf: b.urdf}\n")
    with pytest.raises(ValueError, match="name it differently"):
        ros2_control_files(str(tmp_path / "catalog.yaml"), str(tmp_path / "out"), "m", "p/P")
