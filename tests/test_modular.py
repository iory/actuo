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
