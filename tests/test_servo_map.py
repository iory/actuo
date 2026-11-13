"""Tests for ``servo_map``: a robot config taking its Kondo ids from rcb4's map."""

import textwrap

import pytest

from actuo.drivers.config import joints_with_servo_map, load_document

RCB4_MAP = """\
joint_name_to_servo_id: {
  head_joint0: {id: 32, direction: -1, offset: 2.5},
  rarm_joint0: 0,
  mechanum_joint1: {id: 20, direction: 1, offset: 0, type: continuous},
}
"""


def write_config(tmp_path, joints, servo_map=True):
    (tmp_path / "servo_config.yaml").write_text(RCB4_MAP)
    text = ""
    if servo_map:
        text += "servo_map: {format: rcb4, file: servo_config.yaml}\n"
    text += "joints:\n" + textwrap.indent(textwrap.dedent(joints), "  ")
    path = tmp_path / "robot.yaml"
    path.write_text(text)
    return str(path)


def joints_of(path):
    return joints_with_servo_map(load_document(path), path)


def test_ids_direction_and_offset_come_from_the_map(tmp_path):
    path = write_config(tmp_path, """\
        - name: head_joint0
          actuator: {plugin: kondo_ics, velocity: 1}
        - name: rarm_joint0
          actuator: {plugin: kondo_ics, velocity: 1}
        """)

    head, rarm = joints_of(path)

    assert head["actuator"] == {"plugin": "kondo_ics", "velocity": 1, "id": 32,
                                "direction": -1, "offset_deg": 2.5}
    # A bare id in rcb4's map means direction 1, offset 0.
    assert rarm["actuator"]["id"] == 0
    assert rarm["actuator"]["direction"] == 1
    assert rarm["actuator"]["offset_deg"] == 0.0


def test_the_order_is_the_configs_not_the_maps(tmp_path):
    path = write_config(tmp_path, """\
        - name: rarm_joint0
          actuator: {plugin: kondo_ics}
        - name: head_joint0
          actuator: {plugin: kondo_ics}
        """)

    assert [j["name"] for j in joints_of(path)] == ["rarm_joint0", "head_joint0"]


def test_an_id_given_in_both_places_is_refused(tmp_path):
    path = write_config(tmp_path, """\
        - name: head_joint0
          actuator: {plugin: kondo_ics, id: 32}
        """)

    with pytest.raises(ValueError, match="one source"):
        joints_of(path)


def test_a_joint_missing_from_the_map_is_refused(tmp_path):
    path = write_config(tmp_path, """\
        - name: larm_joint0
          actuator: {plugin: kondo_ics}
        """)

    with pytest.raises(KeyError, match="larm_joint0"):
        joints_of(path)


def test_a_wheel_in_the_map_driven_as_a_joint_is_refused(tmp_path):
    path = write_config(tmp_path, """\
        - name: mechanum_joint1
          actuator: {plugin: kondo_ics}
        """)

    with pytest.raises(ValueError, match="continuous"):
        joints_of(path)


def test_a_joint_in_the_map_driven_as_a_wheel_is_refused(tmp_path):
    path = write_config(tmp_path, """\
        - name: head_joint0
          actuator: {plugin: kondo_ics_wheel}
        """)

    with pytest.raises(ValueError, match="continuous"):
        joints_of(path)


def test_a_wheel_takes_its_id_from_the_map(tmp_path):
    path = write_config(tmp_path, """\
        - name: mechanum_joint1
          actuator: {plugin: kondo_ics_wheel, deadband_deg: 6.0}
        """)

    (wheel,) = joints_of(path)

    assert wheel["actuator"]["id"] == 20
    assert wheel["actuator"]["deadband_deg"] == 6.0


def test_without_a_servo_map_the_joints_are_untouched(tmp_path):
    path = write_config(tmp_path, """\
        - name: head_joint0
          actuator: {plugin: kondo_ics, id: 7}
        """, servo_map=False)

    (joint,) = joints_of(path)

    assert joint["actuator"] == {"plugin": "kondo_ics", "id": 7}
