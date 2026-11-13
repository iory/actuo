"""In-place calibration edits must change the numbers and NOTHING else.

The comments in a robot config are the calibration's reasoning -- which power-up
an offset was measured on, why an envelope has no inward margin. A yaml
round-trip deletes all of it, and a number without its reasoning gets
re-derived wrongly later. So these tests mostly assert about what did NOT
change.

    pytest tests/test_config_edit.py -q
"""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))

from actuo.drivers import config_edit  # noqa: E402
from actuo.drivers.config_edit import ConfigEditError  # noqa: E402

CONFIG = """\
joints:
  - name: joint0
    actuator:
      plugin: cubemars
      id: 103
      gear_ratio: 9.0
      # joint0's offset: measured 2026-07-27, do not "tidy" this number.
      zero_offset_deg: 1.50
      position_limit_deg: [-9.40, 33.30]
  - name: joint3
    actuator:
      plugin: cubemars
      id: 106
      gear_ratio: 6.0
      # A long explanation nobody may delete.
      #
      # Second paragraph of it.
      zero_offset_deg: 28.30
      startup_reference_deg: 0.0
      # Why this envelope is what it is.
      position_limit_deg: [-139.90, 19.40]   # interim
"""


def test_only_the_targeted_line_changes():
    out = config_edit.set_joint_value(CONFIG, "joint3", "zero_offset_deg", "-115.20")
    before = CONFIG.splitlines()
    after = out.splitlines()
    assert len(before) == len(after)
    differing = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    assert len(differing) == 1, [before[i] for i in differing]
    assert after[differing[0]].strip() == "zero_offset_deg: -115.20"


def test_every_comment_survives():
    out = CONFIG
    out = config_edit.set_joint_value(out, "joint3", "zero_offset_deg", "-115.20")
    out = config_edit.set_joint_value(
        out, "joint3", "position_limit_deg", "[-10.00, 120.00]")
    for comment in ("A long explanation nobody may delete.",
                    "Second paragraph of it.",
                    "Why this envelope is what it is.",
                    'joint0\'s offset: measured 2026-07-27'):
        assert comment in out, comment


def test_a_trailing_inline_comment_is_kept():
    out = config_edit.set_joint_value(
        CONFIG, "joint3", "position_limit_deg", "[-10.00, 120.00]")
    line = [l for l in out.splitlines() if "position_limit_deg" in l
            and "joint0" not in l][-1]
    assert "[-10.00, 120.00]" in line
    assert "# interim" in line, "dropped the inline comment: " + line


def test_the_SAME_key_on_another_joint_is_untouched():
    """The bug that would silently recalibrate the wrong joint."""
    out = config_edit.set_joint_value(CONFIG, "joint3", "zero_offset_deg", "-115.20")
    assert "zero_offset_deg: 1.50" in out, "joint0's offset was overwritten"
    assert config_edit.find_joint_value(out, "joint0", "zero_offset_deg") == "1.50"
    assert config_edit.find_joint_value(out, "joint3", "zero_offset_deg") == "-115.20"


def test_a_missing_key_is_inserted_into_the_right_block():
    out = config_edit.set_joint_value(
        CONFIG, "joint0", "startup_reference_deg", "0.0")
    assert config_edit.find_joint_value(
        out, "joint0", "startup_reference_deg") == "0.0"
    # ...and joint3's existing one is still its own.
    assert config_edit.find_joint_value(
        out, "joint3", "startup_reference_deg") == "0.0"
    assert out.count("startup_reference_deg") == 2


def test_indentation_is_taken_from_the_file():
    out = config_edit.set_joint_value(
        CONFIG, "joint0", "startup_reference_deg", "0.0")
    line = [l for l in out.splitlines() if "startup_reference_deg" in l][0]
    assert line.startswith("      "), repr(line)


def test_an_unknown_joint_is_refused_not_guessed():
    with pytest.raises(ConfigEditError, match="no joint named"):
        config_edit.set_joint_value(CONFIG, "elbow", "zero_offset_deg", "0")


def test_the_result_still_parses_as_yaml():
    yaml = pytest.importorskip("yaml")
    out = config_edit.set_joint_value(CONFIG, "joint3", "zero_offset_deg", "-115.20")
    out = config_edit.set_joint_value(
        out, "joint3", "position_limit_deg", "[-10.00, 120.00]")
    document = yaml.safe_load(out)
    joints = {j["name"]: j["actuator"] for j in document["joints"]}
    assert joints["joint3"]["zero_offset_deg"] == -115.20
    assert joints["joint3"]["position_limit_deg"] == [-10.00, 120.00]
    assert joints["joint0"]["zero_offset_deg"] == 1.50


URDF = """\
<?xml version="1.0"?>
<robot name="arm">
  <joint name="joint2" type="revolute">
    <limit lower="-0.778317" upper="1.724285" effort="6.0" velocity="1.0"/>
  </joint>
  <joint name="joint3" type="revolute">
    <!-- a comment that must survive -->
    <limit lower="-2.441716" upper="0.338594" effort="6.0" velocity="1.2840"/>
  </joint>
  <ros2_control name="arm" type="system">
    <joint name="joint2">
      <command_interface name="position">
        <param name="min">-0.778317</param>
        <param name="max">1.724285</param>
      </command_interface>
    </joint>
    <joint name="joint3">
      <command_interface name="position">
        <param name="min">-2.441716</param>
        <param name="max">0.338594</param>
      </command_interface>
    </joint>
  </ros2_control>
</robot>
"""


def test_both_copies_of_the_urdf_limits_move_together():
    """Updating one and not the other is the drift this prevents."""
    out = config_edit.set_urdf_joint_limits(URDF, "joint3", -0.5, 1.25)
    assert 'lower="-0.500000" upper="1.250000"' in out
    assert '<param name="min">-0.500000</param>' in out
    assert '<param name="max">1.250000</param>' in out
    assert "a comment that must survive" in out


def test_another_joints_urdf_limits_are_untouched():
    out = config_edit.set_urdf_joint_limits(URDF, "joint3", -0.5, 1.25)
    assert 'lower="-0.778317" upper="1.724285"' in out, "joint2 was overwritten"
    assert '<param name="min">-0.778317</param>' in out


def test_a_urdf_with_no_ros2_control_block_is_REFUSED():
    """Silently updating half of it is the failure mode, so refuse instead."""
    partial = URDF[:URDF.index("<ros2_control")] + "</robot>\n"
    with pytest.raises(ConfigEditError, match="ros2_control"):
        config_edit.set_urdf_joint_limits(partial, "joint3", -0.5, 1.25)


def test_an_unknown_joint_in_the_urdf_is_refused():
    with pytest.raises(ConfigEditError, match="no <joint"):
        config_edit.set_urdf_joint_limits(URDF, "wrist", -0.5, 1.25)


def test_the_urdf_result_still_parses_as_xml():
    import xml.etree.ElementTree as ET
    out = config_edit.set_urdf_joint_limits(URDF, "joint3", -0.5, 1.25)
    root = ET.fromstring(out)
    limits = {j.get("name"): j.find("limit") for j in root.findall("joint")}
    assert float(limits["joint3"].get("lower")) == pytest.approx(-0.5)
    assert float(limits["joint3"].get("upper")) == pytest.approx(1.25)
    assert float(limits["joint2"].get("lower")) == pytest.approx(-0.778317)
