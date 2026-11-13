"""actuo_presence_markers: the label text and the URDF parsing, without a node."""

import importlib.util
import os

import pytest

pytest.importorskip("actuo_msgs.msg")

from visualization_msgs.msg import Marker  # noqa: E402

_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "scripts", "actuo_presence_markers.py")


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "actuo_presence_markers", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


markers = _load_script()

_URDF = """<robot name="r">
  <link name="base"/>
  <link name="upper">
    <visual><geometry><box size="0.1 0.2 0.3"/></geometry></visual>
  </link>
  <link name="lower"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="upper"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="upper"/><child link="lower"/>
  </joint>
</robot>"""


def test_label_has_no_space():
    """RViz 2 draws a space far wider than a letter: the label has none."""
    text = markers.label_text("rarm_joint3")
    assert " " not in text
    assert text.splitlines() == ["rarm_joint3", "NOT_ANSWERING"]


def test_parse_model_maps_joints_to_child_links_and_visuals():
    child_of, children, visuals = markers.parse_model(_URDF)
    assert child_of == {"j1": "upper", "j2": "lower"}
    assert children == {"base": ["upper"], "upper": ["lower"]}
    kind, _pose, scale, mesh = visuals["upper"][0]
    assert kind == Marker.CUBE
    assert (scale.x, scale.y, scale.z) == pytest.approx((0.1, 0.2, 0.3))
    assert mesh == ""
    # A link with no visual still gets something to flash.
    assert visuals["lower"][0][0] == Marker.SPHERE
