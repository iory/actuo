#!/usr/bin/env python3
"""Make the links of servos that are not answering blink in RViz.

On a robot whose model is fixed, a servo that does not answer -- unplugged,
unpowered, a broken cable -- leaves its link drawn exactly where it was, which
looks like a healthy robot standing still. This node draws the links it moves
over the robot model, flashing red, for as long as the servicer reports the
joint absent, and labels each one with the bus's reason.

Inputs
------
* ``robot_description`` (std_msgs/String, transient local): which link each
  joint moves, and what that link looks like.
* ``<servicer>/servo/state`` (actuo_msgs/ServoState): which joints are present.

Output
------
* ``~/markers`` (visualization_msgs/MarkerArray), frames are the links
  themselves, so robot_state_publisher's TF places them.

Parameters
----------
servo_state : str
    The servicer's ServoState topic.
blink_hz : float
    Flash rate of a missing link.
subtree : bool
    Also flash every link downstream of the missing joint. Off by default:
    those links still have their own servos, and lighting them up would hide
    which one is actually gone.
"""

import math
import sys
import xml.etree.ElementTree as ET

import rclpy
from actuo_msgs.msg import ServoState
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray

NS_BODY = "missing_link"
NS_LABEL = "missing_label"
#: How much larger than the link a missing link's marker is drawn.
GROW = 1.25


def label_text(joint):
    """The 3-D label over a missing joint's link: two lines, no spaces.

    RViz 2's TEXT_VIEW_FACING draws a space far wider than a letter, a width
    that does not shrink with the text height: at the 12 mm used here every
    gap is tens of centimetres, so "rarm_joint3: not answering" showed only the
    middle word, "not", over the link, with the other two far off to the sides
    and usually out of view. A line break and an underscore have normal widths
    (checked in RViz, Jazzy). The text deliberately contains no space.
    """
    return "{}\nNOT_ANSWERING".format(joint)


def _floats(text, default):
    return [float(v) for v in text.split()] if text else list(default)


def _rpy_to_quaternion(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return Quaternion(x=sr * cp * cy - cr * sp * sy,
                      y=cr * sp * cy + sr * cp * sy,
                      z=cr * cp * sy - sr * sp * cy,
                      w=cr * cp * cy + sr * sp * sy)


def _origin_pose(element):
    origin = element.find("origin") if element is not None else None
    xyz = _floats(origin.get("xyz") if origin is not None else None, (0, 0, 0))
    rpy = _floats(origin.get("rpy") if origin is not None else None, (0, 0, 0))
    return Pose(position=Point(x=xyz[0], y=xyz[1], z=xyz[2]),
                orientation=_rpy_to_quaternion(*rpy))


def parse_model(urdf_xml):
    """joint -> child link, link -> visuals, link -> child links.

    Each visual is (marker type, pose, scale, mesh resource or ''). Only what
    a Marker can draw is kept; a link with no visual gets a small sphere so a
    missing joint is never invisible.
    """
    root = ET.fromstring(urdf_xml)
    child_of, children = {}, {}
    for joint in root.findall("joint"):
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        child_of[joint.get("name")] = child
        children.setdefault(parent, []).append(child)
    visuals = {}
    for link in root.findall("link"):
        shapes = []
        for visual in link.findall("visual"):
            geometry = visual.find("geometry")
            if geometry is None or len(geometry) == 0:
                continue
            shape = geometry[0]
            pose = _origin_pose(visual)
            if shape.tag == "box":
                size = _floats(shape.get("size"), (0.02, 0.02, 0.02))
                shapes.append((Marker.CUBE, pose, Vector3(x=size[0], y=size[1], z=size[2]), ""))
            elif shape.tag == "cylinder":
                r, length = float(shape.get("radius")), float(shape.get("length"))
                shapes.append((Marker.CYLINDER, pose, Vector3(x=2 * r, y=2 * r, z=length), ""))
            elif shape.tag == "sphere":
                d = 2 * float(shape.get("radius"))
                shapes.append((Marker.SPHERE, pose, Vector3(x=d, y=d, z=d), ""))
            elif shape.tag == "mesh":
                scale = _floats(shape.get("scale"), (1, 1, 1))
                shapes.append((Marker.MESH_RESOURCE, pose,
                               Vector3(x=scale[0], y=scale[1], z=scale[2]),
                               shape.get("filename")))
        if not shapes:
            shapes.append((Marker.SPHERE, Pose(orientation=Quaternion(w=1.0)),
                           Vector3(x=0.03, y=0.03, z=0.03), ""))
        visuals[link.get("name")] = shapes
    return child_of, children, visuals


class PresenceMarkers(Node):
    def __init__(self):
        super().__init__("actuo_presence_markers")
        self.declare_parameter("servo_state", "/actuo_bus_servicer/servo/state")
        self.declare_parameter("blink_hz", 1.0)
        self.declare_parameter("subtree", False)
        self.blink_hz = float(self.get_parameter("blink_hz").value)
        self.subtree = bool(self.get_parameter("subtree").value)
        self.model = None
        self.missing = {}          # joint -> reason
        self.drawn = set()         # (ns, id) currently published
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "robot_description", self._on_model, latched)
        self.create_subscription(ServoState, self.get_parameter("servo_state").value,
                                 self._on_state, latched)
        self.pub = self.create_publisher(MarkerArray, "~/markers", 10)
        self.create_timer(0.1, self._draw)

    def _on_model(self, msg):
        # A new model (a tool change, a reconfiguration) can move a joint to
        # another link: everything drawn for the old one goes.
        self.model = parse_model(msg.data)
        self.get_logger().info("robot model: {} joints".format(len(self.model[0])))

    def _on_state(self, msg):
        missing = {j.name: (j.absence_reason or "not answering")
                   for j in msg.joints if not j.present}
        for name in set(missing) - set(self.missing):
            self.get_logger().warn("{} is not answering: {}".format(name, missing[name]))
        for name in set(self.missing) - set(missing):
            self.get_logger().info("{} answers again".format(name))
        self.missing = missing

    def _links(self, joint):
        child_of, children, _ = self.model
        if joint not in child_of:
            return []
        links, todo = [], [child_of[joint]]
        while todo:
            link = todo.pop()
            links.append(link)
            if self.subtree:
                todo.extend(children.get(link, []))
        return links

    def _draw(self):
        if self.model is None:
            return
        _, _, visuals = self.model
        phase = (self.get_clock().now().nanoseconds * 1e-9 * self.blink_hz) % 1.0
        # Opaque, then gone: a translucent marker over the opaque link is
        # drawn through by it and barely shows.
        alpha = 1.0 if phase < 0.5 else 0.0
        markers, keep = [], set()
        for joint, reason in sorted(self.missing.items()):
            links = self._links(joint)
            if not links:
                self.get_logger().warn(
                    "{} is missing but is not a joint of the robot model".format(joint),
                    throttle_duration_sec=10.0)
            for link in links:
                for k, (kind, pose, scale, mesh) in enumerate(visuals.get(link, [])):
                    # A little larger than the link itself: drawn at the same
                    # size the two surfaces fight and the flash is lost.
                    marker = Marker(ns=NS_BODY, type=kind, action=Marker.ADD,
                                    pose=pose, scale=Vector3(x=scale.x * GROW, y=scale.y * GROW,
                                                             z=scale.z * GROW),
                                    color=ColorRGBA(r=1.0, g=0.1, b=0.1, a=alpha))
                    marker.header.frame_id = link
                    # Stamp 0 = "the latest transform": stamped with now, RViz
                    # hides the marker on every update whose TF has not
                    # arrived yet, and the whole flash (label too) flickered
                    # out at random instead of blinking.
                    marker.frame_locked = True
                    marker.id = hash((joint, link, k)) & 0x7FFFFFFF
                    if mesh:
                        marker.mesh_resource = mesh
                    markers.append(marker)
                    keep.add((NS_BODY, marker.id))
            if links:
                label = Marker(ns=NS_LABEL, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD,
                               pose=Pose(position=Point(z=0.06), orientation=Quaternion(w=1.0)),
                               scale=Vector3(z=0.012),
                               color=ColorRGBA(r=1.0, g=0.3, b=0.3, a=1.0),
                               # Short: the full reason is a sentence, which
                               # in the view is a line across the screen. It
                               # is in this node's log and in servo/state.
                               text=label_text(joint))
                label.header.frame_id = links[0]
                label.frame_locked = True  # stamp 0, as above
                label.id = hash(joint) & 0x7FFFFFFF
                markers.append(label)
                keep.add((NS_LABEL, label.id))
        for ns, marker_id in self.drawn - keep:
            markers.append(Marker(ns=ns, id=marker_id, action=Marker.DELETE))
        self.drawn = keep
        if markers:
            self.pub.publish(MarkerArray(markers=markers))


def main():
    rclpy.init(args=sys.argv)
    node = PresenceMarkers()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
