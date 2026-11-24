"""ROS shell around actuo.modular.Switching: the declare service and the state topic.

Lives in the bus servicer for as long as it runs a catalog, across every
configuration (the servo node is rebuilt per configuration, this one is not).
Like :class:`~actuo_ros.servo_node.ServoNode` it is single-threaded: the
servicer's loop pumps :meth:`spin_once`, so a declaration is taken between two
cycles and acted on by :meth:`actuo.modular.Switching.poll` in the next.

* ``~/declare`` (actuo_msgs/DeclareConfiguration): make a configuration the
  robot now; an empty name resumes detection.
* ``~/configuration`` (actuo_msgs/Configuration, latched): what runs now; the
  configuration manager follows it with controllers and the robot model.
"""

from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from actuo_msgs.msg import Configuration
from actuo_msgs.srv import DeclareConfiguration

from actuo.modular import Switching

LATCHED = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                     reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)


class ConfigurationNode:
    """Publishes Configuration and serves DeclareConfiguration for one Switching."""

    def __init__(self, switching: Switching, node_name: str = "actuo_configuration"):
        self.switching = switching
        self.node = Node(node_name)
        self._pub = self.node.create_publisher(Configuration, "~/configuration", LATCHED)
        self.node.create_service(DeclareConfiguration, "~/declare", self._declare)
        self._last = None

    def _declare(self, request, response):
        if not request.name:
            self.switching.resume_detection()
            response.success, response.message = True, "detection resumed"
        else:
            response.success, response.message = self.switching.declare(request.name)
        return response

    def publish(self, force: bool = False) -> None:
        """Publish the state if it changed (or always, with ``force``)."""
        s = self.switching
        key = (s.current, s.select, tuple(s.problems), tuple(s.answering),
               tuple(s.held_joints), s.last_switch_s)
        if key == self._last and not force:
            return
        self._last = key
        msg = Configuration()
        msg.stamp = self.node.get_clock().now().to_msg()
        msg.name = s.current
        msg.select = s.select
        msg.problems = list(s.problems)
        msg.answering_ids = [int(i) for i in s.answering]
        msg.held_joints = list(s.held_joints)
        msg.switch_seconds = (math.nan if s.last_switch_s is None
                              else float(s.last_switch_s))
        self._pub.publish(msg)

    def spin_once(self) -> None:
        rclpy.spin_once(self.node, timeout_sec=0.0)
        self.publish()

    def destroy(self) -> None:
        self.node.destroy_node()
