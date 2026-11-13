"""ROS shell around ServoSupervisor: the SetServo action and the state topic.

Deliberately thin. Every safety decision lives in :mod:`actuo.drivers.servo`,
which is ROS-free and unit-tested; this file only translates. If a rule starts
appearing here, it has escaped the place it can be proven.

Single-threaded on purpose: the caller pumps :meth:`ServoNode.spin_once` from
the same loop that services the bus, so an action callback can never observe or
mutate supervisor state halfway through a control cycle. No locks, because there
is nothing to race.
"""

from __future__ import annotations

from typing import Callable, List

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from actuo_msgs.action import SetServo
from actuo_msgs.msg import JointServoState, ServoState

from actuo.drivers.servo import JointServo, ServoSupervisor


def _to_msg(supervisor: ServoSupervisor, joint: JointServo) -> JointServoState:
    msg = JointServoState()
    msg.name = joint.name
    # Per joint: a wheel's `commanded` is rad/s while the arm joint beside it in
    # the same array is rad. A consumer must never have to guess which quantity
    # it is reading.
    msg.mode = joint.mode.value
    msg.enabled = joint.enabled
    msg.present = joint.present
    msg.absence_reason = joint.absence_reason
    msg.measured = joint.measured
    msg.commanded = joint.commanded
    msg.mismatch = joint.mismatch
    # Precomputed server-side so a client cannot re-derive the safety rule and
    # get it wrong -- an RViz panel must not be free to decide what is safe.
    msg.would_jump = supervisor.would_jump(joint)
    msg.error = joint.error
    msg.error_name = joint.error_name
    msg.temperature = joint.temperature
    msg.velocity = joint.velocity
    msg.current_a = joint.current_a
    msg.effort = joint.effort
    # Sorted so a bag's column order is stable across runs: an endurance log
    # whose fields move between restarts is a log nobody can plot.
    names = sorted(joint.aux)
    msg.aux_names = names
    msg.aux_values = [float(joint.aux[k]) for k in names]
    return msg


class ServoNode:
    """Publishes ServoState and serves SetServo for one ServoSupervisor.

    Parameters
    ----------
    supervisor : ServoSupervisor
        Owns the actual rule. This class never decides anything.
    read_shm : callable
        Returns the shm bridge. Passed lazily because the servicer attaches to
        shared memory only after the controller has created it.
    """

    def __init__(self, supervisor: ServoSupervisor, read_shm: Callable,
                 node_name: str = "actuo_bus_servicer"):
        self.supervisor = supervisor
        self._read_shm = read_shm
        self.node = Node(node_name)

        # transient_local: a panel or a bag that joins late must immediately
        # learn whether the robot is holding, not wait for the next change.
        # This topic is the audit trail -- an action records what was *asked*
        # (including refusals); only this records what the robot *was*.
        self._pub = self.node.create_publisher(
            ServoState, "~/servo/state",
            QoSProfile(depth=1,
                       history=HistoryPolicy.KEEP_LAST,
                       reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self._action = ActionServer(
            self.node, SetServo, "~/servo/set",
            execute_callback=self._execute,
            goal_callback=lambda _goal: GoalResponse.ACCEPT,
            cancel_callback=lambda _goal: CancelResponse.REJECT,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        # A separate service, not another field on SetServo: clearing a safety
        # latch and applying torque are different acts, and merging them is how
        # one gets performed while the operator believes they asked for the
        # other. Trigger, because the release takes no parameters -- the pose it
        # adopts is the one the filter is already holding, and letting a caller
        # name one would let them name the pose it stopped for.
        self._release = None
        if supervisor.command_filter is not None:
            from std_srvs.srv import Trigger
            self._release = self.node.create_service(
                Trigger, "~/filter/release", self._on_release,
                callback_group=MutuallyExclusiveCallbackGroup())
            self.node.get_logger().info(
                "command filter active. If it latches, release it with: "
                "ros2 service call {}/filter/release std_srvs/srv/Trigger"
                .format(self.node.get_name()))

        self.node.get_logger().info(
            "servo: torque is OFF. Enable with: ros2 action send_goal "
            "{}/servo/set actuo_msgs/action/SetServo \"{{enable: true}}\""
            .format(self.node.get_name()))

    def _on_release(self, _request, response):
        """Clear a latched command filter. Logged either way.

        A release is an operator overriding a safety stop, so it belongs in the
        log next to the stop it clears -- at WARN when it succeeds, because
        "someone released the collision stop" is exactly the line you want to
        find afterwards, and INFO would put it among the cycle counters.
        """
        ok, message = self.supervisor.release_filter()
        response.success = bool(ok)
        response.message = message
        if ok:
            self.node.get_logger().warn("filter RELEASED by request: " + message)
        else:
            self.node.get_logger().info("filter release refused: " + message)
        return response

    # --- the loop calls these ---------------------------------------------

    def spin_once(self) -> None:
        rclpy.spin_once(self.node, timeout_sec=0.0)

    def publish(self) -> None:
        joints: List[JointServo] = self.supervisor.inspect(self._read_shm())
        msg = ServoState()
        msg.stamp = self.node.get_clock().now().to_msg()
        msg.jump_threshold = self.supervisor.jump_threshold
        # getattr: an older supervisor has no `mode`, and an empty string is the
        # honest answer to "which quantity is this" when nothing said.
        mode = getattr(self.supervisor, "mode", None)
        msg.mode = mode.name.lower() if mode is not None else ""
        msg.joints = [_to_msg(self.supervisor, j) for j in joints]
        msg.all_enabled = self.supervisor.all_enabled
        # Only a joint that is currently limp can be thrown by enabling it; one
        # already holding is, by definition, already where it is being told to
        # be. Reporting a held joint as "would jump" would light up the panel's
        # warning permanently and train operators to ignore it.
        msg.any_would_jump = any(m.would_jump for m in msg.joints
                                 if not m.enabled)

        # Bus health, straight off the counters the bus already keeps. A bus
        # that does not keep them (not every transport can) simply reports 0,
        # which for a COUNTER is honest -- unlike a measurement, "none seen" and
        # "cannot see" agree at zero here only because the field means "how many
        # have I counted", and a bus that counts nothing has counted none.
        bus = getattr(self.supervisor, "bus", None)
        msg.cycles = int(getattr(self.supervisor, "cycles", 0) or 0)
        msg.error_frames = int(getattr(bus, "error_frames", 0) or 0)
        msg.unmatched_frames = int(getattr(bus, "unmatched_frames", 0) or 0)
        self._pub.publish(msg)

    def destroy(self) -> None:
        self._action.destroy()
        self.node.destroy_node()

    # --- action -----------------------------------------------------------

    def _execute(self, goal_handle):
        request = goal_handle.request
        result = SetServo.Result()

        feedback = SetServo.Feedback()
        feedback.stage = "checking"
        goal_handle.publish_feedback(feedback)

        shm = self._read_shm()
        # `joints` is part of the goal ("empty = every joint"), so it must reach
        # the supervisor. Ignoring it -- as this did -- turned "release joint2"
        # into "release the whole arm" while still reporting success, which on
        # an arm means the elbow and everything past it drops.
        selected = list(request.joints)
        if request.enable:
            feedback.stage = "enabling"
            goal_handle.publish_feedback(feedback)
            ok, reason = self.supervisor.enable(shm, joints=selected,
                                                force=request.force)
        else:
            feedback.stage = "releasing"
            goal_handle.publish_feedback(feedback)
            ok, reason = self.supervisor.disable(joints=selected)

        result.ok = ok
        result.reason = reason
        result.joints = [_to_msg(self.supervisor, j)
                         for j in self.supervisor.inspect(shm)]

        # A refusal is a real outcome, not an error: the goal ran and the answer
        # was no. abort() makes that impossible for a client to mistake for
        # success, while the reason says exactly what to do about it.
        if ok:
            self.node.get_logger().info("servo: {}".format(reason))
            goal_handle.succeed()
        else:
            self.node.get_logger().warn("servo REFUSED: {}".format(reason))
            goal_handle.abort()
        return result
