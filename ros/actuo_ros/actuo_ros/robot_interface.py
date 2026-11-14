"""Drive a ROS 2 robot whose bus is serviced by actuo_bus_node from scikit-robot.

The robot is expected to run robot_state_publisher, a controller manager with
``joint_trajectory_controller`` instances, and (for the torque switch)
``actuo_bus_node``'s ``<servicer>/servo/set`` action.

``connect()`` returns two things, as in the euslisp robot interfaces
(``*robot*`` and ``*ri*``):

* ``robot``: a scikit-robot model of the robot. Moving its joints moves
  nothing; it is where a pose is made and checked. It is built from the
  running URDF, as ``model_class`` if one is given (a ``RobotModelFromURDF``
  subclass that adds end coordinates and ``reset_pose``) or as a plain
  ``RobotModelFromURDF``.
* ``ri``: the robot interface. ``ri.angle_vector(av, time)`` sends a pose to
  the robot, ``ri.wait_interpolation()`` waits until it gets there, and
  ``ri.servo_on()`` / ``ri.servo_off()`` switch the torque.

Nothing about the robot is written here. The model is the URDF the running
system publishes on ``/robot_description``, and each controller's joints are
what the controller manager says that controller commands.
"""

import atexit
import os
import tempfile
import threading
from time import monotonic

import control_msgs.action
import numpy as np
import rclpy
from controller_manager_msgs.srv import ListControllers
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from skrobot.interfaces.ros2.base import ROS2RobotInterfaceBase
from skrobot.models.urdf import RobotModelFromURDF
from std_msgs.msg import String


class ServoRefused(RuntimeError):
    """The servo switch answered, and said no (the reason is the message)."""


class PoseRefused(RuntimeError):
    """A trajectory controller rejected the pose it was sent."""

#: robot_state_publisher publishes the description once, transient local.
LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)

#: The controller type this interface drives.
TRAJECTORY_CONTROLLER = 'joint_trajectory_controller/JointTrajectoryController'

#: [rad] How far past a URDF limit a measured joint still counts as at it
#: (see RobotInterface.outside_limits). Well under actuo_bus_node's 0.05 rad
#: --max-start-jump, so holding such a joint at the limit cannot be a jump.
LIMIT_TOLERANCE = 0.01

#: [rad/s] Speed at which servo_on brings a joint found outside its URDF limits
#: back to the nearest limit (about 11 deg/s).
LIMIT_RETURN_SPEED = 0.2

#: [rad/s] A limp joint slower than this counts as come to rest: after a
#: refused servo_on, the re-sync waits for it. A joint still falling moves
#: past the pose sent to its controller before the switch checks it.
STILL_SPEED = 0.1

#: [s] of joint_states stamps the speed is measured over, and how long
#: servo_on waits at most for the joints to come to rest.
STILL_WINDOW = 0.1
STILL_TIMEOUT = 3.0


# [s] of wall time the robot's clock may stand still while something is
# waited for on it, before the wait gives up: a stopped simulator (or no
# /clock at all) is reported instead of waiting forever.
CLOCK_STALL = 30.0

# [ns] the robot's clock may go back while something is waited for on it
# before the wait gives up: more is a restarted simulator (its clock starts
# again near 0), less may be a wall clock adjusted by NTP.
CLOCK_BACK_NS = int(1e9)

#: Name of the interface's node; also the logger whose level :func:`connect`
#: lowers while the base class connects.
NODE_NAME = 'actuo_robot_interface'


def _wait_future(future, timeout, what, clock=None):
    """Block until ``future`` is done; another thread spins its node.

    Parameters
    ----------
    future : rclpy.task.Future
    timeout : float or None
        Seconds; ``None`` waits forever.
    what : str
        What is being waited for, for the error.
    clock : rclpy.clock.Clock or None
        Count ``timeout`` on this clock (the node's: simulation time under
        ``use_sim_time``, so a simulator running slower than real time gets
        the same simulated seconds as a fast one). It must keep advancing:
        standing still for :data:`CLOCK_STALL` wall seconds is an error,
        and so is going back (a restarted simulator starts its clock again:
        what was being waited for went with the old one, and would never be
        answered). ``None``: wall time.

    Returns
    -------
    object
        ``future.result()``.
    """
    done = threading.Event()
    future.add_done_callback(lambda _f: done.set())
    if clock is None:
        if not done.wait(timeout):
            raise RuntimeError('{}: no answer within {} s'.format(what, timeout))
        return future.result()
    last = clock.now().nanoseconds
    start = last
    moved_at = monotonic()
    while not done.wait(0.02):
        now = clock.now().nanoseconds
        if now < last - CLOCK_BACK_NS:
            raise RuntimeError(
                '{}: the robot clock went back from {:.1f} s to {:.1f} s '
                '(the simulator was restarted?): this was lost with it, '
                'send it again'.format(what, last * 1e-9, now * 1e-9))
        if now != last:
            last = now
            moved_at = monotonic()
            if start == 0:  # sim time read 0 until the first /clock
                start = now
        elif monotonic() - moved_at > CLOCK_STALL:
            raise RuntimeError(
                '{}: no answer, and the clock has not moved for {} s '
                '(is the simulator running and publishing /clock?)'.format(
                    what, CLOCK_STALL))
        if (timeout is not None and start != 0
                and (now - start) * 1e-9 > timeout):
            raise RuntimeError('{}: no answer within {} s (robot clock)'.format(
                what, timeout))
    return future.result()


def fetch_robot(controllers=None, manager='/controller_manager',
                timeout=10.0, load_meshes=True, model_class=None,
                description_topic='/robot_description'):
    """The model and controller joints of the running robot.

    Parameters
    ----------
    controllers : sequence of str or None
        Trajectory controllers to drive; ``None``: every loaded one.
    manager : str
        The controller manager node.
    timeout : float
        Seconds to wait for each answer.
    load_meshes : bool
        Load the meshes (needed to draw the model, not for kinematics).
    model_class : type or None
        A ``RobotModelFromURDF`` subclass to build the model as; ``None``:
        ``RobotModelFromURDF``.
    description_topic : str
        Where robot_state_publisher publishes the URDF. A namespaced bring-up
        (``/my_robot``) publishes it as ``/my_robot/robot_description``.

    Returns
    -------
    robot : skrobot.models.urdf.RobotModelFromURDF
    controller_joints : dict
        ``{controller name: [joint name, ...]}`` in the controller's order.
    """
    from skrobot.utils.urdf import no_mesh_load_mode

    node = rclpy.create_node(NODE_NAME + '_fetch')
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    got = []
    sub = node.create_subscription(String, description_topic, got.append, LATCHED)
    try:
        client = node.create_client(ListControllers, manager + '/list_controllers')
        if not client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(
                'no {}/list_controllers within {} s: is the robot\'s launch '
                'running?'.format(manager, timeout))
        future = client.call_async(ListControllers.Request())
        executor.spin_until_future_complete(future, timeout_sec=timeout)
        if not future.done():
            raise RuntimeError('{}/list_controllers did not answer'.format(manager))
        deadline = monotonic() + timeout
        while not got and monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
        if not got:
            raise RuntimeError('no {} within {} s'.format(description_topic, timeout))
    finally:
        node.destroy_subscription(sub)
        executor.shutdown()
        node.destroy_node()

    loaded = {c.name: c for c in future.result().controller}
    if controllers is None:
        controllers = sorted(name for name, c in loaded.items()
                             if c.type == TRAJECTORY_CONTROLLER)
        if not controllers:
            raise RuntimeError('no {} is loaded'.format(TRAJECTORY_CONTROLLER))
    missing = [name for name in controllers if name not in loaded]
    if missing:
        raise RuntimeError(
            'controllers not loaded: {} (loaded: {})'.format(
                ', '.join(missing), ', '.join(sorted(loaded))))
    controller_joints = {
        name: [i.split('/')[0] for i in loaded[name].required_command_interfaces]
        for name in controllers}

    handle = tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False)
    handle.write(got[0].data)
    handle.close()
    model_class = model_class or RobotModelFromURDF
    try:
        if load_meshes:
            robot = model_class(urdf_file=handle.name)
        else:
            with no_mesh_load_mode():
                robot = model_class(urdf_file=handle.name)
    finally:
        os.unlink(handle.name)
    return robot, controller_joints


class RobotInterface(ROS2RobotInterfaceBase):
    """The robot's trajectory controllers, and actuo's torque switch.

    ``controller_type`` names one controller (``'arm_controller'``,
    ``'arm_gripper_controller'``, ...); left out, a pose goes to all of them.

    Parameters
    ----------
    robot : skrobot.models.urdf.RobotModelFromURDF
    controller_joints : dict
        ``{controller name: [joint name, ...]}``, from :func:`fetch_robot`.
    servo_ns : str or None
        The actuo bus servicer, which switches the torque. ``None`` for a
        robot whose torque is not switched (the simulator, the mock backend).
    **kwargs
        Passed to :class:`ROS2RobotInterfaceBase`.
    """

    def __init__(self, robot, controller_joints, servo_ns='/actuo_bus_servicer',
                 **kwargs):
        self._controller_joints = dict(controller_joints)
        self._servo_ns = servo_ns
        kwargs.setdefault('node_name', NODE_NAME)
        # 3 s (the default) can be shorter than discovering the action servers.
        kwargs.setdefault('controller_timeout', 15)
        super(RobotInterface, self).__init__(robot, **kwargs)
        if not self.joint_action_enable:
            raise RuntimeError(
                'the trajectory controllers did not answer: check '
                '`ros2 control list_controllers`')
        self._servo_client = None
        if servo_ns is not None:
            from actuo_msgs.action import SetServo
            self._servo_client = ActionClient(
                self, SetServo, servo_ns.rstrip('/') + '/servo/set')

    def default_controller(self):
        # The base prefixes `namespace` as '{ns}/{action}', so under a
        # namespace the names must be relative: an absolute one would give
        # '/ns//name', which rclpy refuses.
        prefix = '' if self.namespace else '/'
        return [dict(
            controller_type=name,
            controller_action='{}{}/follow_joint_trajectory'.format(prefix, name),
            # No controller_state: nothing here reads it, and each
            # controller publishes it at its update rate (100 Hz), which
            # every other callback of this process then waits behind.
            controller_state=None,
            action_type=control_msgs.action.FollowJointTrajectory,
            joint_names=list(joints))
            for name, joints in self._controller_joints.items()]

    @property
    def controller_names(self):
        """The controllers this interface drives."""
        return list(self._controller_joints)

    def joints_of(self, controller):
        """The joints ``controller`` commands, in its order."""
        return list(self._controller_joints[controller])

    def controller_of(self, joint_name):
        """The controller that commands ``joint_name``."""
        for name, joints in self._controller_joints.items():
            if joint_name in joints:
                return name
        raise KeyError('no controller commands {}'.format(joint_name))

    def wait_interpolation(self, controller_type=None, timeout=None):
        """Wait until the last pose sent is reached.

        Raises when a controller refused the pose or did not reach it (a
        servo with its torque off, a servo that is not answering), instead of
        returning as if it had moved.

        Parameters
        ----------
        controller_type : str or None
            One controller; ``None`` waits for all of them.
        timeout : float or None
            Seconds; ``None`` waits as long as the motion takes.
        """
        controller_type = controller_type or self.controller_type
        actions = self.controller_table[controller_type]
        params = self.controller_param_table[controller_type]
        for action, param in zip(actions, params):
            future = getattr(action, '_current_future', None)
            if future is None:
                continue
            name = param['controller_type']
            handle = _wait_future(future, timeout, name, self.get_clock())
            action._current_future = None
            if not handle.accepted:
                raise PoseRefused(
                    '{} refused the pose: is it active? '
                    '(`ros2 control list_controllers`)'.format(name))
            result = _wait_future(handle.get_result_async(), timeout, name,
                                  self.get_clock()).result
            if result.error_code != 0:
                raise RuntimeError('{} did not reach the pose: {} {}'.format(
                    name, result.error_code, result.error_string))
        self.update_robot_state(wait_until_update=True)

    def outside_limits(self, joint_names=None, tolerance=LIMIT_TOLERANCE):
        """Joints the robot is at but the model cannot represent.

        The model clamps every angle to the URDF limits, so for these joints
        ``ri.angle_vector()`` is not where the robot is, and sending it back
        would move the joint.

        Parameters
        ----------
        joint_names : list of str or None
        tolerance : float
            [rad] How far past a limit still counts as at it. A joint that
            fell onto its stop with the torque off reads a few mrad beyond the
            URDF limit (e.g. -90.03 deg against a -90.0 deg limit), and
            holding it at the limit moves it by no more than this, well inside
            the servicer's jump threshold.

        Returns
        -------
        dict
            ``{joint name: (measured, lower, upper)}`` in radians.
        """
        msg = self._joint_state_msg
        if msg is None:
            raise RuntimeError('no joint_states yet')
        measured = dict(zip(msg.name, msg.position))
        out = {}
        for name in joint_names or measured:
            joint = getattr(self.robot, name, None)
            if joint is None or name not in measured:
                continue
            q = measured[name]
            if not (joint.min_angle - tolerance <= q
                    <= joint.max_angle + tolerance):
                out[name] = (q, joint.min_angle, joint.max_angle)
        return out

    def _servo(self, enable, joints, timeout):
        from actuo_msgs.action import SetServo
        if self._servo_client is None:
            self.get_logger().info(
                'servo_ns is None: this interface does not switch torque')
            return
        if not self._servo_client.wait_for_server(timeout_sec=timeout):
            raise RuntimeError('no {}/servo/set: is the bus node running?'.format(
                self._servo_ns))
        goal = SetServo.Goal(enable=enable, joints=list(joints or []), force=False)
        handle = _wait_future(self._servo_client.send_goal_async(goal), timeout,
                              'servo/set', self.get_clock())
        if not handle.accepted:
            raise RuntimeError('servo/set refused the request')
        result = _wait_future(handle.get_result_async(), timeout, 'servo/set',
                              self.get_clock()).result
        if not result.ok:
            raise ServoRefused(result.reason)
        self.get_logger().info(result.reason)

    def _measured_angles(self):
        """``{joint name: angle}`` from the latest joint_states, unclamped."""
        msg = self._joint_state_msg
        if msg is None:
            raise RuntimeError('no joint_states yet')
        return dict(zip(msg.name, msg.position))

    def _send_measured(self, controllers, targets, duration):
        """Send each controller its joints' measured angles, or ``targets``.

        Through the trajectory sender directly, not ``angle_vector``: that goes
        through the model, which clamps every angle to the URDF limits, so a
        joint outside them would be sent its limit instead of where it is.

        Parameters
        ----------
        controllers : list of str
        targets : dict
            ``{joint name: angle}`` to send instead of the measured angle.
        duration : float
            [s]
        """
        measured = self._measured_angles()
        for name in controllers:
            names = list(self._controller_joints[name])
            positions = np.array([[targets.get(j, measured[j]) for j in names]])
            self._send_trajectory(name, names, positions,
                                  np.zeros_like(positions), [duration], 0.0)

    def servo_on(self, joints=None, time=0.1, timeout=10.0, attempts=3,
                 limit_speed=LIMIT_RETURN_SPEED):
        """Turn the torque on, holding where the robot is now.

        The servicer refuses to turn a joint on while its controller commands
        another angle, because the joint would jump there. So each controller
        is first sent the pose the robot is in now, which moves nothing.

        A joint outside its URDF limits (moved there by hand while limp) is
        switched on where it is too, and then brought slowly to the nearest
        limit, at ``limit_speed``. The limits are the model's, a software
        range: the joint is not driven past them later, but being found
        outside is not a fault.

        A limp joint may still be moving (an arm sliding down under gravity):
        then the pose sent a moment ago is already behind it, and the switch
        refuses. The remedy is the same re-sync again, once the joints have come
        to rest (slower than :data:`STILL_SPEED`; re-syncing while they still
        fall loses the race again), so a refusal is retried up to ``attempts``
        times, each from a fresh reading; one that persists is raised as
        :class:`ServoRefused`, with the switch's reason.

        The hold itself can be rejected too: it is stamped when sent and lasts
        only ``time``, so a controller that receives it more than ``time``
        later (the first goal right after start-up, before the action is fully
        matched) finds it already over. That is retried the same way and,
        if it persists, raised as :class:`PoseRefused`.

        Parameters
        ----------
        joints : list of str or None
            ``None``: every joint of the controllers.
        time : float
            Seconds for that hold. Short: the joints are limp until the
            switch answers, and a moving one gets that much further away.
        timeout : float
            Seconds to wait for each answer.
        attempts : int
            Re-syncs to try before giving up.
        limit_speed : float
            [rad/s] How fast a joint found outside its limits is brought back.
        """
        if joints is None:
            joints = [j for js in self._controller_joints.values() for j in js]
        controllers = sorted({self.controller_of(j) for j in joints})
        for attempt in range(1, attempts + 1):
            if attempt > 1 and not self._wait_until_still(joints, STILL_TIMEOUT):
                self.get_logger().warning(
                    'servo_on: the joints are still moving after {} s; '
                    're-syncing anyway'.format(STILL_TIMEOUT))
            self.update_robot_state(wait_until_update=True)
            self._send_measured(controllers, {}, time)
            try:
                for name in controllers:
                    self.wait_interpolation(name, timeout)
                self._servo(True, joints, timeout)
                break
            except (ServoRefused, PoseRefused) as err:
                if attempt == attempts:
                    raise
                self.get_logger().warning(
                    'servo_on attempt {}/{} refused, re-syncing once the '
                    'joints are at rest: {}'.format(attempt, attempts, err))
        self._bring_into_limits(joints, limit_speed, timeout)

    def _wait_until_still(self, joints, timeout):
        """Wait until ``joints`` move slower than :data:`STILL_SPEED`.

        The speed is taken from two joint_states messages at least
        :data:`STILL_WINDOW` apart by their stamps, so it is in the robot's
        time (simulated under ``use_sim_time``).

        Returns
        -------
        bool
            ``False`` when they were still moving after ``timeout`` wall
            seconds.
        """
        def stamp(msg):
            return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        deadline = monotonic() + timeout
        self.update_robot_state(wait_until_update=True)
        before = self._joint_state_msg
        while monotonic() < deadline:
            self.update_robot_state(wait_until_update=True)
            now = self._joint_state_msg
            dt = stamp(now) - stamp(before)
            if dt < STILL_WINDOW:
                continue
            q0 = dict(zip(before.name, before.position))
            q1 = dict(zip(now.name, now.position))
            moved = [abs(q1[j] - q0[j]) for j in joints if j in q0 and j in q1]
            if not moved or max(moved) / dt <= STILL_SPEED:
                return True
            before = now
        return False

    def _bring_into_limits(self, joints, speed, timeout):
        """Move joints that are outside their URDF limits to the nearest one."""
        self.update_robot_state(wait_until_update=True)
        bad = self.outside_limits(joints)
        if not bad:
            return
        targets = {n: float(np.clip(q, lo, hi)) for n, (q, lo, hi) in bad.items()}
        distance = max(abs(bad[n][0] - targets[n]) for n in bad)
        duration = max(1.0, distance / speed)
        self.get_logger().info(
            'outside the URDF limits, bringing back over {:.1f} s: {}'.format(
                duration, ', '.join(
                    '{} {:.1f} -> {:.1f} deg'.format(
                        n, np.rad2deg(bad[n][0]), np.rad2deg(targets[n]))
                    for n in sorted(bad))))
        controllers = sorted({self.controller_of(n) for n in bad})
        self._send_measured(controllers, targets, duration)
        for name in controllers:
            self.wait_interpolation(name, timeout + duration)
        self.update_robot_state(wait_until_update=True)

    def servo_off(self, joints=None, timeout=10.0):
        """Turn the torque off. The joints go limp and fall under gravity.

        Parameters
        ----------
        joints : list of str or None
            ``None``: every joint.
        timeout : float
            Seconds to wait for the answer.
        """
        self._servo(False, joints, timeout)


def connect(servo_ns='/actuo_bus_servicer', load_meshes=True, model_class=None,
            namespace=None, interface_class=None, **kwargs):
    """Connect to the running robot.

    Parameters
    ----------
    servo_ns : str or None
        See :class:`RobotInterface`.
    load_meshes : bool
        See :func:`fetch_robot`.
    model_class : type or None
        See :func:`fetch_robot`.
    namespace : str or None
        The namespace the robot's launch put the controller manager under
        (``'/my_robot'``). The description, the controller manager,
        ``joint_states`` and the trajectory actions are all looked up in it.
    interface_class : type or None
        A :class:`RobotInterface` subclass to build ``ri`` as.
    **kwargs
        Passed to ``interface_class``.

    Returns
    -------
    robot : skrobot.models.urdf.RobotModelFromURDF
        A model of your own, set to the robot's current pose.
    ri : RobotInterface
    """
    if not rclpy.ok():
        rclpy.init()
    ns = '/' + namespace.strip('/') if namespace and namespace.strip('/') else ''
    robot, controller_joints = fetch_robot(
        manager=ns + '/controller_manager', load_meshes=load_meshes,
        model_class=model_class, description_topic=ns + '/robot_description')
    # The base logs two lines per controller while it connects; keep the
    # samples' output to what they print.
    rclpy.logging.set_logger_level(NODE_NAME, rclpy.logging.LoggingSeverity.WARN)
    interface_class = interface_class or RobotInterface
    ri = interface_class(robot, controller_joints, servo_ns=servo_ns,
                         namespace=ns or None, wait_for_joint_states=True,
                         **kwargs)
    rclpy.logging.set_logger_level(NODE_NAME, rclpy.logging.LoggingSeverity.INFO)
    executor = SingleThreadedExecutor()
    executor.add_node(ri)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()

    def _close():
        executor.shutdown()
        thread.join()
        ri.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    atexit.register(_close)

    robot.angle_vector(ri.angle_vector())
    return robot, ri
