#!/usr/bin/env python3
"""Teach and playback on an actuo robot.

Record: the controllers' joints from ``/joint_states`` while someone moves the
arm (typically in free drive), saved as ``<directory>/<name>.yaml``.
Play: take the recorded joints out of free drive (re-syncing their
controllers), move to where the recording starts at ``approach_speed``, then
send the recording to each trajectory controller as one goal, time-scaled by
the requested speed, with velocities so it does not stop at every sample.

    ros2 run actuo_ros actuo_teach_node [--ros-args -p directory:=...]

Interfaces (relative to the node, ``/actuo_teach/...``):

* ``~/start_recording``   std_srvs/Trigger
* ``~/save``              actuo_msgs/TeachSave: stop recording and save it
* ``~/play``              actuo_msgs/TeachPlay: returns once playback started
* ``~/stop``              std_srvs/Trigger: stop playback, or drop a recording
* ``~/list``              actuo_msgs/TeachList
* ``~/state``             actuo_msgs/TeachState, at 10 Hz

Ported from attachable_arm's feetech_teach_node.
"""

import os
import threading
import time

import numpy as np
import rclpy
from actuo_msgs.msg import ServoState, TeachState
from actuo_msgs.srv import TeachList, TeachPlay, TeachSave
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

from actuo import teach
from actuo_ros.robot_interface import connect


class Stopped(Exception):
    pass


class TeachNode(Node):

    def __init__(self, ri):
        super().__init__('actuo_teach')
        self.declare_parameter('directory', '~/.ros/actuo_teach')
        self.declare_parameter('record_rate', 25.0)
        # [rad/s] fastest joint speed of the move to where a recording starts.
        self.declare_parameter('approach_speed', 0.4)
        self.ri = ri
        self.directory = os.path.expanduser(self.get_parameter('directory').value)
        self.record_rate = float(self.get_parameter('record_rate').value)
        self.approach_speed = float(self.get_parameter('approach_speed').value)
        self.joints = [j for c in ri.controller_names for j in ri.joints_of(c)]

        group = ReentrantCallbackGroup()
        self.lock = threading.Lock()
        self.servo_state = None
        self.recorder = None
        self.mode = 'idle'
        self.name = ''
        self.started = 0.0
        self.duration = 0.0
        self.message = ''
        self.playing = set()
        self.stop_requested = threading.Event()

        self.create_subscription(
            ServoState, '{}/servo/state'.format(ri._servo_ns.rstrip('/')),
            self._on_servo_state,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL),
            callback_group=group)
        self.create_subscription(JointState, '/joint_states', self._on_joint_state,
                                 50, callback_group=group)
        self.state_pub = self.create_publisher(TeachState, '~/state', 10)
        self.create_service(Trigger, '~/start_recording', self.on_start_recording,
                            callback_group=group)
        self.create_service(TeachSave, '~/save', self.on_save, callback_group=group)
        self.create_service(TeachPlay, '~/play', self.on_play, callback_group=group)
        self.create_service(Trigger, '~/stop', self.on_stop, callback_group=group)
        self.create_service(TeachList, '~/list', self.on_list, callback_group=group)
        self.create_timer(0.1, self._publish_state, callback_group=group)
        self.get_logger().info('joints {}; recordings in {}'.format(
            self.joints, self.directory))

    # -- inputs -------------------------------------------------------------

    def _on_servo_state(self, msg):
        with self.lock:
            self.servo_state = {j.name: j for j in msg.joints}

    def _on_joint_state(self, msg):
        with self.lock:
            if self.recorder is not None:
                self.recorder.add(time.monotonic(), msg.name, msg.position)

    def _publish_state(self):
        with self.lock:
            busy = self.mode != 'idle'
            msg = TeachState(state=self.mode, name=self.name, message=self.message,
                             elapsed=time.monotonic() - self.started if busy else 0.0,
                             duration=self.duration)
        self.state_pub.publish(msg)

    def _set(self, mode, message=None, **fields):
        with self.lock:
            self.mode = mode
            if message is not None:
                self.message = message
            for key, value in fields.items():
                setattr(self, key, value)
        if message:
            self.get_logger().info(message)

    # -- recording ----------------------------------------------------------

    def on_start_recording(self, request, response):
        with self.lock:
            if self.mode != 'idle':
                response.success, response.message = False, 'busy: {}'.format(self.mode)
                return response
            self.recorder = teach.Recorder(self.joints, self.record_rate)
            self.mode, self.name, self.started = 'recording', '', time.monotonic()
            self.duration = 0.0
            self.message = 'recording {}'.format(self.joints)
        response.success, response.message = True, self.message
        self.get_logger().info(self.message)
        return response

    def on_save(self, request, response):
        try:
            teach.check_name(request.name)
        except ValueError as e:
            # Keep recording: a bad name can be fixed and saved again.
            response.success, response.message = False, str(e)
            return response
        with self.lock:
            recorder = self.recorder
            if self.mode != 'recording' or recorder is None:
                response.success, response.message = False, 'not recording'
                return response
            self.recorder = None
        try:
            path = teach.save(self.directory, recorder.to_dict(request.name))
        except (ValueError, OSError) as e:
            self._set('idle', 'recording lost: {}'.format(e))
            response.success, response.message = False, self.message
            return response
        message = 'saved {} ({:.1f} s, {} points) to {}'.format(
            request.name, recorder.duration, len(recorder.times), path)
        self._set('idle', message)
        response.success, response.message = True, message
        return response

    def on_list(self, request, response):
        for name, duration in teach.listing(self.directory):
            response.names.append(name)
            response.durations.append(duration)
        return response

    def on_stop(self, request, response):
        with self.lock:
            mode = self.mode
            if mode == 'recording':
                self.recorder = None
            playing = set(self.playing)
        if mode == 'recording':
            self._set('idle', 'recording dropped')
        elif mode != 'idle':
            self.stop_requested.set()
            for controller in playing:
                self.ri._cancel(controller)
        response.success = True
        response.message = 'stopped {}'.format(mode)
        return response

    # -- playback -----------------------------------------------------------

    def on_play(self, request, response):
        speed = request.speed or 1.0
        try:
            joints, times, positions = teach.load(self.directory, request.name)
            play_times = teach.playback_times(times, speed)
        except (ValueError, KeyError) as e:
            response.success, response.message = False, str(e)
            return response
        with self.lock:
            if self.mode != 'idle':
                response.success, response.message = False, 'busy: {}'.format(self.mode)
                return response
            servos = self.servo_state or {}
            missing = [j for j in joints if j not in self.joints]
            off = [j for j in joints if j in servos and not servos[j].enabled]
            if missing or off or not servos:
                response.success = False
                response.message = (
                    'joints {} have no trajectory controller'.format(missing) if missing
                    else 'torque is off for {}'.format(off) if off
                    else 'no servo state from the servicer yet')
                return response
            free = [j for j in joints if servos[j].free_drive]
            self.mode, self.name, self.started = 'moving_to_start', request.name, \
                time.monotonic()
            self.duration = float(play_times[-1])
            self.message = ''
        self.stop_requested.clear()
        threading.Thread(target=self._play, daemon=True,
                         args=(request.name, joints, play_times, positions, free,
                               speed)).start()
        response.success = True
        response.message = 'playing {} at {:g}x ({:.1f} s)'.format(
            request.name, speed, play_times[-1])
        return response

    def _play(self, name, joints, play_times, positions, free, speed):
        ri = self.ri
        try:
            if free:
                # Out of free drive, controllers re-synced to where it is.
                ri.free_drive(free, enable=False)
            ri.update_robot_state(wait_until_update=True)
            current = ri._measured_angles()
            approach = teach.approach_duration([current[j] for j in joints],
                                               positions[0], self.approach_speed)
            controllers = sorted({ri.controller_of(j) for j in joints})
            self._check_stop()
            ri._send_measured(controllers, dict(zip(joints, positions[0])), approach)
            self._wait(controllers, approach)
            self._set('playing', started=time.monotonic())
            for controller in controllers:
                names = ri.joints_of(controller)
                # A joint of the controller that was not recorded stays put.
                rows = np.array([[row[joints.index(n)] if n in joints else current[n]
                                  for n in names] for row in positions])
                ri._send_trajectory(controller, names, rows,
                                    teach.playback_velocities(play_times, rows),
                                    list(play_times), 0.0)
            self._wait(controllers, float(play_times[-1]))
            self._set('idle', 'played {} at {:g}x'.format(name, speed), name='')
        except Stopped:
            self._set('idle', 'stopped {}'.format(name), name='')
        except Exception as e:                                    # noqa: BLE001
            if self.stop_requested.is_set():
                self._set('idle', 'stopped {}'.format(name), name='')
            else:
                self._set('idle', 'playing {} failed: {}'.format(name, e), name='')

    def _check_stop(self):
        if self.stop_requested.is_set():
            raise Stopped()

    def _wait(self, controllers, duration):
        with self.lock:
            self.playing = set(controllers)
        try:
            for controller in controllers:
                self.ri.wait_interpolation(controller, duration + 10.0)
        finally:
            with self.lock:
                self.playing = set()
        self._check_stop()


def main():
    rclpy.init()
    _robot, ri = connect(load_meshes=False)
    node = TeachNode(ri)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
