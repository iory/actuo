#!/usr/bin/env python3
"""Calibrate a robot's joints against a ghost: line each joint up, confirm.

Joint by joint from the root, the joint is put in free drive and the ghost (the
URDF drawn at the target angles; launch the robot with ``ghost:=true`` and add
a RobotModel display with TF prefix ``ghost``) shows where it should be. Move
the joint by hand onto the ghost and press Enter: its present pose is declared
to be the target angle, in the actuator itself (``<servicer>/set_origin``), and
the joint holds there.

    ros2 run actuo_ros actuo_calibrate                       # every joint, at 0
    ros2 run actuo_ros actuo_calibrate --joints joint1 joint2 --angles 0 90

At each joint: Enter calibrates, ``s`` skips (the joint holds where it is),
``b`` goes back one joint, ``q`` quits.

    ros2 run actuo_ros actuo_calibrate --joints joint0 --angles 15 --now

declares the pose joint0 is in right now to be 15 deg, with no free drive and
no prompt.

Ported from the calibration mode of attachable_arm's feetech servo panel.
"""

import argparse
import math
import sys
import threading
import time

import rclpy
from sensor_msgs.msg import JointState

from actuo_ros.robot_interface import ServoRefused, connect


def _parse(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--joints", nargs="+", default=None,
                   help="joints to calibrate, root first (default: every "
                        "joint of the trajectory controllers)")
    p.add_argument("--angles", nargs="+", type=float, default=None,
                   help="target angle of each joint [deg] (default: 0)")
    p.add_argument("--servo-ns", default="/actuo_bus_servicer")
    p.add_argument("--now", action="store_true",
                   help="declare the present poses at once: no free drive, "
                        "no prompt")
    args = p.parse_args(argv)
    if args.angles is not None and args.joints is None:
        p.error("--angles needs --joints")
    if args.angles is not None and len(args.angles) != len(args.joints):
        p.error("--angles needs one angle per joint")
    return args


class Ghost:
    """Publishes /ghost/joint_states: the targets over the measured pose."""

    def __init__(self, ri):
        self.ri = ri
        self.targets = {}
        self.node = rclpy.create_node("actuo_calibrate_ghost")
        self.pub = self.node.create_publisher(JointState, "/ghost/joint_states", 10)
        self.names = [j.name for j in ri.robot.joint_list]
        self._stop = False
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self._stop:
            measured = self.ri._measured_angles()
            msg = JointState()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.name = list(self.names)
            msg.position = [float(self.targets.get(n, measured.get(n, 0.0)))
                            for n in self.names]
            self.pub.publish(msg)
            time.sleep(0.1)

    def stop(self):
        self._stop = True


def main(argv=None):
    args = _parse(sys.argv[1:] if argv is None else argv)
    rclpy.init()
    _robot, ri = connect(load_meshes=False, servo_ns=args.servo_ns)
    joints = args.joints or [j for c in ri.controller_names for j in ri.joints_of(c)]
    angles = [math.radians(a) for a in (args.angles or [0.0] * len(joints))]
    targets = dict(zip(joints, angles))

    if args.now:
        for joint in joints:
            try:
                print(ri.set_origin(joint, targets[joint]))
            except ServoRefused as err:
                print("{}: REFUSED: {}".format(joint, err))
                return 1
        return 0

    ghost = Ghost(ri)
    ghost.targets = dict(targets)
    ri.servo_on()
    step = 0
    try:
        while 0 <= step < len(joints):
            joint = joints[step]
            ri.free_drive([joint], enable=True)
            answer = input(
                "\n[{}/{}] {} is in free drive. Move it by hand onto the ghost at "
                "{:+.1f} deg.\n  Enter: calibrate   s: skip   b: back   q: quit > "
                .format(step + 1, len(joints), joint,
                        math.degrees(targets[joint]))).strip().lower()
            if answer == "":
                try:
                    print("  " + ri.set_origin(joint, targets[joint]))
                except ServoRefused as err:
                    print("  REFUSED: {}".format(err))
                    ri.free_drive([joint], enable=False)
                    continue
                step += 1
            elif answer == "s":
                ri.free_drive([joint], enable=False)
                step += 1
            elif answer == "b":
                ri.free_drive([joint], enable=False)
                step = max(0, step - 1)
            elif answer == "q":
                ri.free_drive([joint], enable=False)
                break
        print("\ncalibration {}; the joints hold torque.".format(
            "finished" if step >= len(joints) else "stopped"))
    finally:
        ghost.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
