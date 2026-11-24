#!/usr/bin/env python3
"""Hardware check: scikit-robot through actuo_ros.robot_interface on a modular robot.

Moves the servos of examples/rigs/feetech_4port (0.3 rad each, towards the
middle of its turn, and back) through ModularConnection while the robot
becomes another: gripper, then drill once id 3 is swapped for id 4 (robot and
ri are built again, with drill's model, joints and controllers), then gripper
again on a declaration (built again once the hand module is up). Run it by
hand, on the rig, with the launch on its own ROS domain and in a Python that
has scikit-robot (actuo_ros's venv)::

    export ROS_DOMAIN_ID=73
    # board: PORT 1-3 powered, PORT 4 off (the gripper configuration)
    ros2 launch actuo_ros modular.launch.py \\
        catalog:=examples/rigs/feetech_4port/modular/catalog.yaml &
    <actuo_ros venv>/bin/python ros/actuo_ros/test/hardware/check_modular_robot_interface.py

THE SERVOS MOVE. Prints PASS / FAIL per check.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
import rclpy  # noqa: E402
from actuo_msgs.srv import DeclareConfiguration  # noqa: E402
from actuo_ros.robot_interface import ModularConnection  # noqa: E402
from actuo.testing.fault_injectors import ServoFaultInjector  # noqa: E402

DELTA = 0.3
ok_all = True


def check(cond, label):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + label, flush=True)


def exercise(robot, ri, name):
    names = [j.name for j in robot.joint_list]
    print("INFO {}: joints {} controllers {}".format(name, names, ri.controller_names), flush=True)
    ri.servo_on()
    time.sleep(0.3)
    start = np.array(ri.angle_vector(), dtype=float)
    # Towards the middle of each servo's turn: an uncalibrated HLS rests at
    # +-pi, the end of the range a goal can be written in.
    target = start - DELTA * np.sign(start + 1e-9)
    ri.angle_vector(target, time=1.5)
    ri.wait_interpolation()
    time.sleep(0.3)
    reached = np.array(ri.angle_vector(), dtype=float)
    err = np.abs(reached - target)
    check(np.all(err < 0.05), "{}: moved {} by {} rad (error {})".format(
        name, names, np.round(target - start, 2).tolist(), np.round(err, 3).tolist()))
    ri.angle_vector(start, time=1.5)
    ri.wait_interpolation()
    time.sleep(0.3)
    back = np.abs(np.array(ri.angle_vector(), dtype=float) - start)
    check(np.all(back < 0.05), "{}: back to start (error {})".format(name, np.round(back, 3).tolist()))


sfi = ServoFaultInjector()
conn = ModularConnection()
try:
    robot, ri = conn.wait("gripper", timeout=90)
    exercise(robot, ri, "gripper")

    t0 = time.monotonic()
    sfi.unplug(3)
    sfi.plug(4)
    robot, ri = conn.wait("drill", timeout=30)
    print("INFO drill robot/ri ready {:.2f} s after the swap".format(time.monotonic() - t0), flush=True)
    check({j.name for j in robot.joint_list} == {"shoulder_pitch", "elbow_pitch", "drill_spin"},
          "model is the drill robot")
    exercise(robot, ri, "drill")

    sfi.plug(3)
    node = rclpy.create_node("declare_check")
    cli = node.create_client(DeclareConfiguration, "/actuo_configuration/declare")
    cli.wait_for_service(timeout_sec=10)
    f = cli.call_async(DeclareConfiguration.Request(name="gripper"))
    rclpy.spin_until_future_complete(node, f, timeout_sec=5)
    t0 = time.monotonic()
    robot, ri = conn.wait("gripper", timeout=30)
    print("INFO gripper robot/ri ready {:.2f} s after the declaration ({})".format(
        time.monotonic() - t0, ri.controller_names), flush=True)
    while "gripper_hand_gripper_controller" not in ri.controller_names and time.monotonic() - t0 < 15:
        time.sleep(0.1)
        robot, ri = conn.wait("gripper", timeout=30)
    check("gripper_hand_gripper_controller" in ri.controller_names,
          "hand module came up after id 3 answered; ri rebuilt with it ({:.2f} s)".format(
              time.monotonic() - t0))
    exercise(robot, ri, "gripper again")
    ri.servo_off()
    time.sleep(0.3)
finally:
    sfi.restore()
    sfi.close()
print("RESULT", "PASS" if ok_all else "FAIL", flush=True)
