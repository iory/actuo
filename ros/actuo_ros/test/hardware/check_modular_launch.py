#!/usr/bin/env python3
"""Hardware check: modular.launch.py on examples/rigs/feetech_4port, end to end.

Not collected by pytest or colcon test (its name does not start with test_):
run it by hand, on the rig, while the launch runs on its own ROS domain::

    export ROS_DOMAIN_ID=73          # not the domain another robot's stack uses
    # board: PORT 1-3 powered, PORT 4 off (the gripper configuration)
    ros2 launch actuo_ros modular.launch.py \\
        catalog:=examples/rigs/feetech_4port/modular/catalog.yaml &
    python3 ros/actuo_ros/test/hardware/check_modular_launch.py

Checks, through the servicer, the configuration manager and controller_manager:
both gripper modules come up; pulling id 3 takes only the hand down and
plugging it back brings it up; swapping id 3 for id 4 switches to drill with
ids 1 and 2 keeping torque under their drill names, /robot_description becomes
the drill model and only drill controllers run; all four ids answering is a
conflict; declaring gripper switches back. Prints PASS / FAIL per check.
Imports actuo.testing from this repository (the servo-fault-injector board
switches the ports).
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
import rclpy  # noqa: E402
from rclpy.action import ActionClient  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from controller_manager_msgs.srv import ListControllers  # noqa: E402

from actuo_msgs.action import SetServo  # noqa: E402
from actuo_msgs.msg import Configuration, ServoState  # noqa: E402
from actuo_msgs.srv import DeclareConfiguration  # noqa: E402
from actuo.testing.fault_injectors import ServoFaultInjector  # noqa: E402

LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
ok_all = True


def check(cond, label):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + label, flush=True)


class Watch(Node):
    def __init__(self):
        super().__init__("modular_check")
        self.config = None
        self.config_t = None
        self.active = None
        self.active_t = None
        self.model = None
        self.model_t = None
        self.servo = None
        self.create_subscription(Configuration, "/actuo_configuration/configuration", self._c, LATCHED)
        self.create_subscription(String, "/actuo_configuration_manager/active", self._a, LATCHED)
        self.create_subscription(String, "/robot_description", self._m, LATCHED)
        self.create_subscription(ServoState, "/actuo_bus_servicer/servo/state", self._s, LATCHED)
        self.servo_cli = ActionClient(self, SetServo, "/actuo_bus_servicer/servo/set")
        self.declare_cli = self.create_client(DeclareConfiguration, "/actuo_configuration/declare")
        self.list_cli = self.create_client(ListControllers, "/controller_manager/list_controllers")

    def _c(self, m):
        self.config, self.config_t = m, time.monotonic()

    def _a(self, m):
        self.active, self.active_t = m.data, time.monotonic()

    def _m(self, m):
        self.model, self.model_t = m.data, time.monotonic()

    def _s(self, m):
        self.servo = m

    def spin_until(self, cond, timeout):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            if cond():
                return time.monotonic() - t0
        return None

    def call(self, client, req, timeout=10.0):
        client.wait_for_service(timeout_sec=timeout)
        f = client.call_async(req)
        self.spin_until(f.done, timeout)
        return f.result()

    def set_servo(self, enable, joints):
        self.servo_cli.wait_for_server(timeout_sec=10)
        g = SetServo.Goal(enable=enable, joints=joints, force=False)
        f = self.servo_cli.send_goal_async(g)
        self.spin_until(f.done, 10)
        h = f.result()
        if not h.accepted:
            return False, "rejected"
        r = h.get_result_async()
        self.spin_until(r.done, 10)
        return r.result().result.ok, r.result().result.reason

    def active_controllers(self):
        res = self.call(self.list_cli, ListControllers.Request())
        return sorted(c.name for c in res.controller if c.state == "active")

    def enabled(self):
        rclpy.spin_once(self, timeout_sec=0.2)
        return sorted(j.name for j in (self.servo.joints if self.servo else []) if j.enabled)


rclpy.init()
w = Watch()
sfi = ServoFaultInjector()
try:
    t = w.spin_until(lambda: w.active == "gripper", 60)
    check(t is not None, "launch up: ros2_control runs gripper")
    up = lambda want: set(want) <= set(w.active_controllers())  # noqa: E731
    GRIPPER = ["gripper_arm_controller", "gripper_hand_gripper_controller"]
    t = w.spin_until(lambda: up(GRIPPER), 30)
    check(t is not None, "both gripper modules up")
    print("INFO active controllers:", w.active_controllers())

    sfi.unplug(3)
    t = w.spin_until(lambda: "gripper_hand_gripper_controller" not in w.active_controllers(), 10)
    check(t is not None and "gripper_arm_controller" in w.active_controllers() and w.active == "gripper",
          "id 3 pulled: only the hand module went down ({})".format(w.active_controllers()))
    sfi.plug(3)
    t = w.spin_until(lambda: up(GRIPPER), 15)
    check(t is not None, "id 3 back: hand module up again ({})".format(
        "%.2f s" % t if t is not None else "timeout"))
    check("arm_shoulder" in (w.model or ""), "robot_description is the gripper model")
    w.spin_until(lambda: w.servo is not None and any(j.present for j in w.servo.joints), 10)
    print("INFO servo on:", w.set_servo(True, ["arm_shoulder", "arm_elbow"]))
    print("INFO enabled:", w.enabled())

    model_t_before = w.model_t
    t0 = time.monotonic()
    sfi.unplug(3)
    sfi.plug(4)
    t_cfg = w.spin_until(lambda: w.config is not None and w.config.name == "drill", 10)
    t_act = w.spin_until(lambda: w.active == "drill", 10)
    check(t_cfg is not None and t_act is not None,
          "ports 3->4: servicer switched to drill and ros2_control follows")
    if t_cfg is not None:
        print("INFO servicer switch_seconds={:.3f}; detection+switch {:.2f}s after the ports; "
              "ros2_control active {:.2f}s after the servicer".format(
                  w.config.switch_seconds, w.config_t - t0, (w.active_t or t0) - w.config_t))
    check(sorted(w.config.held_joints) == ["elbow_pitch", "shoulder_pitch"],
          "held across the switch: {}".format(list(w.config.held_joints)))
    w.spin_until(lambda: w.model_t != model_t_before, 5)
    check("shoulder_pitch" in (w.model or "") and "arm_shoulder" not in (w.model or ""),
          "/robot_description now the drill model (RViz follows it)")
    DRILL = ["drill_arm_controller", "drill_tool_controller"]
    t_mod = w.spin_until(lambda: up(DRILL), 10)
    ac = w.active_controllers()
    print("INFO active controllers:", ac, "-- drill modules up {} after ros2_control switched".format(
        "%.2f s" % (time.monotonic() - w.active_t) if t_mod is not None else "never"))
    check(t_mod is not None and all(c.startswith("drill_") for c in ac),
          "only drill controllers active, both drill modules up")
    check(w.enabled() == ["elbow_pitch", "shoulder_pitch"], "servo/state: drill arm joints enabled")

    sfi.plug(3)
    w.spin_until(lambda: False, 2.5)
    check(w.config.name == "drill" and any("only one" in p for p in w.config.problems),
          "all ids answer: conflict reported, stays drill")
    t0 = time.monotonic()
    res = w.call(w.declare_cli, DeclareConfiguration.Request(name="gripper"))
    t_act = w.spin_until(lambda: w.active == "gripper", 10)
    t_up = w.spin_until(lambda: up(GRIPPER), 10)
    check(res.success and t_act is not None and t_up is not None,
          "declare gripper -> gripper modules up {:.2f}s after the call".format(time.monotonic() - t0))
    print("INFO servicer switch_seconds={:.3f}".format(w.config.switch_seconds))
    check(w.enabled() == ["arm_elbow", "arm_shoulder"], "torque carried back by id")
    check("arm_shoulder" in (w.model or ""), "/robot_description back to gripper")
    w.set_servo(False, [])
finally:
    sfi.restore()
    sfi.close()
    w.destroy_node()
    rclpy.shutdown()
print("RESULT", "PASS" if ok_all else "FAIL")
