#!/usr/bin/env python3
"""Run MoveIt for whichever configuration of a modular robot is running.

move_group reads its robot model and semantic description once, at start, so
it cannot follow a modular robot that becomes another robot. This node keeps
one move_group running for the configuration ros2_control runs (the
configuration manager's ``~/active``): when that changes, the running
move_group is stopped and one is started with the new configuration's
parameters (:func:`actuo.modular.moveit_files`: model, SRDF with a group per
module, KDL, the configuration's trajectory controllers), plus the planning
pipelines from MoveIt's own defaults. ``~/ready`` (latched String) names the
configuration once its move_group answers planning requests; until then
nothing can plan.

    ros2 run actuo_ros actuo_moveit_follower --ros-args \\
        -p manifest:=/tmp/modular/manifest.yaml
"""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import tempfile
import threading
import time

import rclpy
import yaml
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from moveit_msgs.srv import GetMotionPlan
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

LATCHED = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                     reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
#: Seconds a stopping move_group gets before it is killed.
STOP_TIMEOUT = 5.0
#: Seconds a starting move_group gets to answer planning requests.
READY_TIMEOUT = 60.0


def planning_pipelines(pipeline: str) -> dict:
    """MoveIt's default parameters for one planning pipeline."""
    defaults = os.path.join(get_package_share_directory("moveit_configs_utils"),
                            "default_configs")
    with open(os.path.join(defaults, pipeline + "_planning.yaml")) as f:
        params = yaml.safe_load(f)
    if pipeline == "ompl" and "planner_configs" not in params:
        with open(os.path.join(defaults, "ompl_defaults.yaml")) as f:
            params.update(yaml.safe_load(f))
    return {"planning_pipelines": [pipeline], "default_planning_pipeline": pipeline,
            pipeline: params}


class MoveItFollower(Node):
    def __init__(self):
        super().__init__("actuo_moveit_follower")
        self.declare_parameter("manifest", "")
        self.declare_parameter("active_topic", "/actuo_configuration_manager/active")
        self.declare_parameter("pipeline", "ompl")
        path = self.get_parameter("manifest").value
        if not path:
            raise SystemExit("actuo_moveit_follower: parameter manifest is required")
        with open(path) as f:
            manifest = yaml.safe_load(f)
        pipelines = planning_pipelines(self.get_parameter("pipeline").value)
        out = tempfile.mkdtemp(prefix="actuo_move_group_")
        # Every configuration's parameter file is ready before the first switch.
        self.params = {}
        for name, spec in manifest["configurations"].items():
            if "moveit" not in spec:
                raise SystemExit("actuo_moveit_follower: {} has no move_group parameters "
                                 "(actuo-modular-files --moveit)".format(name))
            with open(spec["moveit"]) as f:
                params = yaml.safe_load(f)
            params["move_group"]["ros__parameters"].update(pipelines)
            self.params[name] = os.path.join(out, name + ".yaml")
            with open(self.params[name], "w") as f:
                yaml.safe_dump(params, f, sort_keys=False)
        self.executable = os.path.join(get_package_prefix("moveit_ros_move_group"), "lib",
                                       "moveit_ros_move_group", "move_group")
        self.ready_pub = self.create_publisher(String, "~/ready", LATCHED)
        self.plan_cli = self.create_client(GetMotionPlan, "/plan_kinematic_path")
        self.process = None
        self.running = None
        self.pending: "queue.Queue[str]" = queue.Queue()
        self.create_subscription(String, self.get_parameter("active_topic").value,
                                 lambda msg: self.pending.put(msg.data), LATCHED)
        threading.Thread(target=self._work, daemon=True).start()

    def _work(self):
        while rclpy.ok():
            name = self.pending.get()
            while not self.pending.empty():          # only the latest matters
                name = self.pending.get_nowait()
            if name == self.running or name not in self.params:
                continue
            t0 = time.monotonic()
            self.stop()
            stopped = time.monotonic() - t0
            self.process = subprocess.Popen(
                [self.executable, "--ros-args", "--params-file", self.params[name]],
                start_new_session=True)
            if not self.plan_cli.wait_for_service(timeout_sec=READY_TIMEOUT):
                self.get_logger().error("move_group for {} did not come up within {} s"
                                        .format(name, READY_TIMEOUT))
                continue
            self.running = name
            self.ready_pub.publish(String(data=name))
            self.get_logger().info("MoveIt plans for {} ({:.2f} s: {:.2f} s to stop the last "
                                   "move_group, {:.2f} s to start this one)".format(
                                       name, time.monotonic() - t0, stopped,
                                       time.monotonic() - t0 - stopped))

    def stop(self):
        """Stop the running move_group, if any.

        Also at shutdown, after rclpy's signal handler has shut the context
        down: publishing then raises, so it is only done while the context is
        up, and never before the move_group is stopped (a publish that raised
        first left move_group running after the launch had ended).
        """
        if self.process is None:
            return
        self.running = None
        if self.context.ok():
            self.ready_pub.publish(String(data=""))
        try:
            os.killpg(self.process.pid, signal.SIGINT)
            self.process.wait(timeout=STOP_TIMEOUT)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()
        except ProcessLookupError:
            pass
        self.process = None
        # Its services go with it; wait until the name is free for the next one.
        deadline = time.monotonic() + STOP_TIMEOUT
        while (self.context.ok() and self.plan_cli.service_is_ready()
               and time.monotonic() < deadline):
            time.sleep(0.02)


def main():
    rclpy.init()
    node = MoveItFollower()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
