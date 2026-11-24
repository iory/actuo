#!/usr/bin/env python3
"""Follow a modular robot's configuration with ros2_control and the robot model.

The bus servicer (``actuo_bus_node --real --catalog ...``) decides which
configuration runs and publishes it on ``<servicer>/configuration``. One
controller_manager holds every configuration's components and controllers
(:func:`actuo.modular.ros2_control_files`); this node makes the running one the
active one:

1. when the configuration changes, every other configuration's controllers are
   deactivated, then its components (their servicer segment is gone already),
   and robot_state_publisher gets the new configuration's model, so RViz shows
   the robot that is there;
2. each module (component) of the running configuration is brought up on its
   own as soon as every one of its servos answers (the servicer's
   ``servo/state``): the component is activated -- it adopts the pose the
   servos are measured in, so nothing moves -- then its controllers, its
   joint_state_broadcaster first. A module plugged in later comes up then; one
   whose servo stops answering is taken down by controller_manager (the
   component deactivates itself) and comes up again here when it is back.

Reads the manifest ``ros2_control_files`` wrote. ``~/active`` (latched String)
says which configuration ros2_control runs. With ``superset_topic`` it also
publishes, latched, the description with every configuration's components,
for controller_manager (which takes its description from a topic, once).

    ros2 run actuo_ros actuo_configuration_manager --ros-args \\
        -p manifest:=/tmp/modular/manifest.yaml
"""

from __future__ import annotations

import threading
import time

import rclpy
import yaml
from controller_manager_msgs.msg import ControllerManagerActivity
from controller_manager_msgs.srv import (
    ListHardwareComponents, SetHardwareComponentState, SwitchController)
from lifecycle_msgs.msg import State
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from actuo_msgs.msg import Configuration, ServoState

LATCHED = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                     reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
#: Seconds a controller_manager or robot_state_publisher call may take.
CALL_TIMEOUT = 10.0
#: Seconds between two looks at what should be up.
PERIOD = 0.05
#: Seconds a module's controllers are not asked for again, while the activity
#: topic catches up with the last request.
SETTLE = 0.5


class ConfigurationManager(Node):
    def __init__(self):
        super().__init__("actuo_configuration_manager")
        self.declare_parameter("manifest", "")
        self.declare_parameter("controller_manager", "/controller_manager")
        self.declare_parameter("robot_state_publisher", "/robot_state_publisher")
        self.declare_parameter("configuration_topic", "/actuo_configuration/configuration")
        self.declare_parameter("servo_state_topic", "/actuo_bus_servicer/servo/state")
        self.declare_parameter("superset_topic", "")
        path = self.get_parameter("manifest").value
        if not path:
            raise SystemExit("actuo_configuration_manager: parameter manifest is required")
        with open(path) as f:
            self.manifest = yaml.safe_load(f)
        self.configurations = self.manifest["configurations"]
        cm = self.get_parameter("controller_manager").value
        rsp = self.get_parameter("robot_state_publisher").value
        group = ReentrantCallbackGroup()
        self.switch_cli = self.create_client(SwitchController, cm + "/switch_controller",
                                             callback_group=group)
        self.hw_cli = self.create_client(SetHardwareComponentState,
                                         cm + "/set_hardware_component_state",
                                         callback_group=group)
        self.list_hw_cli = self.create_client(ListHardwareComponents,
                                              cm + "/list_hardware_components",
                                              callback_group=group)
        self.rsp_cli = self.create_client(SetParameters, rsp + "/set_parameters",
                                          callback_group=group)
        self.active_pub = self.create_publisher(String, "~/active", LATCHED)
        superset_topic = self.get_parameter("superset_topic").value
        if superset_topic:
            with open(self.manifest["urdf"]) as f:
                superset = f.read()
            self.superset_pub = self.create_publisher(String, superset_topic, LATCHED)
            self.superset_pub.publish(String(data=superset))
        self.models = {}
        for name, spec in self.configurations.items():
            with open(spec["urdf"]) as f:
                self.models[name] = f.read()
        self._lock = threading.Lock()
        self.wanted = None          # configuration the servicer runs
        self.present = set()        # joints whose servos answer (servicer's view)
        self.active = None          # configuration ros2_control runs
        # controller_manager's own account of every controller and component,
        # kept from its activity topic (published on every change).
        self.controller_state = {}
        self.component_state = None
        self._asked = {}            # component -> when its controllers were last requested
        self.create_subscription(Configuration,
                                 self.get_parameter("configuration_topic").value,
                                 self._on_configuration, LATCHED, callback_group=group)
        self.create_subscription(ServoState, self.get_parameter("servo_state_topic").value,
                                 self._on_servo_state, LATCHED, callback_group=group)
        self.create_subscription(ControllerManagerActivity, cm + "/activity",
                                 self._on_activity, LATCHED, callback_group=group)
        threading.Thread(target=self._work, daemon=True).start()

    def _on_configuration(self, msg):
        if msg.name not in self.configurations:
            self.get_logger().error("unknown configuration {!r}".format(msg.name))
            return
        with self._lock:
            if msg.name != self.wanted:
                self.wanted, self.wanted_at = msg.name, time.monotonic()

    def _on_activity(self, msg):
        with self._lock:
            self.controller_state = {c.name: c.state.id for c in msg.controllers}
            self.component_state = {c.name: c.state.id for c in msg.hardware_components}

    def _on_servo_state(self, msg):
        with self._lock:
            self.present = {j.name for j in msg.joints if j.present}

    # -- calls ----------------------------------------------------------------

    def _call(self, client, request, what):
        if not client.wait_for_service(timeout_sec=CALL_TIMEOUT):
            raise RuntimeError("{}: {} not available".format(what, client.srv_name))
        future = client.call_async(request)
        deadline = time.monotonic() + CALL_TIMEOUT
        while not future.done():
            if time.monotonic() > deadline:
                raise RuntimeError("{}: no answer within {} s".format(what, CALL_TIMEOUT))
            time.sleep(0.002)
        return future.result()

    def _switch_controllers(self, activate, deactivate):
        request = SwitchController.Request()
        request.activate_controllers = list(activate)
        request.deactivate_controllers = list(deactivate)
        request.strictness = SwitchController.Request.BEST_EFFORT
        request.timeout.sec = int(CALL_TIMEOUT)
        return self._call(self.switch_cli, request, "switch_controller").ok

    def _component(self, name, state_id, label):
        request = SetHardwareComponentState.Request()
        request.name = name
        request.target_state = State(id=state_id, label=label)
        return self._call(self.hw_cli, request, "set_hardware_component_state " + name).ok

    def _component_states(self):
        result = self._call(self.list_hw_cli, ListHardwareComponents.Request(),
                            "list_hardware_components")
        return {c.name: c.state.id for c in result.component}

    def _states(self):
        """(component states, active controllers), from the activity topic; asked
        for once if no activity has been heard yet."""
        with self._lock:
            components = self.component_state
            controllers = dict(self.controller_state)
        if components is None:
            components = self._component_states()
        running = {n for n, s in controllers.items() if s == State.PRIMARY_STATE_ACTIVE}
        return components, running

    def _set_model(self, name):
        request = SetParameters.Request()
        request.parameters = [Parameter(
            name="robot_description",
            value=ParameterValue(type=ParameterType.PARAMETER_STRING,
                                 string_value=self.models[name]))]
        result = self._call(self.rsp_cli, request, "robot_state_publisher")
        if not all(r.successful for r in result.results):
            raise RuntimeError("robot_state_publisher refused the model of {}".format(name))

    # -- reconciling ----------------------------------------------------------

    def _work(self):
        while rclpy.ok():
            try:
                self._reconcile()
            except Exception as error:               # noqa: BLE001 -- reported, tried again
                self.get_logger().error(str(error), throttle_duration_sec=5.0)
            time.sleep(PERIOD)

    def _reconcile(self):
        with self._lock:
            wanted, present = self.wanted, set(self.present)
        if wanted is None:
            return
        if wanted != self.active:
            self._leave_all_but(wanted)
            self._set_model(wanted)
            self.active = wanted
            self.active_pub.publish(String(data=wanted))
            self.get_logger().info("ros2_control runs {} ({:.3f} s after the servicer said so)"
                                   .format(wanted, time.monotonic() - self.wanted_at))
        states, running = self._states()
        for component, module in self.configurations[wanted]["modules"].items():
            if states.get(component) != State.PRIMARY_STATE_ACTIVE:
                if not set(module["joints"]) <= present:
                    continue                         # comes up when its servos answer
                if not self._component(component, State.PRIMARY_STATE_ACTIVE, "active"):
                    continue
            # Up means its controllers too: a switch asked for before they were
            # all loaded (start-up) activates some of them, or none.
            missing = [c for c in module["controllers"] if c not in running]
            if not missing or time.monotonic() - self._asked.get(component, 0.0) < SETTLE:
                continue
            self._asked[component] = time.monotonic()
            if self._switch_controllers(missing, []):
                self.get_logger().info("{} up: {}".format(component, ", ".join(missing)))

    def _leave_all_but(self, keep):
        others = [n for n in self.configurations if n != keep]
        controllers = [c for n in others for c in self.configurations[n]["controllers"]]
        self._switch_controllers([], controllers)
        states, _ = self._states()
        for n in others:
            for component in self.configurations[n]["components"]:
                if states.get(component) == State.PRIMARY_STATE_ACTIVE:
                    self._component(component, State.PRIMARY_STATE_INACTIVE, "inactive")


def main():
    rclpy.init()
    node = ConfigurationManager()
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


if __name__ == "__main__":
    main()
