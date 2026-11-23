"""Bring up one robot from its actuo config: bus servicer + ros2_control.

The config (see ``actuo/drivers/robot_document.py``) names the URDF and groups
the bus joints into ``modules``; everything else is generated from it by
``actuo-robot-files``: the URDF with one ``actuo_hardware/ActuoSystem``
component per module, and controller_manager's parameters (a
joint_state_broadcaster and a trajectory controller per module). So a robot
needs no hand-written ``<ros2_control>`` block or controllers yaml, and the
shm joint order cannot drift from the bus config's.

    ros2 launch actuo_ros robot.launch.py config:=<robot>.yaml
    ros2 launch actuo_ros robot.launch.py config:=<robot>.yaml \\
        port:=/dev/ttyACM0 rviz_config:=<robot>.rviz

Started here:

* the stale shm segment file is removed (it outlives every process; a
  leftover lets the servicer finish its handshake against nobody);
* ``actuo_bus_node --real``, which waits up to 90 s for ros2_control to
  create the segment;
* 3 s later ``ros2_control_node``, ``robot_state_publisher`` (with the
  URDF joints that are not on the bus at zero), a spawner per controller and,
  with ``rviz_config:=``, RViz.

Every joint comes up limp; servo it on with the servicer's ``servo_on`` action
or the RViz servo panel. ``rate`` is both the servicer's loop rate and
controller_manager's update rate (the config's ``update_rate`` by default):
on a request/response bus one control period should be one bus cycle.
"""

import os
import subprocess
import tempfile

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.actions import TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackagePrefix

HARDWARE_PLUGIN = "actuo_hardware/ActuoSystem"


def _shm_file(name):
    """The servicer's segment file, as actuo.drivers.shm names it."""
    return os.path.join("/tmp", "{}{}.shm".format(
        os.environ.get("ACTUO_SHM_PREFIX", "actuo_"), name))


def _root_link(urdf_xml):
    """The URDF's root link: the one no joint has as its child."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(urdf_xml)
    children = {j.find("child").get("link") for j in root.findall("joint")}
    roots = [link.get("name") for link in root.findall("link")
             if link.get("name") not in children]
    if len(roots) != 1:
        raise RuntimeError("the URDF has {} root links: {}".format(len(roots), roots))
    return roots[0]


def _setup(context):
    config = os.path.abspath(LaunchConfiguration("config").perform(context))
    shm_name = (LaunchConfiguration("shm_name").perform(context)
                or os.path.splitext(os.path.basename(config))[0])
    prefix = FindPackagePrefix("actuo_ros").perform(context)
    out_dir = tempfile.mkdtemp(prefix="actuo_{}_".format(shm_name))
    result = subprocess.run(
        [os.path.join(prefix, "lib", "actuo_ros", "actuo-robot-files"), config,
         "--out", out_dir, "--shm-name", shm_name,
         "--hardware-plugin", HARDWARE_PLUGIN],
        capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("actuo-robot-files failed on {}:\n{}".format(
            config, result.stderr))
    controllers = result.stdout.split()
    urdf_path = os.path.join(out_dir, "robot.urdf")
    params_path = os.path.join(out_dir, "controllers.yaml")
    with open(urdf_path) as handle:
        robot_description = handle.read()

    rate = LaunchConfiguration("rate").perform(context)
    if not rate:
        with open(params_path) as handle:
            rate = str(yaml.safe_load(handle)["controller_manager"]
                       ["ros__parameters"]["update_rate"])
    port = LaunchConfiguration("port").perform(context)
    rviz_config = LaunchConfiguration("rviz_config").perform(context)
    ghost = LaunchConfiguration("ghost").perform(context).lower() == "true"

    servicer = Node(
        package="actuo_ros", executable="actuo_bus_node", output="screen",
        arguments=["--real", "--config", config, "--shm-name", shm_name,
                   "--rate", rate, "--timeout", "90"]
        + (["--channel", port] if port else []))
    control = [
        Node(package="controller_manager", executable="ros2_control_node",
             output="screen",
             parameters=[{"robot_description": robot_description}, params_path,
                         {"update_rate": int(float(rate))}]),
        # The URDF's joints that are not on the bus (a gripper not fitted yet,
        # passive joints) are in no JointState, so robot_state_publisher would
        # publish no TF below them. joint_state_publisher takes the measured
        # ones from the broadcaster and adds the rest at their zero. It runs in
        # a sub-namespace: remapping its output to a new name in the robot's
        # own namespace would also catch its source_list subscription, which
        # resolves to the same name, and it would subscribe to itself.
        Node(package="joint_state_publisher", executable="joint_state_publisher",
             namespace="state_merge", output="screen",
             parameters=[{"source_list": ["/joint_states"], "rate": 30}],
             remappings=[("robot_description", "/robot_description")]),
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             output="screen",
             parameters=[{"robot_description": robot_description}],
             remappings=[("joint_states", "/state_merge/joint_states")]),
    ]
    if ghost:
        # Calibration ghost: the same URDF under the TF prefix ghost/, posed by
        # /ghost/joint_states (actuo_calibrate publishes it), its root pinned
        # to the robot's. Show it with a second RobotModel display whose TF
        # prefix is "ghost", transparent, over the robot.
        root = _root_link(robot_description)
        control += [
            Node(package="robot_state_publisher", executable="robot_state_publisher",
                 name="ghost_state_publisher", namespace="ghost", output="log",
                 parameters=[{"robot_description": robot_description,
                              "frame_prefix": "ghost/"}],
                 remappings=[("joint_states", "/ghost/joint_states")]),
            Node(package="tf2_ros", executable="static_transform_publisher",
                 name="ghost_root", output="log",
                 arguments=["--frame-id", root, "--child-frame-id",
                            "ghost/" + root]),
        ]
    control += [
        Node(package="controller_manager", executable="spawner", output="screen",
             arguments=[name, "--service-call-timeout", "60"])
        for name in controllers]
    if rviz_config:
        control.append(Node(package="rviz2", executable="rviz2", output="log",
                            arguments=["-d", rviz_config]))
    return [
        ExecuteProcess(cmd=["rm", "-f", _shm_file(shm_name)], output="log"),
        servicer,
        TimerAction(period=3.0, actions=control),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("config", description="robot config YAML "
                              "(bus, urdf, modules)"),
        DeclareLaunchArgument("shm_name", default_value="",
                              description="shm segment; default: the config's "
                                          "file name"),
        DeclareLaunchArgument("rate", default_value="",
                              description="servicer loop and controller "
                                          "update rate [Hz]; default: the "
                                          "config's update_rate"),
        DeclareLaunchArgument("port", default_value="",
                              description="serial device, overriding the "
                                          "config's bus.transport.channel"),
        DeclareLaunchArgument("rviz_config", default_value="",
                              description="start RViz with this config"),
        DeclareLaunchArgument("ghost", default_value="false",
                              choices=["true", "false"],
                              description="publish a calibration ghost (TF "
                                          "prefix ghost/, posed by "
                                          "/ghost/joint_states)"),
        OpaqueFunction(function=_setup),
    ])
