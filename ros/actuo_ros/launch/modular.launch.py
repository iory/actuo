"""Bring up a modular robot: every configuration in one ros2_control stack.

A catalog (``actuo.modular``) lists the robot's configurations, each an
ordinary robot config with ``urdf`` and ``modules``. ``actuo-modular-files``
writes one controller_manager description holding all of them side by side;
the bus servicer runs whichever configuration is current and says so on
``/actuo_configuration/configuration``; ``actuo_configuration_manager``
follows it, activating that configuration's components and controllers and
handing its model to robot_state_publisher (and so to RViz).

    ros2 launch actuo_ros modular.launch.py catalog:=<catalog>.yaml
    ros2 launch actuo_ros modular.launch.py catalog:=<catalog>.yaml \\
        configuration:=drill rviz_config:=<robot>.rviz

Switch by plugging another configuration's servos in, or declare one::

    ros2 service call /actuo_configuration/declare \\
        actuo_msgs/srv/DeclareConfiguration "{name: drill}"

Every joint comes up limp, as with ``robot.launch.py``; servos that held torque
keep it across a switch.

With ``moveit:=true``, ``actuo_moveit_follower`` keeps a move_group running for
the configuration that runs, restarting it with that configuration's model and
SRDF (a planning group per module) on every switch; ``/actuo_moveit_follower/ready``
says when it plans. Needs MoveIt 2 (moveit_ros_move_group, moveit_configs_utils).
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
#: Where controller_manager takes its description from: every configuration's
#: components (the configuration manager publishes it), while /robot_description
#: carries the model of the configuration that runs, for RViz.
SUPERSET_TOPIC = "/actuo_modular/robot_description"


def _shm_file(name):
    """A servicer segment's file, as actuo.drivers.shm names it."""
    return os.path.join("/tmp", "{}{}.shm".format(
        os.environ.get("ACTUO_SHM_PREFIX", "actuo_"), name))


def _setup(context):
    catalog = os.path.abspath(LaunchConfiguration("catalog").perform(context))
    shm_name = (LaunchConfiguration("shm_name").perform(context)
                or os.path.splitext(os.path.basename(catalog))[0])
    prefix = FindPackagePrefix("actuo_ros").perform(context)
    out_dir = tempfile.mkdtemp(prefix="actuo_{}_".format(shm_name))
    moveit = LaunchConfiguration("moveit").perform(context).lower() == "true"
    result = subprocess.run(
        [os.path.join(prefix, "lib", "actuo_ros", "actuo-modular-files"), catalog,
         "--out", out_dir, "--shm-name", shm_name, "--hardware-plugin", HARDWARE_PLUGIN]
        + (["--moveit"] if moveit else []),
        capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("actuo-modular-files failed on {}:\n{}".format(
            catalog, result.stderr))
    manifest_path = result.stdout.strip()
    with open(manifest_path) as handle:
        manifest = yaml.safe_load(handle)
    initial = LaunchConfiguration("configuration").perform(context) or manifest["initial"]
    with open(manifest["configurations"][initial]["urdf"]) as handle:
        initial_model = handle.read()
    rate = LaunchConfiguration("rate").perform(context)
    if not rate:
        with open(manifest["controllers_yaml"]) as handle:
            rate = str(yaml.safe_load(handle)["controller_manager"]
                       ["ros__parameters"]["update_rate"])
    port = LaunchConfiguration("port").perform(context)
    rviz_config = LaunchConfiguration("rviz_config").perform(context)

    servicer = Node(
        package="actuo_ros", executable="actuo_bus_node", output="screen",
        arguments=["--real", "--catalog", catalog, "--configuration", initial,
                   "--shm-name", shm_name, "--rate", rate, "--timeout", "90"]
        + (["--channel", port] if port else []))
    control = [
        Node(package="controller_manager", executable="ros2_control_node",
             output="screen",
             parameters=[manifest["controllers_yaml"], {"update_rate": int(float(rate))}],
             remappings=[("robot_description", SUPERSET_TOPIC)]),
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             output="screen", parameters=[{"robot_description": initial_model}]),
        Node(package="actuo_ros", executable="actuo_configuration_manager",
             output="screen", parameters=[{"manifest": manifest_path,
                                           "superset_topic": SUPERSET_TOPIC}]),
    ]
    if moveit:
        control.append(Node(package="actuo_ros", executable="actuo_moveit_follower",
                            output="screen", parameters=[{"manifest": manifest_path}]))
    # Loaded and configured, all of them, by one spawner (spawners take turns
    # on one lock): the configuration manager activates a module's
    # controllers when the module comes up.
    every = [c for spec in manifest["configurations"].values() for c in spec["controllers"]]
    control.append(Node(package="controller_manager", executable="spawner", output="screen",
                        arguments=every + ["--inactive", "--service-call-timeout", "60"]))
    if rviz_config:
        control.append(Node(package="rviz2", executable="rviz2", output="log",
                            arguments=["-d", rviz_config]))
    stale = [_shm_file(spec["segment"]) for spec in manifest["configurations"].values()]
    return [
        ExecuteProcess(cmd=["rm", "-f"] + stale, output="log"),
        servicer,
        TimerAction(period=3.0, actions=control),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("catalog", description="modular robot catalog YAML"),
        DeclareLaunchArgument("configuration", default_value="",
                              description="configuration to start in; default: "
                                          "the catalog's default"),
        DeclareLaunchArgument("shm_name", default_value="",
                              description="segment base name; default: the "
                                          "catalog's file name"),
        DeclareLaunchArgument("rate", default_value="",
                              description="servicer loop and controller update "
                                          "rate [Hz]; default: the configs' update_rate"),
        DeclareLaunchArgument("port", default_value="",
                              description="serial device, overriding the configs' "
                                          "bus.transport.channel"),
        DeclareLaunchArgument("moveit", default_value="false", choices=["true", "false"],
                              description="run move_group for the running configuration"),
        DeclareLaunchArgument("rviz_config", default_value="",
                              description="start RViz with this config"),
        OpaqueFunction(function=_setup),
    ])
