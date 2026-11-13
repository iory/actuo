"""A whole robot in one config: its bus, its model and its modules.

ROS-free and light (stdlib + PyYAML, and xacro for a ``.xacro`` model), so a
launch file can import it as well as the bus node.

A bus config (:mod:`actuo.drivers.config`) lists every joint with its actuator.
For a robot brought up from a URDF most of that is repeated: the joint names
are the model's, the limits are in its ``<limit>`` tags, and the actuator
settings are the same for every servo. So the config may also carry::

    urdf: package://my_description/urdf/my_robot.xacro   # .xacro is expanded
    modules:                       # hardware components, in bus (shm) order
      head:
        joints: [head_joint0, head_joint1]                # -> head_controller
      rarm:
        joints: [rarm_joint0, rarm_joint1]                # -> rarm_controller
        gripper: [rarm_gripper_joint]           # -> rarm_gripper_controller
    joint_defaults:                # under every joint
      max_velocity: 2.0
      actuator: {plugin: kondo_ics, velocity: 1}
    joints:                        # only what differs, by name
      - {name: rarm_gripper_joint, actuator: {stretch: 40}}

With ``modules``, the bus's joints are the modules' joints in that order, each
``joint_defaults`` with its ``joints`` entry merged over it. With ``urdf``,
an actuator without ``position_limit_deg`` takes the joint's URDF limit.
:func:`expand_document` does this, and :func:`actuo.drivers.config.load_document`
calls it, so every reader of a config sees the full ``joints`` list.

A generic ROS 2 launch file builds the rest from the same document:

* :func:`hardware_urdf` -- the model with one ros2_control component per
  module (the shared-memory hardware component or mock), replacing the model's own ros2_control and gazebo
  elements;
* :func:`controller_parameters` -- controller_manager's parameters: a
  joint_state_broadcaster over every bus joint and a trajectory controller per
  module (``<module>_controller``) and per module gripper
  (``<module>_gripper_controller``). Optional keys: ``update_rate`` (Hz,
  default :data:`DEFAULT_UPDATE_RATE`), ``trajectory_tolerance`` (``path``,
  ``goal`` in rad and ``goal_time`` in s, defaults
  :data:`DEFAULT_TOLERANCE`), and ``controller_parameters``, merged over what
  is generated;
* :func:`robot_modes` -- the fixed robot-modes document the ROS 2 module
  manager runs.

A gripper's controller has no path or goal tolerance: a gripper closed on an
object stops short of its command, and that is the grasp, not a failure.
"""

from __future__ import annotations

import copy
import math
import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

#: controller_manager's update rate when the config gives none, Hz.
DEFAULT_UPDATE_RATE = 50
#: Trajectory controller tolerances when the config gives none: rad, rad, s.
DEFAULT_TOLERANCE = {"path": 0.5, "goal": 0.1, "goal_time": 2.0}

TRAJECTORY_CONTROLLER = "joint_trajectory_controller/JointTrajectoryController"
STATE_BROADCASTER = "joint_state_broadcaster/JointStateBroadcaster"

#: :func:`hardware_urdf` backends: the caller's C++ shared-memory component,
#: or ros2_control's own mock.
SHM_BACKEND = "shm"
MOCK_BACKEND = "mock"
MOCK_PLUGIN = "mock_components/GenericSystem"


#: Directories holding ``<package>/`` trees, separated like ``PATH``: where
#: ``package://`` references are looked up first, without ROS.
PACKAGE_PATH_ENV = "ACTUO_PACKAGE_PATH"


def package_share_directory(package: str, value: str, where: str) -> str:
    """The directory a ``package://<package>/`` reference names.

    Looked up in :data:`PACKAGE_PATH_ENV` first (each entry a directory that
    contains ``<package>/``), then in the ROS 2 ament index if ROS 2 is
    installed. actuo does not need ROS; the ament index is only one place a
    package can be found.

    Raises
    ------
    FileNotFoundError
        Naming every place that was looked in, and saying so when ROS 2 is
        not installed, instead of a bare "not on the ament index".
    """
    roots = [r for r in os.environ.get(PACKAGE_PATH_ENV, "").split(os.pathsep) if r]
    for root in roots:
        candidate = os.path.join(root, package)
        if os.path.isdir(candidate):
            return candidate
    searched = "{}={}".format(PACKAGE_PATH_ENV, os.pathsep.join(roots) or "(unset)")
    try:
        from ament_index_python.packages import get_package_share_directory
    except ImportError:
        raise FileNotFoundError(
            "{}: {!r} names package {!r}, which is not in {} and ROS 2 is not "
            "installed (no ament_index_python) to look it up. Set {} to a "
            "directory containing {}/, or write the path without package://."
            .format(where, value, package, searched, PACKAGE_PATH_ENV, package))
    try:
        return get_package_share_directory(package)
    except Exception as error:
        raise FileNotFoundError(
            "{}: {!r} names package {!r}, which is not in {} nor on the ament "
            "index ({}). Build and source the workspace that provides it, or "
            "set {}.".format(where, value, package, searched, error,
                             PACKAGE_PATH_ENV))


def resolve_data_path(value: str, config_path: str, where: str) -> str:
    """Resolve a data file named by a robot config.

    Three forms, in order:

    * absolute -- used as written.
    * ``package://<pkg>/<rel>`` -- ``<rel>`` inside the directory
      :func:`package_share_directory` finds for ``<pkg>``: in
      ``$ACTUO_PACKAGE_PATH``, else in the ROS 2 ament index (its ``share/``).
    * anything else -- relative to the directory holding the config, so a robot
      directory stays self-contained and movable.

    The ``package://`` form exists because a robot's config does not always live
    beside the data it needs. A firmware symbol map, for example, belongs to the
    board's firmware build and is shared by every robot with that board;
    copying it next to each robot's config would make as many sources of truth
    as there are robots, so each config names the one copy instead.

    Raises
    ------
    FileNotFoundError
        If the package is not found, or the resolved file does not exist.
        Both are reported with the original string, because "no such file:
        kondoh7_v0.6.5_symbols.yaml" after silent path joining tells the
        operator nothing about where it was looked for.
    """
    if value.startswith("package://"):
        remainder = value[len("package://"):]
        package, _, relative = remainder.partition("/")
        if not package or not relative:
            raise ValueError(
                "{}: {!r} is not a usable package:// reference; it needs "
                "package://<package>/<path within its share directory>"
                .format(where, value))
        resolved = os.path.join(package_share_directory(package, value, where),
                                relative)
    elif os.path.isabs(value):
        resolved = value
    else:
        resolved = os.path.join(os.path.dirname(os.path.abspath(config_path)),
                                value)

    if not os.path.exists(resolved):
        raise FileNotFoundError(
            "{}: {!r} resolves to {!r}, which does not exist."
            .format(where, value, resolved))
    return resolved


def merged(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    """``base`` with ``over`` merged in, mappings recursively; neither is changed."""
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merged(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def module_joints(document: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    """``{module: {"joints": [...], "gripper": [...]}}``, in config order.

    Raises
    ------
    ValueError
        If a module has no joints, a key other than ``joints`` / ``gripper``,
        or a joint is in two modules.
    """
    out: Dict[str, Dict[str, List[str]]] = {}
    seen: Dict[str, str] = {}
    for module, spec in (document.get("modules") or {}).items():
        spec = spec or {}
        unknown = set(spec) - {"joints", "gripper"}
        if unknown:
            raise ValueError("modules.{}: unknown keys {} (joints, gripper)".format(
                module, sorted(unknown)))
        joints = [str(j) for j in spec.get("joints") or []]
        gripper = [str(j) for j in spec.get("gripper") or []]
        if not joints and not gripper:
            raise ValueError("modules.{}: no joints".format(module))
        for name in joints + gripper:
            if name in seen:
                raise ValueError("joint {} is in modules {} and {}".format(
                    name, seen[name], module))
            seen[name] = module
        out[str(module)] = {"joints": joints, "gripper": gripper}
    return out


def load_model(document: Dict[str, Any], path: str) -> str:
    """The URDF text of the config's ``urdf``, a ``.xacro`` expanded."""
    model = resolve_data_path(str(document["urdf"]), path, "urdf")
    if model.endswith(".xacro"):
        import xacro
        return xacro.process_file(model).toxml()
    with open(model) as handle:
        return handle.read()


def urdf_limits(model_xml: str) -> Dict[str, tuple]:
    """``{joint: (lower, upper)}`` in rad, for every joint whose limit has both."""
    limits = {}
    for joint in ET.fromstring(model_xml).findall("joint"):
        limit = joint.find("limit")
        if (joint.get("type") in ("revolute", "prismatic") and limit is not None
                and limit.get("lower") is not None and limit.get("upper") is not None):
            limits[joint.get("name")] = (float(limit.get("lower")),
                                         float(limit.get("upper")))
    return limits


def expand_document(document: Dict[str, Any], path: str) -> Dict[str, Any]:
    """The config with its ``joints`` list built from ``modules``, ``urdf`` and
    ``joint_defaults`` (see the module docstring).

    A config with none of the three is returned as it is.

    Raises
    ------
    ValueError
        If a ``joints`` entry names a joint no module has, or a module joint is
        not in the model.
    """
    if not any(key in document for key in ("modules", "urdf", "joint_defaults")):
        return document
    out = dict(document)
    defaults = document.get("joint_defaults") or {}
    given = list(document.get("joints") or [])
    modules = module_joints(document)
    if modules:
        names = [n for m in modules.values() for n in m["joints"] + m["gripper"]]
        by_name = {}
        for entry in given:
            name = str(entry.get("name"))
            if name not in names:
                raise ValueError(
                    "{}: joints entry {!r} is in no module; with `modules` the "
                    "joints are the modules' and `joints` only overrides them"
                    .format(path, name))
            by_name[name] = entry
        joints = [merged(merged(defaults, by_name.get(n, {})), {"name": n})
                  for n in names]
    else:
        joints = [merged(defaults, entry) for entry in given]
    if "urdf" in document:
        model_xml = load_model(document, path)
        model_joints = {j.get("name") for j in ET.fromstring(model_xml).findall("joint")}
        missing = [j["name"] for j in joints if j["name"] not in model_joints]
        if missing:
            raise ValueError("{}: joints not in the model {}: {}".format(
                path, document["urdf"], ", ".join(missing)))
        limits = urdf_limits(model_xml)
        for joint in joints:
            actuator = joint.setdefault("actuator", {})
            if "position_limit_deg" not in actuator and joint["name"] in limits:
                lower, upper = limits[joint["name"]]
                actuator["position_limit_deg"] = [math.degrees(lower),
                                                  math.degrees(upper)]
    out["joints"] = joints
    return out


def controllers_of(document: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """``{controller: {"module", "joints", "gripper"}}`` the modules make."""
    out = {}
    for module, spec in module_joints(document).items():
        if spec["joints"]:
            out[module + "_controller"] = {
                "module": module, "joints": spec["joints"], "gripper": False}
        if spec["gripper"]:
            out[module + "_gripper_controller"] = {
                "module": module, "joints": spec["gripper"], "gripper": True}
    return out


def controller_parameters(document: Dict[str, Any]) -> Dict[str, Any]:
    """controller_manager's parameters, as a ROS parameter file's mapping."""
    tolerance = merged(DEFAULT_TOLERANCE, document.get("trajectory_tolerance") or {})
    controllers = controllers_of(document)
    every_joint = [n for c in controllers.values() for n in c["joints"]]
    params: Dict[str, Any] = {
        "controller_manager": {"ros__parameters": dict(
            update_rate=int(document.get("update_rate", DEFAULT_UPDATE_RATE)),
            # A module whose servos are not all there must not take the stack
            # down: its component stays inactive until the module manager
            # brings it up.
            hardware_components_initial_state={
                "shutdown_on_initial_state_failure": False},
            joint_state_broadcaster={"type": STATE_BROADCASTER},
            **{name: {"type": TRAJECTORY_CONTROLLER} for name in controllers})},
        "joint_state_broadcaster": {"ros__parameters": {
            "joints": every_joint, "interfaces": ["position"]}},
    }
    for name, spec in controllers.items():
        path_tol, goal_tol, goal_time = (
            (0.0, 0.0, 0.0) if spec["gripper"]
            else (tolerance["path"], tolerance["goal"], tolerance["goal_time"]))
        constraints = {"stopped_velocity_tolerance": 0.1, "goal_time": goal_time}
        constraints.update({j: {"trajectory": path_tol, "goal": goal_tol}
                            for j in spec["joints"]})
        params[name] = {"ros__parameters": {
            "joints": spec["joints"],
            "command_interfaces": ["position"],
            # ICS and most hobby servos report a position and nothing else.
            "state_interfaces": ["position"],
            "state_publish_rate": float(document.get("update_rate", DEFAULT_UPDATE_RATE)),
            "action_monitor_rate": 20.0,
            "allow_partial_joints_goal": False,
            "allow_nonzero_velocity_at_trajectory_end": False,
            "constraints": constraints,
        }}
    return merged(params, document.get("controller_parameters") or {})


def hardware_urdf(model_xml: str, document: Dict[str, Any], shm_name: str,
                  backend: str = SHM_BACKEND,
                  hardware_plugin: Optional[str] = None) -> str:
    """The model with one ros2_control component per module.

    Parameters
    ----------
    model_xml : str
        The URDF (expanded).
    document : dict
        The expanded config (:func:`expand_document`).
    shm_name : str
        The bus servicer's shm segment.
    backend : str
        ``"shm"`` (a C++ ros2_control component that reads the shared-memory
        segment, :mod:`actuo.drivers.shm`, each joint at its bus slot) or
        ``"mock"`` (``mock_components/GenericSystem``).
    hardware_plugin : str or None
        The ros2_control plugin name of the C++ shared-memory component, as
        its package exports it (``<package>/<Class>``). Required for
        ``backend="shm"``: this package does not ship that component and so
        cannot name it. Not used for ``"mock"``.

    Returns
    -------
    str

    Raises
    ------
    ValueError
        For an unknown ``backend``, or ``"shm"`` without ``hardware_plugin``.
    """
    if backend not in (SHM_BACKEND, MOCK_BACKEND):
        raise ValueError("backend must be {} or {}, got {!r}".format(
            SHM_BACKEND, MOCK_BACKEND, backend))
    if backend == SHM_BACKEND and not hardware_plugin:
        raise ValueError(
            "backend {!r} needs hardware_plugin, the ros2_control plugin name "
            "of the shared-memory hardware component".format(SHM_BACKEND))
    robot = ET.fromstring(model_xml)
    for tag in ("ros2_control", "gazebo"):
        for element in robot.findall(tag):
            robot.remove(element)
    limits = urdf_limits(model_xml)
    slots = [j["name"] for j in document["joints"]]
    for module, spec in module_joints(document).items():
        block = ET.SubElement(robot, "ros2_control", name=module, type="system")
        hardware = ET.SubElement(block, "hardware")
        if backend == MOCK_BACKEND:
            ET.SubElement(hardware, "plugin").text = MOCK_PLUGIN
        else:
            ET.SubElement(hardware, "plugin").text = hardware_plugin
            ET.SubElement(hardware, "param", name="shm_name").text = shm_name
            ET.SubElement(hardware, "param", name="use_sim").text = "false"
        for name in spec["joints"] + spec["gripper"]:
            joint = ET.SubElement(block, "joint", name=name)
            if backend == SHM_BACKEND:
                ET.SubElement(joint, "param", name="slot").text = str(slots.index(name))
            command = ET.SubElement(joint, "command_interface", name="position")
            if name in limits:
                ET.SubElement(command, "param", name="min").text = str(limits[name][0])
                ET.SubElement(command, "param", name="max").text = str(limits[name][1])
            state = ET.SubElement(joint, "state_interface", name="position")
            ET.SubElement(state, "param", name="initial_value").text = "0.0"
    return ET.tostring(robot, encoding="unicode")


def robot_modes(document: Dict[str, Any], urdf_path: str) -> Dict[str, Any]:
    """The fixed robot-modes document (actuo.drivers.robot_modes) of the config."""
    controllers = controllers_of(document)
    return {"mode": "fixed", "base": {"urdf": urdf_path, "modules": {
        module: {"joints": spec["joints"] + spec["gripper"],
                 "controllers": [c for c, s in controllers.items() if s["module"] == module]}
        for module, spec in module_joints(document).items()}}}
