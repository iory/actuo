"""A robot that is several robots: configurations switched at run time, without ROS.

A modular robot is put together from modules, and each way of putting it
together is a different robot -- other joints, other names, other calibration,
another model -- even when some of the servos are the same. Each such robot is
a **configuration**: an ordinary actuo robot config (bus, joints with their ids
and calibration) plus its URDF. A catalog lists them::

    configurations:
      gripper:
        robot: gripper.yaml      # actuo robot config
        urdf: gripper.urdf
        ids: [1, 2, 3]           # detection signature (default: the config's ids)
      drill:
        robot: drill.yaml
        urdf: drill.urdf
    default: gripper             # the configuration to start in
    select: detect               # or declare
    hold: same_id                # or none

Which configuration is current is decided one of two ways:

``detect``
    By the servos that answer: the configuration all of whose ids answer, the
    one with the most ids when several do. Two complete configurations of the
    same size are a conflict, reported and not guessed: the robot stays where
    it is. So is none being complete (a module half pulled out): the current
    configuration keeps running, its missing joints detached.
``declare``
    By whoever drives the robot: :meth:`ModularRobot.declare` names the
    configuration, for robots whose configurations have the same servos
    (a module mounted another way round, a tool swapped for one of identical
    servos). Detection stops until :meth:`ModularRobot.resume_detection`.

Switching keeps the serial port's owner single and the arm up: the current
:class:`~actuo.robot_interface.ActuoRobotInterface` releases the servos the next
configuration does not have, then closes without releasing the others
(``close`` leaves torque as it is), and the next one starts on the same port
with ``hold_joints``: with ``hold: same_id`` every servo that held torque and
is in the next configuration, under whatever joint name and calibration it has
there, is switched on at its measured pose before the first write. Nothing
moves. Every configuration's model is loaded once at start, so a switch costs
the next interface's start-up, not a URDF parse.

Each configuration's bus runs ``hot_plug`` with the other configurations' ids
as ``watch_ids``, so the servos of a configuration that is not current are
still seen coming and going. Any bus that offers both
(:class:`actuo.core.contracts.Bus`: ``supports_hot_plug``,
``supports_watch``) will do -- FEETECH, DYNAMIXEL, RCB-4mini, CAN servo-mode
drives; one that does not is refused when the configuration's bus is built.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import yaml

_LOG = logging.getLogger(__name__)

SELECT_MODES = ("detect", "declare")
HOLD_POLICIES = ("same_id", "none")
#: Seconds between two looks at which servos answer.
DEFAULT_CHECK_PERIOD = 0.2
#: Service cycles a release is given to reach the wire before the old
#: interface closes.
RELEASE_CYCLES = 3


@dataclass(frozen=True)
class Configuration:
    """One robot the modules can make."""

    name: str
    robot: str                      # actuo robot config path
    urdf: str
    ids: FrozenSet[int]             # detection signature
    joint_ids: Dict[str, int] = field(hash=False, compare=False, default_factory=dict)


class Catalog:
    """The configurations of one modular robot, from a catalog YAML.

    Parameters
    ----------
    path : str
        The catalog. Relative paths in it are taken from its directory.
    """

    def __init__(self, path: str) -> None:
        with open(path) as f:
            spec = yaml.safe_load(f) or {}
        here = os.path.dirname(os.path.abspath(path))
        self.path = path
        raw = spec.get("configurations") or {}
        if not raw:
            raise ValueError("{}: configurations is empty".format(path))
        self.configurations: Dict[str, Configuration] = {}
        transports = {}
        for name, c in raw.items():
            robot = os.path.join(here, str(c["robot"]))
            urdf = os.path.join(here, str(c["urdf"]))
            with open(robot) as f:
                document = yaml.safe_load(f) or {}
            joint_ids = {str(j["name"]): int(j["actuator"]["id"])
                         for j in document.get("joints") or ()
                         if "id" in (j.get("actuator") or {})}
            ids = frozenset(int(i) for i in c.get("ids", joint_ids.values()))
            if not ids:
                raise ValueError("{}: configuration {} has no ids".format(path, name))
            transport = (document.get("bus") or {}).get("transport") or {}
            transports[name] = (transport.get("channel"), transport.get("bitrate"))
            self.configurations[str(name)] = Configuration(
                name=str(name), robot=robot, urdf=urdf, ids=ids, joint_ids=joint_ids)
        if len(set(transports.values())) > 1:
            raise ValueError(
                "{}: every configuration must be on the same serial line (one "
                "port is kept open across switches); they name {}".format(
                    path, transports))
        self.default = spec.get("default")
        if self.default is not None and self.default not in self.configurations:
            raise ValueError("{}: default {!r} is not a configuration".format(
                path, self.default))
        self.select = str(spec.get("select", "detect"))
        if self.select not in SELECT_MODES:
            raise ValueError("{}: select must be one of {}".format(path, SELECT_MODES))
        self.hold = str(spec.get("hold", "same_id"))
        if self.hold not in HOLD_POLICIES:
            raise ValueError("{}: hold must be one of {}".format(path, HOLD_POLICIES))

    @property
    def names(self) -> List[str]:
        return list(self.configurations)

    def all_ids(self) -> Set[int]:
        out: Set[int] = set()
        for c in self.configurations.values():
            out |= c.ids
        return out

    def detect(self, answering: Set[int]) -> Tuple[Optional[str], List[str]]:
        """The configuration the answering ids make, and what is wrong.

        Returns
        -------
        tuple
            ``(name, problems)``. ``name`` is None when no configuration is
            complete or the largest complete ones tie.
        """
        complete = [c for c in self.configurations.values() if c.ids <= set(answering)]
        if not complete:
            return None, ["no configuration has all its servos answering "
                          "(answering: {})".format(sorted(answering))]
        size = max(len(c.ids) for c in complete)
        best = [c.name for c in complete if len(c.ids) == size]
        if len(best) > 1:
            return None, ["{} are all complete; only one can be the robot".format(
                " and ".join(best))]
        return best[0], []


@dataclass
class ModularState:
    """What the robot is right now."""

    configuration: str
    select: str
    problems: List[str]
    answering: List[int]
    attached: List[str]
    urdf: str
    switches: int
    last_switch_s: Optional[float]


class ModularRobot:
    """Run whichever configuration of a modular robot is current.

    Parameters
    ----------
    catalog : str
        Catalog YAML (see the module docstring).
    initial : str, optional
        Configuration to start in; default the catalog's ``default``.
    rate : float
        Service rate of the robot interface [Hz].
    check_period : float
        Seconds between looks at which servos answer (``detect``).
    on_change : callable, optional
        Called with the new :class:`ModularState` after every switch.
    transport_overrides : dict, optional
        Passed to every configuration's interface.
    startup_timeout : float
        Seconds a configuration's interface may take to attach its held joints.
    """

    def __init__(self, catalog: str, initial: Optional[str] = None,
                 rate: float = 50.0, check_period: float = DEFAULT_CHECK_PERIOD,
                 on_change: Optional[Callable[[ModularState], None]] = None,
                 transport_overrides: Optional[dict] = None,
                 startup_timeout: float = 5.0) -> None:
        from skrobot.model import RobotModel

        self.catalog = Catalog(catalog)
        name = initial or self.catalog.default
        if name is None:
            raise ValueError("no initial configuration: pass initial or set default")
        if name not in self.catalog.configurations:
            raise KeyError(name)
        self.rate = float(rate)
        self.check_period = float(check_period)
        self.on_change = on_change
        self.transport_overrides = transport_overrides
        self.startup_timeout = float(startup_timeout)
        self.select = self.catalog.select
        # Loaded once: a switch must not wait for a URDF parse.
        self.models = {n: RobotModel.from_urdf(c.urdf)
                       for n, c in self.catalog.configurations.items()}
        self._lock = threading.RLock()
        self._problems: List[str] = []
        self._switches = 0
        self._last_switch_s: Optional[float] = None
        self._current = name
        self._ri = self._start(name, hold=())
        self._running = True
        self._thread = threading.Thread(target=self._watch, name="actuo-modular",
                                        daemon=True)
        self._thread.start()

    # -- what is current --------------------------------------------------------

    @property
    def current(self) -> str:
        return self._current

    @property
    def ri(self):
        """The current configuration's ActuoRobotInterface (replaced on a switch)."""
        return self._ri

    @property
    def robot(self):
        """The current configuration's skrobot model."""
        return self.models[self._current]

    def state(self) -> ModularState:
        with self._lock:
            ri = self._ri
            return ModularState(
                configuration=self._current, select=self.select,
                problems=list(self._problems),
                answering=list(ri.bus.answering_ids()),
                attached=ri.attached_joints(),
                urdf=self.catalog.configurations[self._current].urdf,
                switches=self._switches, last_switch_s=self._last_switch_s)

    # -- switching --------------------------------------------------------------

    def declare(self, name: str) -> ModularState:
        """Make ``name`` the robot now, and stop detecting until resumed."""
        if name not in self.catalog.configurations:
            raise KeyError("no configuration {!r}; have {}".format(
                name, ", ".join(self.catalog.names)))
        with self._lock:
            self.select = "declare"
            missing = sorted(self.catalog.configurations[name].ids
                             - set(self._ri.bus.answering_ids()))
            self._problems = ([] if not missing else
                              ["declared {} but ids {} do not answer".format(name, missing)])
            if name != self._current:
                self._switch(name)
            return self.state()

    def resume_detection(self) -> None:
        """Let the answering servos decide again."""
        with self._lock:
            self.select = "detect"

    def _start(self, name: str, hold: Sequence[str]):
        from actuo.robot_interface import ActuoRobotInterface

        config = self.catalog.configurations[name]
        watch = sorted(self.catalog.all_ids() - set(config.joint_ids.values()))
        ri = ActuoRobotInterface(
            self.models[name], config.robot, rate=self.rate,
            transport_overrides=self.transport_overrides,
            startup_timeout=self.startup_timeout, required_joints=[],
            bus_overrides={"hot_plug": True, "watch_ids": watch},
            hold_joints=list(hold))
        return ri

    def _switch(self, name: str) -> None:
        """Hand the port over to configuration ``name``. Holds ``self._lock``."""
        old_ri, old_name = self._ri, self._current
        new = self.catalog.configurations[name]
        enabled = {old_ri.bus.actuators[old_ri._index[j]].id: j
                   for j in old_ri.supervisor.enabled_joints}
        carry = (set(enabled) & set(new.joint_ids.values())
                 if self.catalog.hold == "same_id" else set())
        # Servos the next configuration does not drive are released now: after
        # the switch nobody would write to them, and they would hold forever.
        release = [j for i, j in enabled.items() if i not in carry]
        if release:
            old_ri.servo_off(release)
            time.sleep(RELEASE_CYCLES * old_ri.period)
        hold = [j for j, i in new.joint_ids.items() if i in carry]
        t0 = time.monotonic()
        old_ri.close()
        try:
            self._ri = self._start(name, hold)
        except Exception as error:
            _LOG.error("switch %s -> %s failed (%s); back to %s", old_name, name,
                       error, old_name)
            self._problems.append("switch to {} failed: {}".format(name, error))
            back = [j for j in enabled.values() if j not in release]
            self._ri = self._start(old_name, back)
            raise
        self._current = name
        self._switches += 1
        self._last_switch_s = time.monotonic() - t0
        _LOG.info("configuration %s -> %s in %.3f s (held: %s)", old_name, name,
                  self._last_switch_s, ", ".join(hold) or "none")
        if self.on_change is not None:
            self.on_change(self.state())

    def _watch(self) -> None:
        while self._running:
            time.sleep(self.check_period)
            with self._lock:
                if not self._running or self.select != "detect":
                    continue
                target, problems = self.catalog.detect(set(self._ri.bus.answering_ids()))
                self._problems = problems
                if target is not None and target != self._current:
                    try:
                        self._switch(target)
                    except Exception:  # reported in problems; keep watching
                        pass

    def close(self) -> None:
        """Stop watching and close the current interface (torque left as it is)."""
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        with self._lock:
            self._ri.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@dataclass(frozen=True)
class SwitchRequest:
    """Leave the current configuration for ``target``; ``hold_ids`` keep torque."""

    target: str
    hold_ids: FrozenSet[int]


class Switching:
    """Configuration switching for a process that owns the bus loop itself.

    :class:`ModularRobot` runs :class:`~actuo.robot_interface.ActuoRobotInterface`
    and switches between them; a bus servicer that runs its own loop uses this
    instead: once per cycle it calls :meth:`poll`, and when that returns a
    :class:`SwitchRequest` it services one more cycle (so the releases it asked
    for reach the wire), leaves its loop without releasing anything else, and
    starts the target configuration holding ``hold_ids``. The rules are
    :class:`ModularRobot`'s (detect / declare, ``hold``).

    Parameters
    ----------
    catalog : Catalog
    current : str
        The configuration running now.
    """

    def __init__(self, catalog: Catalog, current: str) -> None:
        if current not in catalog.configurations:
            raise KeyError(current)
        self.catalog = catalog
        self.current = current
        self.select = catalog.select
        self.problems: List[str] = []
        self.answering: List[int] = []
        #: Ids the starting configuration takes over with torque on.
        self.hold_ids: FrozenSet[int] = frozenset()
        #: Joints that kept torque across the last switch (set by the servicer).
        self.held_joints: List[str] = []
        self.last_switch_s: Optional[float] = None
        self._requested: Optional[str] = None
        self._lock = threading.Lock()
        self._left_at: Optional[float] = None

    def declare(self, name: str) -> Tuple[bool, str]:
        """Ask for ``name`` and stop detecting. Thread-safe; acted on at the
        next :meth:`poll`."""
        if name not in self.catalog.configurations:
            return False, "no configuration {!r}; have {}".format(
                name, ", ".join(self.catalog.names))
        with self._lock:
            self.select = "declare"
            self._requested = name
        return True, ("already {}".format(name) if name == self.current
                      else "switching to {}".format(name))

    def resume_detection(self) -> None:
        with self._lock:
            self.select = "detect"
            self._requested = None

    def poll(self, bus, supervisor, joint_names: Sequence[str]) -> Optional[SwitchRequest]:
        """Decide whether to leave the current configuration.

        Releases (``supervisor.disable``) the enabled joints whose servos the
        target does not have; the caller services one more cycle before
        leaving so those releases reach the wire.
        """
        answering = set(bus.answering_ids())
        self.answering = sorted(answering)
        with self._lock:
            requested, self._requested = self._requested, None
            select = self.select
        if requested is not None:
            missing = sorted(self.catalog.configurations[requested].ids - answering)
            self.problems = ([] if not missing else
                             ["declared {} but ids {} do not answer".format(
                                 requested, missing)])
            target = requested
        elif select == "detect":
            target, self.problems = self.catalog.detect(answering)
        else:
            return None
        if target is None or target == self.current:
            return None
        enabled = set(supervisor.enabled_joints)
        id_of = {name: int(a.id) for name, a in zip(joint_names, bus.actuators)}
        target_ids = set(self.catalog.configurations[target].joint_ids.values())
        carry = ({id_of[n] for n in enabled} & target_ids
                 if self.catalog.hold == "same_id" else set())
        release = [n for n in joint_names if n in enabled and id_of[n] not in carry]
        if release:
            supervisor.disable(release)
        self._left_at = time.monotonic()
        return SwitchRequest(target=target, hold_ids=frozenset(carry))

    def started(self, request: SwitchRequest) -> None:
        """The target configuration's loop is about to run."""
        self.current = request.target
        self.hold_ids = request.hold_ids
        if self._left_at is not None:
            self.last_switch_s = time.monotonic() - self._left_at
            self._left_at = None

    def hold_joints(self, joint_names: Sequence[str], actuators) -> List[str]:
        """Joints of the starting configuration whose servos keep torque."""
        return [n for n, a in zip(joint_names, actuators) if int(a.id) in self.hold_ids]


__all__ = ("Catalog", "Configuration", "ModularRobot", "ModularState", "component_name",
           "moveit_files", "ros2_control_files", "shm_segment", "srdf",
           "SwitchRequest", "Switching")


# -- ros2_control files ---------------------------------------------------------

def component_name(configuration: str, module: str) -> str:
    """The ros2_control component of ``module`` in ``configuration``."""
    return "{}_{}".format(configuration, module)


def shm_segment(base: str, configuration: str) -> str:
    """The shared-memory segment the servicer creates for ``configuration``."""
    return "{}_{}".format(base, configuration)


def ros2_control_files(catalog_path: str, out_dir: str, shm_base: str,
                       hardware_plugin: str) -> Dict[str, object]:
    """Write one controller_manager's files for every configuration at once.

    controller_manager takes one robot description, once. So every
    configuration's modules go into it side by side, as components named
    ``<configuration>_<module>`` reading the segment the servicer creates for
    that configuration (:func:`shm_segment`); a switch then only deactivates
    one configuration's components and controllers and activates another's.
    Each configuration's robot config needs ``urdf`` and ``modules``
    (:mod:`actuo.drivers.robot_document`). Joint names must differ between
    configurations -- ros2_control gives a joint to one component -- while
    links of the same name are taken as the same link (the robot's base).

    Writes ``robot.urdf`` (the superset, with the components),
    ``controllers.yaml`` and, per configuration, ``<name>.urdf`` (its own model,
    for robot_state_publisher), and returns the manifest the configuration
    manager and the launch file read::

        {"urdf": ..., "controllers_yaml": ..., "initial": <default>,
         "configurations": {name: {"urdf": ..., "segment": ...,
                                   "components": [...], "controllers": [...],
                                   "modules": {component: {"joints": [...],
                                                           "controllers": [...]}}}}}

    Each module (component) has its own joint_state_broadcaster, first in its
    ``controllers``.

    Raises
    ------
    ValueError
        A configuration has no urdf/modules, or two configurations name the
        same joint or put one link under different parents.
    """
    import xml.etree.ElementTree as ET

    from actuo.drivers.config import load_document
    from actuo.drivers.robot_document import (
        STATE_BROADCASTER, controller_parameters, controllers_of, hardware_urdf,
        load_model, module_joints)

    catalog = Catalog(catalog_path)
    os.makedirs(out_dir, exist_ok=True)
    superset = ET.Element("robot", name=os.path.splitext(os.path.basename(catalog_path))[0])
    joint_owner: Dict[str, str] = {}
    parent_of: Dict[str, str] = {}
    present_links: Set[str] = set()
    cm: Dict[str, object] = {}
    params: Dict[str, object] = {}
    manifest_configs: Dict[str, object] = {}
    update_rate = 0
    for name, config in catalog.configurations.items():
        document = load_document(config.robot)
        for key in ("urdf", "modules"):
            if key not in document:
                raise ValueError("{}: {} has no {!r}".format(catalog_path, config.robot, key))
        model = load_model(document, config.robot)
        own = os.path.join(out_dir, name + ".urdf")
        with open(own, "w") as f:
            f.write(model)
        segment = shm_segment(shm_base, name)
        tree = ET.fromstring(hardware_urdf(model, document, segment,
                                           hardware_plugin=hardware_plugin))
        components = []
        for element in tree:
            if element.tag == "link":
                if element.get("name") in present_links:
                    continue
                present_links.add(element.get("name"))
            elif element.tag == "joint":
                joint = element.get("name")
                if joint in joint_owner:
                    raise ValueError(
                        "{}: joint {} is in configurations {} and {}; ros2_control "
                        "gives a joint to one component, so name it differently "
                        "in each".format(catalog_path, joint, joint_owner[joint], name))
                joint_owner[joint] = name
                child = element.find("child").get("link")
                parent = element.find("parent").get("link")
                if parent_of.setdefault(child, parent) != parent:
                    raise ValueError(
                        "{}: link {} hangs from {} in one configuration and from "
                        "{} in {}".format(catalog_path, child, parent_of[child], parent, name))
            elif element.tag == "ros2_control":
                element.set("name", component_name(name, element.get("name")))
                components.append(element.get("name"))
            elif element.get("name") is not None and any(
                    e.tag == element.tag and e.get("name") == element.get("name")
                    for e in superset):
                continue
            superset.append(element)
        generated = controller_parameters(document)
        ros_params = generated["controller_manager"]["ros__parameters"]
        update_rate = max(update_rate, int(ros_params["update_rate"]))
        # Per module, broadcaster included: a module whose servos are missing
        # stays down without keeping the others from coming up (a broadcaster
        # over the whole configuration would not activate without it).
        modules = {}
        owner = {c: spec["module"] for c, spec in controllers_of(document).items()}
        for module, spec in module_joints(document).items():
            component = component_name(name, module)
            broadcaster = "{}_joint_state_broadcaster".format(component)
            cm[broadcaster] = {"type": STATE_BROADCASTER}
            params[broadcaster] = {"ros__parameters": {
                "joints": spec["joints"] + spec["gripper"], "interfaces": ["position"]}}
            controllers = [broadcaster]
            for controller, of in owner.items():
                if of != module:
                    continue
                renamed = "{}_{}".format(name, controller)
                cm[renamed] = {"type": ros_params[controller]["type"]}
                params[renamed] = generated[controller]
                controllers.append(renamed)
            modules[component] = {"joints": spec["joints"] + spec["gripper"],
                                  "controllers": controllers}
        manifest_configs[name] = {
            "urdf": own, "segment": segment, "components": components,
            "controllers": [c for m in modules.values() for c in m["controllers"]],
            "modules": modules}
    initial = catalog.default or catalog.names[0]
    cm.update(
        update_rate=update_rate,
        hardware_components_initial_state={
            "shutdown_on_initial_state_failure": False,
            # Every component starts inactive: the configuration manager brings
            # each module up once the servicer runs its configuration and all
            # of its servos answer -- at start-up as after a switch.
            "inactive": [c for spec in manifest_configs.values()
                         for c in spec["components"]]})
    controllers_yaml = os.path.join(out_dir, "controllers.yaml")
    with open(controllers_yaml, "w") as f:
        yaml.safe_dump({"controller_manager": {"ros__parameters": cm}, **params}, f,
                       sort_keys=False)
    urdf = os.path.join(out_dir, "robot.urdf")
    with open(urdf, "w") as f:
        f.write(ET.tostring(superset, encoding="unicode"))
    manifest = {"urdf": urdf, "controllers_yaml": controllers_yaml, "initial": initial,
                "configurations": manifest_configs}
    with open(os.path.join(out_dir, "manifest.yaml"), "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    return manifest


# -- MoveIt files ---------------------------------------------------------------

#: The IK plugin each planning group gets.
KINEMATICS_SOLVER = "kdl_kinematics_plugin/KDLKinematicsPlugin"
#: rad/s^2 a joint is planned to accelerate at when nothing says otherwise. URDF
#: has no acceleration limit, and MoveIt's time parameterization needs one.
DEFAULT_MAX_ACCELERATION = 5.0


def joint_limits(model_xml: str, max_acceleration: float) -> Dict[str, Dict[str, object]]:
    """MoveIt's ``joint_limits`` for the URDF's movable joints: its velocity
    limit, and ``max_acceleration``."""
    import xml.etree.ElementTree as ET

    out = {}
    for joint in ET.fromstring(model_xml).findall("joint"):
        if joint.get("type") in ("fixed", "floating", "planar"):
            continue
        limit = joint.find("limit")
        velocity = None if limit is None else limit.get("velocity")
        out[joint.get("name")] = {
            "has_velocity_limits": velocity is not None,
            "max_velocity": float(velocity) if velocity is not None else 0.0,
            "has_acceleration_limits": True,
            "max_acceleration": float(max_acceleration)}
    return out


def srdf(model_xml: str, modules: Dict[str, List[str]]) -> str:
    """A semantic description: one planning group per module, adjacent links
    never checked against each other.

    Parameters
    ----------
    model_xml : str
        The configuration's URDF.
    modules : dict
        Module name -> its joints, in order.
    """
    import xml.etree.ElementTree as ET

    model = ET.fromstring(model_xml)
    root = ET.Element("robot", name=model.get("name", "robot"))
    for module, joints in modules.items():
        group = ET.SubElement(root, "group", name=module)
        for joint in joints:
            ET.SubElement(group, "joint", name=joint)
    for joint in model.findall("joint"):
        ET.SubElement(root, "disable_collisions", link1=joint.find("parent").get("link"),
                      link2=joint.find("child").get("link"), reason="Adjacent")
    return ET.tostring(root, encoding="unicode")


def moveit_files(catalog_path: str, out_dir: str, manifest: Dict[str, object],
                 max_acceleration: float = DEFAULT_MAX_ACCELERATION) -> Dict[str, object]:
    """Write each configuration's move_group parameters; add them to ``manifest``.

    MoveIt reads a robot model once, at start, so a modular robot gets one
    move_group parameter file per configuration, and the running move_group
    is replaced when the configuration changes. Each file holds the
    configuration's model and :func:`srdf`, a KDL solver per group, and its
    controllers for ``moveit_simple_controller_manager`` -- which MoveIt does
    not switch (``moveit_manage_controllers: false``): the configuration
    manager does. Planning pipelines are left to whoever starts move_group
    (they live in MoveIt's own share directory). Joint limits are the URDF's
    velocity limits and ``max_acceleration`` [rad/s^2] for every joint: MoveIt's
    time parameterization fails a plan without an acceleration limit.

    Returns
    -------
    dict
        ``manifest``, with ``configurations[name]["moveit"]`` (the parameter
        file) and ``["groups"]`` added, and rewritten in ``out_dir``.
    """
    from actuo.drivers.config import load_document
    from actuo.drivers.robot_document import controllers_of, module_joints

    catalog = Catalog(catalog_path)
    for name, config in catalog.configurations.items():
        spec = manifest["configurations"][name]
        document = load_document(config.robot)
        with open(spec["urdf"]) as f:
            model = f.read()
        groups = {m: s["joints"] + s["gripper"] for m, s in module_joints(document).items()}
        semantic = srdf(model, groups)
        with open(os.path.join(out_dir, name + ".srdf"), "w") as f:
            f.write(semantic)
        controllers = {"{}_{}".format(name, c): s for c, s in controllers_of(document).items()}
        params = {
            "robot_description": model,
            "robot_description_semantic": semantic,
            "robot_description_planning": {
                "joint_limits": joint_limits(model, max_acceleration)},
            "robot_description_kinematics": {
                g: {"kinematics_solver": KINEMATICS_SOLVER,
                    "kinematics_solver_search_resolution": 0.005,
                    "kinematics_solver_timeout": 0.05} for g in groups},
            "moveit_manage_controllers": False,
            "moveit_controller_manager":
                "moveit_simple_controller_manager/MoveItSimpleControllerManager",
            "moveit_simple_controller_manager": {
                "controller_names": list(controllers),
                **{c: {"type": "FollowJointTrajectory", "action_ns": "follow_joint_trajectory",
                       "default": True, "joints": s["joints"]}
                   for c, s in controllers.items()}},
            "trajectory_execution": {"allowed_execution_duration_scaling": 1.2,
                                     "allowed_goal_duration_margin": 0.5,
                                     "allowed_start_tolerance": 0.05},
            "publish_robot_description_semantic": True,
        }
        path = os.path.join(out_dir, name + "_move_group.yaml")
        with open(path, "w") as f:
            yaml.safe_dump({"move_group": {"ros__parameters": params}}, f, sort_keys=False)
        spec["moveit"] = path
        spec["groups"] = groups
    with open(os.path.join(out_dir, "manifest.yaml"), "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    return manifest
