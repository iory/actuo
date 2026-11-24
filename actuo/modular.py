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
still seen coming and going (FEETECH buses; see
:class:`~actuo.drivers.buses.feetech_bus.FeetechBus`).
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
        if not callable(getattr(ri.bus, "answering_ids", None)):
            ri.close()
            raise TypeError(
                "{}: {} cannot report which ids answer outside its own servos; "
                "ModularRobot needs a bus with watch_ids (FeetechBus)".format(
                    name, type(ri.bus).__name__))
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


__all__ = ("Catalog", "Configuration", "ModularRobot", "ModularState")
