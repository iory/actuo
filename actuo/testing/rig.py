"""A test rig: a robot config, its model, and a fault injector in its servo line.

A rig file names the robot and says where each joint's servo is plugged into
the injector, so a test can say "unplug the elbow" and mean it on any rig::

    robot: four.yaml          # actuo robot config (bus.hot_plug: true)
    urdf: four.urdf           # its model
    modes: [tool_change.yaml] # robot_modes descriptions to check (optional)
    modular: catalog.yaml     # actuo.modular catalog to check (optional)
    injector:
      plugin: servo_fault_injector   # or ics_fault_relay
      port: /dev/serial/by-id/...    # optional when there is only one
      unplug_by: power
    targets: {j1: 1, j2: 2, j3: 3, j4: 4}   # joint -> port (or ICS id)
    drop_joints: [j1, j2, j3] # where drop_replies works (optional, default all)

Relative paths are taken from the rig file's directory.
"""

from __future__ import annotations

import itertools
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import yaml

from actuo.testing.fault_injectors import (
    UNPLUG_MANY,
    FaultInjector,
    FaultInjectorError,
    make_injector,
)

#: Seconds to wait for the attached set to follow a change on the line.
DEFAULT_SETTLE_TIMEOUT = 5.0
POLL_S = 0.05


@dataclass
class ServoRig:
    """A fault injector plus which target each joint's servo is behind.

    Parameters
    ----------
    injector : FaultInjector
        The hardware in the servo line.
    targets : dict
        Joint name -> injector target (a port, or an ICS id).
    servo_ids : dict
        Joint name -> servo id, for :meth:`drop_replies` (from the robot
        config; see :meth:`from_file`).
    drop_joints : list of str
        Joints whose replies the injector can cut. A rig with a known fault
        (a port whose signal switch does not open) leaves those out, so the
        tests skip them instead of reporting the same broken port each run.
        Default: every joint.
    """

    injector: FaultInjector
    targets: Dict[str, int]
    servo_ids: Dict[str, int] = field(default_factory=dict)
    robot_config: Optional[str] = None
    urdf: Optional[str] = None
    modes: List[str] = field(default_factory=list)
    drop_joints: Optional[List[str]] = None
    modular: Optional[str] = None
    _unplugged: set = field(default_factory=set, init=False, repr=False)

    @classmethod
    def from_file(cls, path: str) -> "ServoRig":
        """Load a rig file (see the module docstring) and open its injector."""
        with open(path) as f:
            spec = yaml.safe_load(f) or {}
        here = os.path.dirname(os.path.abspath(path))

        def resolve(p: Optional[str]) -> Optional[str]:
            return None if p is None else os.path.join(here, p)

        robot_config = resolve(spec.get("robot"))
        servo_ids: Dict[str, int] = {}
        if robot_config is not None:
            with open(robot_config) as f:
                robot = yaml.safe_load(f) or {}
            for joint in robot.get("joints") or ():
                actuator = joint.get("actuator") or {}
                if "id" in actuator:
                    servo_ids[str(joint["name"])] = int(actuator["id"])
        targets = {str(k): int(v) for k, v in (spec.get("targets") or {}).items()}
        if not targets:
            raise ValueError("{}: targets (joint -> port or id) is required".format(path))
        return cls(injector=make_injector(spec.get("injector") or {}),
                   targets=targets, servo_ids=servo_ids,
                   robot_config=robot_config, urdf=resolve(spec.get("urdf")),
                   modes=[resolve(m) for m in spec.get("modes") or ()],
                   drop_joints=(None if spec.get("drop_joints") is None
                                else [str(j) for j in spec["drop_joints"]]),
                   modular=resolve(spec.get("modular")))

    def __post_init__(self) -> None:
        if self.drop_joints is None:
            self.drop_joints = list(self.targets)

    # -- line ------------------------------------------------------------------

    @property
    def joints(self) -> List[str]:
        return list(self.targets)

    def _target(self, joint: str) -> int:
        if joint not in self.targets:
            raise KeyError("joint {} is not on this rig ({})".format(
                joint, ", ".join(self.targets)))
        return self.targets[joint]

    def unplug(self, *joints: str) -> None:
        """Make these joints' servos vanish."""
        for joint in joints:
            self.injector.unplug(self._target(joint))
            self._unplugged.add(joint)

    def plug(self, *joints: str) -> None:
        """Bring these joints' servos back."""
        for joint in joints:
            self.injector.plug(self._target(joint))
            self._unplugged.discard(joint)

    def only(self, joints: Iterable[str]) -> None:
        """Leave exactly these joints plugged in, every other one unplugged.

        Raises
        ------
        FaultInjectorError
            More than one joint would be unplugged and the injector can only
            unplug one at a time.
        """
        wanted = set(joints)
        unknown = wanted - set(self.targets)
        if unknown:
            raise KeyError("not on this rig: {}".format(", ".join(sorted(unknown))))
        off = [j for j in self.targets if j not in wanted]
        if len(off) > 1:
            self.injector.require(UNPLUG_MANY)
        self.plug(*[j for j in self.targets if j in wanted and j in self._unplugged])
        self.unplug(*[j for j in off if j not in self._unplugged])

    def only_ids(self, ids: Iterable[int]) -> None:
        """Leave exactly the servos with these ids plugged in (by the robot
        config's ids), whichever joint names a configuration gives them."""
        by_id = {i: j for j, i in self.servo_ids.items() if j in self.targets}
        unknown = set(ids) - set(by_id)
        if unknown:
            raise KeyError("ids not on this rig: {}".format(sorted(unknown)))
        self.only([by_id[i] for i in ids])

    def drop_replies(self, joint: str) -> None:
        """The joint's servo still gets commands but its replies are cut."""
        if joint not in self.servo_ids:
            raise KeyError("no servo id for {} (give the rig a robot config)".format(joint))
        self.injector.drop_replies(self._target(joint), self.servo_ids[joint])

    def restore(self) -> None:
        """Every servo plugged and answering."""
        self.injector.restore()
        self._unplugged.clear()

    def close(self) -> None:
        self.injector.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            self.restore()
        finally:
            self.close()

    # -- what the software sees ------------------------------------------------

    def combinations(self, joints: Optional[Sequence[str]] = None) -> Iterator[Tuple[str, ...]]:
        """Every subset of ``joints`` (default: all on the rig), smallest first."""
        joints = list(joints or self.targets)
        for size in range(len(joints) + 1):
            yield from itertools.combinations(joints, size)


def wait_attached(ri, expected: Iterable[str],
                  timeout: float = DEFAULT_SETTLE_TIMEOUT) -> Tuple[bool, List[str], float]:
    """Wait until ``ri.attached_joints()`` is exactly ``expected``.

    Parameters
    ----------
    ri : ActuoRobotInterface
        The interface on the rig's bus (``hot_plug``).
    expected : iterable of str
        Joint names.
    timeout : float
        Seconds.

    Returns
    -------
    tuple
        ``(reached, attached_now, seconds_waited)``.
    """
    want = sorted(expected)
    t0 = time.monotonic()
    while True:
        got = sorted(ri.attached_joints())
        waited = time.monotonic() - t0
        if got == want:
            return True, got, waited
        if waited >= timeout:
            return False, got, waited
        time.sleep(POLL_S)


__all__ = ("DEFAULT_SETTLE_TIMEOUT", "FaultInjectorError", "ServoRig", "wait_attached")
