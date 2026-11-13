"""Which robot is plugged together right now, and what its model is.

A robot here is a fixed **base** plus **slots**. A slot is a place modules
attach (a link on the base, or on another module), and each slot lists the
**candidates** that can be attached there. A module is one ros2_control
hardware component: a set of joints, the URDF fragment they live in, and the
controllers that command them.

One description covers three ways of running:

``fixed``
    The model never changes. Every module must be there; a missing one is an
    error (a ROS 2 layer above can flag its links in RViz), and it
    rejoins when it answers again.
``tool_change``
    The base is required and fixed; a slot holds whichever of its candidates
    is plugged in, and the model follows.
``reconfigurable``
    Nothing is required; every slot follows what is plugged in.

Which candidate is plugged in is decided by which joints answer: a candidate
is attached when all its joints are present. ICS cannot tell where a servo is
plugged, so two candidates that could be attached at once must have different
servo ids -- that is the ID convention this relies on. Candidates of one robot
never share a joint: ros2_control cannot give one command interface to two
hardware components.

The controller_manager is given :meth:`RobotModes.superset_urdf` once, with
every candidate in it (it ignores robot descriptions after the first, so the
superset is the only description that works across changes); the
robot_state_publisher is given :meth:`RobotModes.compose_urdf` of what is
attached, and again whenever that changes.
"""

from __future__ import annotations

import copy
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import yaml

MODES = ("fixed", "tool_change", "reconfigurable")


@dataclass(frozen=True)
class Module:
    """One hardware component: its joints, URDF fragment and controllers."""

    name: str                       # the <ros2_control name=...> of its component
    joints: Tuple[str, ...]
    controllers: Tuple[str, ...]
    urdf: str                       # path of the URDF fragment
    root_link: str                  # fragment link that attaches to the parent


@dataclass(frozen=True)
class Slot:
    name: str
    parent_link: str
    xyz: Tuple[float, float, float]
    rpy: Tuple[float, float, float]
    required: bool
    candidates: Tuple[Module, ...]


@dataclass
class Selection:
    """What is attached, slot by slot, and what is wrong."""

    attached: Dict[str, Optional[str]] = field(default_factory=dict)  # slot -> module
    problems: List[str] = field(default_factory=list)                 # errors
    missing_base: List[str] = field(default_factory=list)             # base modules


def _vec(value, default) -> Tuple[float, float, float]:
    if value is None:
        return default
    out = tuple(float(v) for v in value)
    if len(out) != 3:
        raise ValueError("expected three numbers, got {!r}".format(value))
    return out


def _root_link(fragment: ET.Element, path: str) -> str:
    children = {j.find("child").get("link") for j in fragment.findall("joint")}
    roots = [link.get("name") for link in fragment.findall("link")
             if link.get("name") not in children]
    if len(roots) != 1:
        raise ValueError("{}: a module fragment needs exactly one root link, "
                         "found {}".format(path, roots))
    return roots[0]


class RobotModes:
    """A robot description loaded from a modes YAML.

    Parameters
    ----------
    path : str
        The YAML. Relative URDF paths in it are taken from its directory.
    """

    def __init__(self, path: str) -> None:
        with open(path) as f:
            spec = yaml.safe_load(f) or {}
        here = os.path.dirname(os.path.abspath(path))
        self.path = path
        self.mode = str(spec.get("mode", "fixed"))
        if self.mode not in MODES:
            raise ValueError("{}: mode {!r} is not one of {}".format(
                path, self.mode, ", ".join(MODES)))
        base = spec.get("base") or {}
        if "urdf" not in base:
            raise ValueError("{}: base.urdf is required".format(path))
        self.base_urdf = os.path.join(here, base["urdf"])
        self._base_tree = ET.parse(self.base_urdf).getroot()
        self.base_modules = tuple(
            self._module(name, m, here, fragment=False)
            for name, m in (base.get("modules") or {}).items())
        self.slots = tuple(self._slot(name, s, here)
                           for name, s in (spec.get("slots") or {}).items())
        self._validate()

    # -- loading -----------------------------------------------------------

    def _module(self, name, m, here, fragment=True) -> Module:
        joints = tuple(m.get("joints") or ())
        if not joints:
            raise ValueError("{}: module {} has no joints".format(self.path, name))
        urdf, root = "", ""
        if fragment:
            if "urdf" not in m:
                raise ValueError("{}: module {} needs a urdf fragment".format(
                    self.path, name))
            urdf = os.path.join(here, m["urdf"])
            root = _root_link(ET.parse(urdf).getroot(), urdf)
        return Module(name=name, joints=joints,
                      controllers=tuple(m.get("controllers") or ()),
                      urdf=urdf, root_link=root)

    def _slot(self, name, s, here) -> Slot:
        candidates = tuple(self._module(n, m, here)
                           for n, m in (s.get("candidates") or {}).items())
        if not candidates:
            raise ValueError("{}: slot {} has no candidates".format(self.path, name))
        required = bool(s.get("required", self.mode == "fixed"))
        if self.mode == "fixed" and (len(candidates) != 1 or not required):
            raise ValueError(
                "{}: slot {} -- a fixed robot has exactly one, required, "
                "module per slot".format(self.path, name))
        return Slot(name=name, parent_link=str(s["parent_link"]),
                    xyz=_vec(s.get("xyz"), (0.0, 0.0, 0.0)),
                    rpy=_vec(s.get("rpy"), (0.0, 0.0, 0.0)),
                    required=required, candidates=candidates)

    def _validate(self) -> None:
        owner: Dict[str, str] = {}
        for module in self.modules():
            for joint in module.joints:
                if joint in owner:
                    raise ValueError(
                        "{}: joint {} is in both {} and {}; ros2_control cannot "
                        "give one joint to two hardware components".format(
                            self.path, joint, owner[joint], module.name))
                owner[joint] = module.name
        names = [m.name for m in self.modules()]
        if len(set(names)) != len(names):
            raise ValueError("{}: module names repeat: {}".format(self.path, names))

    # -- queries -----------------------------------------------------------

    def modules(self) -> List[Module]:
        """Every module, base first."""
        return list(self.base_modules) + [c for s in self.slots for c in s.candidates]

    def module(self, name: str) -> Module:
        for m in self.modules():
            if m.name == name:
                return m
        raise KeyError(name)

    def select(self, present: Set[str]) -> Selection:
        """Decide what is attached from the set of joints that answer.

        A candidate is attached when every one of its joints answers. Two
        candidates of one slot both complete is a conflict: neither is taken,
        because driving the wrong one's controllers is worse than driving none.
        """
        out = Selection()
        for module in self.base_modules:
            if not set(module.joints) <= present:
                out.missing_base.append(module.name)
                if self.mode != "reconfigurable":
                    out.problems.append("{} is missing ({} not answering)".format(
                        module.name, ", ".join(sorted(set(module.joints) - present))))
        for slot in self.slots:
            complete = [c.name for c in slot.candidates if set(c.joints) <= present]
            if len(complete) > 1:
                out.attached[slot.name] = None
                out.problems.append(
                    "slot {}: {} all answer; only one can be attached".format(
                        slot.name, " and ".join(complete)))
            else:
                out.attached[slot.name] = complete[0] if complete else None
                if not complete and slot.required:
                    out.problems.append("slot {}: {} is missing".format(
                        slot.name, " or ".join(c.name for c in slot.candidates)))
        return out

    def running(self, selection: Selection) -> List[Module]:
        """Modules whose controllers should run under ``selection``."""
        names = [m.name for m in self.base_modules if m.name not in selection.missing_base]
        names += [n for n in selection.attached.values() if n]
        return [self.module(n) for n in names]

    def static_selection(self) -> Dict[str, Optional[str]]:
        """A fixed robot's one and only selection."""
        return {s.name: s.candidates[0].name for s in self.slots}

    # -- URDF --------------------------------------------------------------

    def superset_urdf(self) -> str:
        """Base plus every candidate, each attached at its slot: for the
        controller_manager, which loads one description and keeps it."""
        return self._compose({s.name: [c.name for c in s.candidates] for s in self.slots})

    def compose_urdf(self, attached: Dict[str, Optional[str]]) -> str:
        """Base plus what is attached: for the robot_state_publisher."""
        return self._compose({slot: [name] if name else [] for slot, name in attached.items()})

    def _compose(self, chosen: Dict[str, Sequence[str]]) -> str:
        robot = copy.deepcopy(self._base_tree)
        links = {link.get("name") for link in robot.findall("link")}
        for slot in self.slots:
            for name in chosen.get(slot.name, []):
                module = self.module(name)
                if slot.parent_link not in links:
                    raise ValueError(
                        "slot {}: parent link {} is not in the model (is the "
                        "module it belongs to attached?)".format(slot.name, slot.parent_link))
                fragment = ET.parse(module.urdf).getroot()
                for element in fragment:
                    robot.append(copy.deepcopy(element))
                    if element.tag == "link":
                        links.add(element.get("name"))
                mount = ET.SubElement(robot, "joint", name="{}_mount".format(module.name),
                                      type="fixed")
                ET.SubElement(mount, "parent", link=slot.parent_link)
                ET.SubElement(mount, "child", link=module.root_link)
                ET.SubElement(mount, "origin", xyz=" ".join(map(str, slot.xyz)),
                              rpy=" ".join(map(str, slot.rpy)))
        return ET.tostring(robot, encoding="unicode")
