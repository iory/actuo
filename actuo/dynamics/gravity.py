"""Static gravity torques of a URDF robot, from scikit-robot's inverse dynamics.

Needs the ``dynamics`` extra (scikit-robot >= 0.4.0: its inverse dynamics
returns the torque a joint must exert, and a zero-mass link weighs nothing --
https://github.com/iory/scikit-robot/pull/912).

Links that are on the URDF but not on the robot (a gripper not fitted yet) are
dropped with everything below them, so the model weighs what is really there.
A moving link with no ``<inertial>`` is refused: scikit-robot would give it
1 g, and a model that silently underweighs the arm is exactly the plausible but
wrong number this package does not produce.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Dict, Sequence

import numpy as np


def mounted_inertia_only(urdf_text: str, unmounted_links: Sequence[str]) -> str:
    """The URDF as fitted, without visual and collision geometry.

    Parameters
    ----------
    urdf_text : str
        URDF XML.
    unmounted_links : sequence of str
        Links that are not on the robot; each goes with everything below it.

    Returns
    -------
    str
        URDF XML.

    Raises
    ------
    ValueError
        If an unmounted link is not in the URDF, or a joint that moves moves a
        link with no ``<inertial>``.
    """
    root = ET.fromstring(urdf_text)
    names = {link.get("name") for link in root.findall("link")}
    unknown = set(unmounted_links) - names
    if unknown:
        raise ValueError("unmounted_links not in the URDF: {}".format(sorted(unknown)))
    children: Dict[str, list] = {}
    for joint in root.findall("joint"):
        children.setdefault(joint.find("parent").get("link"), []).append(
            joint.find("child").get("link"))
    dropped = set()
    stack = list(unmounted_links)
    while stack:
        link = stack.pop()
        dropped.add(link)
        stack += children.get(link, [])
    for joint in root.findall("joint"):
        if joint.find("child").get("link") in dropped:
            root.remove(joint)
    for link in root.findall("link"):
        if link.get("name") in dropped:
            root.remove(link)
            continue
        for tag in ("visual", "collision"):
            for element in link.findall(tag):
                link.remove(element)
    for joint in root.findall("joint"):
        child = joint.find("child").get("link")
        mass = root.find("link[@name='{}']/inertial/mass".format(child))
        if joint.get("type") != "fixed" and mass is None:
            raise ValueError("{} moves link {}, which has no <inertial>; "
                             "scikit-robot would give it 1 g".format(
                                 joint.get("name"), child))
    return ET.tostring(root, encoding="unicode")


class GravityModel:
    """Joint torques that hold a URDF robot still against gravity.

    Parameters
    ----------
    urdf_text : str
        URDF XML.
    unmounted_links : sequence of str
        Links that are not on the real robot (see :func:`mounted_inertia_only`).
    gravity : sequence of float
        Gravity in the root link's frame [m/s^2].
    backend : str
        scikit-robot backend: ``"numpy"`` (about 3 ms for a 5-joint arm) or
        ``"jax"`` (jitted; needs jax installed).
    """

    def __init__(self, urdf_text: str, unmounted_links: Sequence[str] = (),
                 gravity: Sequence[float] = (0.0, 0.0, -9.81),
                 backend: str = "numpy"):
        from skrobot.backend import get_backend
        from skrobot.dynamics import build_inverse_dynamics_fn
        from skrobot.models.urdf import RobotModelFromURDF

        self.robot = RobotModelFromURDF(
            urdf=mounted_inertia_only(urdf_text, unmounted_links))
        self.names = [joint.name for joint in self.robot.joint_list]
        self.gravity = np.asarray(gravity, dtype=np.float64)
        if self.gravity.shape != (3,):
            raise ValueError("gravity must be 3 numbers, got {}".format(list(gravity)))
        self._fn = build_inverse_dynamics_fn(self.robot,
                                             backend=get_backend(backend))
        # Compile (jax) or warm up now rather than in the first control cycle.
        self.torques({}, self.names[:1])

    def torques(self, angles: Dict[str, float],
                joint_names: Sequence[str]) -> np.ndarray:
        """Torques [N m] the given joints must exert to hold the pose.

        Positive is along the joint axis, the sense of a positive angle.

        Parameters
        ----------
        angles : dict of str to float
            Joint angles [rad]; a joint not given is at 0.
        joint_names : sequence of str
            Joints to report, in this order. A joint dropped with an unmounted
            link carries nothing and reports 0.

        Returns
        -------
        numpy.ndarray
        """
        q = np.array([angles.get(name, 0.0) for name in self.names])
        tau = np.asarray(self._fn(q, gravity=self.gravity))
        return np.array([tau[self.names.index(name)] if name in self.names else 0.0
                         for name in joint_names])
