"""Self-collision monitor.

The same capability exists upstream in scikit-robot as
``skrobot.collision.SelfCollisionMonitor``. This copy does not depend on the
installed scikit-robot version at import time: the class only calls
``robot_model.self_collision_check()`` at runtime, so importing it needs numpy
only. The distance and fast boolean paths use trimesh's collision managers,
which need python-fcl (the ``safety`` extra).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Set, Tuple

import numpy as np

Pair = Tuple[str, str]


class SelfCollisionMonitor:
    def __init__(
        self,
        robot_model,
        reference_angle_vector: Optional[Sequence[float]] = None,
        extra_ignore_pairs: Optional[Sequence[Pair]] = None,
    ) -> None:
        self.robot = robot_model
        self.joint_names: List[str] = [j.name for j in robot_model.joint_list]

        if reference_angle_vector is None:
            reference = np.zeros(len(self.joint_names))
        else:
            reference = np.asarray(reference_angle_vector, dtype=float)
        # Pairs colliding in the reference pose are treated as structural
        # (adjacent / permanent overlap) and ignored forever after.
        _, base_pairs = self._check_at(reference)
        self._ignore: Set[frozenset] = {frozenset(p) for p in base_pairs}
        if extra_ignore_pairs:
            self._ignore |= {frozenset(p) for p in extra_ignore_pairs}

    @property
    def ignored_pairs(self) -> Set[frozenset]:
        return set(self._ignore)

    def _check_at(self, angle_vector: np.ndarray):
        self.robot.angle_vector(np.asarray(angle_vector, dtype=float).copy())
        return self.robot.self_collision_check()

    def in_collision(self) -> Tuple[bool, List[Pair]]:
        """Check the robot's CURRENT configuration. Returns (is_colliding, pairs)
        with structural/adjacent pairs removed."""
        _, pairs = self.robot.self_collision_check()
        real = [tuple(p) for p in pairs if frozenset(p) not in self._ignore]
        return (len(real) > 0), real

    def check_angle_vector(self, angle_vector: Sequence[float]) -> Tuple[bool, List[Pair]]:
        """Set the full angle vector, then report real self-collisions."""
        self.robot.angle_vector(np.asarray(angle_vector, dtype=float).copy())
        return self.in_collision()

    def index_of(self, joint_name: str) -> int:
        return self.joint_names.index(joint_name)

    # --- distance queries (for graceful, distance-based deceleration) ---

    def enable_distance(self, groups) -> None:
        """Prepare per-group min-distance queries. ``groups`` is either a list of
        moving link names (one unnamed group) or a dict ``{name: [links]}``.

        For each group the distance is computed between the group's links and ALL
        OTHER links, so a group only slows down for what IS near IT -- decoupling
        e.g. the two arms. Within-group adjacencies and the structural boundary
        links across the split (a shoulder root permanently touching the torso)
        are excluded, mirroring the boolean ignore set at the group level.
        """
        import trimesh

        if not isinstance(groups, dict):
            groups = {"default": list(groups)}

        all_links = [ln for ln in self.robot.link_list
                     if getattr(ln, "collision_mesh", None) is not None
                     and len(getattr(ln.collision_mesh, "faces", [])) > 0]

        self._groups = {}
        for name, moving_names in groups.items():
            moving = {ln.name for ln in all_links} & set(moving_names)
            boundary = set()
            for pair in self._ignore:
                members = tuple(pair)
                if len(members) != 2:
                    continue
                a, b = members
                if (a in moving) != (b in moving):
                    boundary.add(a if a in moving else b)

            mgr_move = trimesh.collision.CollisionManager()
            mgr_rest = trimesh.collision.CollisionManager()
            links = []
            for link in all_links:
                if link.name in boundary:
                    continue
                mgr = mgr_move if link.name in moving else mgr_rest
                mgr.add_object(link.name, link.collision_mesh)
                links.append((link, link.name, link.name in moving))
            self._groups[name] = (mgr_move, mgr_rest, links)

    # --- the fast boolean path (what a 100 Hz gate must use) ---

    def in_collision_fast(self, group=None,
                          angle_vector: Optional[Sequence[float]] = None) -> bool:
        """Is ``group`` colliding with the rest? Boolean only, no pair names.

        This is the check a servicer can afford. Measured on a humanoid (27 collision
        links, 76k triangles after CoACD, one 10 ms control cycle):

            all-pairs check, WITH names      114 ms
            all-pairs check, without names     0.0 ms
            this call (group vs rest)          0.22 ms
            min_distance for the same group    3.87 ms

        The entire cost of the ordinary check is ``return_names=True``: fcl
        collects every contact point of every colliding pair. A gate does not
        need them to decide to stop -- it needs a boolean, and the names belong
        in the message it writes *after* the robot is already stopped.

        Names cannot simply be dropped from the all-pairs check, though, because
        the ignore set is applied to them: without names there is no way to tell
        a structural touch (a shoulder resting on its torso mount) from a real
        collision, and the check would report True forever. That is why this
        goes through the group managers, which EXCLUDE the boundary links
        instead of filtering afterwards.

        Requires :meth:`enable_distance` (it builds the same managers).
        """
        if not hasattr(self, "_groups"):
            raise RuntimeError(
                "call enable_distance() before in_collision_fast(): the fast "
                "path needs the per-group managers, which is also what makes "
                "it able to skip the structural pairs.")
        if angle_vector is not None:
            self.robot.angle_vector(np.asarray(angle_vector, dtype=float).copy())
        if group is None:
            group = next(iter(self._groups))
        mgr_move, mgr_rest, links = self._groups[group]
        for link, name, is_move in links:
            transform = link.worldcoords().T()
            (mgr_move if is_move else mgr_rest).set_transform(name, transform)
        return bool(mgr_move.in_collision_other(mgr_rest, return_names=False))

    def check_angle_vector_fast(
            self, angle_vector: Sequence[float]) -> Tuple[bool, List[Pair]]:
        """Same answer as :meth:`check_angle_vector`, at a fraction of the cost.

        Every enabled group is tested with the boolean path; only if one of them
        reports a hit is the expensive named check run, to say WHAT is touching.
        So the common case (no collision) costs a few hundred microseconds and
        the rare case pays for its own diagnosis.

        Falls back to the plain check when no groups are enabled, rather than
        refusing: a caller that has not configured groups still deserves a
        correct answer, just a slow one.
        """
        self.robot.angle_vector(np.asarray(angle_vector, dtype=float).copy())
        if not hasattr(self, "_groups"):
            return self.in_collision()
        for group in self._groups:
            if self.in_collision_fast(group):
                return self.in_collision()
        return False, []

    def min_distance(self, group=None,
                     angle_vector: Optional[Sequence[float]] = None) -> float:
        """Minimum distance [m] between ``group``'s links and the rest at the given
        configuration (default group when unnamed). Requires
        :meth:`enable_distance`. Zero/negative means contact/penetration."""
        if not hasattr(self, "_groups"):
            raise RuntimeError("call enable_distance() before min_distance()")
        if angle_vector is not None:
            self.robot.angle_vector(np.asarray(angle_vector, dtype=float).copy())
        if group is None:
            group = next(iter(self._groups))
        mgr_move, mgr_rest, links = self._groups[group]
        for link, name, is_move in links:
            T = link.worldcoords().T()
            (mgr_move if is_move else mgr_rest).set_transform(name, T)
        return float(mgr_move.min_distance_other(mgr_rest))
