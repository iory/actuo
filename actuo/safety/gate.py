"""CollisionGate -- the last software stage before actuation.

Placed in the bus servicer just before commands become motor setpoints, this
is the analog of hrpsys's ``CollisionDetector``: it watches the *commanded*
configuration and, if it would self-collide, holds the last collision-free
command instead of passing the colliding one through. Motion freezes at the last
safe pose rather than driving into a self-collision.

Fail-safe: if the collision check itself raises, the command is treated as
colliding and held (fail to STOP, never fail-open). This lives in
``actuo.safety``, never in ``actuo.core`` -- geometry/collision is a separate
layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

# NOTE: SelfCollisionMonitor / Pair are used only in string annotations (this
# module uses `from __future__ import annotations`), so they are intentionally
# not imported here -- that keeps CollisionGate importable (and unit-testable
# with a fake monitor) without pulling in scikit-robot.
Pair = Tuple[str, str]


@dataclass
class GateResult:
    applied: np.ndarray            # command actually forwarded (subset order)
    stopped: bool                  # True if this cycle was gated
    colliding_pairs: List[Pair] = field(default_factory=list)


class CollisionGate:
    def __init__(
        self,
        monitor: SelfCollisionMonitor,
        joint_names: Sequence[str],
        base_angle_vector: Sequence[float],
        initial_safe: Optional[Sequence[float]] = None,
        latched: bool = True,
        mode: str = "stop",
        moving_links: Optional[Sequence[str]] = None,
        scale_groups: Optional[dict] = None,
        d_stop: float = 0.02,
        d_slow: float = 0.10,
        max_step: float = 0.03,
    ) -> None:
        self.monitor = monitor
        # Prefer the monitor's fast boolean path when it has one. Resolved ONCE
        # here rather than per cycle, and recorded, because which path a gate is
        # on can differ by two orders of magnitude per cycle and must not be a
        # silent property. `check_angle_vector_fast` gives the same verdict --
        # it consults the group managers first and only pays for pair names when
        # one of them reports a hit -- but an upstreamed or duck-typed monitor
        # may not have it, and falling back must be visible rather than
        # mysterious slowness. See `fast_path`.
        self._check = getattr(monitor, "check_angle_vector_fast", None)
        #: Whether this gate got the fast path. False means every cycle pays for
        #: pair names; report it rather than discovering it in a loop-rate graph.
        self.fast_path = self._check is not None
        if self._check is None:
            self._check = monitor.check_angle_vector
        self.joint_names = list(joint_names)
        self._jidx = [monitor.index_of(n) for n in self.joint_names]
        self._base_av = np.asarray(base_angle_vector, dtype=float).copy()
        self._last_safe: Optional[np.ndarray] = (
            None if initial_safe is None else np.asarray(initial_safe, dtype=float).copy()
        )
        # 'scale' = distance-based deceleration (smooth slow-down before contact);
        # 'stop' = boolean hold-last-safe. Scale needs a moving-link group.
        self.mode = mode
        self.d_stop = d_stop
        self.d_slow = d_slow
        self.max_step = max_step
        self._applied = (np.asarray(initial_safe, dtype=float).copy()
                         if initial_safe is not None else None)
        # per-joint group name (for grouped deceleration); None = no scaling
        self._joint_group = [None] * len(self.joint_names)
        self._group_names = []
        if mode == "scale":
            if scale_groups is not None:
                # {name: {"joints": [...], "links": [...]}} -- each group decelerates
                # only its own joints, based only on its own links' proximity.
                monitor.enable_distance(
                    {n: g["links"] for n, g in scale_groups.items()})
                self._group_names = list(scale_groups.keys())
                for gname, g in scale_groups.items():
                    for jn in g["joints"]:
                        if jn in self.joint_names:
                            self._joint_group[self.joint_names.index(jn)] = gname
            elif moving_links is not None:
                monitor.enable_distance(list(moving_links))
                self._group_names = ["default"]
                self._joint_group = ["default"] * len(self.joint_names)
            else:
                raise ValueError("mode='scale' requires scale_groups or moving_links")
        self.distance = float("inf")
        self.distances = {}
        # Latched (default): once a collision is detected, stay stopped at the last
        # safe pose until reset(). This is the conservative hrpsys-like behaviour --
        # auto-resuming when a later command happens to be collision-free is unsafe,
        # because the *path* back to it may still sweep through the collision zone.
        self.latched = latched
        self.stopped = False
        self._latch = False
        self._last_pairs: List[Pair] = []

    def _full_av(self, subset: np.ndarray) -> np.ndarray:
        av = self._base_av.copy()
        for k, i in enumerate(self._jidx):
            av[i] = subset[k]
        return av

    def filter(self, command: Sequence[float]) -> GateResult:
        cmd = np.asarray(command, dtype=float)
        if self.mode == "scale":
            return self._filter_scale(cmd)

        if self.latched and self._latch:
            # already latched-stopped: keep holding the last safe pose.
            applied = self._last_safe if self._last_safe is not None else cmd
            self.stopped = True
            return GateResult(applied=np.asarray(applied, dtype=float).copy(),
                              stopped=True, colliding_pairs=self._last_pairs)

        try:
            hit, pairs = self._check(self._full_av(cmd))
        except Exception:  # fail-safe: any checker failure -> treat as collision
            hit, pairs = True, [("<check-error>", "<check-error>")]

        if hit:
            self.stopped = True
            self._latch = True
            self._last_pairs = pairs
            # retreat to last safe; if we never had one, hold the incoming command
            # (cannot invent a safe pose) and report the stop.
            applied = self._last_safe if self._last_safe is not None else cmd
            return GateResult(applied=np.asarray(applied, dtype=float).copy(),
                              stopped=True, colliding_pairs=pairs)

        self._last_safe = cmd.copy()
        self.stopped = False
        return GateResult(applied=cmd.copy(), stopped=False, colliding_pairs=[])

    def _filter_scale(self, cmd: np.ndarray) -> GateResult:
        """Distance-based deceleration, per group: each joint's per-cycle step is
        limited by ITS group's proximity, so one arm nearing the body doesn't slow
        the other. Full speed far away, slowing through [d_stop, d_slow], zero at
        d_stop. A boolean check is the backstop for anything distance doesn't cover."""
        if self._applied is None:
            self._applied = cmd.copy()

        # evaluate every group's distance at the current applied pose (set once)
        self.monitor.robot.angle_vector(self._full_av(self._applied).copy())
        span = max(self.d_slow - self.d_stop, 1e-6)
        scale_by_group = {}
        for g in self._group_names:
            try:
                d = self.monitor.min_distance(g, angle_vector=None)
            except Exception:
                d = -1.0  # fail-safe: treat as contact
            self.distances[g] = d
            scale_by_group[g] = float(np.clip((d - self.d_stop) / span, 0.0, 1.0))

        max_delta = np.full(len(self.joint_names), self.max_step)
        for k, gname in enumerate(self._joint_group):
            if gname is not None:
                max_delta[k] = self.max_step * scale_by_group.get(gname, 1.0)
        candidate = self._applied + np.clip(cmd - self._applied, -max_delta, max_delta)

        try:
            hit, pairs = self._check(self._full_av(candidate))
        except Exception:
            hit, pairs = True, [("<check-error>", "<check-error>")]
        if hit:
            self.stopped = True
            return GateResult(applied=self._applied.copy(), stopped=True,
                              colliding_pairs=pairs)

        self._applied = candidate
        self.stopped = any(s <= 0.0 for s in scale_by_group.values())
        self.distance = min(self.distances.values()) if self.distances else float("inf")
        return GateResult(applied=candidate.copy(), stopped=self.stopped,
                          colliding_pairs=[])

    def reset(self, safe: Sequence[float]) -> None:
        """Clear the latch, adopting ``safe`` as the last collision-free pose.

        ``safe`` is REQUIRED, and passing the pose the gate is currently holding
        is the only correct thing to pass. It used to default to None, which set
        ``_last_safe = None`` -- and look at what :meth:`filter` does with a
        collision when it has no safe pose to retreat to: it applies the
        INCOMING command, because it cannot invent one. That fallback is right
        for a gate that has never seen a safe pose, and a footgun immediately
        after a reset: the next colliding command would be passed through to the
        robot while still being reported as stopped. A release must not be able
        to do that, so there is no longer a way to ask for it.
        """
        if safe is None:
            raise ValueError(
                "reset() needs the pose to adopt as safe -- normally the one "
                "the gate is holding. Without it the next colliding command is "
                "applied rather than blocked (see filter's no-safe-pose "
                "fallback), which is the opposite of releasing a stop.")
        self._last_safe = np.asarray(safe, dtype=float).copy()
        self.stopped = False
        self._latch = False
        self._last_pairs = []
