"""The fast boolean path must give the SAME answer as the named one.

On a 27-link humanoid (76k triangles), per 10 ms control cycle: all-pairs check
with names 114 ms; fast path, collision-free 0.42 ms; fast path, colliding
114 ms (paid once, after the stop).

These tests use fakes rather than a robot: the equivalence is a property of the
control flow (when each checker is consulted), so it is pinned here without
fcl or meshes.
"""

from actuo.safety.monitor import SelfCollisionMonitor
class FakeRobot:
    """Enough of a RobotModel for the monitor: joints, a pose, and a checker."""

    def __init__(self, colliding_pairs_by_pose=None):
        self.joint_list = [type("J", (), {"name": n})()
                           for n in ("j0", "j1")]
        self.link_list = []
        self._pose = (0.0, 0.0)
        self._pairs = colliding_pairs_by_pose or {}
        self.named_checks = 0

    def angle_vector(self, av=None):
        if av is not None:
            self._pose = tuple(float(x) for x in av)
        return list(self._pose)

    def self_collision_check(self):
        self.named_checks += 1
        pairs = self._pairs.get(self._pose, [])
        return (len(pairs) > 0), list(pairs)


class FakeManager:
    """Stands in for a trimesh CollisionManager pair."""

    def __init__(self, hit=False):
        self.hit = hit
        self.calls = 0

    def set_transform(self, name, transform):
        pass

    def in_collision_other(self, other, return_names=False):
        self.calls += 1
        assert return_names is False, \
            "the fast path must never ask for names -- names ARE the cost"
        return self.hit


def _monitor(robot, group_hits=None):
    monitor = SelfCollisionMonitor(robot, reference_angle_vector=[0.0, 0.0])
    if group_hits is not None:
        monitor._groups = {
            name: (FakeManager(hit), FakeManager(), [])
            for name, hit in group_hits.items()
        }
    return monitor


def test_no_collision_never_pays_for_names():
    robot = FakeRobot()
    monitor = _monitor(robot, {"rarm": False, "larm": False})
    before = robot.named_checks

    hit, pairs = monitor.check_angle_vector_fast([1.0, 2.0])

    assert hit is False and pairs == []
    assert robot.named_checks == before, \
        "the every-cycle case must not run the 114 ms named check"


def test_a_hit_falls_back_to_the_named_check_to_say_what_touched():
    """Stopping needs a boolean; the operator needs the pair. Both, in order."""
    robot = FakeRobot({(1.0, 2.0): [("hand", "head")]})
    monitor = _monitor(robot, {"rarm": True})
    before = robot.named_checks

    hit, pairs = monitor.check_angle_vector_fast([1.0, 2.0])

    assert hit is True
    assert pairs == [("hand", "head")]
    assert robot.named_checks == before + 1


def test_every_group_is_consulted_until_one_hits():
    robot = FakeRobot()
    monitor = _monitor(robot, {"rarm": False, "larm": False})

    monitor.check_angle_vector_fast([0.0, 0.0])

    assert all(mgr[0].calls == 1 for mgr in monitor._groups.values())


def test_without_groups_it_still_answers_correctly_just_slowly():
    """A caller who never configured groups gets the right answer, not a refusal
    and not a silent False."""
    robot = FakeRobot({(1.0, 2.0): [("a", "b")]})
    monitor = _monitor(robot)

    hit, pairs = monitor.check_angle_vector_fast([1.0, 2.0])

    assert hit is True and pairs == [("a", "b")]


def test_the_structural_ignore_set_still_applies_through_the_fast_path():
    """A shoulder resting on its torso mount collides at the reference pose and
    must stay ignored. This is why the fast path goes through GROUPS (which drop
    the boundary links) rather than just dropping names from the global check."""
    robot = FakeRobot({(0.0, 0.0): [("shoulder", "torso")],
                       (1.0, 1.0): [("shoulder", "torso")]})
    monitor = SelfCollisionMonitor(robot, reference_angle_vector=[0.0, 0.0])
    monitor._groups = {"rarm": (FakeManager(True), FakeManager(), [])}

    hit, pairs = monitor.check_angle_vector_fast([1.0, 1.0])

    assert hit is False and pairs == [], \
        "the pair that collides in the reference pose is structural"


def test_in_collision_fast_refuses_before_groups_are_enabled():
    """Silently answering False would be a gate that never fires."""
    monitor = SelfCollisionMonitor(FakeRobot(), reference_angle_vector=[0.0, 0.0])
    try:
        monitor.in_collision_fast()
    except RuntimeError as error:
        assert "enable_distance" in str(error)
    else:
        raise AssertionError("must refuse, not return False")
