"""AirCommandReceiver: the queue-vs-overwrite behaviour, without a real node.

The class is deliberately thin (every safety decision lives in the ROS-free
layer; this only translates) -- there is no real decision logic here
beyond "the latest command wins", but that property is exactly the one that
matters for safety (a stale queued "grip" must never fire after a more recent
"release"), so it is pinned rather than left to look obviously true.
"""

from actuo_ros.telemetry_node import AirCommandReceiver


class FakeNode:
    """The one method AirCommandReceiver needs from a real rclpy Node."""

    def __init__(self):
        self.subscriptions = []   # (msg_type, topic, callback, qos)

    def create_subscription(self, msg_type, topic, callback, qos):
        sub = (msg_type, topic, callback, qos)
        self.subscriptions.append(sub)
        return sub


class FakeMsg:
    def __init__(self, data):
        self.data = data


def _deliver(node, topic, data):
    """Find the subscription for `topic` and invoke its callback, as DDS would."""
    for _msg_type, sub_topic, callback, _qos in node.subscriptions:
        if sub_topic == topic:
            callback(FakeMsg(data))
            return
    raise AssertionError("no subscription for {!r}".format(topic))


def test_subscribes_one_topic_per_sensor():
    node = FakeNode()
    AirCommandReceiver(node, [("air_board", object()), ("other", object())])
    topics = {t for _, t, _, _ in node.subscriptions}
    assert topics == {"~/air_board/air/command", "~/other/air/command"}


def test_pending_is_empty_until_a_command_arrives():
    node = FakeNode()
    receiver = AirCommandReceiver(node, [("air_board", object())])
    assert receiver.pending() == {}


def test_a_command_is_delivered_exactly_once():
    """pending() DRAINS -- a second call before a new message must be empty,
    or a stale command would re-fire on the servicer's very next cycle."""
    node = FakeNode()
    receiver = AirCommandReceiver(node, [("air_board", object())])
    _deliver(node, "~/air_board/air/command", "grip")
    assert receiver.pending() == {"air_board": "grip"}
    assert receiver.pending() == {}


def test_a_second_press_before_drain_overwrites_not_queues():
    """The safety property this class exists for: a stale 'grip' queued behind
    an operator's very next 'release' must never fire after the release."""
    node = FakeNode()
    receiver = AirCommandReceiver(node, [("air_board", object())])
    _deliver(node, "~/air_board/air/command", "grip")
    _deliver(node, "~/air_board/air/command", "release")
    assert receiver.pending() == {"air_board": "release"}


def test_two_sensors_are_independent():
    node = FakeNode()
    receiver = AirCommandReceiver(
        node, [("air_board", object()), ("other_board", object())])
    _deliver(node, "~/air_board/air/command", "grip")
    _deliver(node, "~/other_board/air/command", "release")
    assert receiver.pending() == {"air_board": "grip", "other_board": "release"}
