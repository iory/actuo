"""BusDiagnosticsPublisher: translation and failure handling without ROS spin."""

import os
import tempfile

from builtin_interfaces.msg import Time
from diagnostic_msgs.msg import DiagnosticStatus
import pytest

from actuo.core.diagnostics import DiagItem, DiagLevel

from actuo_ros.diagnostics_node import BusDiagnosticsPublisher


class _FakePublisher:
    def __init__(self, topic_name):
        self.topic_name = topic_name
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


class _FakeTimer:
    def __init__(self, period, callback):
        self.period = period
        self.callback = callback
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def reset(self):
        pass


class _FakeParameter:
    def __init__(self, value):
        self.value = value


class _FakeLogger:
    def warning(self, *_args, **_kwargs):
        pass

    def info(self, *_args, **_kwargs):
        pass


class _FakeClock:
    class _Now:
        def to_msg(self):
            return Time(sec=0, nanosec=0)

    def now(self):
        return self._Now()


class _FakeNode:
    def __init__(self):
        self._parameters = {}
        self.publishers = []
        self.timers = []
        self._logger = _FakeLogger()
        self._clock = _FakeClock()

    def create_publisher(self, _msg_type, topic, _qos):
        pub = _FakePublisher(topic)
        self.publishers.append(pub)
        return pub

    def create_timer(self, period, callback):
        timer = _FakeTimer(period, callback)
        self.timers.append(timer)
        return timer

    def declare_parameter(self, name, value):
        self._parameters[name] = value
        return _FakeParameter(value)

    def has_parameter(self, name):
        return name in self._parameters

    def get_parameter(self, name):
        return _FakeParameter(self._parameters[name])

    def get_namespace(self):
        return "/robot"

    def get_name(self):
        return "actuo_bus_servicer"

    def get_logger(self):
        return self._logger

    def get_clock(self):
        return self._clock


class _Bus:
    def __init__(self, items=None, error=None):
        self._items = list(items or [])
        self._error = error

    def diagnose(self):
        if self._error is not None:
            raise self._error
        return list(self._items)


def _status_by_suffix(array_msg, suffix):
    for status in array_msg.status:
        if status.name.endswith(suffix):
            return status
    raise AssertionError("status with suffix {!r} not found".format(suffix))


def test_diag_items_translate_to_status_levels_and_keyvalues():
    node = _FakeNode()
    generic = lambda: [DiagItem(  # noqa: E731
        name="bus/generic_metrics",
        level=DiagLevel.OK,
        message="ok",
        values={"target_rate_hz": "50.0"},
    )]
    bus = _Bus(items=[
        DiagItem("servo/j0", DiagLevel.WARN, "warn", {"k1": "v1"}),
        DiagItem("servo/j1", DiagLevel.ERROR, "error", {"k2": "v2"}),
        DiagItem("servo/j2", DiagLevel.STALE, "stale", {"k3": "v3"}),
    ])
    publisher = BusDiagnosticsPublisher(
        node, bus, generic_items=generic, hardware_id="transport=can0")

    publisher.publish()
    array = node.publishers[-1].messages[-1]

    ok = _status_by_suffix(array, "bus/generic_metrics")
    warn = _status_by_suffix(array, "servo/j0")
    error = _status_by_suffix(array, "servo/j1")
    stale = _status_by_suffix(array, "servo/j2")

    assert ok.level == DiagnosticStatus.OK
    assert warn.level == DiagnosticStatus.WARN
    assert error.level == DiagnosticStatus.ERROR
    assert stale.level == DiagnosticStatus.STALE
    assert {kv.key: kv.value for kv in warn.values} == {"k1": "v1"}
    assert {kv.key: kv.value for kv in error.values} == {"k2": "v2"}
    assert {kv.key: kv.value for kv in stale.values} == {"k3": "v3"}


def test_diagnose_exception_is_published_as_error_not_raised():
    node = _FakeNode()
    bus = _Bus(error=RuntimeError("probe failed"))
    publisher = BusDiagnosticsPublisher(
        node, bus, generic_items=lambda: [], hardware_id="transport=ttyUSB0")

    publisher.publish()  # must not raise
    array = node.publishers[-1].messages[-1]
    status = _status_by_suffix(array, "bus/diagnostics_runtime")

    assert status.level == DiagnosticStatus.ERROR
    assert "failed" in status.message


def test_real_rclpy_round_trip_uses_global_topic_and_byte_levels():
    rclpy = pytest.importorskip("rclpy", reason="rclpy unavailable in this env")
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
    from rclpy.executors import SingleThreadedExecutor

    did_init = False
    if not rclpy.ok():
        log_dir = os.path.join(tempfile.gettempdir(), "actuo_ros_test_logs")
        os.makedirs(log_dir, exist_ok=True)
        os.environ["ROS_LOG_DIR"] = log_dir
        rclpy.init()
        did_init = True

    pub_node = rclpy.create_node("diag_pub", namespace="my_robot")
    sub_node = rclpy.create_node("diag_sub")
    seen = []
    sub_node.create_subscription(
        DiagnosticArray, "/diagnostics", lambda msg: seen.append(msg), 10
    )
    executor = SingleThreadedExecutor()
    executor.add_node(pub_node)
    executor.add_node(sub_node)

    try:
        bus = _Bus(items=[
            DiagItem("servo/ok", DiagLevel.OK, "ok", {"k_ok": "v_ok"}),
            DiagItem("servo/warn", DiagLevel.WARN, "warn", {"k_warn": "v_warn"}),
            DiagItem("servo/error", DiagLevel.ERROR, "error", {"k_err": "v_err"}),
            DiagItem("servo/stale", DiagLevel.STALE, "stale", {"k_stale": "v_stale"}),
        ])
        publisher = BusDiagnosticsPublisher(
            pub_node, bus, generic_items=lambda: [], hardware_id="hw"
        )
        assert publisher._updater.publisher.topic_name == "/diagnostics"
        publisher.publish()

        deadline = pub_node.get_clock().now().nanoseconds + int(2e9)
        while not seen and pub_node.get_clock().now().nanoseconds < deadline:
            executor.spin_once(timeout_sec=0.05)
        assert seen, "no diagnostics message received from /diagnostics"

        array = seen[-1]
        ok = _status_by_suffix(array, "servo/ok")
        warn = _status_by_suffix(array, "servo/warn")
        error = _status_by_suffix(array, "servo/error")
        stale = _status_by_suffix(array, "servo/stale")

        assert ok.level == DiagnosticStatus.OK
        assert warn.level == DiagnosticStatus.WARN
        assert error.level == DiagnosticStatus.ERROR
        assert stale.level == DiagnosticStatus.STALE
        assert {kv.key: kv.value for kv in ok.values} == {"k_ok": "v_ok"}
        assert {kv.key: kv.value for kv in warn.values} == {"k_warn": "v_warn"}
        assert {kv.key: kv.value for kv in error.values} == {"k_err": "v_err"}
        assert {kv.key: kv.value for kv in stale.values} == {"k_stale": "v_stale"}
    finally:
        executor.shutdown()
        pub_node.destroy_node()
        sub_node.destroy_node()
        if did_init and rclpy.ok():
            rclpy.shutdown()
