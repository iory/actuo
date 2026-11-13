"""Plain-Python coverage for actuo_bus_node.py's non-ROS helpers.

The script imports rclpy only lazily, inside run_real()/run_sim() -- module
scope needs nothing ROS provides -- so its top-level helpers are importable and
testable exactly like any library module. It is a script, not a package
module, so it is loaded from its file.
"""

import importlib.util
import os
import sys
import types

import pytest

from actuo.core.diagnostics import DiagLevel

_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "scripts", "actuo_bus_node.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("actuo_bus_node", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


actuo_bus_node = _load_script()


class _ActuatorWithoutFaultNames:
    pass


class _ActuatorWithFaultNames:
    def fault_name(self, code):
        return "vendor fault {}".format(int(code))


def test_fault_name_uses_the_actuator_hook_not_a_bus_family_switch():
    assert actuo_bus_node._fault_name(_ActuatorWithFaultNames(), 5) == \
        "vendor fault 5"


def test_fault_name_falls_back_to_unknown_when_the_actuator_has_no_hook():
    assert actuo_bus_node._fault_name(_ActuatorWithoutFaultNames(), 5) == \
        "unknown fault code 5"


# --------------------------------------------------------------------------
# TxStallPolicy: what a full CAN transmit queue means for a setpoint stream.
#
# A servicer that dies of "Transmit buffer full" mid-motion takes the whole
# stack with it (it carries on_exit=Shutdown()), and the arm drops.
# --------------------------------------------------------------------------

def test_a_single_dropped_cycle_is_not_fatal():
    """The next cycle is 20 ms away with a fresher setpoint. Drop, don't die."""
    policy = actuo_bus_node.TxStallPolicy(max_consecutive=10)
    assert policy.dropped() is False
    assert policy.streak == 1
    assert policy.total == 1


def test_an_unbroken_streak_eventually_becomes_fatal():
    """A queue that never drains means torque is on and commands are not."""
    policy = actuo_bus_node.TxStallPolicy(max_consecutive=3)
    assert policy.dropped() is False
    assert policy.dropped() is False
    assert policy.dropped() is True, "must abort at the configured limit"


def test_only_a_clean_write_clears_the_streak():
    policy = actuo_bus_node.TxStallPolicy(max_consecutive=3)
    policy.dropped()
    policy.dropped()
    assert policy.succeeded() == 2, "must report the streak it ended"
    assert policy.streak == 0
    # ...and the budget is whole again afterwards.
    assert policy.dropped() is False
    assert policy.dropped() is False
    assert policy.dropped() is True


def test_an_intermittent_stall_never_aborts_but_is_still_counted():
    """The distinction the streak exists to draw.

    A bus that drops one cycle in two is degraded, not dead: every command is
    only 20 ms stale. It must keep running -- but `total` has to keep the
    evidence, or a degrading bus reads as a healthy one.
    """
    policy = actuo_bus_node.TxStallPolicy(max_consecutive=3)
    for _ in range(50):
        assert policy.dropped() is False
        policy.succeeded()
    assert policy.total == 50
    assert policy.streak == 0


def test_succeeded_reports_nothing_when_there_was_no_streak():
    policy = actuo_bus_node.TxStallPolicy(max_consecutive=3)
    assert policy.succeeded() == 0


# --------------------------------------------------------------------------
# --window-reference parsing. An operator assertion about where a joint is,
# used to pick a gear window. A typo must be a startup error, never a joint
# quietly placed 60 deg from the truth.
# --------------------------------------------------------------------------

def test_a_window_reference_parses_to_radians():
    import math
    out = actuo_bus_node.parse_window_references(["joint3=-23.5"])
    assert set(out) == {"joint3"}
    assert out["joint3"] == pytest.approx(math.radians(-23.5))


def test_several_joints_and_whitespace():
    out = actuo_bus_node.parse_window_references([" joint3 =10", "joint0=0"])
    assert sorted(out) == ["joint0", "joint3"]


def test_no_overrides_is_an_empty_mapping_not_an_error():
    assert actuo_bus_node.parse_window_references([]) == {}
    assert actuo_bus_node.parse_window_references(None) == {}


def test_a_typo_is_refused_rather_than_silently_truncated():
    """`-23,5` must not become -23. That is a 0.5 deg lie at best."""
    for bad in ["joint3=-23,5", "joint3=", "joint3", "=10", "joint3=abc",
                "joint3=nan", "joint3=inf"]:
        with pytest.raises(ValueError):
            actuo_bus_node.parse_window_references([bad])


# --------------------------------------------------------------------------
# _diagnose_stall: which of two identical-looking faults was it?
#
# "The transmit queue filled and the servicer gave up" is produced by a failing
# bus AND by a stall below the CAN layer, and the fixes are opposite; without
# this, the cause can only be inferred from stats lines a second apart.
# --------------------------------------------------------------------------

def test_bus_errors_during_the_stall_point_at_the_PHYSICAL_layer():
    before = {"error-warn": 10, "error-pass": 100, "bus-off": 0, "tx_dropped": 0}
    after = {"error-warn": 13, "error-pass": 4200, "bus-off": 0, "tx_dropped": 0}
    out = actuo_bus_node._diagnose_stall(before, after)
    assert "PHYSICAL" in out
    assert "error-pass +4100" in out
    assert "termination" in out
    assert "60 ohm" in out


def test_NO_bus_errors_points_below_the_CAN_layer():
    before = {"error-warn": 10, "error-pass": 100, "bus-off": 0, "tx_dropped": 0}
    after = {"error-warn": 10, "error-pass": 100, "bus-off": 0, "tx_dropped": 7}
    out = actuo_bus_node._diagnose_stall(before, after, channel="can1")
    assert "NO bus errors" in out
    assert "USB" in out
    assert "txqueuelen" in out and "can1" in out
    assert "wiring is not" in out
    # the one counter that DID move is still named -- silence about it would
    # make "nothing happened" and "I did not look" indistinguishable
    assert "tx_dropped +7" in out


def test_missing_counters_say_UNKNOWN_rather_than_guessing():
    """A non-socketcan bus, or no `ip`. The honest answer is not a verdict."""
    for before, after in ((None, {"error-pass": 1}),
                          ({"error-pass": 1}, None),
                          (None, None)):
        out = actuo_bus_node._diagnose_stall(before, after)
        assert "UNKNOWN" in out
        assert "PHYSICAL" not in out


def test_bus_off_alone_is_enough_to_call_it_physical():
    before = {"error-warn": 0, "error-pass": 0, "bus-off": 0}
    after = {"error-warn": 0, "error-pass": 0, "bus-off": 1}
    assert "PHYSICAL" in actuo_bus_node._diagnose_stall(before, after)


def test_servicer_metrics_computes_window_stats_drop_rate_and_warn_level():
    metrics = actuo_bus_node.ServicerDiagnostics(
        target_rate_hz=100.0,          # target period 10 ms
        window_cycles=8,
        loop_warn_ratio=1.10,
        loop_error_ratio=1.50,
        latency_warn_ratio=0.70,
        latency_error_ratio=1.00,
        drop_rate_warn=0.25,
        drop_rate_error=0.75,
        consecutive_miss_warn=2,
        consecutive_miss_error=4,
        stale_warn_s=0.02,
        stale_error_s=0.08,
        cmd_stall_warn_s=0.02,
        cmd_stall_error_s=0.08,
    )

    for period in (0.010, 0.011, 0.012, 0.013):
        metrics.observe_cycle_period(period)
    for latency in (0.001, 0.002, 0.003, 0.004):
        metrics.observe_read_latency(latency)

    metrics.observe_read_result(had_miss=False, had_fresh_state=True, now_mono=0.01)
    metrics.observe_shm(cmd_seq=1, state_seq=2, now_mono=0.01)
    metrics.observe_read_result(had_miss=False, had_fresh_state=True, now_mono=0.02)
    metrics.observe_shm(cmd_seq=2, state_seq=4, now_mono=0.02)
    metrics.observe_read_result(had_miss=True, had_fresh_state=False, now_mono=0.03)
    metrics.observe_shm(cmd_seq=2, state_seq=6, now_mono=0.03)
    metrics.observe_read_result(had_miss=True, had_fresh_state=False, now_mono=0.04)
    metrics.observe_shm(cmd_seq=2, state_seq=8, now_mono=0.04)

    item = metrics.snapshot_item(now_mono=0.05)
    values = item.values

    assert item.level == DiagLevel.WARN
    assert float(values["loop_period_mean_s"]) == pytest.approx(0.0115, rel=1e-4)
    assert float(values["loop_period_p95_s"]) == pytest.approx(0.01285, rel=1e-4)
    assert float(values["drop_rate"]) == pytest.approx(0.5, rel=1e-6)
    assert int(values["consecutive_misses"]) == 2


def _loop_metrics(**overrides):
    """ServicerDiagnostics with only the loop-period knobs that matter here."""
    kwargs = dict(
        target_rate_hz=100.0,          # target period 10 ms
        window_cycles=64,
        loop_warn_ratio=1.10,          # 11.0 ms
        loop_error_ratio=1.30,         # 13.0 ms
        latency_warn_ratio=0.50,
        latency_error_ratio=0.90,
        drop_rate_warn=0.25,
        drop_rate_error=0.75,
        consecutive_miss_warn=2,
        consecutive_miss_error=4,
        stale_warn_s=10.0,
        stale_error_s=20.0,
        cmd_stall_warn_s=10.0,
        cmd_stall_error_s=20.0,
    )
    kwargs.update(overrides)
    return actuo_bus_node.ServicerDiagnostics(**kwargs)


def _run_loop(metrics, periods):
    """Feed a run of cycles and return the resulting item.

    Reads and shm sequences are fed too, not just periods: without them the
    staleness and command-stall checks see `inf` and every one of these cases
    comes back ERROR for a reason that has nothing to do with loop period.
    """
    now = 0.0
    for index, period in enumerate(periods, start=1):
        now += period
        metrics.observe_cycle_period(period)
        metrics.observe_read_latency(0.002)
        metrics.observe_read_result(
            had_miss=False, had_fresh_state=True, now_mono=now)
        metrics.observe_shm(cmd_seq=index, state_seq=index, now_mono=now)
    return metrics.snapshot_item(now_mono=now)


def test_one_late_cycle_does_not_call_a_healthy_loop_an_error():
    """A healthy Kondo bus: p95 10.06 ms at a 10 ms target, with a single
    13.08 ms cycle. Reporting that as ERROR forever would send the operator
    hunting a slow loop that is not there.
    """
    metrics = _loop_metrics()
    item = _run_loop(metrics, [0.01003] * 60 + [0.013076])  # one spike, >1.30x

    assert item.level == DiagLevel.WARN
    assert "one spike" in item.message
    # The p95 has to be IN the message: "worst cycle 13.1 ms" alone reads as a
    # slow loop, and the number that says otherwise is the one to show.
    assert "p95" in item.message


def test_a_sustained_slow_loop_is_still_an_error():
    metrics = _loop_metrics()
    item = _run_loop(metrics, [0.020] * 60)   # 2x the target, every cycle

    assert item.level == DiagLevel.ERROR
    assert "not keeping up" in item.message


def test_a_single_cycle_long_enough_to_be_a_stall_is_an_error():
    """A spike is forgiven for being brief, not for being any size at all."""
    metrics = _loop_metrics()
    item = _run_loop(metrics, [0.01003] * 60 + [0.40])   # 40x: the bus stopped

    assert item.level == DiagLevel.ERROR
    assert "stall" in item.message


def test_a_loop_running_exactly_on_target_is_ok():
    """A loop cannot average faster than its period, so OK must be reachable."""
    metrics = _loop_metrics()
    item = _run_loop(metrics, [0.010] * 60)

    assert _loop_metrics().loop_spike_error_ratio == 3.0
    assert item.level == DiagLevel.OK


def test_servicer_metrics_drop_rate_is_rolling_not_lifetime():
    metrics = actuo_bus_node.ServicerDiagnostics(
        target_rate_hz=100.0,
        window_cycles=4,
        loop_warn_ratio=1.10,
        loop_error_ratio=1.50,
        latency_warn_ratio=0.70,
        latency_error_ratio=1.00,
        drop_rate_warn=0.25,
        drop_rate_error=0.75,
        consecutive_miss_warn=2,
        consecutive_miss_error=4,
        stale_warn_s=0.02,
        stale_error_s=0.08,
        cmd_stall_warn_s=0.02,
        cmd_stall_error_s=0.08,
    )

    # Two drops happened, but only in the first half.
    for now_mono, miss in enumerate((True, True, False, False,
                                     False, False, False, False), start=1):
        metrics.observe_read_result(
            had_miss=miss,
            had_fresh_state=not miss,
            now_mono=float(now_mono) * 0.01,
        )

    values = metrics.snapshot_item(now_mono=0.09).values
    assert int(values["drop_cycles_total"]) == 2
    assert float(values["drop_rate"]) == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# Names shared with the C++ side and the python-can transport.
# --------------------------------------------------------------------------

def test_the_servicer_lock_is_the_one_the_hardware_component_checks():
    from actuo.drivers.shm import lock_path
    assert actuo_bus_node.servicer_lock_path("arm") == lock_path("arm")


def test_transmit_queue_full_is_inert_without_the_can_transport(monkeypatch):
    """A non-CAN robot must not need python-can to run the servicer."""
    monkeypatch.delitem(sys.modules, actuo_bus_node._PYTHON_CAN_MODULE,
                        raising=False)
    assert actuo_bus_node.transmit_queue_full_type() is \
        actuo_bus_node._NeverRaised


def test_transmit_queue_full_comes_from_the_loaded_can_transport(monkeypatch):
    full = type("TransmitQueueFull", (RuntimeError,), {})
    monkeypatch.setitem(sys.modules, actuo_bus_node._PYTHON_CAN_MODULE,
                        types.SimpleNamespace(TransmitQueueFull=full))
    assert actuo_bus_node.transmit_queue_full_type() is full
