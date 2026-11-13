"""Tests for the read-only Kondo probe CLI script."""

from __future__ import annotations

import importlib.util
import os
import sys

_HERE = os.path.dirname(__file__)

from actuo.core.contracts import Frame  # noqa: E402
from actuo.drivers.buses.kondo_h7_bus import KondoDiscovery  # noqa: E402


def _load():
    path = os.path.join(_HERE, "..", "actuo", "cli", "kondo_probe.py")
    spec = importlib.util.spec_from_file_location("actuo_kondo_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_scan(monkeypatch, capsys, expected_ids, present_ids=(0, 2)):
    """Run ``--scan`` against a board that discovered ``present_ids``.

    The fake discovery always leaves the inferred ``Control.device_port`` scan
    unavailable (``timed_out_ids=None``), which is the routine case for any
    symbol map without the ``Control`` symbol.
    """
    probe = _load()

    class FakeSymbols:
        version = "v-test"

        def require_version(self, reported):
            if str(reported).strip() != "v-test":
                raise probe.ProtocolError("firmware mismatch")

        def address(self, _name):
            return 0x24000100

    class FakeBus:
        def __init__(self, transport, actuators, symbols):
            self.transport = transport
            self.actuators = actuators
            self.symbols = symbols

        def discover(self):
            return KondoDiscovery(
                present_ids=tuple(present_ids),
                ports={servo_id: 1 for servo_id in present_ids},
                per_port=None,
                timed_out_ids=None,
                firmware_version="v-test",
                ics_channel_scan_error="Control.device_port unavailable",
            )

        def scan_servo_flags(self):
            return {servo_id: (servo_id in set(present_ids))
                    for servo_id in range(probe.SERVO_VECTOR_LENGTH)}

        def scan_servo_ports(self):
            return {servo_id: (1 if servo_id in set(present_ids) else 0)
                    for servo_id in range(probe.SERVO_VECTOR_LENGTH)}

        def compare_servo_ports_with_inferred_ics_ports(self, servo_ports, inferred):
            del servo_ports
            del inferred
            return {}

    class FakeTransport:
        checksum_errors = 0
        framing_errors = 0

        def __init__(self, channel, bitrate, timeout):
            del channel
            del bitrate
            del timeout

        def open(self):
            return None

        def close(self):
            return None

        def transact(self, frames, timeout):
            del frames
            del timeout
            return [Frame(payload=b"v-test")]

    def read_slot(*args, **kwargs):
        del args
        del kwargs
        return [0] * probe.SERVO_VECTOR_LENGTH

    monkeypatch.setattr(probe, "KondoH7Transport", FakeTransport)
    monkeypatch.setattr(probe, "KondoH7Bus", FakeBus)
    monkeypatch.setattr(probe, "load_symbol_map", lambda path: FakeSymbols())
    monkeypatch.setattr(probe, "load_expected_ids", lambda path: list(expected_ids))
    monkeypatch.setattr(probe, "read_slot", read_slot)

    exit_code = probe.main([
        "--channel", "/dev/fake",
        "--scan",
        "--symbols", "/tmp/symbols.yaml",
        "--config", "/tmp/config.yaml",
    ])
    return exit_code, capsys.readouterr().out


def test_scan_config_verdict_is_not_ok_when_timeout_scan_is_unavailable(
        monkeypatch, capsys):
    exit_code, output = _run_scan(monkeypatch, capsys, expected_ids=[0])

    assert exit_code == 0
    assert "VERDICT: OK" not in output
    assert "VERDICT: INCOMPLETE" in output
    assert "unexpected" in output


def test_a_confirmed_missing_servo_outranks_an_unavailable_scan(
        monkeypatch, capsys):
    """INCOMPLETE means "could not tell". A servo missing from the VERIFIED
    flag scan is something we did tell, so it must not be softened to
    INCOMPLETE just because the inferred device_port scan was unavailable --
    and that scan is unavailable for every symbol map without `Control`, so
    this pair is the common case rather than an exotic one."""
    exit_code, output = _run_scan(
        monkeypatch, capsys, expected_ids=[0, 7], present_ids=(0, 2))

    assert exit_code == 0
    assert "VERDICT: SUSPICIOUS" in output
    assert "VERDICT: INCOMPLETE" not in output
    # the unavailable scan is still reported, just not as the headline
    assert "device_port timeout scan unavailable" in output
