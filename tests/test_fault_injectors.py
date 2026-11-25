"""Fault injectors and rigs without hardware: what goes on the wire, and what is refused."""

import pytest

from actuo.testing import fault_injectors as fi
from actuo.testing.rig import ServoRig


class FakeSerial:
    """A serial port that answers each written line from a script."""

    def __init__(self, answer):
        self.answer = answer
        self.sent = []
        self._out = []

    def reset_input_buffer(self):
        self._out.clear()

    def write(self, data):
        line = data.decode().strip()
        self.sent.append(line)
        self._out.extend(x.encode() + b"\n" for x in self.answer(line))

    def readline(self):
        return self._out.pop(0) if self._out else b""

    def close(self):
        pass


def board(monkeypatch, answer=lambda line: ["ok " + line]):
    port = FakeSerial(answer)
    monkeypatch.setattr(fi.serial, "Serial", lambda *a, **k: port)
    return port


def test_unplug_by_power_switches_the_port_off_and_back_on(monkeypatch):
    port = board(monkeypatch)
    sfi = fi.ServoFaultInjector(port="/dev/x", timeout=0.1)
    sfi.unplug(3)
    sfi.plug(3)
    sfi.restore()
    assert port.sent == ["power 3 off", "power 3 on", "power all on", "pass"]


def test_signal_after_power_cuts_the_signal_until_the_servo_has_booted(monkeypatch):
    def answer(line):
        if line == "status":
            return ["port n=1 power=1 isolated=0", "port n=2 power=0 isolated=1", "ok status"]
        return ["ok " + line]
    port = board(monkeypatch, answer)
    sleeps = []
    monkeypatch.setattr(fi.time, "sleep", sleeps.append)
    sfi = fi.ServoFaultInjector(port="/dev/x", timeout=0.1, signal_after_s=1.0)
    sfi.plug(3)
    assert port.sent == ["power 3 on", "isolate 3", "unisolate 3"]
    assert sleeps == [1.0]
    port.sent.clear()
    sfi.restore()
    # Only the port that was off is powered, and its signal follows.
    assert port.sent == ["status", "power 2 on", "isolate 2", "unisolate 2", "pass"]


def test_signal_after_power_needs_unplug_by_power(monkeypatch):
    board(monkeypatch)
    with pytest.raises(ValueError, match="unplug_by: power"):
        fi.ServoFaultInjector(port="/dev/x", unplug_by="signal", signal_after_s=1.0)


def test_unplug_by_signal_isolates_the_port(monkeypatch):
    port = board(monkeypatch)
    sfi = fi.ServoFaultInjector(port="/dev/x", unplug_by="signal", timeout=0.1)
    sfi.unplug(2)
    sfi.plug(2)
    sfi.drop_replies(1, 7)
    assert port.sent == ["isolate 2", "unisolate 2", "drop 1 7"]


def test_a_board_error_is_raised_not_swallowed(monkeypatch):
    board(monkeypatch, answer=lambda line: ["error port 2 has no power"])
    sfi = fi.ServoFaultInjector(port="/dev/x", timeout=0.1)
    with pytest.raises(fi.FaultInjectorError, match="no power"):
        sfi.plug(2)


def test_a_port_outside_the_board_is_refused_before_sending(monkeypatch):
    port = board(monkeypatch)
    sfi = fi.ServoFaultInjector(port="/dev/x", timeout=0.1)
    with pytest.raises(fi.FaultInjectorError):
        sfi.unplug(5)
    assert port.sent == []


def test_status_records_are_numbers(monkeypatch):
    board(monkeypatch, answer=lambda line: [
        "cfg baud=1000000", "port n=1 power=1 isolated=0 volt=11.990 ma=27.0", "ok status"])
    sfi = fi.ServoFaultInjector(port="/dev/x", timeout=0.1)
    assert sfi.status() == [{"n": 1, "power": 1, "isolated": 0, "volt": 11.99, "ma": 27.0}]


def test_the_ics_relay_unplugs_one_servo_at_a_time(monkeypatch):
    port = board(monkeypatch)
    relay = fi.IcsFaultRelay(port="/dev/x", timeout=0.1)
    relay.unplug(2)
    with pytest.raises(fi.FaultInjectorError, match="one servo at a time"):
        relay.unplug(8)
    relay.plug(2)
    relay.unplug(8)
    assert port.sent == ["!relay", "!isolate 2", "!nodrop", "!relay", "!isolate 8"]


class RecordingInjector(fi.FaultInjector):
    capabilities = frozenset({fi.UNPLUG, fi.UNPLUG_MANY, fi.DROP_REPLIES})

    def __init__(self):
        self.calls = []

    def unplug(self, target):
        self.calls.append(("unplug", target))

    def plug(self, target):
        self.calls.append(("plug", target))

    def drop_replies(self, target, servo_id):
        self.calls.append(("drop", target, servo_id))

    def restore(self):
        self.calls.append(("restore",))

    def close(self):
        pass


def test_only_unplugs_the_rest_and_plugs_back_what_is_wanted():
    inj = RecordingInjector()
    rig = ServoRig(inj, {"j1": 1, "j2": 2, "j3": 3}, servo_ids={"j1": 11})
    rig.only(["j1"])
    assert inj.calls == [("unplug", 2), ("unplug", 3)]
    inj.calls.clear()
    rig.only(["j1", "j3"])
    assert inj.calls == [("plug", 3)]
    rig.drop_replies("j1")
    assert inj.calls[-1] == ("drop", 1, 11)


def test_only_refuses_what_a_one_at_a_time_injector_cannot_do():
    class OneAtATime(RecordingInjector):
        capabilities = frozenset({fi.UNPLUG})

    rig = ServoRig(OneAtATime(), {"j1": 1, "j2": 2, "j3": 3})
    with pytest.raises(fi.FaultInjectorError, match="unplug_many"):
        rig.only(["j1"])


def test_combinations_cover_every_subset():
    rig = ServoRig(RecordingInjector(), {"a": 1, "b": 2, "c": 3})
    combos = list(rig.combinations())
    assert len(combos) == 8 and combos[0] == () and combos[-1] == ("a", "b", "c")


def test_a_rig_file_takes_servo_ids_from_the_robot_config(tmp_path, monkeypatch):
    board(monkeypatch)
    (tmp_path / "robot.yaml").write_text(
        "bus: {transport: {plugin: serial, channel: /dev/null}}\n"
        "joints:\n"
        "  - {name: j1, actuator: {plugin: feetech_sts, id: 5, model: STS3215}}\n")
    (tmp_path / "rig.yaml").write_text(
        "robot: robot.yaml\nurdf: r.urdf\n"
        "injector: {plugin: servo_fault_injector, port: /dev/x}\n"
        "targets: {j1: 2}\n")
    rig = ServoRig.from_file(str(tmp_path / "rig.yaml"))
    assert rig.servo_ids == {"j1": 5} and rig.targets == {"j1": 2}
    assert rig.urdf == str(tmp_path / "r.urdf")
