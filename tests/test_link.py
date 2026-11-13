"""Coming back from the robot's power being switched off.

The behaviour that matters is not "it reconnects" -- it is what state it
reconnects INTO. A power cut leaves the robot limp, gravity moves it, and the
controller upstream keeps commanding where it used to be. Resuming would apply
full authority across that gap.
"""

import errno
import os
import sys

import pytest

_HERE = os.path.dirname(__file__)

from actuo.drivers.link import (  # noqa: E402
    BoardSilent, LinkRecovery, is_link_error)


class FakeSerialException(Exception):
    pass


FakeSerialException.__name__ = "SerialException"


class FakeTransport:
    def __init__(self, opens_after=0):
        self.opens_after = opens_after
        self.open_attempts = 0
        self.closes = 0
        self.is_open = False

    def open(self):
        self.open_attempts += 1
        if self.open_attempts <= self.opens_after:
            raise OSError(errno.ENOENT, "no such device")
        self.is_open = True

    def close(self):
        self.closes += 1
        self.is_open = False


class FakeBus:
    def __init__(self, transport):
        self.transport = transport
        self.reads = 0
        self.verified = 0

    def verify_servos(self):
        if not self.transport.is_open:
            raise OSError(errno.ENODEV, "gone")
        self.verified += 1
        return []

    def read_batch(self, timeout=None):
        if not self.transport.is_open:
            raise OSError(errno.ENODEV, "gone")
        self.reads += 1
        return []


class FakeSupervisor:
    def __init__(self, holding=("joint1", "joint2")):
        self.enabled_joints = list(holding)
        self.disables = 0

    def disable(self, joints=None):
        self.disables += 1
        self.enabled_joints = []
        return True, "off"


def make(opens_after=0, holding=("joint1", "joint2"), usb_reset=None,
         usb_reset_after=3):
    transport = FakeTransport(opens_after)
    transport.channel = "/dev/serial/by-id/fake"
    bus = FakeBus(transport)
    supervisor = FakeSupervisor(holding)
    messages = []
    recovery = LinkRecovery(
        bus, supervisor, interval=0.0,
        log=lambda message, is_error: messages.append(message),
        sleep=lambda _seconds: None,
        usb_reset=usb_reset, usb_reset_after=usb_reset_after)
    return recovery, bus, supervisor, messages


# -- what counts as the link dying -----------------------------------------


def test_a_vanished_usb_device_is_a_link_error():
    for code in (errno.ENODEV, errno.ENXIO, errno.EIO, errno.ENOENT):
        assert is_link_error(OSError(code, os.strerror(code)))


def test_pyserials_exception_is_a_link_error_by_name():
    """pyserial is not imported here, and importing it to catch it would make a
    core policy depend on one transport's package."""
    assert is_link_error(FakeSerialException("device reports readiness..."))


def test_an_ordinary_error_is_NOT_a_link_error():
    """A servo fault, a command outside its envelope, a bug -- swallowing any of
    those into a reconnect loop hides exactly what must never be hidden."""
    assert not is_link_error(ValueError("commanded 3.4 rad outside its limit"))
    assert not is_link_error(RuntimeError("servo 3 reports overload"))
    assert not is_link_error(OSError(errno.EACCES, "permission denied"))


# -- what it comes back as -------------------------------------------------


def test_torque_goes_off_BEFORE_anything_else():
    """From that moment there is no path from a controller setpoint to a frame,
    whatever the reconnect does next."""
    recovery, _, supervisor, _ = make()
    recovery.recover(OSError(errno.ENODEV, "gone"))
    assert supervisor.disables == 1
    assert supervisor.enabled_joints == []


def test_it_does_NOT_come_back_holding():
    """The robot was limp and gravity moved it while the controller kept
    commanding the old pose. Resuming would close that gap at full authority."""
    recovery, _, supervisor, _ = make(holding=("joint1",))
    assert recovery.recover(OSError(errno.ENODEV, "gone")) is True
    assert supervisor.enabled_joints == []


def test_it_says_which_joints_were_holding():
    recovery, _, _, messages = make(holding=("joint1", "joint3"))
    recovery.recover(OSError(errno.ENODEV, "gone"))
    assert "joint1, joint3" in messages[0]


def test_the_old_descriptor_is_closed_before_reopening():
    """A descriptor onto a device that no longer exists cannot be reused, and
    reopening without closing leaks one per power cycle."""
    recovery, bus, _, _ = make()
    recovery.recover(OSError(errno.ENODEV, "gone"))
    assert bus.transport.closes >= 1
    assert bus.transport.is_open


def test_the_board_gets_its_COLD_START_checks_again():
    """It lost power too: its ServoStruct is back at defaults, so the feedback
    flags that leave a servo accepting commands and not moving are set again."""
    recovery, bus, _, _ = make()
    recovery.recover(OSError(errno.ENODEV, "gone"))
    assert bus.verified == 1
    assert bus.reads == 1


def test_it_keeps_trying_while_the_robot_is_switched_off():
    recovery, bus, _, _ = make(opens_after=5)
    assert recovery.recover(OSError(errno.ENODEV, "gone")) is True
    assert bus.transport.open_attempts == 6


def test_a_half_open_port_is_closed_again_between_attempts():
    """open() can succeed onto a port whose board is not answering yet. Leaving
    it open would leak a descriptor per attempt, once a second, forever."""
    recovery, bus, _, _ = make()
    bus.transport.opens_after = 0
    original = bus.transport.open
    calls = {"n": 0}

    def flaky_open():
        calls["n"] += 1
        original()
        if calls["n"] < 3:
            bus.transport.is_open = False   # the board is not answering yet
    bus.transport.open = flaky_open

    assert recovery.recover(OSError(errno.ENODEV, "gone")) is True
    assert bus.transport.closes >= 3


def test_recoveries_are_counted():
    """A robot that reconnects every few minutes has a loose cable, and that
    reads as "it works" unless someone is counting."""
    recovery, _, _, _ = make()
    for _ in range(3):
        recovery.recover(OSError(errno.ENODEV, "gone"))
    assert recovery.recoveries == 3


def test_a_deadline_bounds_the_wait():
    recovery, bus, _, messages = make(opens_after=10**9)
    import time
    assert recovery.recover(OSError(errno.ENODEV, "gone"),
                            deadline=time.monotonic() + 0.05) is False
    assert "gave up" in messages[-1]


def test_the_message_points_at_re_syncing_the_controller():
    """Not at forcing, and not at moving the robot. Direction is the whole
    safety argument."""
    recovery, _, _, messages = make()
    recovery.recover(OSError(errno.ENODEV, "gone"))
    back = messages[-1]
    assert "Torque is OFF" in back
    assert "Re-sync the controller" in back


# -- the failure that actually happens -------------------------------------


def test_silence_is_not_an_exception_and_must_not_need_one():
    """Measured on a Kondo humanoid: with the robot switched off the CH340 is still
    enumerated and /dev/ttyACM0 keeps its inode, so NOTHING raises. A reconnect
    that only watched for exceptions would never fire on the case an operator
    causes every day."""
    assert not is_link_error(BoardSilent("no joint answered for 1.0s"))


def test_a_silent_board_recovers_by_the_same_path_and_says_so_differently():
    """Same remedy -- release, wait, cold-start the board -- but the operator
    must be told which of the two happened, because one means "check the cable"
    and the other means "you switched it off"."""
    recovery, bus, supervisor, messages = make(holding=("joint1",))
    assert recovery.recover(BoardSilent("no joint has answered for 1.0s")) is True
    assert messages[0].startswith("BOARD SILENT")
    assert supervisor.enabled_joints == []
    assert bus.verified == 1


def test_an_exception_still_reads_as_a_lost_link():
    recovery, _, _, messages = make()
    recovery.recover(OSError(errno.ENODEV, "gone"))
    assert messages[0].startswith("LINK LOST")


# -- when reopening is not enough ------------------------------------------


def test_a_transport_used_after_close_is_a_link_error():
    """The cycle after a failed reopen hits this. Without it the recovery loop
    re-raises a plain RuntimeError and kills the process one cycle after
    deciding to survive -- observed exactly once, which was enough."""
    from actuo.drivers.link import TransportNotOpen
    assert is_link_error(TransportNotOpen("kondo_h7 transport is not open"))


def test_reopening_alone_never_escalates():
    """A device reset is a heavier hammer than a board that was briefly busy."""
    resets = []
    recovery, _, _, _ = make(usb_reset=lambda channel: resets.append(channel))
    recovery.recover(BoardSilent("silent"))
    assert resets == []


def test_a_wedged_adapter_is_reset_after_enough_failed_reopens():
    """Measured: a wedged link does not answer a reopened port, and did not
    answer a physical replug either. Reopening forever cannot succeed."""
    resets = []

    def fake_reset(channel):
        resets.append(channel)
        return "1a86:55d3 on bus 3 port 1"

    recovery, bus, _, messages = make(opens_after=5, usb_reset=fake_reset,
                                      usb_reset_after=3)
    assert recovery.recover(BoardSilent("silent")) is True
    assert resets == ["/dev/serial/by-id/fake"] * 3      # attempts 4, 5 and 6
    assert recovery.usb_resets == 3
    assert any("USB reset of 1a86:55d3" in m for m in messages)


def test_a_reset_that_is_not_permitted_is_said_ONCE_and_not_fatal():
    """It is usually a udev rule the operator has to add. Repeating it every
    second buries it, and giving up would lose the robot over a permission."""
    def refuses(channel):
        raise RuntimeError("no permission to reset the USB device")

    recovery, _, _, messages = make(opens_after=8, usb_reset=refuses,
                                    usb_reset_after=3)
    assert recovery.recover(BoardSilent("silent")) is True
    complaints = [m for m in messages if "cannot reset" in m]
    assert len(complaints) == 1


def test_the_board_gets_several_chances_to_answer_after_reopening():
    """rcb4: "after powering on, the ACK value becomes unstable for some
    reason". One probe would throw away a link that works on the third try."""
    recovery, bus, _, _ = make()
    tries = {"n": 0}
    real_verify = bus.verify_servos

    def flaky_verify():
        tries["n"] += 1
        if tries["n"] < 3:
            raise OSError(errno.EIO, "no reply")
        return real_verify()
    bus.verify_servos = flaky_verify

    assert recovery.recover(BoardSilent("silent")) is True
    assert tries["n"] == 3
    assert bus.transport.open_attempts == 1        # ...without reopening twice
