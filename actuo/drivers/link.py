"""Surviving the robot's power being switched off, and coming back SAFE.

There are TWO ways to lose a robot, and they look nothing like each other.
Measured on a humanoid with Kondo ICS servos on a USB-serial (CH340) link,
2026-08-01:

* **The device goes away** (cable out, hub reset). Reads and writes on the open
  descriptor raise. Unhandled, that takes the bus-servicing process down and,
  in a ROS 2 launch with ``on_exit=Shutdown()``, the whole ros2_control stack
  with it.
* **The link wedges while everything looks fine.** Nothing raises at all. The
  device stays enumerated, the port still opens, writes still succeed, and the
  board simply stops answering. Observed: 30 consecutive version requests with
  no reply, through an open port, with the robot powered on the whole time --
  and **a physical unplug and replug did not fix it** (the device re-enumerated,
  new inode and new bus address, and the board stayed silent).

The second is the dangerous one, because a servicer watching only for exceptions
sees a perfectly healthy loop: it keeps cycling, keeps writing into a void, keeps
publishing the last pose it read, and commands that stale pose the moment the
board comes back -- into servos that may have gone limp and drooped in the
meantime. Silence is therefore a first-class trigger here
(:class:`BoardSilent`), not an afterthought.

And since a wedged link does not answer a reopened port either, recovery
escalates to a USB-level reset -- which is exactly what rcb4 does from its
second connection attempt onward. See :mod:`actuo.drivers.usb_reset`.

A note on what this does NOT diagnose: it deliberately says "the board stopped
answering" and not why. Power, cable, wedged adapter and crashed firmware all
look identical from the host, so asserting a cause (say, "the power is off")
on evidence that only shows the board is silent would be a guess. Reporting
the observation and the remedy is honest; naming the cause is not.

What kxr_controller does, and what is kept
------------------------------------------
``kxr_controller`` is the ROS package for Kondo KXR robots in the public rcb4
repository (https://github.com/iory/rcb4, ``ros/kxr_controller``). Its
``rcb4_ros_bridge`` counts the success rate of its board reads and, every two
seconds, reinitialises the interface if the rate falls below 80 %: close, reopen
in a retry loop until the port exists again, re-apply the board-side settings,
and re-seed the controller's initial position from the MEASURED angles. The
shape is kept: detect from repeated failure, reopen patiently, re-run the
board's own startup.

What is deliberately different: it comes back with torque OFF
-------------------------------------------------------------
A power cut is not a dropped packet. The servos were unpowered, so the robot
went limp and gravity moved it -- and the controller upstream has been holding
its old command the whole time. That is exactly the divergence the servo design
exists to make impossible: reconnecting and resuming would apply full authority
across a gap nobody watched open.

So recovery ends in the safe state, not the previous one. Every joint is
released in the supervisor's model, the first cycle after reconnect therefore
writes the off sentinel, and the operator re-enables through the ordinary
guarded transition -- which refuses unless the controller is already commanding
where the robot now is, and whose remedy re-syncs the CONTROLLER to the robot.
Coming back holding would be a decision this layer is not entitled to make.
"""

from __future__ import annotations

import errno
import time
from typing import Callable, Optional

class TransportNotOpen(RuntimeError):
    """A transport was used after being closed.

    Exists so recovery can recognise it. Without it, a reopen that fails leaves
    the transport closed, the next cycle's read raises a plain ``RuntimeError``,
    and the recovery loop -- which correctly refuses to swallow unrecognised
    errors -- re-raises it and kills the process one cycle after deciding to
    survive. Observed exactly once, which was enough.
    """


class BoardSilent(RuntimeError):
    """The transport is fine and the board is not answering.

    A distinct type from the OS errors above because the cause is different --
    the robot's power switch, not its cable -- and because nothing raises this
    on its own: the servicer decides, from consecutive cycles with no reply, that
    silence has gone on long enough to mean the board is gone rather than busy.
    """


#: Errno values that mean "the device is gone", not "the device said no".
_GONE = frozenset({errno.ENODEV, errno.ENXIO, errno.EIO, errno.ENOENT,
                   errno.EBADF, errno.ESHUTDOWN})


def is_link_error(error: BaseException) -> bool:
    """Is this the transport dying, rather than the robot misbehaving?

    The distinction matters because the two demand opposite responses. A servo
    reporting a fault, a command outside its envelope, a garbled reply -- those
    are the robot talking, and swallowing them into a reconnect loop would hide
    exactly the failures that must always surface loudly rather than be
    masked by a fallback that makes things look healthy. A vanished USB
    device is not the robot talking at all.

    Recognised: pyserial's ``SerialException`` (which is how a removed CH340
    surfaces on Linux), and any ``OSError`` whose errno says the device is gone.
    An ``OSError`` with some other errno is NOT treated as a link failure --
    guessing would mean retrying forever through a real bug.
    """
    if isinstance(error, TransportNotOpen):
        return True
    name = type(error).__name__
    if name in ("SerialException", "SerialTimeoutException", "PortNotOpenError"):
        return True
    if isinstance(error, OSError):
        return error.errno in _GONE
    return False


class LinkRecovery:
    """Close, wait for the port to come back, restart the board, stay OFF.

    Parameters
    ----------
    bus : Bus
        Reopened and re-verified here. Its transport is closed first, because a
        file descriptor onto a device that no longer exists cannot be reused and
        reopening without closing leaks one per power cycle.
    supervisor : ServoSupervisor
        Released before anything else. Doing it first matters: from that moment
        there is no path from a controller setpoint to a frame, whatever the
        reconnect does next.
    interval : float
        Seconds between reopen attempts.
    log : callable
        Takes ``(message, is_error)``. Everything here is worth saying out loud;
        a robot that silently reconnected with torque off would look like one
        that is simply ignoring its controller.
    """

    def __init__(self, bus, supervisor, interval: float = 1.0,
                 log: Optional[Callable[[str, bool], None]] = None,
                 sleep: Callable[[float], None] = time.sleep,
                 usb_reset_after: int = 3, probe_attempts: int = 10,
                 usb_reset: Optional[Callable[[str], str]] = None) -> None:
        self.bus = bus
        self.supervisor = supervisor
        self.interval = float(interval)
        self._log = log or (lambda message, is_error: None)
        self._sleep = sleep
        #: Reopen attempts before escalating to a USB-level reset. Not the first
        #: attempt: most outages are the board being busy or briefly unpowered,
        #: and resetting the adapter for those would be a heavier hammer than
        #: the problem. rcb4 escalates from its second attempt; this waits a
        #: little longer because it retries once a second rather than three
        #: times total.
        self.usb_reset_after = int(usb_reset_after)
        #: How many times to ask the board for its version after reopening,
        #: before calling the attempt a failure. rcb4 uses ten, with the comment
        #: "after powering on, the ACK value becomes unstable for some reason";
        #: taken at face value because it is the observation of someone who ran
        #: this board for years, and one probe would throw away a link that
        #: works on the third try.
        self.probe_attempts = int(probe_attempts)
        self._usb_reset = usb_reset
        #: USB resets issued. Counted separately from ``recoveries`` because
        #: they mean something worse: the port reopened and the board still
        #: would not talk.
        self.usb_resets = 0
        #: How many times this servicer has come back. Published in the stats
        #: line: a robot that reconnects every few minutes has a loose cable,
        #: and that reads as "it works" unless someone is counting.
        self.recoveries = 0

    def recover(self, error: BaseException, deadline: Optional[float] = None) -> bool:
        """Release, reopen, re-verify. Blocks until the board answers again.

        Returns ``True`` once the bus is usable. Returns ``False`` only if
        ``deadline`` (a ``time.monotonic`` value) passes first, which exists so
        a test does not hang and an operator can bound the wait; the default is
        to keep trying, because the usual cause is a robot that will be switched
        back on in a minute.
        """
        was_holding = self.supervisor.enabled_joints
        self.supervisor.disable()
        self._log(
            "{}: {}. Torque is now OFF for every joint{}. Waiting for the board "
            "to come back; nothing will be commanded until you enable it "
            "again.".format(
                "BOARD SILENT" if isinstance(error, BoardSilent) else "LINK LOST",
                error,
                " (it was holding {})".format(", ".join(was_holding))
                if was_holding else ""),
            True)

        try:
            self.bus.transport.close()
        except Exception:                                      # noqa: BLE001
            # Closing a descriptor onto a device that is already gone can fail,
            # and that failure says nothing useful. What matters is that the
            # transport no longer believes it is open, which close() sets even
            # on the error path.
            pass

        attempts = 0
        while deadline is None or time.monotonic() < deadline:
            self._sleep(self.interval)
            attempts += 1

            # Escalation. A wedged adapter does not answer a reopened port
            # either -- measured, including across a physical replug -- so
            # reopening forever would be a loop that cannot succeed.
            if self._usb_reset is not None and attempts > self.usb_reset_after:
                try:
                    self.bus.transport.close()
                except Exception:                              # noqa: BLE001
                    pass
                try:
                    described = self._usb_reset(self.bus.transport.channel)
                    self.usb_resets += 1
                    self._log("USB reset of {} (attempt {})".format(
                        described, attempts), True)
                    # rcb4 waits two seconds here. The device re-enumerates, and
                    # opening it before udev has recreated the node fails in a
                    # way that looks like the board still being dead.
                    self._sleep(2.0)
                except Exception as reset_error:               # noqa: BLE001
                    if self.usb_resets == 0 and attempts == self.usb_reset_after + 1:
                        # Said once, in full: this is usually a udev rule the
                        # operator has to add, and repeating it every second
                        # buries it.
                        self._log("cannot reset the USB device: {}".format(
                            reset_error), True)

            try:
                self.bus.transport.open()
            except Exception as open_error:                    # noqa: BLE001
                self._log_retry(attempts, open_error)
                continue

            try:
                # The board may have restarted: its ServoStruct is back at
                # defaults, so this is a cold start and gets the cold-start
                # checks -- firmware version, servo discovery, and the feedback
                # flags that otherwise leave a servo accepting commands and not
                # moving.
                verify = getattr(self.bus, "verify_servos", None)
                if callable(verify):
                    for item in self._verify_with_retries(verify):
                        self._log("reconnect {}: {}".format(item.name,
                                                            item.message),
                                  True)
                self.bus.read_batch(timeout=0.5)
            except Exception as retry_error:                   # noqa: BLE001
                try:
                    self.bus.transport.close()
                except Exception:                              # noqa: BLE001
                    pass
                self._log_retry(attempts, retry_error)
                continue

            self.recoveries += 1
            # Back to the startup state, not to "everything released": a joint
            # with no pose cannot be thrown anywhere, and a base that stops
            # answering after a cable hiccup -- and stays that way until someone
            # notices -- is its own hazard.
            reset = getattr(self.supervisor, "reset_to_default", None)
            if callable(reset):
                reset()
            self._log(
                "BOARD BACK after {} attempt(s). Torque is OFF and the robot has "
                "probably moved while it was unpowered, so enabling goes "
                "through the usual guard: it will refuse unless the controller "
                "already commands where the robot now is. Re-sync the "
                "controller (the RViz servo panel's ON does it for you).".format(
                    attempts),
                False)
            return True

        self._log("gave up reconnecting after {} attempt(s)".format(attempts),
                  True)
        return False

    def _verify_with_retries(self, verify):
        """Run the board's startup checks, tolerating an unstable first reply.

        rcb4 retries its ACK ten times right after a power-on. Not doing so
        throws away a board that would have answered on the third try and sends
        the recovery round another full interval -- or, past the threshold,
        another USB reset it did not need.
        """
        last = None
        for attempt in range(max(1, self.probe_attempts)):
            try:
                return verify()
            except Exception as error:                         # noqa: BLE001
                last = error
                self._sleep(0.1)
        raise last

    def _log_retry(self, attempts: int, error: BaseException) -> None:
        # Once, then at 10, then once a minute. A line a second says nothing the
        # first line did not, and buries the USB-reset message when it comes.
        if attempts in (1, 10) or attempts % 60 == 0:
            self._log("still waiting for the board ({} attempt(s)): {}".format(
                attempts, error), True)
