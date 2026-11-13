#!/usr/bin/env python3
"""Switch the bench power relay that feeds a board under test.

The relay is a USB-RELAY1 (`04d8:000a`, serial `X-RL1`) on a USB hub. It
enumerates as a CDC serial port and gates the supply to whatever is wired to its
contacts -- for example a Kondo ARM-H7 board.

This exists because power-cycling is part of the hardware workflow, not a
side quest: a board that has locked up, a firmware that needs a clean boot, a
USB adapter that has wedged. Doing it from a script means it can be scripted
into a test, and means nobody has to remember which byte means on.

Protocol
--------
Determined empirically -- no datasheet was available for this unit:

===========  ==========================================================
Sent         Observed
===========  ==========================================================
``ver``      ``v1.1.0\\r\\n`` -- a real firmware version
``1``        echoes ``1``
``0``        echoes ``0``
garbage      **no response at all**
===========  ==========================================================

The last row is what makes the rest meaningful: the board ignores what it does
not understand, so the digit echo is an acknowledgement rather than a terminal
echoing everything back. A first attempt at Numato's ``relay on 0`` syntax
looked like it worked -- it returned ``0`` -- until ``relay read 1`` returned
``1``. That was the trailing digit of the command being echoed, not a state.
Which is the whole reason this file documents evidence rather than a guess.

**Unverified, and it needs eyes on the hardware:** that ``1`` is ON rather than
OFF. Both digits are acknowledged identically, the unit exposes no state
read-back command, and nothing downstream of the contacts is visible to this
machine, so software cannot close the loop. Check the module's LED (or listen
for the click) once, and record the answer here.
"""

from __future__ import annotations

import argparse
import glob
import sys
import time

import serial

#: The relay presents itself with this serial number, which is stable across
#: replugs -- unlike the /dev name, which is not on every OS.
DEFAULT_PORT_GLOB = "/dev/cu.usbmodem*RL1*"

CMD_ON = b"1"
CMD_OFF = b"0"


def find_port(explicit: str = None) -> str:
    if explicit:
        return explicit
    matches = sorted(glob.glob(DEFAULT_PORT_GLOB))
    if not matches:
        raise SystemExit(
            "no relay found matching {}.\n"
            "Plug it in, or pass --port. Present serial devices:\n  {}".format(
                DEFAULT_PORT_GLOB,
                "\n  ".join(sorted(glob.glob("/dev/cu.*")) or ["(none)"])))
    if len(matches) > 1:
        raise SystemExit(
            "more than one relay matches {}: {}. Pass --port to say which.".format(
                DEFAULT_PORT_GLOB, matches))
    return matches[0]


def send(port: str, command: bytes, timeout: float = 1.0) -> bytes:
    """Send one command and return whatever was echoed.

    The echo is the only acknowledgement this unit gives, so it is returned
    rather than discarded: a command that produces no echo did not land, and
    silently reporting success would be worse than useless on something that
    switches power.
    """
    with serial.Serial(port, 19200, timeout=timeout) as link:
        link.reset_input_buffer()
        link.write(command)
        link.flush()
        return link.read(len(command))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Switch the bench power relay feeding the board under test.")
    parser.add_argument("action", choices=("on", "off", "cycle", "version"),
                        help="'cycle' turns it off, waits, and turns it back on")
    parser.add_argument("--port", default=None,
                        help="serial device (default: autodetect {})".format(
                            DEFAULT_PORT_GLOB))
    parser.add_argument("--off-time", type=float, default=2.0,
                        help="seconds to stay off during 'cycle'. Long enough "
                             "for the board's rails to actually discharge; too "
                             "short and it never really rebooted")
    args = parser.parse_args(argv)

    port = find_port(args.port)

    if args.action == "version":
        with serial.Serial(port, 19200, timeout=1.0) as link:
            link.reset_input_buffer()
            link.write(b"ver\r")
            link.flush()
            print("{}: {}".format(port, link.read(32).decode(errors="replace").strip()))
        return 0

    def switch(command: bytes, label: str) -> int:
        echo = send(port, command)
        if echo != command:
            print("{}: sent {!r} for {} but got echo {!r} -- the relay did not "
                  "acknowledge, so do NOT assume it switched".format(
                      port, command, label, echo), file=sys.stderr)
            return 1
        print("{}: {}".format(port, label))
        return 0

    if args.action == "on":
        return switch(CMD_ON, "on")
    if args.action == "off":
        return switch(CMD_OFF, "off")

    rc = switch(CMD_OFF, "off")
    if rc:
        return rc
    time.sleep(args.off_time)
    return switch(CMD_ON, "on")


if __name__ == "__main__":
    sys.exit(main())
