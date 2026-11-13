#!/usr/bin/env python3
"""Find out what is actually on a DYNAMIXEL bus, without being told.

This exists because every other question about a DYNAMIXEL robot -- which
control table, which units, which modes -- is answered by the model number, and
the servo will tell you its model number if you ask it correctly. Asking the
operator instead produces answers like "AX-12, and I think protocol 2.0", which
cannot both be true (an AX speaks 1.0 only). So this probe asks the hardware.

It is read-only. It sends PING and READs the model-number register; it never
writes a control-table value, never enables torque, and therefore cannot move a
joint or change a servo's configuration. Running it on an unknown robot is safe.

What it reports, and why each line is worth printing
----------------------------------------------------
* **protocol and baudrate that worked.** The sweep is the point: an unlabelled
  bus is identified by trying, not by assuming.
* **model number, product name, and whether the table knows it.** An unknown
  model prints the exact YAML row to add and the e-Manual URL to fill it from,
  so "unsupported" is a five-minute fix rather than a wall.
* **whether the adapter echoes our own transmission.** This stack has twice
  diagnosed a fault from a frame it sent itself (once as a CubeMars "encoder
  fault" that was really the low byte of our own command frame read back as a
  status). On protocol 1.0 an echoed instruction packet decodes as a
  plausible status packet, so it is tested for explicitly and the fix
  (``discard_echo: true``) is named.
* **round-trip latency.** An FTDI adapter's default 16 ms latency timer caps a
  request/response bus far below its baud rate, and it is invisible unless
  measured -- so it is measured, not assumed.

Examples
--------
    actuo-dxl-probe --channel /dev/ttyUSB0
    actuo-dxl-probe --channel /dev/tty.usbserial-FT1234 \
        --protocol 1 --baud 1000000 --ids 1-30
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional, Sequence, Tuple

from actuo.drivers import dynamixel_protocol as dxl
from actuo.drivers.dynamixel_models import UnknownModelError, load_model_table
from actuo.drivers.transports.serial_transport import SerialTransport

#: Rates to sweep when none is given. A search list, not a claim about any
#: model: it is ordered by how often these buses actually run at them.
CANDIDATE_BAUDRATES = (1000000, 57600, 115200, 2000000, 3000000, 4000000,
                       500000, 400000, 250000, 200000, 19200, 9600)

#: Scanned by default. The full range is 0..252, but a 253-id sweep at every
#: candidate baudrate is minutes of wall clock; ids above 30 are rare on a hand
#: built robot and ``--ids`` covers the exception.
DEFAULT_IDS = tuple(range(0, 31))


def parse_ids(text: str) -> Tuple[int, ...]:
    """Parse ``1,2,5-9`` into ids."""
    ids: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, high = part.split("-", 1)
            ids.extend(range(int(low), int(high) + 1))
        else:
            ids.append(int(part))
    bad = [i for i in ids if not 0 <= i <= dxl.MAX_ID]
    if bad:
        raise ValueError("ids out of range 0..{}: {}".format(dxl.MAX_ID, bad))
    return tuple(sorted(set(ids)))


def _ping(transport: SerialTransport, protocol: int, dxl_id: int,
          timeout: float) -> Optional[Tuple[bytes, float]]:
    """PING one id. Returns (reply packet, round-trip seconds) or None."""
    from actuo.core.contracts import Frame

    request = Frame(payload=dxl.build_ping(protocol, dxl_id), address=dxl_id)
    started = time.monotonic()
    reply = transport.transact([request], timeout)[0]
    elapsed = time.monotonic() - started
    if reply is None:
        return None
    return reply.payload, elapsed


def _read_model_number(transport: SerialTransport, protocol: int, dxl_id: int,
                       timeout: float) -> Optional[int]:
    """READ address 0 (Model Number) -- the protocol 1.0 path.

    Protocol 2.0 returns the model in the PING reply; 1.0's PING carries no
    parameters, so it must be read. Address 0 / size 2 is the one control-table
    entry that is identical across every DYNAMIXEL, which is exactly why it is
    safe to read before knowing which servo this is.
    """
    from actuo.core.contracts import Frame

    request = Frame(payload=dxl.build_read(protocol, dxl_id, 0, 2), address=dxl_id)
    reply = transport.transact([request], timeout)[0]
    if reply is None:
        return None
    parsed = dxl.parse_status(protocol, reply.payload)
    if parsed is None or parsed[0] != dxl_id or len(parsed[2]) < 2:
        return None
    return dxl.from_le(parsed[2][:2])


def _detect_echo(transport: SerialTransport, protocol: int,
                 timeout: float) -> bool:
    """Does the adapter loop our transmission back into the receiver?

    Probed by pinging the broadcast id, which no servo answers with a status
    packet on protocol 1.0 -- so anything that comes back byte-identical to what
    was sent came from the adapter, not from a servo.
    """
    from actuo.core.contracts import Frame

    sent = dxl.build_ping(protocol, dxl.BROADCAST_ID)
    transport.reset_input()
    transport.write([Frame(payload=sent, address=None)])
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        left = deadline - time.monotonic()
        reply = transport.recv(left)
        if reply is None:
            break
        if reply.payload == sent:
            return True
    return False


def scan(channel: str, protocols: Sequence[int], baudrates: Sequence[int],
         ids: Sequence[int], timeout: float, discard_echo: bool) -> int:
    """Sweep the bus. Returns a process exit code."""
    table = load_model_table()
    found_any = False

    for protocol in protocols:
        for baudrate in baudrates:
            transport = SerialTransport(channel, baudrate, timeout=timeout,
                                        protocol=protocol,
                                        discard_echo=discard_echo)
            try:
                transport.open()
            except Exception as error:            # noqa: BLE001 - report and move on
                print("cannot open {} at {} baud: {}".format(
                    channel, baudrate, error), file=sys.stderr)
                return 2

            try:
                echoes = _detect_echo(transport, protocol, timeout)
                hits = []
                latencies = []
                for dxl_id in ids:
                    result = _ping(transport, protocol, dxl_id, timeout)
                    if result is None:
                        continue
                    packet, elapsed = result
                    parsed = dxl.parse_status(protocol, packet)
                    if parsed is None or parsed[0] != dxl_id:
                        continue
                    latencies.append(elapsed)

                    ping_info = dxl.parse_ping_status(protocol, packet)
                    if ping_info is not None:
                        _id, model_number, firmware = ping_info
                    else:
                        model_number = _read_model_number(
                            transport, protocol, dxl_id, timeout)
                        firmware = None
                    hits.append((dxl_id, model_number, firmware, parsed[1]))

                if not hits:
                    continue

                found_any = True
                print("")
                print("=== protocol {}.0, {} baud, {} ===".format(
                    protocol, baudrate, channel))
                if echoes:
                    print("  !! this adapter ECHOES transmitted packets back.")
                    print("     On protocol 1.0 an echoed instruction packet has")
                    print("     the same shape as a status packet and decodes as")
                    print("     plausible state. Set `discard_echo: true` on the")
                    print("     transport in the robot config, or fix the wiring.")
                if latencies:
                    print("  round trip: min {:.2f} ms, max {:.2f} ms".format(
                        1000 * min(latencies), 1000 * max(latencies)))
                    if min(latencies) > 0.010:
                        print("     (>10 ms floor: on an FTDI adapter this is the")
                        print("      16 ms latency timer, not the servo. It caps a")
                        print("      request/response bus far below its baud rate.)")
                print("")
                _report_hits(hits, table)
            finally:
                transport.close()

    if not found_any:
        print("No servo answered on {}.".format(channel))
        print("")
        print("Before suspecting the servos, rule out in this order:")
        print("  1. the port      -- is {} the adapter? (ls /dev/tty.*)".format(
            channel))
        print("  2. servo power   -- a DYNAMIXEL with only data connected is")
        print("                      silent, not slow. Check the supply rail.")
        print("  3. the baudrate  -- rerun without --baud to sweep all of {}".format(
            list(CANDIDATE_BAUDRATES)))
        print("  4. the protocol  -- rerun without --protocol to try both")
        print("  5. the id range  -- rerun with --ids 0-252 (slow but complete)")
        return 1
    return 0


def _report_hits(hits, table) -> None:
    """Print one line per servo, then the YAML for any model we cannot drive."""
    unknown = []
    for dxl_id, model_number, firmware, error in hits:
        if model_number is None:
            print("  id {:3d}  model number unreadable".format(dxl_id))
            continue
        try:
            model = table.by_number(model_number)
            name = model.name
            detail = "resolution {} over {:.0f} deg, modes {}".format(
                model.resolution, model.range_deg,
                ",".join(sorted(m.value for m in model.modes)))
        except UnknownModelError:
            name = "UNKNOWN"
            detail = "not in the model table -- see below"
            unknown.append(model_number)
        print("  id {:3d}  model {:5d}  {:<16s} fw {:<4s} err {:#04x}  {}".format(
            dxl_id, model_number, name,
            "?" if firmware is None else str(firmware), error, detail))

    print("")
    print("Config lines for the joints you keep (your robot config):")
    for dxl_id, model_number, _firmware, _error in hits:
        try:
            name = table.by_number(model_number).name if model_number else "?"
        except UnknownModelError:
            name = "<add this model first>"
        print("      - name: <urdf joint name>")
        print("        actuator: {{plugin: dynamixel, id: {}, model: {}}}".format(
            dxl_id, name))

    for model_number in sorted(set(unknown)):
        print("")
        try:
            table.by_number(model_number)
        except UnknownModelError as error:
            print(error)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Identify the servos on a DYNAMIXEL bus (read-only).")
    parser.add_argument("--channel", required=True,
                        help="serial device, e.g. /dev/ttyUSB0")
    parser.add_argument("--protocol", type=int, choices=(1, 2), default=None,
                        help="only try this protocol version (default: both)")
    parser.add_argument("--baud", type=int, default=None,
                        help="only try this baudrate (default: sweep)")
    parser.add_argument("--ids", type=str, default=None,
                        help="ids to scan, e.g. '1,2,5-9' (default: 0-30)")
    parser.add_argument("--timeout", type=float, default=0.05,
                        help="per-request timeout in seconds")
    parser.add_argument("--discard-echo", action="store_true",
                        help="drop packets identical to what was just sent")
    args = parser.parse_args(argv)

    protocols = (args.protocol,) if args.protocol else (dxl.PROTOCOL_2,
                                                        dxl.PROTOCOL_1)
    baudrates = (args.baud,) if args.baud else CANDIDATE_BAUDRATES
    ids = parse_ids(args.ids) if args.ids else DEFAULT_IDS

    return scan(args.channel, protocols, baudrates, ids, args.timeout,
                args.discard_echo)


if __name__ == "__main__":
    sys.exit(main())
