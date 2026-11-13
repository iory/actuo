#!/usr/bin/env python3
"""Runs one scenario against a gateway and prints a diffable transcript.

The same scenario runs against the firmware's simulator and against a flashed
board, and the two transcripts are diffed. That is the whole idea: hardware time
is the scarce resource, the firmware's native tests already settle everything
the fake bus can decide, and what remains for a real board is line settings,
turnaround latency, and whether an ICS servo behaves the way the fake assumes.
A divergence means either the fake is lying about the hardware or the firmware
has a bug the fake cannot see -- and the divergence itself is the next test.

Usage
-----
::

    # simulated board
    actuo-gateway-probe --sim path/to/gateway_sim --servos 0,3 > sim.txt

    # real board, same scenario
    actuo-gateway-probe --channel /dev/ttyUSB0 > hw.txt

    diff sim.txt hw.txt

What is compared, and what is not
---------------------------------
Two lines of output, deliberately separated:

* ``STRUCT`` lines are **structural facts** -- which servos are present, which
  flags are set, which error code came back, whether a joint actually went
  limp. These must be identical on both sides; a difference is a finding.
* ``#`` lines are **measurements and identity** -- uptime, cycle time, raw
  counts, board name. These legitimately differ (a real servo does not sit at
  exactly the commanded count, and the boards have different names), so diffing
  them would drown the signal. They are printed for the operator to read.

Mixing the two is what makes hardware transcripts undiffable in practice, so
they are kept apart from the start.
"""

from __future__ import annotations

import argparse
import os
import sys


from actuo.drivers import gateway_protocol as gw  # noqa: E402
from actuo.drivers.gateway_client import (  # noqa: E402
    GatewayClient, GatewayError, NackError, SerialLink, SimulatorLink)


def struct_line(text):
    print("STRUCT {}".format(text))


def note(text):
    print("# {}".format(text))


def flag_text(servo):
    """The flags, as a stable string. Order is fixed so a diff is meaningful."""
    names = []
    if servo.present:
        names.append("present")
    if servo.holding:
        names.append("holding")
    if servo.position_valid:
        names.append("valid")
    if servo.stale:
        names.append("stale")
    return "|".join(names) if names else "none"


def report_state(label, state, baselines):
    note("{}: uptime={} cycle_us={} dropped={} generation={}".format(
        label, state.uptime_us, state.cycle_us, state.dropped,
        state.topology_generation))
    for servo in sorted(state.servos, key=lambda s: (s.port, s.id)):
        struct_line("{} servo {}.{} flags={} error={}".format(
            label, servo.port, servo.id, flag_text(servo), servo.error))
        base = baselines.get((servo.port, servo.id))
        delta = "" if base is None else " delta={}".format(servo.position - base)
        note("  {}.{} position={}{} age_us={}".format(
            servo.port, servo.id, servo.position, delta, servo.age_us))


def expect_nack(client, label, command, payload, expected):
    """Runs a command that must be refused, and records which refusal."""
    try:
        client.request(command, payload)
        struct_line("{} UNEXPECTEDLY ACCEPTED".format(label))
    except NackError as error:
        struct_line("{} nack={}".format(label, error.nack.code.name))
        if error.nack.code is not expected:
            struct_line("{} EXPECTED {}".format(label, expected.name))
    except GatewayError as error:
        struct_line("{} NO REPLY".format(label))
        note("  {}".format(error))


def run(client, port):
    hello = client.hello()
    note("board={} fw={}.{}.{} git={} ports={} max_servos={}".format(
        hello.board_name, *hello.fw_version, hello.git_hash, hello.port_count,
        hello.max_servos))
    note("link counters: resyncs={} crc={} version={} overlong={} aborted={}"
         .format(hello.rx_resyncs, hello.rx_crc_errors, hello.rx_version_errors,
                 hello.rx_overlong, hello.rx_aborted))
    struct_line("hello proto={} ports={}".format(hello.proto_version,
                                                 hello.port_count))

    uptime, echo = client.ping(b"\xde\xad\xbe\xef")
    struct_line("ping echo_ok={}".format(echo == b"\xde\xad\xbe\xef"))
    note("ping uptime={}".format(uptime))

    topology = client.enumerate(1 << port)
    baselines = {}
    for descriptor in topology.ports:
        struct_line("port {} protocol={} servos={} caps=w{}b{}c{}t{}".format(
            descriptor.port, descriptor.protocol, descriptor.servo_count,
            int(descriptor.write_returns_position), int(descriptor.batch_write),
            int(bool(descriptor.caps & gw.CAP_CURRENT_SENSE)),
            int(bool(descriptor.caps & gw.CAP_TEMPERATURE_SENSE))))
        note("port {} baud={} counts_per_rev_x100={} range={}..{} centre={}"
             .format(descriptor.port, descriptor.baudrate,
                     descriptor.counts_per_rev_x100, descriptor.pos_min,
                     descriptor.pos_max, descriptor.pos_center))
        for servo_id, position in descriptor.servos:
            struct_line("found {}.{}".format(descriptor.port, servo_id))
            note("  {}.{} enumerated at {}".format(descriptor.port, servo_id,
                                                   position))
            baselines[(descriptor.port, servo_id)] = position

    if not baselines:
        struct_line("no servos; stopping")
        return

    # Move every servo a small, identical distance from where it was found. A
    # relative move is the only kind that means the same thing on a simulator
    # and on a robot whose joints are wherever the operator left them.
    targets = [(p, i, base + 100, True) for (p, i), base in sorted(baselines.items())]
    state = client.write_targets(targets)
    report_state("after-write", state, baselines)

    state = client.read_state(1 << port)
    report_state("after-settle", state, baselines)

    # Torque off. This is the path that was silently broken until a review
    # caught it: the table said limp while the servo kept holding.
    client.set_torque([(p, i, False) for (p, i) in sorted(baselines)])
    state = client.read_state(1 << port)
    report_state("after-release", state, baselines)
    for servo in state.servos:
        struct_line("released {}.{} holding={}".format(
            servo.port, servo.id, int(servo.holding)))

    # Parameters, round trip.
    first = sorted(baselines)[0]
    try:
        before = client.get_param(first[0], first[1], gw.ICS_PARAM_STRETCH)
        client.set_param(first[0], first[1], gw.ICS_PARAM_STRETCH, 40)
        after = client.get_param(first[0], first[1], gw.ICS_PARAM_STRETCH)
        struct_line("param stretch write_readback_ok={}".format(int(after == 40)))
        note("stretch before={} after={}".format(before, after))
        client.set_param(first[0], first[1], gw.ICS_PARAM_STRETCH, before)
    except GatewayError as error:
        struct_line("param FAILED")
        note("  {}".format(error))

    # Refusals. Each of these must be rejected, and with the same code on both
    # sides -- an error path that only the simulator exercises is an error path
    # nobody has tested.
    expect_nack(client, "bad-port", gw.Command.READ_STATE,
                gw.encode_port_mask(0x80), gw.ErrorCode.BAD_PORT)
    expect_nack(client, "bad-command", gw.Command(0x7F) if False else 0x7F,
                b"", gw.ErrorCode.BAD_COMMAND)
    expect_nack(client, "bad-id", gw.Command.GET_PARAM,
                gw.encode_param_request(port, 200, gw.ICS_PARAM_STRETCH),
                gw.ErrorCode.BAD_ID)
    expect_nack(client, "unknown-flag", gw.Command.WRITE_TARGETS,
                bytes([1, port, first[1], 0x00, 0x1D, 0x02]),
                gw.ErrorCode.BAD_PAYLOAD)
    expect_nack(client, "out-of-range", gw.Command.WRITE_TARGETS,
                bytes([1, port, first[1], 0xFF, 0xFF, gw.TARGET_HOLD]),
                gw.ErrorCode.BAD_PAYLOAD)

    hello = client.hello()
    struct_line("final link errors crc={} version={} overlong={}".format(
        hello.rx_crc_errors, hello.rx_version_errors, hello.rx_overlong))
    note("final: resyncs={} aborted={}".format(hello.rx_resyncs,
                                               hello.rx_aborted))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run one scenario against a gateway and print a diffable "
                    "transcript.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sim", help="path to the firmware's gateway_sim")
    source.add_argument("--channel", help="serial device of a flashed board")
    parser.add_argument("--servos", default="0",
                        help="simulator only: ids to populate the fake bus with")
    parser.add_argument("--fail-after",
                        help="simulator only: ID:FRAMES, make a servo go silent")
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    parser.add_argument("--port", type=int, default=0, help="servo bus index")
    parser.add_argument("--timeout", type=float, default=2.0)
    args = parser.parse_args(argv)

    if args.sim:
        link = SimulatorLink(args.sim, servos=args.servos,
                             fail_after=args.fail_after)
    else:
        link = SerialLink(args.channel, args.baudrate)

    client = GatewayClient(link, timeout=args.timeout)
    try:
        run(client, args.port)
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
