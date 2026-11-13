#!/usr/bin/env python3
"""First contact with a Kondo RCB-4mini board.

By default this is read-only: it finds the rate the board answers at without
switching it, then prints the firmware, the config word decoded, the ICS
assignment table, the supply voltage, and the pulse of every assigned servo.
The only packets sent are the ack, the version request and RAM *reads*.

``--scan`` does what ``kondo_rcb4_bus`` does at startup -- switch the COM rate,
stop the ROM program, set the ICS rate, assign ICS numbers -- for every number
0..34, and keeps the ones a servo answers on. Those are RAM writes; they are
printed, and power-cycling the board undoes them. No servo is switched on, so
nothing moves.

Usage
-----
::

    actuo-kondo-rcb4-probe --channel /dev/ttyUSB0
    actuo-kondo-rcb4-probe --channel /dev/ttyUSB0 --scan

See ``actuo/drivers/references.md``, "Kondo RCB-4mini", for every address and
bit used here.
"""

from __future__ import annotations

import argparse
import os
import sys


from actuo.drivers.actuators.kondo_ics import (  # noqa: E402
    CENTER_PULSE, KondoIcsActuator, KondoIcsCalibration, PULSE_PER_DEG)
from actuo.drivers.buses.kondo_rcb4_bus import (  # noqa: E402
    DEFAULT_CONNECT_BITRATES, KondoRcb4Bus)
from rcb4.protocol.rcb4mini import (  # noqa: E402
    COM_BITRATE_MASK, COM_BITRATE_SHIFT, CONFIG_RUN_ROM, CONFIG_VECTOR_JUMP,
    FIRMWARE_VERSION_REPLY, ICS_BITRATE_CODES, ICS_BITRATE_MASK,
    ICS_BITRATE_SHIFT, ICS_SLOT_COUNT, ICS_UNASSIGNED, ProtocolError,
    com_bitrate_of, decode_servo_records, plan_record_reads, servo_record_address)
from actuo.drivers.transports.kondo_rcb4 import KondoRcb4Transport  # noqa: E402

#: Config bits 4-5, frame length in ms, as kxreus's ``:sysreg`` names them.
#: Only ``01`` (15 ms) was measured on a board; see references.md.
FRAME_MS_BY_CODE = {0: 10, 1: 15, 2: 20, 3: 25}
FRAME_MASK = 0x0030
FRAME_SHIFT = 4


def describe_config(config: int) -> list:
    """One line per config field this package uses or reports."""
    frame_code = (config & FRAME_MASK) >> FRAME_SHIFT
    com_code = (config & COM_BITRATE_MASK) >> COM_BITRATE_SHIFT
    ics_code = (config & ICS_BITRATE_MASK) >> ICS_BITRATE_SHIFT
    try:
        com = "{} bit/s".format(com_bitrate_of(config))
    except ProtocolError:
        com = "unknown"
    ics = next((str(rate) + " bit/s" for rate, code in ICS_BITRATE_CODES.items()
                if code == ics_code), "unknown")
    return [
        "ICS on        : {}".format(bool(config & 0x0001)),
        "ROM program   : {}".format("running" if config & CONFIG_RUN_ROM else "stopped"),
        "vector jumps  : {}".format("on" if config & CONFIG_VECTOR_JUMP else "off"),
        "frame         : code {} ({} ms per kxreus)".format(
            frame_code, FRAME_MS_BY_CODE[frame_code]),
        "COM rate      : code {} ({})".format(com_code, com),
        "ICS rate      : code {} ({})".format(ics_code, ics),
    ]


def connect_read_only(bus: KondoRcb4Bus, rates) -> int:
    """Open at the first rate the board acks at, without switching it."""
    for rate in rates:
        bus.transport.reopen(rate)
        if bus._ack():
            return rate
    raise RuntimeError(
        "board answers at none of {} bit/s. Check power, the cable, and that "
        "the adapter's LED is green (serial mode, not ICS).".format(
            ", ".join(str(r) for r in rates)))


def print_servos(bus: KondoRcb4Bus, ids) -> None:
    if not ids:
        print("servos        : none assigned")
        return
    for address, size in plan_record_reads(ids, ("current_angle",)):
        data = bus._read_ram(address, size)
        for servo_id, values in sorted(decode_servo_records(
                address, data, ids, ("current_angle",)).items()):
            pulse = values["current_angle"]
            print("  ICS {:2d}  pulse {:5d}  {:+7.1f} deg from neutral".format(
                servo_id, pulse, (pulse - CENTER_PULSE) / PULSE_PER_DEG))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="First contact with a Kondo RCB-4mini (read-only unless --scan).")
    parser.add_argument("--channel", required=True,
                        help="serial device, e.g. /dev/ttyUSB0")
    parser.add_argument("--bitrate", type=int, default=1250000,
                        help="rate tried first, and switched to by --scan")
    parser.add_argument("--scan", action="store_true",
                        help="assign every ICS number in RAM and keep the ones a "
                             "servo answers on (RAM writes; nothing moves)")
    args = parser.parse_args(argv)

    transport = KondoRcb4Transport(args.channel, bitrate=args.bitrate)
    if args.scan:
        actuators = [KondoIcsActuator(i, KondoIcsCalibration())
                     for i in range(ICS_SLOT_COUNT)]
    else:
        actuators = []
    bus = KondoRcb4Bus(transport, actuators)
    transport.open()
    try:
        if args.scan:
            for item in bus.verify_servos(allow_missing=True):
                if item.name == "bus/startup_missing":
                    continue
                print("CHANGED {}: {}".format(item.name, item.message))
            rate = transport.current_bitrate
        else:
            rate = connect_read_only(
                bus, (args.bitrate,) + tuple(
                    r for r in DEFAULT_CONNECT_BITRATES if r != args.bitrate))
        version = bus.firmware_version()
        print("COM port      : {} at {} bit/s".format(args.channel, rate))
        print("firmware      : {!r}{}".format(
            version, "" if version == FIRMWARE_VERSION_REPLY
            else "  (NOT the RCB-4mini firmware this package knows)"))
        if version != FIRMWARE_VERSION_REPLY:
            return 1
        config = bus.read_config()
        print("config        : {:#06x}".format(config))
        for line in describe_config(config):
            print("  " + line)
        print("supply        : {:.2f} V (analog input 0, kxreus's scale)".format(
            bus.battery_voltage()))
        table = bus.read_ics_assign_table()
        assigned = [i for i, entry in enumerate(table) if entry != ICS_UNASSIGNED]
        foreign = [i for i in assigned if table[i] != servo_record_address(i)]
        print("ICS assigned  : {}".format(assigned or "none"))
        if foreign:
            print("  mapped to another record: {}".format(
                ", ".join("{} -> {:#06x}".format(i, table[i]) for i in foreign)))
        print_servos(bus, [i for i in assigned if i not in foreign])
        print("naks          : {}".format(bus.naks))
        print("unanswered    : {}".format(bus.unanswered))
        return 0
    finally:
        transport.close()


if __name__ == "__main__":
    sys.exit(main())
