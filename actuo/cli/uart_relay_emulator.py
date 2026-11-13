#!/usr/bin/env python3
"""The Arduino UART relay, on the host (see actuo/drivers/uart_relay.py).

    actuo-uart-relay-emulator --serial /tmp/actuo_rcb4_emulator
    actuo-uart-relay-emulator --serial ... --loss 0.05 --delay 0.02
    touch /tmp/relay_cut      # with --cut-file /tmp/relay_cut: the WiFi is gone

Prints the relay's counters every --stats seconds, and every failsafe.
"""

import argparse
import random
import time

import serial

from actuo.drivers.transports.kondo_rcb4_udp import DEFAULT_PORT
from actuo.drivers.uart_relay import UartRelay


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--serial', required=True,
                   help='the device the relay is wired to (a pty for the '
                        'RCB-4mini emulator)')
    p.add_argument('--baud', type=int, default=115200)
    p.add_argument('--port', type=int, default=DEFAULT_PORT)
    p.add_argument('--bind', default='0.0.0.0')
    p.add_argument('--loss', type=float, default=0.0,
                   help='probability of dropping each datagram, each way')
    p.add_argument('--delay', type=float, default=0.0,
                   help='seconds each answer is held back')
    p.add_argument('--cut-file', default=None,
                   help='while this file exists, nothing gets through')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--stats', type=float, default=5.0)
    args = p.parse_args()

    uart = serial.Serial(args.serial, args.baud, parity=serial.PARITY_EVEN,
                         timeout=0)
    relay = UartRelay(uart, args.port, args.bind, args.loss, args.delay,
                      args.cut_file, random.Random(args.seed),
                      log=lambda m: print(m, flush=True))
    print('relay: UDP {}:{} <-> {} (loss {}, delay {} s{})'.format(
        args.bind, args.port, args.serial, args.loss, args.delay,
        ', cut while {} exists'.format(args.cut_file) if args.cut_file
        else ''), flush=True)
    next_stats = time.monotonic() + args.stats
    try:
        while True:
            relay.spin_once()
            if time.monotonic() >= next_stats:
                next_stats += args.stats
                print('relay: {} failsafes {}'.format(
                    relay.stats, relay.failsafes), flush=True)
    except KeyboardInterrupt:
        pass
    print('relay: final {} failsafes {}'.format(relay.stats, relay.failsafes),
          flush=True)


if __name__ == '__main__':
    main()
