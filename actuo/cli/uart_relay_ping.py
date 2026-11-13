#!/usr/bin/env python3
"""Wait until a UART relay (the Arduino WiFi-to-UART relay) answers, then exit 0.

    actuo-uart-relay-ping --relay 192.168.1.50 --timeout 0

The WiFi counterpart of waiting for a serial port to appear: nothing behind
the relay is touched (a ping is answered by the relay itself, not the board).
``--timeout 0`` waits forever; otherwise exits 1 when it runs out.
"""

import argparse
import socket
import time

from actuo.drivers import uart_relay_protocol as relay
from actuo.drivers.transports.kondo_rcb4_udp import DEFAULT_PORT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--relay', required=True, help='HOST[:PORT]')
    p.add_argument('--timeout', type=float, default=0.0)
    p.add_argument('--interval', type=float, default=0.5)
    args = p.parse_args()
    host, _, port = args.relay.partition(':')
    address = (host, int(port) if port else DEFAULT_PORT)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(args.interval)
    start = time.monotonic()
    seq = 0
    said = False
    while args.timeout <= 0 or time.monotonic() - start < args.timeout:
        seq = (seq + 1) & 0xFFFF
        sock.sendto(relay.ping(seq), address)
        try:
            data, _ = sock.recvfrom(64)
            kind, got, status, failsafes, _ = relay.parse_answer(data)
            if got == seq and status == relay.STATUS_OK:
                print('relay at {}:{} answers (failsafes sent since boot: {})'
                      .format(address[0], address[1], failsafes), flush=True)
                return 0
        except (socket.timeout, ConnectionRefusedError, ValueError):
            time.sleep(args.interval if not said else 0.0)
        if not said:
            print('waiting for the relay at {}:{} ...'.format(*address),
                  flush=True)
            said = True
    print('no answer from the relay at {}:{} within {} s'.format(
        address[0], address[1], args.timeout), flush=True)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
