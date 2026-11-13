#!/usr/bin/env python3
"""Run an emulated RCB-4mini on a pty, with the servos a bus config names.

    actuo-kondo-rcb4-emulator --config <bus.yaml> \
        --link /tmp/actuo_rcb4_emulator

then point the config's ``bus.transport.channel`` at the link and open the
bus as you would for the board. ``--write-config OUT`` does that
for you: it writes a copy of the config with the channel set to the link and
every wheel's ``rotation_checked`` marked as emulated, so the copy can never be
mistaken for one checked on a servo. What the emulator does and does not
model is in ``actuo/drivers/kondo_rcb4_emulator.py``; read it before taking a
pass here as evidence about hardware.
"""

import argparse

import yaml

from actuo.drivers.kondo_rcb4_emulator import Rcb4Emulator, serve


def servo_ids(config_path):
    """ICS numbers of every joint in a bus config."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if config["bus"]["plugin"] != "kondo_rcb4_bus":
        raise SystemExit("{} is a {} config, not kondo_rcb4_bus".format(
            config_path, config["bus"]["plugin"]))
    if config["bus"].get("port_detect"):
        raise SystemExit("port_detect configs are not emulated: the id there "
                         "is a servo ID, not the ICS number it answers on")
    return ([int(j["actuator"]["id"]) for j in config["joints"]],
            int(config["bus"].get("ics_bitrate", 1250000)))


def write_config(config_path, link, out_path):
    """A copy of the config that talks to the emulator."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config["bus"]["transport"]["channel"] = link
    for joint in config["joints"]:
        actuator = joint["actuator"]
        if actuator["plugin"] == "kondo_ics_wheel":
            actuator.setdefault(
                "rotation_checked",
                "EMULATED by actuo-kondo-rcb4-emulator -- not a real servo")
    with open(out_path, "w") as f:
        f.write("# Written by actuo-kondo-rcb4-emulator from {}.\n"
                "# For the emulator ONLY: wheels are marked rotation_checked "
                "without a servo.\n".format(config_path))
        yaml.safe_dump(config, f, sort_keys=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="kondo_rcb4_bus config")
    p.add_argument("--link", default="/tmp/actuo_rcb4_emulator",
                   help="symlink to create for the pty")
    p.add_argument("--write-config", metavar="OUT", default=None,
                   help="write a copy of --config that uses the emulator")
    p.add_argument("--trace", type=int, nargs="*", default=(),
                   help="ICS numbers whose commanded pulse is printed (with "
                        "a monotonic time) whenever it changes")
    args = p.parse_args()
    if args.write_config:
        write_config(args.config, args.link, args.write_config)
        print("wrote {}".format(args.write_config))
    ids, ics_bitrate = servo_ids(args.config)
    serve(Rcb4Emulator(ids, servo_ics_bitrate=ics_bitrate, trace=args.trace,
                       log=lambda m: print(m, flush=True)), args.link)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
