#!/usr/bin/env python3
"""Publish what the board reports, as JointState, and command NOTHING.

For the question RViz answers faster than any tool: *is this the wheel I think
it is, and does it turn the way the model says?* Hand-move the robot with torque
off, watch the model, and any disagreement between the config and the machine is
visible in a second -- a wheel drawn at the wrong corner, or turning the wrong
way, is not something a number on a terminal makes obvious.

It is listen-only by construction, not by promise: it opens the transport, calls
``read_batch``, and there is no code path here that builds a motion frame. It
does not run ``verify_servos`` either, because that CLEARS ``ServoStruct.feedback``
and so changes the board -- a watching tool must leave the robot exactly as it
found it.

    ros2 run actuo_ros actuo_watch_pose --config <robot>.yaml

Every joint comes from the config, so this works on any robot actuo can
describe.

What is published, and what it means
------------------------------------
* A joint with real position sense publishes its position. That is a
  measurement.
* A **wheel** publishes its ``pot_rad`` channel: the raw potentiometer in
  radians, which WRAPS once per wheel revolution, starts from wherever the pot
  happened to be, and does not report roughly a sixth of each turn. It is here
  to be looked at and for nothing else, which is why the servicer's own
  ``/joint_states`` still refuses to carry it: a wrapping, arbitrarily zeroed
  angle is not a joint pose. ``--wheels none`` leaves them out.

Reading the answer
------------------
``direction:`` in the robot config is the sign relating a commanded speed to the
joint's positive rotation, and it is exactly what this checks -- because the
other half of that chain is already measured. ``actuo-wheel-calibrate``
records, per wheel, whether a POSITIVE command offset makes the pot count up
(``forward_top_rev_s`` positive). Where it does:

    positive command -> pot rises          (measured by actuo-wheel-calibrate)
    pot -> what RViz draws                 (this tool, with direction applied)
    => RViz turning the way your hand does  <=>  direction is right

Turn a wheel by hand and watch. Right corner, right way: leave it. Right corner,
wrong way: flip that joint's ``direction``. Wrong corner: the ids are swapped in
the config, and no sign will fix that.
"""

from __future__ import annotations

import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from actuo.drivers.config import build_bus
from actuo.drivers.link import is_link_error


def parse_args(argv):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True,
                   help="the robot's actuo config (joints, ids, transport)")
    p.add_argument("--topic", default="joint_states",
                   help="where to publish. The default is relative to this "
                        "node's namespace, so --ros-args -r __ns:=/<robot> "
                        "puts it beside the rest of the robot's topics.")
    p.add_argument("--rate", type=float, default=30.0,
                   help="Hz. This is a thing to look at, not to control from")
    p.add_argument("--wheels", choices=["pot", "none"], default="pot",
                   help="pot (default) publishes a continuous-rotation servo's "
                        "raw potentiometer as its joint angle, so the wheel "
                        "turns on screen. It is NOT a measurement of a joint "
                        "pose; see this tool's docstring. none omits them.")
    p.add_argument("--reconnect-interval", type=float, default=1.0,
                   help="seconds between reopen attempts after the robot's "
                        "power is switched off. This tool commands nothing, so "
                        "there is no torque state to be careful about: it "
                        "simply reopens and carries on. 0 disables.")
    p.add_argument("--ros-args", dest="ros_args", nargs=argparse.REMAINDER,
                   default=[])
    return p.parse_args(argv)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)

    bus, joint_names = build_bus(args.config)
    rclpy.init(args=args.ros_args or None)
    node = Node("actuo_watch_pose")
    publisher = node.create_publisher(JointState, args.topic, 10)

    # Which channel each joint's angle comes from, decided ONCE from what the
    # actuator says it senses rather than per cycle from what happened to
    # arrive -- so a joint cannot silently change what it is reporting.
    sources = []
    for name, actuator in zip(joint_names, bus.actuators):
        if actuator.capabilities.has_position_sense:
            sources.append((name, "position"))
        elif args.wheels == "pot" and any(
                channel.name == "pot_rad" for channel in actuator.channels):
            sources.append((name, "pot_rad"))
    published = [name for name, _ in sources]

    node.get_logger().info(
        "watching {} -- LISTEN ONLY, nothing is commanded".format(args.config))
    for name, source in sources:
        if source == "pot_rad":
            node.get_logger().warn(
                "{}: publishing its raw POTENTIOMETER as a joint angle. It "
                "wraps once per wheel revolution, its zero is arbitrary and "
                "part of each turn is not reported. Use it to see which wheel "
                "turns which way, and for nothing else.".format(name))
    dropped = [n for n in joint_names if n not in published]
    if dropped:
        node.get_logger().info(
            "not published: {} (nothing measurable to report)".format(
                ", ".join(dropped)))

    bus.transport.open()
    try:
        period = 1.0 / max(1e-3, args.rate)
        down = False
        attempts = 0
        silent_since = None
        warned = False
        while rclpy.ok():
            try:
                bus.read_batch(timeout=0.1)
            except Exception as error:                         # noqa: BLE001
                # Switching the robot off pulls its USB serial device out from
                # under an open descriptor. Nothing here commands, so recovery
                # is just close/open -- the servicer's version has to think
                # about torque as well (actuo.drivers.link).
                if args.reconnect_interval <= 0 or not is_link_error(error):
                    raise
                if not down:
                    down = True
                    node.get_logger().error(
                        "link lost ({}); reopening every {:.1f}s. Nothing was "
                        "being commanded.".format(error,
                                                  args.reconnect_interval))
                attempts += 1
                if attempts > 3:
                    # A wedged adapter does not answer a reopened port either.
                    # Same escalation the servicer uses; see
                    # actuo.drivers.usb_reset.
                    try:
                        from actuo.drivers.usb_reset import reset_usb_device
                        node.get_logger().warn("USB reset of {}".format(
                            reset_usb_device(bus.transport.channel)))
                        time.sleep(2.0)
                    except Exception as reset_error:           # noqa: BLE001
                        if attempts == 4:
                            node.get_logger().error(
                                "cannot reset the USB device: {}".format(
                                    reset_error))
                try:
                    bus.transport.close()
                    bus.transport.open()
                except Exception:                              # noqa: BLE001
                    try:
                        bus.transport.close()
                    except Exception:                          # noqa: BLE001
                        pass
                    rclpy.spin_once(node,
                                    timeout_sec=args.reconnect_interval)
                    continue
                down = False
                attempts = 0
                node.get_logger().info("link back")
                continue
            message = JointState()
            message.header.stamp = node.get_clock().now().to_msg()
            for name, source in sources:
                actuator = bus.actuators[joint_names.index(name)]
                state = bus.last_state(actuator.id)
                if state is None:
                    continue
                if source == "position":
                    if state.position is None:
                        continue
                    value = float(state.position)
                else:
                    if "pot_rad" not in state.aux:
                        continue
                    value = float(state.aux["pot_rad"])
                message.name.append(name)
                message.position.append(value)
            if message.name:
                if silent_since is not None:
                    node.get_logger().info(
                        "the board is answering again")
                    silent_since = None
                publisher.publish(message)
            else:
                # Nothing answered. On a board whose USB bridge is powered by
                # the PC, that is what switching the ROBOT off looks like: the
                # port stays open and no exception is raised. Saying so
                # matters: this tool otherwise just stops publishing, which
                # reads as a broken tool rather than an unpowered robot.
                now = time.monotonic()
                if silent_since is None:
                    silent_since = now
                    warned = False
                elif not warned and now - silent_since > 2.0:
                    warned = True
                    node.get_logger().error(
                        "no joint has answered for {:.0f}s. The port is open "
                        "and nothing raised, so this is the board not "
                        "responding -- is the robot switched on? Still "
                        "reading; it will pick up by itself.".format(
                            now - silent_since))
            rclpy.spin_once(node, timeout_sec=period)
    except KeyboardInterrupt:
        pass
    finally:
        bus.transport.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
