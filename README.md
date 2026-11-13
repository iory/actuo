# actuo

One Python layer for Dynamixel, Kondo (RCB-4mini / ARM-H7, ICS), Feetech and
CAN (CubeMars) actuators: batch bus I/O, per-motor models and calibration,
guarded servo on/off, link-loss recovery, emulators and probe tools.
**No ROS.** A ROS 2 layer (ros2_control shim, nodes, messages, RViz panel)
sits on top of it separately.

```
scikit-robot              robot models, kinematics, planning, viewers
    ^ actuo depends on it (models, and the skrobot-style `ri`)
actuo                     this package (PyPI, ROS-free)
    ^ extras
rcb4 / feetech-cli / dynamixel-sdk / python-can     vendor libraries, usable alone
```

## Status: prototype

- `actuo.core` -- contracts (transport, actuator, bus, device), the plugin
  registry, diagnostics
- `actuo.drivers` -- transports, actuator models, buses, sensors, robot config
  loading, the servo supervisor, link-loss recovery and emulators
- `actuo.safety` -- self-collision safety, analogous to hrpsys's
  `CollisionDetector`: `SelfCollisionMonitor` checks a scikit-robot model
  against its own links (pairs touching in a reference pose are ignored as
  structural), and `CollisionGate` sits last before the motors, holding the
  last collision-free command when the next one would collide (latched until
  `reset`, fail-to-stop when the check raises, optional distance-based
  slow-down). Distance queries need the `safety` extra (python-fcl).
- `actuo.cli.*` -- probe, calibration and emulator tools, installed as
  `actuo-<name>` commands

Not done yet: the package layout (`actuo.drivers.*`) is not yet a designed
public API; vendor adapters are not yet behind entry-point plugins.

## A robot without ROS

```python
from skrobot.model import RobotModel
from actuo.robot_interface import ActuoRobotInterface

robot = RobotModel.from_urdf("humanoid.urdf")
ri = ActuoRobotInterface(robot, "humanoid_rcb4.yaml")   # opens the bus
ri.servo_on(["rarm_joint0"])            # holds where it is: never moves the robot
ri.angle_vector({"rarm_joint0": 0.5}, time=1.0)
ri.wait_interpolation()
print(ri.potentio_vector(), ri.outside_limits())
```

`ActuoRobotInterface` subclasses scikit-robot's `RobotInterfaceBase` (vector or
`{joint: angle}` goals, durations from velocity limits, `angle_vector_sequence`,
unclamped `potentio_vector`, `outside_limits`) and implements its hooks
(scikit-robot >= 0.4.14).
A thread in the process services the bus. `angle_vector` plans a minimum-jerk
path; every cycle the sample goes through the same `ServoSupervisor` the ROS 2
layer's bus process uses (limp joints get no position, enable is guarded,
commands are fenced and rate-limited). `examples/ri_with_emulator.py` runs a
real robot config and URDF (an RCB-4mini humanoid) against the RCB-4mini
emulator.

On a bus whose servos may be missing (`hot_plug: true` on the RCB-4mini bus),
the config can list more joints than are plugged in:

```python
ri = ActuoRobotInterface(robot, "arm_rcb4.yaml", required_joints=["base_joint"])
ri.attached_joints()     # the joints whose servos answer now
```

A joint is attached once its servo has answered for a few cycles: its command
starts from where it is, and it is limp until `servo_on`. A vector goal moves
the attached joints; naming a detached joint is an error. A servo that stops
answering is switched off, its motion is dropped and its measurement becomes
NaN, so when it is plugged back in it comes back limp, from where it then is.

## With ROS 2: the servicer and shared memory

The **servicer** is the process that owns the bus and services the
shared-memory segment: every cycle it reads the actuators' state into the
segment and writes the commands it finds there to the bus. A C++ ros2_control
hardware component (not part of the Python package; see `ros/` below) attaches
to the same segment (`actuo.drivers.shm`). Both sides must agree on two values: the file prefix,
`/tmp/<prefix><name>.shm` (default prefix `actuo_`, environment variable
`ACTUO_SHM_PREFIX`, or the `prefix` argument), and the header magic (default
`0x32544341`, the bytes `ACT2`; the `magic` argument of `ShmBridge`). The
servicer's lock is `/tmp/<prefix><name>.lock`; tools that drive the bus
directly take `--shm-name` and refuse to run while it is held.
`robot_document.hardware_urdf(..., backend="shm", hardware_plugin=...)` and
`actuo-teach-limits --hardware-plugin` write the component's plugin name, which
the caller supplies. Remembered gear-window references are kept in
`$XDG_STATE_HOME/actuo` (`~/.local/state/actuo`), or in `$ACTUO_STATE_DIR`.

### The ROS 2 packages (`ros/`)

Clone this repository into a colcon workspace's `src/`; colcon finds the
packages under `ros/` (the repository root itself is not a ROS package, and
the PyPI package contains only `actuo/`).

| package | what it is |
| --- | --- |
| `actuo_msgs` | `ServoState` / `JointServoState` (the audit trail) and the `SetServo` action |
| `actuo_hardware` | `actuo_hardware/ActuoSystem`, the motor-agnostic ros2_control component on the shared memory above |
| `actuo_ros` | `actuo_bus_node` (the servicer as a ROS 2 node: servo on/off action, telemetry, diagnostics), presence markers, `actuo_ros.robot_interface` (a scikit-robot interface that drives the robot through ros2_control), udev rules (`share/actuo_ros/udev/install.sh`), the ARM-H7 symbol map, and every `actuo-*` command as `ros2 run actuo_ros <command>` |
| `actuo_rviz_plugins` | RViz panels: per-joint servo on/off, and `/diagnostics` |

`actuo_ros` builds a venv (`ament_uv_virtualenv`) with this checkout installed
editable, so the ROS layer always runs the actuo next to it. Another package
whose venv needs the same actuo calls `actuo_requirement()` after
`find_package(actuo_ros)` (see `ros/actuo_ros/cmake/actuo_ros-extras.cmake.in`).

`firmware/actuo_uart_relay/` is the Arduino UNO R4 WiFi sketch that relays an
RCB-4mini's serial port over UDP for the `kondo_rcb4_udp` transport.

## Install and test

```bash
uv run --no-project --python 3.12 --with-editable '.[test,kondo,safety]' \
    python -m pytest -q
```

751 passed, 1 skipped (2026-11-12, scikit-robot 0.4.15 and rcb4 0.2.1 from
PyPI). The skipped test needs ROS's ament index. In a shell where ROS 2 is
sourced, run it with `env -u PYTHONPATH`: ROS's pytest plugins are on that
path and fail to import in this environment.

Extras: `kondo` (rcb4), `dynamixel`, `can`, `usb` (adapter reset), `xacro`,
`firmware` (pyelftools, ARM-H7 symbol maps), `safety` (python-fcl,
collision distances), `all`. Install only the vendors
a robot uses: a plugin's module is imported when a config first names it, so a
Kondo robot does not need python-can and a CAN robot does not need rcb4.
Naming a plugin whose library is missing raises an `ImportError` that says
which extra to install.
