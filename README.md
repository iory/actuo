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

On a bus whose servos may be missing (`hot_plug: true`), the config can list
more joints than are plugged in:

```python
ri = ActuoRobotInterface(robot, "arm_rcb4.yaml", required_joints=["base_joint"])
ri.attached_joints()     # the joints whose servos answer now
```

A joint is attached once its servo has answered for a few cycles: its command
starts from where it is, and it is limp until `servo_on`. A vector goal moves
the attached joints; naming a detached joint is an error. A servo that stops
answering is switched off, its motion is dropped and its measurement becomes
NaN, so when it is plugged back in it comes back limp, from where it then is.

Presence is one contract for every bus (`actuo.core.contracts.Bus`:
`present`, `absence_reason`, `watch`, `answering_ids`), and a bus says what it
can do (`supports_hot_plug`, `supports_watch`). A config that asks a bus for
what it cannot do is refused when the bus is built, not ignored:

| bus | `hot_plug` | `watch_ids` (ids it does not drive) |
| --- | --- | --- |
| FEETECH | missed SYNC_READs; probe with a facts read | PING |
| DYNAMIXEL | missed reads; probe with PING (2.0) or a model-number READ (1.0), model checked | same probe |
| RCB-4mini | whether the board keeps rewriting the pulse | the ICS numbers assigned, never driven |
| CAN (servo-mode CubeMars) | a status frame within `stale_after` | status frames arriving (`status_source`) |
| CAN (2.0 MIT) | as above | refused: it answers only when addressed |
| ARM-H7, serial | refused | refused |

Request/response buses share the bookkeeping (`actuo.drivers.buses.presence`).

### One robot, several robots: `actuo.modular`

A modular robot can be several robots: other joints, other names, other
calibration, another model, even when some servos are the same. Each is a
**configuration** (an ordinary robot config and its URDF), listed in a catalog,
and `ModularRobot` runs whichever is current:

```python
from actuo.modular import ModularRobot

mr = ModularRobot("catalog.yaml", on_change=lambda s: print(s.configuration))
mr.ri.servo_on()             # the current configuration's ActuoRobotInterface
mr.robot                     # ... and its skrobot model
mr.declare("drill")          # same servos, another robot: say which
mr.resume_detection()        # or let the servos that answer decide again
```

- **detect**: the configuration all of whose ids answer becomes current (the
  largest when several are complete; two complete of one size is a conflict,
  reported, and the robot stays as it is).
- **declare**: whoever drives the robot names it, for configurations with the
  same servos.
- A switch keeps the arm up: servos that held torque and are in the next
  configuration keep it under their new joint names (`hold_joints`), at their
  measured pose; the others are released first. Every model is loaded at
  start. On `examples/rigs/feetech_4port/modular` (ids 1 and 2 shared under
  other names and calibration) a switch took 18-39 ms, detection noticed the
  servos change within about 1.3 s, and held servos did not move by one count.

Detection needs the bus to see ids outside the current configuration
(`watch_ids`), so it runs on any bus in the table above that supports both.

### Testing it on hardware: fault injectors and rigs

`actuo.testing` makes servos come and go on purpose, so a robot that is
plugged together at run time (`robot_modes`: `fixed`, `tool_change`,
`reconfigurable`) can be tested without anyone pulling cables:

- `actuo.testing.fault_injectors`: one interface (`unplug`, `plug`,
  `drop_replies`, `restore`) over the hardware in the servo line --
  [iory/servo-fault-injector](https://github.com/iory/servo-fault-injector)
  (four ports, each with its own power and signal switch; any FEETECH /
  DYNAMIXEL TTL / KRS speed) and
  [iory/ics-fault-relay](https://github.com/iory/ics-fault-relay)
  (Kondo ICS at 115200 bit/s, one servo at a time).
- `actuo.testing.rig`: a rig file names the robot config, its URDF, the
  `robot_modes` descriptions to check and which port each joint is on, so
  tests speak in joints (`rig.only(["j1", "j3"])`).

```sh
ACTUO_RIG=examples/rigs/feetech_4port/rig.yaml python -m pytest tests/hardware -v
```

The hardware tests (skipped unless `ACTUO_RIG` is set) run the robot interface
on the rig's bus with torque off and check, for **every combination** of
plugged joints, that `attached_joints()` becomes that combination and that
each `robot_modes` description selects from the live bus the same robot, with
the same problems, as from the combination itself; and that a joint whose
replies are cut while its commands still arrive is detached and comes back.
On 2026-11-24 they passed on `examples/rigs/feetech_4port` (two STS3215, two
HLS3950M, 1 Mbit/s): 16 combinations x 2 descriptions in about 20 s.
On `examples/rigs/krs_4port` (four KRS servos on an RCB-4mini, 2026-11-25)
both tests and `tests/hardware/test_modular_hw.py` passed (4 runs of 4) once
the injector connected each port's signal 1 s after its power
(`signal_after_s`). Without that, a KRS powered with its signal on the live
ICS line sometimes started in PWM mode (the line's level during the first
500 ms selects the mode; the board's 10k pull-ups R17-R47 are not fitted):
it no longer answers, was found near an end stop afterwards (pulse 11400-12001),
and blocks the other servos' replies until it is powered off.

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

#### A modular robot under ROS 2: `modular.launch.py`

```sh
ros2 launch actuo_ros modular.launch.py catalog:=<catalog>.yaml
ros2 service call /actuo_configuration/declare actuo_msgs/srv/DeclareConfiguration "{name: drill}"
```

One controller_manager holds every configuration of an `actuo.modular` catalog
side by side (`actuo-modular-files`: each configuration's modules as
components `<configuration>_<module>`, each on the servicer segment
`<shm-name>_<configuration>`; joint names must differ between
configurations). The servicer (`actuo_bus_node --real --catalog`) runs the
current configuration and switches in-process -- its loop is left without
releasing the servos the next configuration has, and those keep torque under
their new names -- and publishes `/actuo_configuration/configuration`.
`actuo_configuration_manager` follows it: the other configurations'
controllers and components go inactive, `/robot_description` becomes the new
configuration's model (RViz follows), and each module of the running
configuration comes up -- component, then its joint_state_broadcaster and
controllers -- as soon as all its servos answer, so a module plugged in later
joins and one pulled out goes down alone. controller_manager takes the
superset description on its own topic (`/actuo_modular/robot_description`).

On `examples/rigs/feetech_4port/modular` (2026-11-24,
`ros/actuo_ros/test/hardware/check_modular_launch.py`): after the servos
changed, the servicer switched in 47 ms, ros2_control followed 80 ms later and
the new modules were up 70 ms after that; a declaration had the modules up
0.26 s after the service call; a module pulled within a configuration went
down alone and was back 0.9 s after it was plugged in.

With `moveit:=true`, `actuo_moveit_follower` runs MoveIt for the configuration
that runs. move_group reads a robot model once, so on every switch it is
restarted with the new configuration's model, SRDF (a planning group per
module, adjacent links not checked; `actuo-modular-files --moveit`), KDL,
joint limits (the URDF's velocities, and an acceleration limit MoveIt's time
parameterization needs) and its trajectory controllers, which MoveIt does not
switch itself. `/actuo_moveit_follower/ready` names the configuration once it
plans. On the same rig, planning succeeded for each configuration's groups
under its own joint names after each switch, and failed for a group of the
other configuration; move_group took 3.0-4.3 s to answer again after a switch.

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
