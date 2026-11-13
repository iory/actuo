"""ActuoRobotInterface on a Kondo humanoid, against the RCB-4mini emulator (no robot, no ROS).

The config is a ``kondo_rcb4_bus`` robot config and the URDF its model; the
joint names used below (``rarm_joint0``, ``head_joint0``) must exist in both.

    python examples/ri_with_emulator.py <humanoid_rcb4.yaml> <humanoid.urdf>
"""

import os
import sys
import tempfile
import threading

import numpy as np
from skrobot.model import RobotModel

from actuo.cli.kondo_rcb4_emulator import servo_ids, write_config
from actuo.drivers.kondo_rcb4_emulator import Rcb4Emulator, serve
from actuo.robot_interface import ActuoRobotInterface

config, urdf = sys.argv[1], sys.argv[2]
work = tempfile.mkdtemp()
link = os.path.join(work, "rcb4")
emulated = os.path.join(work, "config.yaml")
write_config(config, link, emulated)
ids, ics_bitrate = servo_ids(config)
board = Rcb4Emulator(ids, servo_ics_bitrate=ics_bitrate, log=lambda m: None)
threading.Thread(target=serve, args=(board, link, lambda m: None), daemon=True).start()
while not os.path.exists(link):
    pass

robot = RobotModel.from_urdf(urdf)
ri = ActuoRobotInterface(robot, emulated)
print("joints on the bus:", len(ri.bus_joints), " model joints with no actuator:", ri.unactuated)
start = ri.potentio_vector()
print("measured at start (deg):", np.round(np.rad2deg(start[:6]), 2))

# Motion while limp must not move anything.
target = {"rarm_joint0": 0.5, "head_joint0": -0.3}
ri.angle_vector(target, time=1.0)
ri.wait_interpolation()
limp = ri.potentio_vector()
names = [j.name for j in robot.joint_list]
print("limp after a command: rarm_joint0 {:.3f}  head_joint0 {:.3f}".format(
    limp[names.index("rarm_joint0")], limp[names.index("head_joint0")]))

ri.servo_on(["rarm_joint0", "head_joint0"])
ri.angle_vector(target, time=1.0)
ri.wait_interpolation()
ri.angle_vector(target, time=0.0)   # settle: the emulated servos lag their command
import time; time.sleep(0.5)
held = ri.potentio_vector()
print("held after 1.0 s: rarm_joint0 {:.3f} (target 0.5)  head_joint0 {:.3f} (target -0.3)".format(
    held[names.index("rarm_joint0")], held[names.index("head_joint0")]))
print("outside limits:", ri.outside_limits())
ri.servo_off()
ri.close()
