"""Write a robot's ros2_control URDF and controller parameters from its config.

A robot config with ``urdf:`` and ``modules:`` (see
:mod:`actuo.drivers.robot_document`) already says everything a ros2_control
stack needs: which joints, in which shm slot, grouped into which controllers.
This writes the two files a launch hands to ``ros2_control_node``, so a robot
needs no hand-written ``<ros2_control>`` block or controllers yaml, and the
joint order cannot drift from the bus config's.

Examples
--------
    actuo-robot-files arm.yaml --shm-name arm \\
        --hardware-plugin actuo_hardware/ActuoSystem --out /tmp/arm
    # -> /tmp/arm/robot.urdf, /tmp/arm/controllers.yaml; the controller names
    #    are printed, one per line, joint_state_broadcaster first
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

from actuo.drivers.config import load_document
from actuo.drivers.robot_document import (
    MOCK_BACKEND, SHM_BACKEND, controller_parameters, controllers_of,
    hardware_urdf, load_model)


def write_robot_files(config, out_dir, shm_name, backend=SHM_BACKEND,
                      hardware_plugin=None, namespace=""):
    """Write ``robot.urdf`` and ``controllers.yaml`` for ``config``.

    Parameters
    ----------
    config : str
        Robot config with ``urdf`` and ``modules``.
    out_dir : str
        Directory to write into (created).
    shm_name : str
        The bus servicer's segment.
    backend : str
        ``shm`` or ``mock``.
    hardware_plugin : str or None
        The shared-memory component's ros2_control plugin name (``shm``).
    namespace : str
        Namespace of controller_manager; its parameters are keyed under it.

    Returns
    -------
    tuple of (str, str, list of str)
        URDF path, controllers yaml path, controller names to spawn
        (``joint_state_broadcaster`` first).
    """
    document = load_document(config)
    for key in ("urdf", "modules"):
        if key not in document:
            raise ValueError("{} has no {!r}; it needs urdf and modules to "
                             "make a ros2_control stack".format(config, key))
    urdf = hardware_urdf(load_model(document, config), document, shm_name,
                         backend=backend, hardware_plugin=hardware_plugin)
    params = controller_parameters(document)
    if namespace:
        params = {"/{}".format(namespace.strip("/")): params}
    os.makedirs(out_dir, exist_ok=True)
    urdf_path = os.path.join(out_dir, "robot.urdf")
    params_path = os.path.join(out_dir, "controllers.yaml")
    with open(urdf_path, "w") as handle:
        handle.write(urdf)
    with open(params_path, "w") as handle:
        yaml.safe_dump(params, handle, sort_keys=False)
    return (urdf_path, params_path,
            ["joint_state_broadcaster"] + list(controllers_of(document)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", help="robot config YAML (urdf + modules)")
    parser.add_argument("--out", required=True, help="directory to write into")
    parser.add_argument("--shm-name", required=True)
    parser.add_argument("--backend", choices=[SHM_BACKEND, MOCK_BACKEND],
                        default=SHM_BACKEND)
    parser.add_argument("--hardware-plugin", default=None,
                        help="ros2_control plugin of the shm component, e.g. "
                             "actuo_hardware/ActuoSystem")
    parser.add_argument("--namespace", default="")
    args = parser.parse_args(argv)
    try:
        _urdf, _params, controllers = write_robot_files(
            args.config, args.out, args.shm_name, backend=args.backend,
            hardware_plugin=args.hardware_plugin, namespace=args.namespace)
    except (ValueError, KeyError, OSError) as error:
        print("actuo-robot-files: {}".format(error), file=sys.stderr)
        return 1
    print("\n".join(controllers))
    return 0


if __name__ == "__main__":
    sys.exit(main())
