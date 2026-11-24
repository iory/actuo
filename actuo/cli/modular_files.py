"""Write one controller_manager's files for every configuration of a modular robot.

The catalog (:mod:`actuo.modular`) lists configurations; each robot config
names its ``urdf`` and ``modules``. This writes, into ``--out``, the superset
``robot.urdf`` (every configuration's components, each on its own segment
``<shm-name>_<configuration>``), ``controllers.yaml``, each configuration's own
model, and ``manifest.yaml`` -- and prints the manifest's path. See
:func:`actuo.modular.ros2_control_files`.

Examples
--------
    actuo-modular-files catalog.yaml --shm-name modular \\
        --hardware-plugin actuo_hardware/ActuoSystem --out /tmp/modular
"""

from __future__ import annotations

import argparse
import os
import sys

from actuo.modular import ros2_control_files


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("catalog")
    parser.add_argument("--out", required=True)
    parser.add_argument("--shm-name", required=True,
                        help="segment base name; each configuration adds _<name>")
    parser.add_argument("--hardware-plugin", required=True,
                        help="the shared-memory component's ros2_control plugin name")
    args = parser.parse_args(argv)
    try:
        ros2_control_files(args.catalog, args.out, args.shm_name, args.hardware_plugin)
    except (OSError, ValueError, KeyError) as error:
        print("actuo-modular-files: {}".format(error), file=sys.stderr)
        return 1
    print(os.path.join(args.out, "manifest.yaml"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
