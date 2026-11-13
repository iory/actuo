"""A vendor's plugins import only that vendor's library.

Each check runs in a fresh interpreter in which one vendor library cannot be
imported, as on a machine that installed only another extra.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

BLOCK = """
import sys

class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {blocked!r}:
            raise ModuleNotFoundError("No module named " + repr(name), name=name)
        return None

sys.meta_path.insert(0, _Block())
"""


def run(blocked, body):
    code = textwrap.dedent(BLOCK.format(blocked=set(blocked))) + textwrap.dedent(body)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_a_kondo_config_loads_without_python_can(tmp_path):
    pytest.importorskip("rcb4.protocol")
    config = tmp_path / "arm.yaml"
    config.write_text(textwrap.dedent("""\
        bus:
          plugin: kondo_rcb4_bus
          transport: {plugin: kondo_rcb4, channel: /dev/null, bitrate: 1250000}
        joints:
          - name: j0
            actuator: {plugin: kondo_ics, id: 2}
        """))
    out = run({"can"}, """
        from actuo.drivers.config import build_bus
        bus, names = build_bus({path!r})
        print(type(bus).__name__, names, "can" in sys.modules)
        """.format(path=str(config)))
    assert out.split() == ["KondoRcb4Bus", "['j0']", "False"]


def test_a_dynamixel_actuator_loads_without_rcb4_or_python_can():
    out = run({"can", "rcb4"}, """
        from actuo.drivers.config import build_actuators
        from actuo.drivers.buses import DynamixelBus
        actuators, names = build_actuators(
            [{"name": "j0", "actuator": {"plugin": "dynamixel", "id": 1,
                                         "model": "XL430-W250-T"}}])
        print(type(actuators[0]).__name__, DynamixelBus.__name__)
        """)
    assert out.split() == ["DynamixelActuator", "DynamixelBus"]


def test_a_missing_vendor_library_names_the_extra():
    out = run({"can"}, """
        from actuo.core.registry import get_transport
        try:
            get_transport("python_can")
        except ImportError as error:
            print(error)
        """)
    assert "actuo[can]" in out
    assert "python_can" in out
