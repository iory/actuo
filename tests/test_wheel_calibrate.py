"""actuo-wheel-calibrate refuses to spin what it cannot vouch for."""

import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "actuo", "cli", "wheel_calibrate.py")


def load():
    spec = importlib.util.spec_from_file_location("actuo_wheel_calibrate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rcb4mini_without_rotation_checked_commands_nothing(capsys):
    # No port is opened: the refusal comes before any board is touched.
    code = load().main(["--board", "rcb4mini", "--channel", "/dev/null",
                        "--ids", "20", "--yes"])

    assert code == 2
    assert "--rotation-checked" in capsys.readouterr().out


def test_h7_without_symbols_commands_nothing(capsys):
    code = load().main(["--channel", "/dev/null", "--ids", "20", "--yes"])

    assert code == 2
    assert "--symbols" in capsys.readouterr().out
