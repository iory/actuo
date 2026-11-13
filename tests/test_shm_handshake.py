"""The servicer/component handshake that removes the start-up lurch.

No CAN, no ROS: this drives the real shm layout from both sides, standing in for
the C++ ros2_control hardware component (one per module) that reads it. Worth pinning because
the failure modes are a robot commanded to a pose nobody measured, and -- new
with one component per module -- one module's activation wiping another's
command.
"""
from __future__ import annotations

import math
import os
import struct
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)

from actuo.drivers.shm import (  # noqa: E402
    CMD_STRIDE, DEFAULT_PREFIX, MAGIC, PREFIX_ENV, STATE_PRESENT,
    STATE_STRIDE, ShmBridge, default_path, lock_path, wait_for_commands)

MEASURED = [1.7052, 1.3404, -0.2426]   # a real pose read off an arm


def servicer(name, n=3):
    """The servicer's side: it creates the segment."""
    try:
        os.unlink(default_path(name))
    except FileNotFoundError:
        pass
    return ShmBridge(name, n_joints=n, create=True)


def component(name, n=3):
    """A hardware component's side: it attaches."""
    return ShmBridge(name, n_joints=n, create=False)


def measured_block(present=(1.0, 1.0, 1.0)):
    block = np.zeros((3, STATE_STRIDE))
    block[:, 0] = MEASURED
    block[:, STATE_PRESENT] = present
    return block


def test_fresh_segment_holds_no_fiction():
    """An untouched segment must not read as a robot at 0 rad.

    The version-1 shm layout handed back zeros that looked exactly like a
    measurement. The version-2 layout creates
    every command unowned (NaN) and every state absent, so neither side can
    mistake "nobody wrote this" for a value.
    """
    owner = servicer("hs_fresh")
    try:
        assert owner.state_seq() == 0
        assert owner.cmd_seq() == 0
        assert owner.epoch() != 0
        assert np.all(np.isnan(owner.read_command()))
        state = owner.read_state().reshape(3, STATE_STRIDE)
        assert np.all(np.isnan(state[:, 0]))
        assert np.all(state[:, STATE_PRESENT] == 0.0)
    finally:
        owner.close(unlink=True)


def test_component_adopts_measured_state_so_the_start_jump_is_zero():
    owner = servicer("hs_adopt")
    try:
        owner.write_state(measured_block())          # measured state FIRST
        hw = component("hs_adopt")
        state = hw.read_state().reshape(3, STATE_STRIDE)
        rows = np.zeros((3, CMD_STRIDE))
        rows[:, 0] = state[:, 0]                     # command = measured
        hw.write_command_slots([0, 1, 2], rows)
        commanded = owner.read_command()[0::CMD_STRIDE]
        assert commanded == pytest.approx(MEASURED)
        hw.close()
    finally:
        owner.close(unlink=True)


def test_one_module_deactivating_leaves_the_others_command_alone():
    """Two components share one segment; each writes only its own slots."""
    owner = servicer("hs_modules")
    try:
        owner.write_state(measured_block())
        arm = component("hs_modules")       # slots 0, 1
        hand = component("hs_modules")      # slot 2
        arm.write_command_slots([0, 1], [[MEASURED[0], 0, 0], [MEASURED[1], 0, 0]])
        hand.write_command_slots([2], [[MEASURED[2], 0, 0]])
        # The hand's module goes away: it gives its slot back.
        hand.write_command_slots([2], [[math.nan] * CMD_STRIDE])
        command = owner.read_command()[0::CMD_STRIDE]
        assert command[:2] == pytest.approx(MEASURED[:2])
        assert math.isnan(command[2])
        arm.close()
        hand.close()
    finally:
        owner.close(unlink=True)


def test_a_restarted_servicer_wipes_the_old_session_and_changes_the_epoch():
    """A segment file left by a servicer that died cannot be mistaken for
    live state (this race was hit twice, 2026-07-29 and 2026-09-29): only the
    servicer creates, and it clears everything a previous run left, in place. An attached component sees the epoch change."""
    first = servicer("hs_restart")
    first.write_state(measured_block())
    hw = component("hs_restart")
    hw.write_command_slots([0, 1, 2], np.zeros((3, CMD_STRIDE)))
    epoch = hw.epoch()
    first.close()
    second = ShmBridge("hs_restart", n_joints=3, create=True)   # same file
    try:
        assert hw.epoch() != epoch
        assert np.all(np.isnan(hw.read_command()))
        assert hw.state_seq() == 0
        hw.close()
    finally:
        second.close(unlink=True)


def test_absent_slot_is_published_as_absent():
    owner = servicer("hs_absent")
    try:
        owner.write_state(measured_block(present=(1.0, 0.0, 1.0)))
        state = component("hs_absent").read_state().reshape(3, STATE_STRIDE)
        assert list(state[:, STATE_PRESENT]) == [1.0, 0.0, 1.0]
    finally:
        owner.close(unlink=True)


def test_attaching_with_another_layout_is_refused():
    owner = servicer("hs_layout")
    try:
        with pytest.raises(ValueError, match="layout mismatch"):
            ShmBridge("hs_layout", n_joints=2, create=False)
    finally:
        owner.close(unlink=True)


def test_wait_for_commands_needs_every_listed_slot():
    owner = servicer("hs_wait")
    try:
        hw = component("hs_wait")
        hw.write_command_slots([0], [[0.1, 0, 0]])
        assert not wait_for_commands(owner, [0, 1], timeout=0.05, poll=0.01)
        assert wait_for_commands(owner, [0], timeout=0.05, poll=0.01)
        hw.close()
    finally:
        owner.close(unlink=True)


def test_prefix_comes_from_argument_then_environment_then_default(monkeypatch):
    monkeypatch.delenv(PREFIX_ENV, raising=False)
    assert default_path("arm") == "/tmp/{}arm.shm".format(DEFAULT_PREFIX)
    assert lock_path("arm") == "/tmp/{}arm.lock".format(DEFAULT_PREFIX)
    monkeypatch.setenv(PREFIX_ENV, "env_")
    assert default_path("arm") == "/tmp/env_arm.shm"
    assert lock_path("arm") == "/tmp/env_arm.lock"
    assert default_path("arm", prefix="arg_") == "/tmp/arg_arm.shm"
    assert lock_path("arm", prefix="arg_") == "/tmp/arg_arm.lock"


def test_the_magic_is_the_one_both_sides_were_given():
    """The default magic reads ``ACT2`` on disk; another one can be chosen to
    match a C++ reader, and a mismatch is refused rather than attached."""
    assert struct.pack("<I", MAGIC) == b"ACT2"
    other = 0x12345678
    try:
        os.unlink(default_path("hs_magic", prefix="hs_"))
    except FileNotFoundError:
        pass
    owner = ShmBridge("hs_magic", n_joints=1, create=True, prefix="hs_",
                      magic=other)
    try:
        with open(default_path("hs_magic", prefix="hs_"), "rb") as handle:
            assert struct.unpack("<I", handle.read(4))[0] == other
        ShmBridge("hs_magic", n_joints=1, prefix="hs_", magic=other).close()
        with pytest.raises(ValueError, match="bad magic"):
            ShmBridge("hs_magic", n_joints=1, prefix="hs_")
    finally:
        owner.close(unlink=True)
