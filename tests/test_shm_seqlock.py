"""Seqlock reads against a writer that is preempted mid-write.

ros2_control's hardware component writes the command block from a thread that
usually has no real-time priority, so the OS can stop it between "counter odd"
and "counter even" for milliseconds. The servicer's read must ride that out
instead of aborting the bus.
"""
from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np
import pytest

from actuo.drivers import shm as shm_mod
from actuo.drivers.shm import ShmBridge

_CMD_SEQ_OFF = 16
N = 4


def _bridge(path, create):
    return ShmBridge("seqlock_test", N, create=create, path=str(path))


def _slow_writer(path, hold_s, stop):
    """Stand-in for the C++ component: holds the counter odd for hold_s."""
    b = _bridge(path, create=False)
    value = 0.0
    while not stop.is_set():
        value += 1.0
        s = b._load_seq(_CMD_SEQ_OFF)
        b._store_seq(_CMD_SEQ_OFF, s + 1)
        b._cmd[:] = value
        t = time.perf_counter()
        while time.perf_counter() - t < hold_s:
            pass
        b._store_seq(_CMD_SEQ_OFF, s + 2)
        time.sleep(0.005)
    b.close()


def test_preempted_writer_does_not_abort_reader(tmp_path):
    path = tmp_path / "seq.shm"
    reader = _bridge(path, create=True)
    stop = mp.Event()
    proc = mp.Process(target=_slow_writer, args=(path, 0.002, stop))
    proc.start()
    try:
        time.sleep(0.1)
        for _ in range(300):
            cmd = reader.read_command()
            # never a torn mix of two writes
            assert np.all(cmd == cmd[0])
            time.sleep(0.003)
    finally:
        stop.set()
        proc.join()
        reader.close(unlink=True)


def test_stuck_writer_returns_last_copy_then_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(shm_mod, "READ_DEADLINE_S", 0.002)
    monkeypatch.setattr(shm_mod, "STALE_LIMIT_S", 0.05)
    b = _bridge(tmp_path / "seq.shm", create=True)
    try:
        b.write_command(np.full(N * shm_mod.CMD_STRIDE, 0.25))
        assert np.all(b.read_command() == 0.25)
        # writer dies between the two bumps
        b._store_seq(_CMD_SEQ_OFF, b._load_seq(_CMD_SEQ_OFF) + 1)
        assert np.all(b.read_command() == 0.25)
        assert b.torn_reads == 1
        time.sleep(0.06)
        with pytest.raises(RuntimeError, match="did not converge"):
            b.read_command()
    finally:
        b.close(unlink=True)


def test_stuck_before_any_read_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(shm_mod, "READ_DEADLINE_S", 0.002)
    b = _bridge(tmp_path / "seq.shm", create=True)
    try:
        b._store_seq(_CMD_SEQ_OFF, b._load_seq(_CMD_SEQ_OFF) + 1)
        with pytest.raises(RuntimeError, match="did not converge"):
            b.read_command()
    finally:
        b.close(unlink=True)
