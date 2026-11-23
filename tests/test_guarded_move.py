"""Guarded move: the stop rule, and the supervisor running one."""

from __future__ import annotations

import time

import numpy as np
import pytest

from actuo.core.contracts import ActuatorState, Capabilities, OperatingMode
from actuo.drivers.guarded_move import GuardedMove
from actuo.drivers.servo import ServoSupervisor

STRIDE = 3


def test_reaches_the_end_and_settles():
    move = GuardedMove(0.0, 0.1, 1.0, max_tracking_error=0.05, settle_time=0.05)
    command, now = 0.0, 0.0
    while not move.done:
        command = move.step(command, command, 0.0, now, 0.01)   # follows exactly
        now += 0.01
    assert not move.stopped and move.reason == "reached"
    assert move.target == pytest.approx(0.1)


def test_stops_on_tracking_error_and_backs_off():
    move = GuardedMove(0.0, 1.0, 1.0, max_tracking_error=0.05, trip_duration=0.03,
                       backoff=0.02)
    command, now = 0.0, 0.0
    while not move.done:
        command = move.step(command, min(command, 0.2), None, now, 0.01)  # wall at 0.2
        now += 0.01
    assert move.stopped and "tracking error" in move.reason
    assert move.position == pytest.approx(0.2)
    assert move.target == pytest.approx(0.18)
    assert move.peak_error > 0.05


def test_stops_on_load_in_the_direction_of_travel_only():
    move = GuardedMove(0.0, -1.0, 1.0, max_tracking_error=1.0, max_load=0.3,
                       trip_duration=0.0)
    command = move.step(0.0, 0.0, +0.9, 0.0, 0.01)   # pushing the other way
    assert not move.done
    move.step(command, command, -0.4, 0.01, 0.01)
    assert move.done and "load" in move.reason


class Actuator:
    def __init__(self, node_id, limit=None):
        self.id = node_id
        self.capabilities = Capabilities(modes=(OperatingMode.POSITION,))
        self.calibration = type("Cal", (), {"position_limit": limit})()

    def position_refusal(self, position):
        return None


class Bus:
    def __init__(self, positions, limit=None):
        self.actuators = [Actuator(10 + i, limit) for i in range(len(positions))]
        self.positions = list(positions)
        self.load = 0.0
        self.batches = []

    def last_state(self, device_id):
        i = device_id - 10
        return ActuatorState(position=self.positions[i], error=0,
                             aux={"load": self.load})

    def write_batch(self, commands):
        self.batches.append(list(commands))

    def release_all(self):
        pass


class Shm:
    def __init__(self, commands):
        self.c = np.zeros(len(commands) * STRIDE)
        for i, v in enumerate(commands):
            self.c[i * STRIDE] = v

    def read_command(self):
        return self.c


def _supervisor(limit=None):
    bus = Bus([0.0, 0.5], limit)
    shm = Shm([0.0, 0.5])
    sup = ServoSupervisor(bus, ["joint0", "joint1"], 0.05, STRIDE)
    assert sup.enable(shm)[0]
    return sup, bus, shm


def test_supervisor_runs_a_move_into_a_wall_and_holds_backed_off():
    sup, bus, shm = _supervisor()
    ok, reason = sup.start_guarded_move("joint0", 1.0, 5.0, max_tracking_error=0.05,
                                        trip_duration=0.0, backoff=0.02)
    assert ok, reason
    assert sup.needs_service
    for _ in range(200):
        sent = bus.batches[-1][0].position if bus.batches else 0.0
        bus.positions[0] = min(sent, 0.1)          # a wall at 0.1
        sup.service(shm)
        if not sup.guarded_move_running:
            break
        time.sleep(0.002)
    move = sup.pop_guarded_move()
    assert move is not None and move.stopped
    assert move.position == pytest.approx(0.1)
    assert bus.batches[-1][0].position == pytest.approx(0.08)    # backed off
    assert bus.batches[-1][1].position == pytest.approx(0.5)     # the other holds
    # Held there, not thrown back to the controller's 0.0, until re-synced.
    sup.service(shm)
    assert bus.batches[-1][0].position == pytest.approx(0.08)
    assert sup.inspect(shm)[0].resyncing
    assert sup.pop_guarded_move() is None


def test_supervisor_refusals():
    sup, bus, shm = _supervisor(limit=(-0.5, 0.5))
    ok, reason = sup.start_guarded_move("joint0", 1.0, 0.1, 0.05)
    assert not ok and "outside its envelope" in reason
    assert sup.start_guarded_move("joint0", 0.2, 0.1, 0.05)[0]
    ok, reason = sup.start_guarded_move("joint1", 0.1, 0.1, 0.05)
    assert not ok and "is running" in reason
    sup.disable(["joint0"])
    move = sup.pop_guarded_move()
    assert move.stopped and "torque" in move.reason
    ok, reason = sup.start_guarded_move("joint0", 0.1, 0.1, 0.05)
    assert not ok and "no torque" in reason


def test_needs_service_until_the_back_off_is_reached():
    """The back-off is slowed by max_velocity, and a request/response bus is
    written only when asked: it has to keep asking until the pose is reached."""
    bus = Bus([0.0, 0.5])
    shm = Shm([0.0, 0.5])
    sup = ServoSupervisor(bus, ["joint0", "joint1"], 0.05, STRIDE,
                          max_velocity={"joint0": 2.0})
    assert sup.enable(shm)[0]
    assert sup.start_guarded_move("joint0", 1.0, 5.0, max_tracking_error=0.05,
                                  trip_duration=0.0, backoff=0.05)[0]
    for _ in range(2000):
        if not sup.needs_service:
            break
        sent = bus.batches[-1][0].position if bus.batches else 0.0
        bus.positions[0] = min(sent, 0.2)
        sup.service(shm)
        time.sleep(0.002)
    assert not sup.guarded_move_running
    assert bus.batches[-1][0].position == pytest.approx(0.15)
