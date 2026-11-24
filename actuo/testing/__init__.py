"""Test rigs: making servos come and go on purpose, and checking what follows.

:mod:`actuo.testing.fault_injectors` drives the hardware in a servo line
(iory/servo-fault-injector, iory/ics-fault-relay) behind one interface;
:mod:`actuo.testing.rig` maps a robot's joints onto it. The hardware tests in
``tests/hardware`` use both and run only when ``ACTUO_RIG`` names a rig file.
"""
