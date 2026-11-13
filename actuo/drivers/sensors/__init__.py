"""Concrete sensor / custom-board plugins.

Each module registers its device with ``actuo.core.registry`` when it is
imported -- the same discipline the transport and actuator packages follow.
Nothing is imported here: the registry imports a sensor's module the first
time a config names it.
"""
