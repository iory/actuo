"""Concrete actuator model plugins.

Each module registers its actuator with ``actuo.core.registry`` when it is
imported. Nothing is imported here: the registry imports a model's module the
first time a config names it, so only that vendor's library is needed.
"""
