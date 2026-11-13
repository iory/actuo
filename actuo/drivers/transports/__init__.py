"""Concrete transport plugins.

Each module registers its transport with ``actuo.core.registry`` when it is
imported. Nothing is imported here: the registry imports a transport's module
the first time a config names it, so only that vendor's library is needed.
"""
