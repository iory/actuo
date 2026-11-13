#!/usr/bin/env python3
"""CLI wrapper for importing legacy rcb4 Kondo servo maps into an actuo robot config.

    actuo-kondo-import-rcb4 --help
"""

from __future__ import annotations

import os
import sys


from actuo.drivers.kondo_import_rcb4 import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
