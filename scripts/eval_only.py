#!/usr/bin/env python3
"""Compatibility entry point; prefer ``scripts/cli/eval_only.py``."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cli.eval_only import *  # noqa: F401,F403,E402


if __name__ == "__main__":
    main()
