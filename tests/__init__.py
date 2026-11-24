"""Test suite initialization setting environment defaults."""
from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_KERAS_CACHE = _PROJECT_ROOT / ".keras"
if _LOCAL_KERAS_CACHE.is_dir():
    os.environ["KERAS_HOME"] = str(_LOCAL_KERAS_CACHE)
