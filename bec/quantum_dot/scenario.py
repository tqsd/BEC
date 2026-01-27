from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class Scenario:
    drives: Any = None
