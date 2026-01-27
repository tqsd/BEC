from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, Optional

import numpy as np

from bec.simulation.types import ModeRegistryView
from bec.simulation.drive_decode import DriveDecodeContext


class DecodeContextProvider(Protocol):
    """Build everything decode/compile needs from an opaque model."""

    def mode_registry(self, model: Any) -> ModeRegistryView: ...
    def decode_ctx(self, model: Any) -> DriveDecodeContext: ...
    def derived_for_report(self, model: Any) -> Optional[Any]: ...
