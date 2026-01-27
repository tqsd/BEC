from __future__ import annotations

from typing import Any
import numpy as np


HVVec = np.ndarray  # shape (2,), complex
# avoid importing enums at module import time to reduce cycles
TrLike = Any
