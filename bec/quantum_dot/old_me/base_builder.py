from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np
from photon_weave.extra import interpreter

from bec.quantum_dot.kron_pad_utility import KronPad


class BaseBuilder:
    """
    Shared utilities for builders that construct padded Qobj operators from
    symbolic context entries (s_<BRA>_<KET>, etc.).

    This class intentionally contains *no physics* (no term selection),
    only operator-construction plumbing.
    """

    DOT_LEVELS: List[str] = ["G", "X1", "X2", "XX"]

    def __init__(
        self,
        context: Dict[str, Any],
        kron: KronPad,
        *,
        dot_label: str = "i",
        dot_index: int = -1,
    ):
        self._ctx = context
        self._kron = kron
        self._dot_label = dot_label
        self._dot_index = dot_index

    def _require_ctx(self, key: str) -> Callable[..., Any]:
        try:
            return self._ctx[key]
        except KeyError as exc:
            raise KeyError(
                f"Context missing required operator key '{key}'"
            ) from exc

    def _pad_dot(self, local_op: Any) -> Any:
        return self._kron.pad(local_op, self._dot_label, self._dot_index)

    def _eval(self, expr: Any, dims: List[int]) -> np.ndarray:
        mat = interpreter(expr, self._ctx, dims)
        return np.asarray(mat, dtype=np.complex128)

    def op(self, bra: str, ket: str, dims: List[int]) -> np.ndarray:
        loc = self._require_ctx(f"s_{bra}_{ket}")([])
        expr = self._pad_dot(loc)
        return self._eval(expr, dims)
