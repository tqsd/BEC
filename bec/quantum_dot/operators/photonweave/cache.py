from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Sequence, Tuple
import hashlib
import numpy as np

from bec.quantum_dot.operators.photonweave.materialize import materialize_expr


def _expr_fingerprint(expr: Any) -> str:
    """
    Create a stable-ish fingerprint for an expr tree.
    We avoid hashing raw ndarray contents by replacing them with a tag.
    """

    def normalize(x: Any) -> Any:
        if isinstance(x, (tuple, list)):
            return tuple(normalize(y) for y in x)
        if isinstance(x, np.ndarray):
            return ("__ndarray__", x.shape, str(x.dtype))
        return x

    norm = normalize(expr)
    h = hashlib.sha256(repr(norm).encode("utf-8")).hexdigest()
    return h


@dataclass
class PWCache:
    cache: Dict[Tuple[str, Tuple[int, ...]], np.ndarray] = field(
        default_factory=dict
    )

    def get_or_build(
        self, expr: Any, ctx: Dict[str, Callable], dims: Sequence[int]
    ) -> np.ndarray:
        key = (_expr_fingerprint(expr), tuple(int(d) for d in dims))
        if key in self.cache:
            return self.cache[key]
        A = materialize_expr(expr, ctx, dims)
        self.cache[key] = A
        return A
