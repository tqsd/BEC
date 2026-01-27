from __future__ import annotations

from typing import Any, Callable, Dict, Sequence
import numpy as np

from photon_weave.extra import interpreter


def materialize_expr(
    expr: Any, ctx: Dict[str, Callable], dims: Sequence[int]
) -> np.ndarray:
    """
    Materialize a PhotonWeave expression to a concrete ndarray.

    Parameters
    ----------
    expr:
        PhotonWeave expression tuple (nested tuples/strings/matrices)
    ctx:
        Mapping symbol->callable(dims)->matrix (or symbol->matrix)
    dims:
        Full system dims, e.g. [4, d0H, d0V, d1H, d1V, ...]

    Returns
    -------
    np.ndarray:
        The full operator matrix of shape (D, D) where D = prod(dims).
    """
    A = interpreter(expr, ctx, list(dims))
    return np.asarray(A)
