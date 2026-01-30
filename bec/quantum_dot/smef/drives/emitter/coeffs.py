from __future__ import annotations

import numpy as np


class ArrayCoeff:
    """
    Simple coefficient backed by a precomputed array on the solver grid.

    This stays backend-agnostic: SMEF only requires `.eval(tlist)`.
    """

    def __init__(self, values: np.ndarray):
        self._values = np.asarray(values, dtype=complex).reshape(-1)

    def eval(self, tlist: np.ndarray) -> np.ndarray:
        tlist = np.asarray(tlist, dtype=float).reshape(-1)
        if tlist.size != self._values.size:
            raise ValueError(
                "Coeff length mismatch: len(tlist)=%d vs len(values)=%d"
                % (tlist.size, self._values.size)
            )
        return self._values
