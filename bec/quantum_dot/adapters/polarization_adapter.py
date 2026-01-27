from __future__ import annotations

from typing import TYPE_CHECKING, Any
import numpy as np

if TYPE_CHECKING:
    from bec.quantum_dot.dot import QuantumDot


class QDPolarizationAdapter:
    def __init__(self, qd: "QuantumDot"):
        self._dip = qd.dipole_params

    def coupling_weight(self, tr: Any, E_hv: np.ndarray) -> complex:
        E = np.asarray(E_hv, dtype=complex).reshape(
            2,
        )
        nE = np.linalg.norm(E)
        if nE == 0:
            return 0.0 + 0j
        E = E / nE

        # normalized direction only (dimensionless)
        d_dir = np.asarray(self._dip.e_pol_hv(tr), dtype=complex).reshape(
            2,
        )
        # return complex(np.vdot(d_dir, E))

        c = complex(np.vdot(d_dir, E))
        print("coupling", tr, "d=", d_dir, "E=", E, "->", c, "abs", abs(c))
        return c
