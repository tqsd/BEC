from __future__ import annotations

import numpy as np

from bec.units import QuantityLike, Q, as_quantity

from .types import HVVec


class DipolesMixin:
    def mu(self, tr) -> QuantityLike:
        if hasattr(self.qd.dipole_params, "mu"):
            return as_quantity(self.qd.dipole_params.mu(tr), "C*m")
        return Q(float(self.qd.dipole_params.mu_Cm(tr)), "C*m")

    def e_pol_hv(self, tr) -> HVVec:
        return self.qd.dipole_params.e_pol_hv(tr)

    def drive_projection(self, tr, E_pol_hv: HVVec) -> complex:
        E_pol_hv = np.asarray(E_pol_hv, dtype=complex).reshape(2)
        n = np.linalg.norm(E_pol_hv)
        if n == 0:
            return 0.0 + 0.0j
        E_pol_hv = E_pol_hv / n
        e_d = np.asarray(self.e_pol_hv(tr), dtype=complex).reshape(2)
        return complex(np.vdot(e_d, E_pol_hv))
