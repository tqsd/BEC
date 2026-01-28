from __future__ import annotations

import numpy as np

from smef.core.units import QuantityLike, Q, as_quantity

from .types import HVVec


class DipolesMixin:
    def mu(self, tr) -> QuantityLike:
        """
        Return dipole magnitude mu(tr) as a QuantityLike in C*m.

        Supports two styles of dipole_params:
          - provides mu(tr) returning a QuantityLike
          - provides mu_Cm(tr) returning a float in SI (C*m)
        """
        dp = self.qd.dipoles
        if hasattr(dp, "mu"):
            return as_quantity(dp.mu(tr), "C*m")
        return Q(float(dp.mu_Cm(tr)), "C*m")

    def e_pol_hv(self, tr) -> HVVec:
        """
        Return normalized dipole polarization direction in HV basis.
        """
        return self.qd.dipoles.e_pol_hv(tr)

    def drive_projection(self, tr, E_pol_hv: HVVec) -> complex:
        """
        Return <e_d(tr) | E_pol> with both vectors normalized.
        Uses np.vdot, so the first argument is conjugated.
        """
        E = np.asarray(E_pol_hv, dtype=complex).reshape(2)
        n = float(np.linalg.norm(E))
        if n == 0.0:
            return 0.0 + 0.0j
        E = E / n

        e_d = np.asarray(self.e_pol_hv(tr), dtype=complex).reshape(2)
        return complex(np.vdot(e_d, E))
