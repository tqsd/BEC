from __future__ import annotations

import numpy as np


class PolarizationMixin:
    def e_plus_hv(self) -> np.ndarray:
        pol = getattr(self.qd, "polarization", None)
        if pol is None:
            return np.array([1.0 + 0j, 0.0 + 0j], dtype=complex)
        return np.asarray(pol.e_plus_hv(), dtype=complex).reshape(2)

    def e_minus_hv(self) -> np.ndarray:
        pol = getattr(self.qd, "polarization", None)
        if pol is None:
            return np.array([0.0 + 0j, 1.0 + 0j], dtype=complex)
        return np.asarray(pol.e_minus_hv(), dtype=complex).reshape(2)

    def overlap_to_pm(self, tr) -> tuple[float, float]:
        e_d = np.asarray(self.e_pol_hv(tr), dtype=complex).reshape(2)
        e_d = e_d / np.linalg.norm(e_d)
        e_p = self.e_plus_hv()
        e_m = self.e_minus_hv()
        e_p = e_p / np.linalg.norm(e_p)
        e_m = e_m / np.linalg.norm(e_m)
        p_plus = float(abs(np.vdot(e_p, e_d)) ** 2)
        p_minus = float(abs(np.vdot(e_m, e_d)) ** 2)
        return p_plus, p_minus

    def coupled_pm_label(self, tr, *, thresh: float = 0.75) -> str:
        p_plus, p_minus = self.overlap_to_pm(tr)
        if p_plus >= thresh and p_plus > p_minus:
            return f"+ (p+={p_plus:.2f})"
        if p_minus >= thresh and p_minus > p_plus:
            return f"- (p-={p_minus:.2f})"
        return f"mixed (p+={p_plus:.2f}, p-={p_minus:.2f})"
