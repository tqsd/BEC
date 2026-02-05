from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np
from smef.core.units import magnitude

from bec.quantum_dot.enums import Transition

HVVec = np.ndarray  # shape (2,), complex


def _norm2(v: HVVec) -> float:
    v = np.asarray(v, dtype=complex).reshape(2)
    return float(np.vdot(v, v).real)


def _normalize(v: HVVec) -> HVVec:
    v = np.asarray(v, dtype=complex).reshape(2)
    n2 = _norm2(v)
    if n2 <= 0.0:
        return np.array([0.0 + 0.0j, 0.0 + 0.0j], dtype=complex)
    return v / np.sqrt(n2)


class ExcitonBasisReportMixin:
    """
    Reporting-only exciton eigenbasis analysis.

    You simulate in the {X1, X2} basis. This mixin does NOT modify any operators.
    It only derives the exciton eigen-rotation implied by (fss, delta_prime) and
    reports "effective" dipole polarization content for the exciton eigenstates.

    Model:
      In the {X1, X2} basis, exciton Hamiltonian part:
        H = [[ +Delta/2,  delta_prime],
             [ delta_prime, -Delta/2]]

      Delta is the exciton splitting (FSS) in energy units.

      Rotation angle theta satisfies:
        tan(2 theta) = 2*delta_prime / Delta
    """

    def _exciton_delta_eV(self) -> float:
        # Delta = E(X1) - E(X2) in eV
        # Uses your qd energy structure/levels (unitful).
        el = getattr(self.qd, "energy_structure", None) or getattr(
            self.qd, "energy_levels", None
        )
        if el is None:
            raise AttributeError(
                "QuantumDot must provide energy_structure or energy_levels"
            )
        X1 = float(magnitude(el.X1, "eV"))
        X2 = float(magnitude(el.X2, "eV"))
        return X1 - X2

    def _exciton_delta_prime_eV(self) -> float:
        mp = getattr(self.qd, "exciton_mixing_params", None)
        if mp is None:
            return 0.0
        return float(magnitude(getattr(mp, "delta_prime", 0.0), "eV"))

    @cached_property
    def exciton_theta_rad(self) -> float:
        """
        Rotation angle theta (radians) that diagonalizes the exciton 2x2 block.
        """
        Delta = float(self._exciton_delta_eV())
        dp = float(self._exciton_delta_prime_eV())
        # theta = 0.5 * atan2(2 dp, Delta)
        return 0.5 * float(np.arctan2(2.0 * dp, Delta))

    @cached_property
    def exciton_rotation_matrix(self) -> np.ndarray:
        """
        Orthonormal rotation matrix U such that:
          [Xa]   [ cos(theta)  sin(theta)] [X1]
          [Xb] = [-sin(theta)  cos(theta)] [X2]
        """
        th = float(self.exciton_theta_rad)
        c = float(np.cos(th))
        s = float(np.sin(th))
        return np.array([[c, s], [-s, c]], dtype=float)

    def exciton_eigen_energies_eV(self) -> tuple[float, float]:
        """
        Eigen-energies relative to the exciton center in eV:
          (+lambda, -lambda)
        where lambda = sqrt((Delta/2)^2 + delta_prime^2)
        """
        Delta = float(self._exciton_delta_eV())
        dp = float(self._exciton_delta_prime_eV())
        lam = float(np.sqrt((0.5 * Delta) ** 2 + dp**2))
        return (lam, -lam)

    def _effective_exciton_dipoles_hv(self) -> dict[str, HVVec]:
        """
        Build effective HV dipole direction vectors for exciton eigenstates.

        We use the *ground-exciton* dipole directions:
          d(G<->Xa) = c d(G<->X1) + s d(G<->X2)
          d(G<->Xb) = -s d(G<->X1) + c d(G<->X2)

        This is a reporting proxy: it tells you which lab polarization the
        eigenstates would look like if you spectrally addressed them.
        """
        U = self.exciton_rotation_matrix
        c = float(U[0, 0])
        s = float(U[0, 1])

        d1 = _normalize(self.e_pol_hv(Transition.G_X1))
        d2 = _normalize(self.e_pol_hv(Transition.G_X2))

        d_a = _normalize(c * d1 + s * d2)
        d_b = _normalize(-s * d1 + c * d2)

        return {"Xa": d_a, "Xb": d_b}

    def exciton_effective_polarization_report(self) -> dict[str, Any]:
        """
        Return a pure-data dict suitable for rich/plain reporting.

        Includes:
          - theta_rad
          - Delta_eV, delta_prime_eV
          - effective dipole vectors for Xa, Xb in HV
          - H/V power fractions for each (in the HV basis)
        """
        Delta = float(self._exciton_delta_eV())
        dp = float(self._exciton_delta_prime_eV())
        th = float(self.exciton_theta_rad)

        eff = self._effective_exciton_dipoles_hv()
        out: dict[str, Any] = {
            "Delta_eV": Delta,
            "delta_prime_eV": dp,
            "theta_rad": th,
            "theta_deg": float(th * 180.0 / np.pi),
            "eigen_relative_eV": self.exciton_eigen_energies_eV(),
            "effective": {},
        }

        for name, v in eff.items():
            v = _normalize(v)
            pH = float(abs(v[0]) ** 2)
            pV = float(abs(v[1]) ** 2)
            out["effective"][name] = {
                "hv_vec": (complex(v[0]), complex(v[1])),
                "pH": pH,
                "pV": pV,
            }

        return out
