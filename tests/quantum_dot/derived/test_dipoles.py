import unittest
from dataclasses import dataclass

import numpy as np

from smef.core.units import Q

from bec.quantum_dot.derived.dipoles import DipolesMixin
from bec.quantum_dot.derived.core import DerivedQDBase


class _DipoleParamsQuantity:
    def mu(self, tr):
        return Q(2.0e-29, "C*m")

    def e_pol_hv(self, tr):
        return np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex)


class _DipoleParamsFloat:
    def mu_Cm(self, tr) -> float:
        return 3.0e-29

    def e_pol_hv(self, tr):
        return np.array([0.0 + 0.0j, 1.0 + 0.0j], dtype=complex)


@dataclass(frozen=True)
class _QD:
    dipole_params: object


@dataclass(frozen=True)
class _Derived(DerivedQDBase, DipolesMixin):
    pass


class TestDipolesMixin(unittest.TestCase):
    def test_mu_from_quantity_provider(self) -> None:
        d = _Derived(qd=_QD(dipole_params=_DipoleParamsQuantity()))
        mu = d.mu("any")
        self.assertAlmostEqual(
            float(mu.to("C*m").magnitude), 2.0e-29, places=40
        )

    def test_mu_from_float_provider(self) -> None:
        d = _Derived(qd=_QD(dipole_params=_DipoleParamsFloat()))
        mu = d.mu("any")
        self.assertAlmostEqual(
            float(mu.to("C*m").magnitude), 3.0e-29, places=40
        )

    def test_drive_projection_zero_vector(self) -> None:
        d = _Derived(qd=_QD(dipole_params=_DipoleParamsQuantity()))
        p = d.drive_projection("any", np.array([0.0 + 0.0j, 0.0 + 0.0j]))
        self.assertEqual(p, 0.0 + 0.0j)

    def test_drive_projection_normalization_invariant(self) -> None:
        d = _Derived(qd=_QD(dipole_params=_DipoleParamsQuantity()))
        # e_d = [1,0], so projection equals normalized E[0]
        p1 = d.drive_projection("any", np.array([2.0 + 0.0j, 0.0 + 0.0j]))
        p2 = d.drive_projection("any", np.array([1.0 + 0.0j, 0.0 + 0.0j]))
        self.assertAlmostEqual(p1.real, p2.real, places=12)
        self.assertAlmostEqual(p1.imag, p2.imag, places=12)
        self.assertAlmostEqual(p2.real, 1.0, places=12)

    def test_drive_projection_vdot_convention(self) -> None:
        # e_d = [0,1], E = [1, i] -> normalized E is [1/sqrt(2), i/sqrt(2)]
        # vdot(e_d, E) = conj(e_d)^T E = 1 * E[1] = i/sqrt(2)
        d = _Derived(qd=_QD(dipole_params=_DipoleParamsFloat()))
        p = d.drive_projection("any", np.array([1.0 + 0.0j, 0.0 + 1.0j]))
        self.assertAlmostEqual(p.real, 0.0, places=12)
        self.assertAlmostEqual(p.imag, 1.0 / np.sqrt(2.0), places=12)
