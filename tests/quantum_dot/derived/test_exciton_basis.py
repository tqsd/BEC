import unittest
import numpy as np

from smef.core.units import Q
from bec.quantum_dot.enums import Transition
from bec.quantum_dot.spec.energy_structure import EnergyStructure
from bec.quantum_dot.spec.exciton_mixing_params import ExcitonMixingParams
from bec.quantum_dot.spec.dipole_params import DipoleParams
from bec.quantum_dot.derived.exciton_basis import ExcitonBasisReportMixin


class _QDStub:
    def __init__(
        self, *, energy_structure, exciton_mixing_params, dipole_params
    ):
        self.energy_structure = energy_structure
        self.exciton_mixing_params = exciton_mixing_params
        self.dipole_params = dipole_params


class _DerivedStub(ExcitonBasisReportMixin):
    def __init__(self, qd):
        self.qd = qd

    # Provide the method used by the mixin (from DipolesMixin in your real class)
    def e_pol_hv(self, tr):
        return self.qd.dipole_params.e_pol_hv(tr)


class TestExcitonBasisReportMixin(unittest.TestCase):
    def test_theta_zero_when_no_mixing(self):
        es = EnergyStructure.from_params(
            exciton=Q(1.3, "eV"), fss=Q(20e-6, "eV"), binding=Q(0.0, "eV")
        )
        mp = ExcitonMixingParams.from_values(delta_prime_eV=0.0)
        dp = DipoleParams.biexciton_cascade_defaults()

        d = _DerivedStub(
            _QDStub(
                energy_structure=es, exciton_mixing_params=mp, dipole_params=dp
            )
        )
        self.assertAlmostEqual(d.exciton_theta_rad, 0.0, places=12)

        rep = d.exciton_effective_polarization_report()
        self.assertIn("effective", rep)
        xa = rep["effective"]["Xa"]
        xb = rep["effective"]["Xb"]

        # With theta=0, Xa ~ X1 dipole (H) and Xb ~ X2 dipole (V)
        self.assertGreater(xa["pH"], 0.99)
        self.assertGreater(xb["pV"], 0.99)

    def test_theta_near_45deg_when_delta_prime_dominates(self):
        es = EnergyStructure.from_params(
            exciton=Q(1.3, "eV"), fss=Q(1e-9, "eV"), binding=Q(0.0, "eV")
        )
        mp = ExcitonMixingParams.from_values(delta_prime_eV=1e-6)
        dp = DipoleParams.biexciton_cascade_defaults()

        d = _DerivedStub(
            _QDStub(
                energy_structure=es, exciton_mixing_params=mp, dipole_params=dp
            )
        )
        # For Delta ~ 0, theta ~ 45 deg
        self.assertTrue(abs(d.exciton_theta_rad - (0.25 * np.pi)) < 1e-3)

        rep = d.exciton_effective_polarization_report()
        xa = rep["effective"]["Xa"]
        xb = rep["effective"]["Xb"]

        # Each should be roughly mixed H/V
        self.assertTrue(0.35 < xa["pH"] < 0.65)
        self.assertTrue(0.35 < xb["pH"] < 0.65)

    def test_vectors_normalized(self):
        es = EnergyStructure.from_params(
            exciton=Q(1.3, "eV"), fss=Q(5e-6, "eV"), binding=Q(0.0, "eV")
        )
        mp = ExcitonMixingParams.from_values(delta_prime_eV=2e-6)
        dp = DipoleParams.biexciton_cascade_defaults()

        d = _DerivedStub(
            _QDStub(
                energy_structure=es, exciton_mixing_params=mp, dipole_params=dp
            )
        )
        rep = d.exciton_effective_polarization_report()
        for name in ("Xa", "Xb"):
            v0, v1 = rep["effective"][name]["hv_vec"]
            n2 = abs(v0) ** 2 + abs(v1) ** 2
            self.assertAlmostEqual(n2, 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
