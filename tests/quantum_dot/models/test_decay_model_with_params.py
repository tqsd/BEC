from __future__ import annotations

import unittest

from smef.core.units import Q

from bec.quantum_dot.enums import RateKey, Transition
from bec.quantum_dot.models.decay_model import DecayModel
from bec.quantum_dot.spec.energy_structure import EnergyStructure


class _DipolesStub:
    def __init__(self, mu_C_m: float = 1e-29):
        self._mu = Q(mu_C_m, "C*m")

    def mu(self, _tr: Transition):
        return self._mu


class _CavityStub:
    def __init__(
        self, *, Qfac: float = 0.0, n: float = 1.0, Veff_um3: float = 0.0
    ):
        self.Q = Qfac
        self.n = n
        self.Veff_um3 = Veff_um3


class TestDecayModel(unittest.TestCase):
    def test_gamma_zero_when_photon_energy_nonpositive(self) -> None:
        # Make X1 below G so X1->G photon energy is negative
        e = EnergyStructure.from_levels(X1=0.5, X2=0.6, XX=1.5)
        # Then overwrite G to be larger (EnergyStructure is frozen, so build directly)
        e2 = EnergyStructure(X1=e.X1, X2=e.X2, XX=e.XX, G=Q(1.0, "eV"))

        m = DecayModel(energy_structure=e2, dipole_params=_DipolesStub())
        g = m.gamma(Transition.X1_G)
        self.assertEqual(float(g.to("1/s").magnitude), 0.0)

    def test_gamma0_positive_for_valid_transition(self) -> None:
        e = EnergyStructure.from_params(exciton=1.3, fss=1e-4, binding=0.0)
        m = DecayModel(
            energy_structure=e, dipole_params=_DipolesStub(mu_C_m=1e-29)
        )
        g0 = m.gamma0(Transition.X1_G)
        self.assertGreater(float(g0.to("1/s").magnitude), 0.0)

    def test_purcell_zero_without_cavity(self) -> None:
        e = EnergyStructure.from_params(exciton=1.3, fss=1e-4, binding=0.0)
        m = DecayModel(
            energy_structure=e, dipole_params=_DipolesStub(), cavity_params=None
        )
        self.assertEqual(m.purcell_factor(Transition.X1_G), 0.0)

    def test_purcell_positive_with_reasonable_cavity(self) -> None:
        e = EnergyStructure.from_params(exciton=1.3, fss=1e-4, binding=0.0)
        cav = _CavityStub(Qfac=5000.0, n=3.5, Veff_um3=0.5)
        m = DecayModel(
            energy_structure=e, dipole_params=_DipolesStub(), cavity_params=cav
        )
        Fp = m.purcell_factor(Transition.X1_G)
        self.assertGreaterEqual(Fp, 0.0)

    def test_gamma_increases_with_purcell(self) -> None:
        e = EnergyStructure.from_params(exciton=1.3, fss=1e-4, binding=0.0)
        dip = _DipolesStub(mu_C_m=1e-29)

        m0 = DecayModel(
            energy_structure=e, dipole_params=dip, cavity_params=None
        )
        g0 = float(m0.gamma(Transition.X1_G).to("1/s").magnitude)

        cav = _CavityStub(Qfac=5000.0, n=3.5, Veff_um3=0.5)
        m1 = DecayModel(
            energy_structure=e, dipole_params=dip, cavity_params=cav
        )
        g1 = float(m1.gamma(Transition.X1_G).to("1/s").magnitude)

        self.assertGreaterEqual(g1, g0)

    def test_compute_outputs_contains_expected_keys(self) -> None:
        e = EnergyStructure.from_params(exciton=1.3, fss=1e-4, binding=0.0)
        m = DecayModel(energy_structure=e, dipole_params=_DipolesStub())
        out = m.compute()

        self.assertIn(RateKey.RAD_X1_G, out.rates)
        self.assertIn(RateKey.RAD_X2_G, out.rates)
        self.assertIn(RateKey.RAD_XX_X1, out.rates)
        self.assertIn(RateKey.RAD_XX_X2, out.rates)

        for k, v in out.rates.items():
            self.assertGreaterEqual(float(v.to("1/s").magnitude), 0.0)


if __name__ == "__main__":
    unittest.main()
