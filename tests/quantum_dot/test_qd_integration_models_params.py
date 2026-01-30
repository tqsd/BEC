from __future__ import annotations

import unittest

from smef.core.units import Q
from smef.engine import UnitSystem

from bec.quantum_dot.dot import QuantumDot
from bec.quantum_dot.enums import RateKey
from bec.quantum_dot.spec.energy_structure import EnergyStructure
from bec.quantum_dot.spec.dipole_params import DipoleParams
from bec.quantum_dot.spec.phonon_params import (
    PhononParams,
    PhononModelKind,
    PhenomenologicalPhononParams,
    PolaronLAParams,
    PhononCouplings,
    SpectralDensityKind,
)


def _units() -> UnitSystem:
    # 1 solver unit = 1 ps
    try:
        return UnitSystem(time_unit_s=1e-12)
    except TypeError:
        return UnitSystem(time_unit_s=float(Q(1.0, "ps").to("s").magnitude))


class TestQuantumDotWiring(unittest.TestCase):
    def _energy(self) -> EnergyStructure:
        return EnergyStructure.from_params(
            exciton=1.3,  # eV
            fss=1e-4,  # eV
            binding=0.0,  # eV
        )

    def _dipoles(self) -> DipoleParams:
        return DipoleParams.from_values(mu_default_Cm=1e-29)

    def _phonons_with_phenom_rates(self) -> PhononParams:
        return PhononParams(
            kind=PhononModelKind.POLARON_LA,
            temperature=Q(4.0, "K"),
            couplings=PhononCouplings(
                phi_g=0.0,
                phi_x1=1.0,
                phi_x2=1.0,
                phi_xx=2.0,
            ),
            phenomenological=PhenomenologicalPhononParams(
                gamma_phi_x1=Q(1e9, "1/s"),
                gamma_phi_x2=Q(2e9, "1/s"),
                gamma_phi_xx=Q(3e9, "1/s"),
                gamma_relax_x1_x2=Q(4e9, "1/s"),
                gamma_relax_x2_x1=Q(5e9, "1/s"),
            ),
            polaron_la=PolaronLAParams(
                enable_polaron_renorm=False,
                enable_exciton_relaxation=False,
                enable_eid=False,
                spectral_density=SpectralDensityKind.SUPER_OHMIC_GAUSSIAN,
                alpha=Q(0.0, "s**2"),
                omega_c=Q(1e12, "rad/s"),
            ),
        )

    def _phonons_polaron_minimal(self) -> PhononParams:
        return PhononParams(
            kind=PhononModelKind.POLARON_LA,
            temperature=Q(4.0, "K"),
            couplings=PhononCouplings(
                phi_g=0.0,
                phi_x1=1.0,
                phi_x2=1.0,
                phi_xx=2.0,
            ),
            phenomenological=PhenomenologicalPhononParams(
                gamma_phi_x1=Q(0.0, "1/s"),
                gamma_phi_x2=Q(0.0, "1/s"),
                gamma_phi_xx=Q(0.0, "1/s"),
                gamma_relax_x1_x2=Q(0.0, "1/s"),
                gamma_relax_x2_x1=Q(0.0, "1/s"),
            ),
            polaron_la=PolaronLAParams(
                enable_polaron_renorm=True,
                enable_exciton_relaxation=False,
                enable_eid=False,
                spectral_density=SpectralDensityKind.SUPER_OHMIC_GAUSSIAN,
                alpha=Q(1e-26, "s**2"),
                omega_c=Q(1e12, "rad/s"),
            ),
        )

    def test_decay_outputs_present(self) -> None:
        qd = QuantumDot(
            energy=self._energy(),
            dipoles=self._dipoles(),
            cavity=None,
            phonons=None,
            mixing=None,
        )
        out = qd.decay_outputs
        self.assertTrue(hasattr(out, "rates"))
        self.assertIn(RateKey.RAD_X1_G, out.rates)

        g = float(out.rates[RateKey.RAD_X1_G].to("1/s").magnitude)
        self.assertGreaterEqual(g, 0.0)

    def test_phonon_outputs_empty_when_phonons_none(self) -> None:
        qd = QuantumDot(
            energy=self._energy(),
            dipoles=self._dipoles(),
            phonons=None,
        )
        out = qd.phonon_outputs
        self.assertEqual(out.rates, {})
        self.assertEqual(out.b_polaron, {})
        self.assertFalse(out.eid.enabled)

    def test_rates_aggregate_decay_and_phonons(self) -> None:
        qd = QuantumDot(
            energy=self._energy(),
            dipoles=self._dipoles(),
            phonons=self._phonons_with_phenom_rates(),
        )
        rates = qd.rates

        self.assertIn(RateKey.RAD_X1_G, rates)

        self.assertIn(RateKey.PH_DEPH_X1, rates)
        self.assertIn(RateKey.PH_DEPH_X2, rates)
        self.assertIn(RateKey.PH_DEPH_XX, rates)
        self.assertIn(RateKey.PH_RELAX_X1_X2, rates)
        self.assertIn(RateKey.PH_RELAX_X2_X1, rates)

        self.assertAlmostEqual(
            float(rates[RateKey.PH_DEPH_X1].to("1/s").magnitude),
            1e9,
            places=6,
        )

    def test_polaron_b_map_nonempty_when_enabled(self) -> None:
        qd = QuantumDot(
            energy=self._energy(),
            dipoles=self._dipoles(),
            phonons=self._phonons_polaron_minimal(),
        )
        out = qd.phonon_outputs

        self.assertGreater(len(out.b_polaron), 0)

        for tr, b in out.b_polaron.items():
            self.assertIsInstance(b, float)
            self.assertGreaterEqual(b, 0.0)
            self.assertLessEqual(b, 1.0)

    def test_compile_bundle_smoke(self) -> None:
        qd = QuantumDot(
            energy=self._energy(),
            dipoles=self._dipoles(),
            phonons=None,
        )
        bundle = qd.compile_bundle(units=_units())

        self.assertTrue(hasattr(bundle, "hamiltonian"))
        self.assertTrue(hasattr(bundle, "collapse"))
        self.assertTrue(hasattr(bundle, "modes"))

        self.assertTrue(hasattr(bundle.hamiltonian, "all_terms"))
        self.assertTrue(hasattr(bundle.collapse, "all_terms"))


if __name__ == "__main__":
    unittest.main()
