from __future__ import annotations

import unittest

from smef.core.units import Q
from smef.engine import UnitSystem

from bec.quantum_dot.dot import QuantumDot
from bec.quantum_dot.enums import RateKey
from bec.quantum_dot.smef.catalogs.collapse import QDCollapseCatalog
from bec.quantum_dot.smef.modes import QDModes
from bec.quantum_dot.spec.dipole_params import DipoleParams
from bec.quantum_dot.spec.energy_structure import EnergyStructure
from bec.quantum_dot.spec.exciton_mixing_params import ExcitonMixingParams
from bec.quantum_dot.spec.phonon_params import (
    PhenomenologicalPhononParams,
    PhononModelKind,
    PhononParams,
)


def _units() -> UnitSystem:
    # 1 solver unit = 1 ps
    try:
        return UnitSystem(time_unit_s=1e-12)
    except TypeError:
        return UnitSystem(time_unit_s=float(Q(1.0, "ps").to("s").magnitude))


class TestQDCollapseCatalogIntegration(unittest.TestCase):
    def _energy(self) -> EnergyStructure:
        # Choose a nonzero FSS so theta derivation is well-defined
        return EnergyStructure.from_params(
            exciton=1.30,  # eV
            fss=1e-4,  # eV
            binding=0.0,  # eV
        )

    def _dipoles(self) -> DipoleParams:
        return DipoleParams.from_values(mu_default_Cm=1e-29)

    def _qd_no_phonons(self) -> QuantumDot:
        return QuantumDot(
            energy=self._energy(),
            dipoles=self._dipoles(),
            cavity=None,
            phonons=None,
            mixing=None,
        )

    def _qd_with_phonon_rates(self) -> QuantumDot:
        ph = PhononParams(
            kind=PhononModelKind.POLARON_LA,
            temperature=Q(4.0, "K"),
            phenomenological=PhenomenologicalPhononParams(
                gamma_phi_x1=Q(1e9, "1/s"),
                gamma_phi_x2=Q(2e9, "1/s"),
                gamma_phi_xx=Q(3e9, "1/s"),
                gamma_relax_x1_x2=Q(4e9, "1/s"),
                gamma_relax_x2_x1=Q(5e9, "1/s"),
            ),
        )
        return QuantumDot(
            energy=self._energy(),
            dipoles=self._dipoles(),
            cavity=None,
            phonons=ph,
            mixing=None,
        )

    def test_smoke_radiative_terms_exist(self) -> None:
        qd = self._qd_no_phonons()
        units = _units()
        modes = QDModes(fock_dim=2)

        cat = QDCollapseCatalog.from_qd(qd, modes=modes, units=units)
        terms = list(cat.all_terms)

        labels = [t.label for t in terms]
        self.assertIn("L_XX", labels)
        self.assertIn("L_GX", labels)

        # With no phonons, we expect exactly the two radiative terms.
        self.assertEqual(len(terms), 2)

        # Sanity: rates used by the catalog are present
        rates = qd.rates
        self.assertIn(RateKey.RAD_XX_X1, rates)
        self.assertIn(RateKey.RAD_XX_X2, rates)
        self.assertIn(RateKey.RAD_X1_G, rates)
        self.assertIn(RateKey.RAD_X2_G, rates)

    def test_phonon_terms_emitted_when_rates_present(self) -> None:
        qd = self._qd_with_phonon_rates()
        units = _units()
        modes = QDModes(fock_dim=2)

        cat = QDCollapseCatalog.from_qd(qd, modes=modes, units=units)
        terms = list(cat.all_terms)
        labels = [t.label for t in terms]

        # Radiative always
        self.assertIn("L_XX", labels)
        self.assertIn("L_GX", labels)

        # Phonon dephasing projectors
        self.assertIn("L_ph_deph_X1", labels)
        self.assertIn("L_ph_deph_X2", labels)
        self.assertIn("L_ph_deph_XX", labels)

        # Phonon relaxation
        self.assertIn("L_ph_relax_X1_X2", labels)
        self.assertIn("L_ph_relax_X2_X1", labels)

        # Total terms: 2 radiative + 3 deph + 2 relax = 7
        self.assertEqual(len(terms), 7)

    def test_theta_default_from_energy_and_mixing(self) -> None:
        # theta = 0.5 * atan2(2*delta_prime, fss)
        energy = self._energy()
        mixing = ExcitonMixingParams.from_values(delta_prime_eV=2.0e-4)

        qd = QuantumDot(
            energy=energy,
            dipoles=self._dipoles(),
            cavity=None,
            phonons=None,
            mixing=mixing,
        )

        units = _units()
        modes = QDModes(fock_dim=2)

        cat = QDCollapseCatalog.from_qd(qd, modes=modes, units=units)
        terms = list(cat.all_terms)

        # Both radiative terms should include theta in meta and match each other
        t_xx = next(t for t in terms if t.label == "L_XX")
        t_gx = next(t for t in terms if t.label == "L_GX")

        self.assertIn("theta", t_xx.meta)
        self.assertIn("theta", t_gx.meta)

        theta_xx = float(t_xx.meta["theta"])
        theta_gx = float(t_gx.meta["theta"])
        self.assertAlmostEqual(theta_xx, theta_gx, places=12)

        # Just sanity: should be nonzero for nonzero delta_prime
        self.assertNotEqual(theta_xx, 0.0)

    def test_theta_override(self) -> None:
        qd = self._qd_no_phonons()
        units = _units()
        modes = QDModes(fock_dim=2)

        cat = QDCollapseCatalog.from_qd(
            qd, modes=modes, units=units, theta=0.123, phi=0.0
        )
        terms = list(cat.all_terms)
        t_xx = next(t for t in terms if t.label == "L_XX")
        t_gx = next(t for t in terms if t.label == "L_GX")

        self.assertAlmostEqual(float(t_xx.meta["theta"]), 0.123, places=12)
        self.assertAlmostEqual(float(t_gx.meta["theta"]), 0.123, places=12)


if __name__ == "__main__":
    unittest.main()
