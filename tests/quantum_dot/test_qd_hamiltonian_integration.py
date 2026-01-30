from __future__ import annotations

import unittest

from smef.core.units import Q

from smef.engine import UnitSystem
from smef.core.ir.terms import TermKind

from bec.quantum_dot.dot import QuantumDot
from bec.quantum_dot.spec.energy_structure import EnergyStructure
from bec.quantum_dot.spec.dipole_params import DipoleParams
from bec.quantum_dot.spec.exciton_mixing_params import ExcitonMixingParams


def _units_1ps() -> UnitSystem:
    # 1 solver unit = 1 ps
    try:
        return UnitSystem(time_unit_s=1e-12)
    except TypeError:
        return UnitSystem(time_unit_s=float(Q(1.0, "ps").to("s").magnitude))


class TestStaticHamiltonianIntegration(unittest.TestCase):
    def _dipoles(self) -> DipoleParams:
        # Keep it minimal; Hamiltonian catalog does not depend on dipoles.
        return DipoleParams.from_values(mu_default_Cm=1e-29)

    def test_hamiltonian_term_present_when_fss_nonzero(self) -> None:
        # Non-zero FSS via from_params
        energy = EnergyStructure.from_params(
            exciton=1.3,  # eV
            fss=1e-4,  # eV  (non-zero)
            binding=0.0,  # eV
        )
        qd = QuantumDot(
            energy=energy,
            dipoles=self._dipoles(),
            mixing=None,
            phonons=None,
            cavity=None,
        )

        bundle = qd.compile_bundle(units=_units_1ps())
        terms = list(bundle.hamiltonian.all_terms)

        # Expect exactly one static Hamiltonian term
        self.assertEqual(len(terms), 1)
        t = terms[0]
        self.assertEqual(t.kind, TermKind.H)
        self.assertEqual(t.label, "H_exciton_fss_mix")

        # Meta should include Delta and delta_prime in eV
        self.assertTrue(isinstance(t.meta, dict))
        self.assertIn("Delta_eV", t.meta)
        self.assertIn("delta_prime_eV", t.meta)
        self.assertAlmostEqual(float(t.meta["Delta_eV"]), 1e-4, places=12)
        self.assertAlmostEqual(float(t.meta["delta_prime_eV"]), 0.0, places=12)

    def test_hamiltonian_term_present_when_delta_prime_nonzero(self) -> None:
        # Zero FSS, but non-zero mixing
        energy = EnergyStructure.from_params(
            exciton=1.3,
            fss=0.0,
            binding=0.0,
        )
        mixing = ExcitonMixingParams.from_values(delta_prime_eV=2e-6)

        qd = QuantumDot(
            energy=energy,
            dipoles=self._dipoles(),
            mixing=mixing,
            phonons=None,
            cavity=None,
        )

        bundle = qd.compile_bundle(units=_units_1ps())
        terms = list(bundle.hamiltonian.all_terms)

        self.assertEqual(len(terms), 1)
        t = terms[0]
        self.assertEqual(t.kind, TermKind.H)
        self.assertEqual(t.label, "H_exciton_fss_mix")

        self.assertIn("Delta_eV", t.meta)
        self.assertIn("delta_prime_eV", t.meta)
        self.assertAlmostEqual(float(t.meta["Delta_eV"]), 0.0, places=12)
        self.assertAlmostEqual(float(t.meta["delta_prime_eV"]), 2e-6, places=12)

    def test_hamiltonian_terms_empty_when_fss_and_delta_prime_zero(
        self,
    ) -> None:
        energy = EnergyStructure.from_params(
            exciton=1.3,
            fss=0.0,
            binding=0.0,
        )
        mixing = ExcitonMixingParams.from_values(delta_prime_eV=0.0)

        qd = QuantumDot(
            energy=energy,
            dipoles=self._dipoles(),
            mixing=mixing,
            phonons=None,
            cavity=None,
        )

        bundle = qd.compile_bundle(units=_units_1ps())
        terms = list(bundle.hamiltonian.all_terms)
        self.assertEqual(terms, [])

    def test_hamiltonian_is_drive_independent(self) -> None:
        # This is a regression-style test: compile_bundle should always provide
        # a Hamiltonian catalog; drive terms are emitted elsewhere.
        energy = EnergyStructure.from_params(exciton=1.3, fss=1e-4, binding=0.0)
        qd = QuantumDot(energy=energy, dipoles=self._dipoles())

        bundle = qd.compile_bundle(units=_units_1ps())

        # Static hamiltonian terms exist (from catalog)
        self.assertTrue(hasattr(bundle, "hamiltonian"))
        self.assertTrue(hasattr(bundle.hamiltonian, "all_terms"))
        h_terms = list(bundle.hamiltonian.all_terms)

        # All are TermKind.H (static system terms only in this catalog)
        for t in h_terms:
            self.assertEqual(t.kind, TermKind.H)


if __name__ == "__main__":
    unittest.main()
