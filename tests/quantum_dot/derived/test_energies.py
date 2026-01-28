import unittest
from dataclasses import dataclass

from smef.core.units import Q, hbar

from bec.quantum_dot.derived.core import DerivedQDBase
from bec.quantum_dot.derived.energies import EnergiesMixin
from bec.quantum_dot.enums import Transition, TransitionPair
from bec.quantum_dot.transitions import DEFAULT_TRANSITION_REGISTRY


@dataclass(frozen=True)
class _EnergyLevels:
    X1: object
    X2: object
    XX: object


@dataclass(frozen=True)
class _QD:
    energy_levels: _EnergyLevels
    transitions: object = DEFAULT_TRANSITION_REGISTRY


@dataclass(frozen=True)
class _Derived(DerivedQDBase, EnergiesMixin):
    pass


class TestEnergiesMixin(unittest.TestCase):
    def test_energies_dict(self) -> None:
        qd = _QD(
            energy_levels=_EnergyLevels(
                X1=Q(1.305, "eV"),
                X2=Q(1.295, "eV"),
                XX=Q(2.58, "eV"),
            )
        )
        d = _Derived(qd=qd)
        E = d.energies

        self.assertAlmostEqual(float(E["G"].to("eV").magnitude), 0.0, places=12)
        self.assertAlmostEqual(
            float(E["X1"].to("eV").magnitude), 1.305, places=12
        )
        self.assertAlmostEqual(
            float(E["X2"].to("eV").magnitude), 1.295, places=12
        )
        self.assertAlmostEqual(
            float(E["XX"].to("eV").magnitude), 2.58, places=12
        )

    def test_transition_energy_signs(self) -> None:
        qd = _QD(
            energy_levels=_EnergyLevels(
                X1=Q(1.305, "eV"),
                X2=Q(1.295, "eV"),
                XX=Q(2.58, "eV"),
            )
        )
        d = _Derived(qd=qd)

        dE_fwd = float(d.transition_energy[Transition.G_X1].to("eV").magnitude)
        dE_bwd = float(d.transition_energy[Transition.X1_G].to("eV").magnitude)

        self.assertAlmostEqual(dE_fwd, 1.305, places=12)
        self.assertAlmostEqual(dE_bwd, -1.305, places=12)

    def test_omega_matches_energy_over_hbar(self) -> None:
        qd = _QD(
            energy_levels=_EnergyLevels(
                X1=Q(1.305, "eV"),
                X2=Q(1.295, "eV"),
                XX=Q(2.58, "eV"),
            )
        )
        d = _Derived(qd=qd)

        w = d.omega(Transition.G_X1)  # rad/s
        dE_from_w = (w * hbar).to("J")  # should match delta E in J
        dE_direct = d.transition_energy[Transition.G_X1].to("J")

        self.assertAlmostEqual(
            float(dE_from_w.magnitude),
            float(dE_direct.magnitude),
            places=12,
        )

    def test_pair_inputs_use_forward_transition(self) -> None:
        qd = _QD(
            energy_levels=_EnergyLevels(
                X1=Q(1.305, "eV"),
                X2=Q(1.295, "eV"),
                XX=Q(2.58, "eV"),
            ),
            transitions=DEFAULT_TRANSITION_REGISTRY,
        )
        d = _Derived(qd=qd)

        # Pair G<->X1 should resolve to G_X1 (forward)
        w_pair = float(d.omega(TransitionPair.G_X1).to("rad/s").magnitude)
        w_fwd = float(d.omega(Transition.G_X1).to("rad/s").magnitude)
        self.assertAlmostEqual(w_pair, w_fwd, places=12)

    def test_omega_ref_is_absolute(self) -> None:
        qd = _QD(
            energy_levels=_EnergyLevels(
                X1=Q(1.305, "eV"),
                X2=Q(1.295, "eV"),
                XX=Q(2.58, "eV"),
            )
        )
        d = _Derived(qd=qd)

        w_fwd = d.omega_ref_rad_s(Transition.G_X1)
        w_bwd = d.omega_ref_rad_s(Transition.X1_G)

        self.assertAlmostEqual(w_fwd, w_bwd, places=12)
        self.assertGreater(w_fwd, 0.0)
