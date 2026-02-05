from __future__ import annotations

import unittest
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from smef.core.units import Q, hbar

from bec.quantum_dot.enums import RateKey, Transition, TransitionPair

# Import the real QDDerivedView from your code
from bec.quantum_dot.smef.derived_view import QDDerivedView
from bec.quantum_dot.transitions import (
    DEFAULT_TRANSITION_REGISTRY,
    TransitionRegistry,
)


@dataclass(frozen=True)
class _EnergyStub:
    """
    Minimal energy structure stub.

    The derived view accesses attributes by QDState.name: G, X1, X2, XX.
    """

    G: Any
    X1: Any
    X2: Any
    XX: Any


class _DipolesStub:
    """
    Minimal dipoles stub providing e_pol_hv(tr) -> (2,) complex.
    """

    def __init__(self, hv_by_tr: Mapping[Transition, np.ndarray]):
        self._hv_by_tr = {
            k: np.asarray(v, dtype=complex).reshape(2)
            for k, v in hv_by_tr.items()
        }

    def e_pol_hv(self, tr: Transition) -> np.ndarray:
        return np.asarray(self._hv_by_tr[tr], dtype=complex).reshape(2)


@dataclass(frozen=True)
class _DecayOutputsStub:
    rates: Mapping[RateKey, Any] = None  # unused here


@dataclass(frozen=True)
class _PhononOutputsStub:
    rates: Mapping[RateKey, Any] = None  # unused here


class TestQDDerivedView(unittest.TestCase):
    def _make_view(
        self, *, registry: TransitionRegistry | None = None
    ) -> QDDerivedView:
        t_registry = registry or DEFAULT_TRANSITION_REGISTRY

        # Choose simple energies (eV)
        # G = 0 eV, X1=1 eV, X2=0.9 eV, XX=2 eV
        energy = _EnergyStub(
            G=Q(0.0, "eV"),
            X1=Q(1.0, "eV"),
            X2=Q(0.9, "eV"),
            XX=Q(2.0, "eV"),
        )

        # Simple dipole polarizations in HV basis
        dipoles = _DipolesStub(
            hv_by_tr={
                Transition.G_X1: np.array([1.0 + 0.0j, 0.0 + 0.0j]),
                Transition.G_X2: np.array([0.0 + 0.0j, 1.0 + 0.0j]),
                # Provide any others you might test later
            }
        )

        # Minimal stubs for unused fields
        decay_out = _DecayOutputsStub(rates={})
        phonon_out = _PhononOutputsStub(rates={})
        rates: Mapping[RateKey, Any] = {}

        return QDDerivedView(
            energy=energy,  # type: ignore[arg-type]
            dipoles=dipoles,  # type: ignore[arg-type]
            mixing=None,
            phonons=None,
            t_registry=t_registry,
            decay_outputs=decay_out,  # type: ignore[arg-type]
            phonon_outputs=phonon_out,  # type: ignore[arg-type]
            rates=rates,
        )

    def test_drive_targets_returns_drive_allowed_pairs(self) -> None:
        """drive_targets should return exactly the registry pairs with drive_allowed=True."""
        view = self._make_view()

        expected = []
        for pair in view.t_registry.pairs():
            if view.t_registry.spec(pair).drive_allowed:
                expected.append(pair)

        self.assertEqual(view.drive_targets(), tuple(expected))

    def test_drive_kind_maps_order(self) -> None:
        """drive_kind should map spec.order 1->'1ph', 2->'2ph'."""
        view = self._make_view()

        # In default registry: G_X1 is order=1, G_XX is order=2
        self.assertEqual(view.drive_kind(TransitionPair.G_X1), "1ph")
        self.assertEqual(view.drive_kind(TransitionPair.G_XX), "2ph")

    def test_omega_ref_rad_s_matches_energy_gap(self) -> None:
        """omega_ref_rad_s should equal (E_dst - E_src)/hbar in rad/s (physical units)."""
        view = self._make_view()

        # For pair G_X1: src=G, dst=X1 => 1 eV gap
        de_J = Q(1.0, "eV").to("J")
        expected = float((de_J / hbar).to("rad/s").magnitude)

        got = float(view.omega_ref_rad_s(TransitionPair.G_X1))
        self.assertAlmostEqual(got, expected, places=9)

        # For pair G_X2: gap = 0.9 eV
        de_J2 = Q(0.9, "eV").to("J")
        expected2 = float((de_J2 / hbar).to("rad/s").magnitude)
        got2 = float(view.omega_ref_rad_s(TransitionPair.G_X2))
        self.assertAlmostEqual(got2, expected2, places=9)

    def test_drive_projection_vdot(self) -> None:
        """drive_projection should return vdot(mu_hat, E) with mu_hat from dipoles."""
        view = self._make_view()

        # mu_hat for G_X1 is H = [1,0]
        E = np.array([0.5 + 0.0j, 0.25 + 0.0j], dtype=complex)
        got = view.drive_projection(Transition.G_X1, E)
        self.assertAlmostEqual(got.real, 0.5, places=12)
        self.assertAlmostEqual(got.imag, 0.0, places=12)

        # mu_hat for G_X2 is V = [0,1]
        got2 = view.drive_projection(Transition.G_X2, E)
        self.assertAlmostEqual(got2.real, 0.25, places=12)
        self.assertAlmostEqual(got2.imag, 0.0, places=12)

    def test_fss_eV(self) -> None:
        """fss_eV should be E(X1)-E(X2) in eV."""
        view = self._make_view()
        self.assertAlmostEqual(float(view.fss_eV), 0.1, places=12)


if __name__ == "__main__":
    unittest.main()
