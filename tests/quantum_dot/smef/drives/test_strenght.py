from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from smef.core.units import hbar, magnitude
from smef.core.drives.types import DriveSpec, ResolvedDrive

from bec.quantum_dot.enums import Transition, TransitionPair
from bec.quantum_dot.transitions import DEFAULT_TRANSITION_REGISTRY

from bec.quantum_dot.smef.drives.context import QDDriveDecodeContext
from bec.quantum_dot.smef.drives.strength import QDDriveStrengthModel


@dataclass
class _FakeDrivePayload:
    """
    Minimal payload for strength model tests.

    Required:
      - E_env_V_m(t_phys_s) -> float

    Optional:
      - effective_pol() -> (2,) complex
    """

    E0: float
    pol: Optional[np.ndarray] = None

    def E_env_V_m(self, _t_phys_s: float) -> float:
        return float(self.E0)

    def effective_pol(self) -> Optional[np.ndarray]:
        if self.pol is None:
            return None
        return np.asarray(self.pol, dtype=complex).reshape(2)


class _DerivedForStrength:
    """
    Minimal derived view required by QDDriveStrengthModel.

    Provides:
      - t_registry.directed(pair) -> (fwd, bwd)
      - mu_Cm(tr) -> float
      - drive_projection(tr, E) -> complex
      - polaron_B(tr) -> float
    """

    def __init__(
        self,
        *,
        mu_Cm: float,
        polaron_B_value: float = 1.0,
    ):
        self.t_registry = DEFAULT_TRANSITION_REGISTRY
        self._mu_Cm = float(mu_Cm)
        self._B = float(polaron_B_value)

    def mu_Cm(self, _tr: Transition) -> float:
        return float(self._mu_Cm)

    def drive_projection(self, tr: Transition, E: np.ndarray) -> complex:
        # Simple: G_X1 dipole aligned with H, G_X2 aligned with V.
        E = np.asarray(E, dtype=complex).reshape(2)
        if tr is Transition.G_X1:
            mu_hat = np.asarray([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex)
        elif tr is Transition.G_X2:
            mu_hat = np.asarray([0.0 + 0.0j, 1.0 + 0.0j], dtype=complex)
        else:
            mu_hat = np.asarray([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex)
        return complex(np.vdot(mu_hat, E))

    def polaron_B(self, _tr: Transition) -> float:
        return float(self._B)


class TestQDDriveStrengthModel(unittest.TestCase):
    def _ctx(self, derived: Any) -> QDDriveDecodeContext:
        # Note: with_solver_grid isn't required by strength model, only by decoder/emitter,
        # but it's harmless to include.
        return QDDriveDecodeContext(derived=derived).with_solver_grid(
            tlist=np.linspace(0.0, 10.0, 11),
            time_unit_s=1.0e-12,
        )

    def _resolved(
        self, drive_id: Any, pair: TransitionPair
    ) -> Sequence[ResolvedDrive]:
        return (
            ResolvedDrive(
                drive_id=drive_id,
                transition_key=pair,
                carrier_omega_rad_s=0.0,
                t_s=None,
                envelope=None,
                meta={},
            ),
        )

    def test_requires_qd_context_type(self) -> None:
        """Strength model should reject missing/wrong decode_ctx type."""
        model = QDDriveStrengthModel()
        resolved = self._resolved("d0", TransitionPair.G_X1)
        tlist = np.array([0.0, 1.0], dtype=float)

        with self.assertRaises(TypeError):
            _ = model.compute(
                resolved,
                tlist,
                time_unit_s=1.0,
                decode_ctx=None,
            )

    def test_uses_payload_from_ctx_meta_drives(self) -> None:
        """Strength model should read payload from decode_ctx.meta_drives[drive_id]."""
        derived = _DerivedForStrength(mu_Cm=1e-29)
        ctx = self._ctx(derived)
        payload = _FakeDrivePayload(E0=123.0)
        ctx.meta_drives["d_payload"] = payload

        model = QDDriveStrengthModel()
        resolved = self._resolved("d_payload", TransitionPair.G_X1)
        tlist = np.array([0.0, 1.0, 2.0], dtype=float)

        out = model.compute(
            resolved, tlist, time_unit_s=1.0e-12, decode_ctx=ctx
        )
        self.assertIn(("d_payload", TransitionPair.G_X1), out.coeffs)

    def test_omega_solver_basic_real(self) -> None:
        """
        Omega_solver(t) should be:
          (mu_Cm * E(t) / hbar) * time_unit_s
        when no effective_pol is provided.
        """
        mu_Cm = 2.0e-29
        E0 = 10.0
        time_unit_s = 5.0e-12

        derived = _DerivedForStrength(mu_Cm=mu_Cm, polaron_B_value=1.0)
        ctx = self._ctx(derived)
        ctx.meta_drives["d0"] = _FakeDrivePayload(E0=E0, pol=None)

        model = QDDriveStrengthModel()
        tlist = np.linspace(0.0, 2.0, 5)
        resolved = self._resolved("d0", TransitionPair.G_X1)

        out = model.compute(
            resolved, tlist, time_unit_s=time_unit_s, decode_ctx=ctx
        )
        omega = out.coeffs[("d0", TransitionPair.G_X1)]

        hbar_Js = float(magnitude(hbar, "J*s"))
        expected_val = (mu_Cm * E0 / hbar_Js) * float(time_unit_s)

        expected = np.full(tlist.shape, expected_val, dtype=complex)
        np.testing.assert_allclose(omega, expected, rtol=0.0, atol=0.0)

    def test_omega_solver_includes_polarization_overlap(self) -> None:
        """
        If payload.effective_pol is present, Omega_solver should be multiplied by
        pol = drive_projection(fwd, pol_vec).
        """
        mu_Cm = 1.0e-29
        E0 = 3.0
        time_unit_s = 1.0e-12

        derived = _DerivedForStrength(mu_Cm=mu_Cm, polaron_B_value=1.0)
        ctx = self._ctx(derived)

        # For G_X1, projection is vdot([1,0], E) = E[0]
        pol_vec = np.array([0.25 + 0.5j, 0.0 + 0.0j], dtype=complex)
        ctx.meta_drives["d_pol"] = _FakeDrivePayload(E0=E0, pol=pol_vec)

        model = QDDriveStrengthModel()
        tlist = np.array([0.0, 1.0, 2.0], dtype=float)
        resolved = self._resolved("d_pol", TransitionPair.G_X1)

        out = model.compute(
            resolved, tlist, time_unit_s=time_unit_s, decode_ctx=ctx
        )
        omega = out.coeffs[("d_pol", TransitionPair.G_X1)]

        hbar_Js = float(magnitude(hbar, "J*s"))
        base = (mu_Cm * E0 / hbar_Js) * float(time_unit_s)
        expected = np.full(tlist.shape, base, dtype=complex) * complex(
            pol_vec[0]
        )

        np.testing.assert_allclose(omega, expected, rtol=0.0, atol=0.0)

    def test_polaron_B_scales_omega(self) -> None:
        """Polaron B should scale Omega_solver multiplicatively."""
        mu_Cm = 1.0e-29
        E0 = 2.0
        time_unit_s = 2.0e-12
        B = 0.7

        derived = _DerivedForStrength(mu_Cm=mu_Cm, polaron_B_value=B)
        ctx = self._ctx(derived)
        ctx.meta_drives["dB"] = _FakeDrivePayload(E0=E0, pol=None)

        model = QDDriveStrengthModel()
        tlist = np.array([0.0, 1.0], dtype=float)
        resolved = self._resolved("dB", TransitionPair.G_X1)

        out = model.compute(
            resolved, tlist, time_unit_s=time_unit_s, decode_ctx=ctx
        )
        omega = out.coeffs[("dB", TransitionPair.G_X1)]

        hbar_Js = float(magnitude(hbar, "J*s"))
        base = (mu_Cm * E0 / hbar_Js) * float(time_unit_s)
        expected = np.full(tlist.shape, base * B, dtype=complex)

        np.testing.assert_allclose(omega, expected, rtol=0.0, atol=0.0)

    def test_missing_payload_raises(self) -> None:
        """If ctx.meta_drives lacks drive_id, strength model should raise KeyError."""
        derived = _DerivedForStrength(mu_Cm=1e-29)
        ctx = self._ctx(derived)

        model = QDDriveStrengthModel()
        resolved = self._resolved("missing", TransitionPair.G_X1)
        tlist = np.array([0.0], dtype=float)

        with self.assertRaises(KeyError):
            _ = model.compute(
                resolved, tlist, time_unit_s=1.0e-12, decode_ctx=ctx
            )


if __name__ == "__main__":
    unittest.main()
