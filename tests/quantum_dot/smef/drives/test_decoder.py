from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from smef.core.drives.types import DriveSpec
from bec.quantum_dot.enums import Transition, TransitionPair
from bec.quantum_dot.transitions import DEFAULT_TRANSITION_REGISTRY

from bec.quantum_dot.smef.drives.context import (
    DecodePolicy,
    QDDriveDecodeContext,
)
from bec.quantum_dot.smef.drives.decoder import QDDriveDecoder


@dataclass
class _FakeDrivePayload:
    """
    Minimal drive payload used by QDDriveDecoder.

    - omega_L_rad_s(t) is required by the decoder for scoring.
    - effective_pol() is optional; when provided it affects polarization penalty.
    - preferred_kind is optional; when set to "1ph" or "2ph" it filters targets.
    """

    omega0_rad_s: float
    pol: Optional[np.ndarray] = None
    preferred_kind: Optional[str] = None

    def omega_L_rad_s(self, _t_phys_s: float) -> float:
        return float(self.omega0_rad_s)

    def effective_pol(self) -> Optional[np.ndarray]:
        if self.pol is None:
            return None
        return np.asarray(self.pol, dtype=complex).reshape(2)


class _DerivedStub:
    """
    Minimal derived-like object for decoder tests.

    This stub provides only what the decoder uses:
      - t_registry.directed(pair) to get fwd transition
      - drive_targets(), drive_kind(pair), omega_ref_rad_s(pair)
      - drive_projection(fwd_transition, E) for polarization penalty
    """

    def __init__(self, *, omega_ref_by_pair: dict[TransitionPair, float]):
        self.t_registry = DEFAULT_TRANSITION_REGISTRY
        self._omega_ref_by_pair = dict(omega_ref_by_pair)

    def drive_targets(self) -> Sequence[TransitionPair]:
        # Keep stable order for deterministic tests
        return tuple(self._omega_ref_by_pair.keys())

    def drive_kind(self, pair: TransitionPair) -> str:
        spec = self.t_registry.spec(pair)
        return "2ph" if int(spec.order) == 2 else "1ph"

    def omega_ref_rad_s(self, pair: TransitionPair) -> float:
        return float(self._omega_ref_by_pair[pair])

    def drive_projection(self, tr: Transition, E: np.ndarray) -> complex:
        # Simple polarization selectivity:
        # - Treat G_X1 as strongly coupled to H
        # - Treat G_X2 as weakly coupled to H
        E = np.asarray(E, dtype=complex).reshape(2)
        H = np.asarray([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex)

        overlap = complex(np.vdot(H, E))
        if tr is Transition.G_X1:
            return overlap
        if tr is Transition.G_X2:
            return 0.1 * overlap
        return overlap


class TestQDDriveDecodeContext(unittest.TestCase):
    def test_with_solver_grid_sets_fields(self) -> None:
        """with_solver_grid should attach solver grid and time unit to context."""
        derived = _DerivedStub(omega_ref_by_pair={TransitionPair.G_X1: 1.0e12})
        ctx = QDDriveDecodeContext(derived=derived)
        self.assertIsNone(ctx.tlist_solver)
        self.assertIsNone(ctx.time_unit_s)

        tlist = np.linspace(0.0, 10.0, 11)
        ctx2 = ctx.with_solver_grid(tlist=tlist, time_unit_s=2.5e-12)

        self.assertIsNot(ctx, ctx2)
        self.assertEqual(ctx2.tlist_solver.shape, (11,))
        self.assertAlmostEqual(float(ctx2.time_unit_s), 2.5e-12)

        np.testing.assert_allclose(ctx2.tlist_solver, tlist.astype(float))


class TestQDDriveDecoder(unittest.TestCase):
    def test_decode_requires_solver_grid(self) -> None:
        """Decoder should raise if context lacks solver grid/time unit."""
        derived = _DerivedStub(omega_ref_by_pair={TransitionPair.G_X1: 1.0e12})
        ctx = QDDriveDecodeContext(derived=derived)

        payload = _FakeDrivePayload(
            omega0_rad_s=1.0e12,
            pol=np.array([1.0 + 0.0j, 0.0 + 0.0j]),
        )
        spec = DriveSpec(drive_id="d0", payload=payload)

        dec = QDDriveDecoder()
        with self.assertRaises(ValueError):
            _ = dec.decode([spec], ctx=ctx)

    def test_decode_stores_payload_in_context(self) -> None:
        """Decoder should store payload under ctx.meta_drives[drive_id]."""
        derived = _DerivedStub(omega_ref_by_pair={TransitionPair.G_X1: 1.0e12})
        ctx = QDDriveDecodeContext(derived=derived).with_solver_grid(
            tlist=np.linspace(0.0, 10.0, 11),
            time_unit_s=1.0e-12,
        )

        payload = _FakeDrivePayload(omega0_rad_s=1.0e12)
        spec = DriveSpec(drive_id="d_store", payload=payload)

        dec = QDDriveDecoder()
        _ = dec.decode([spec], ctx=ctx)

        self.assertIn("d_store", ctx.meta_drives)
        self.assertIs(ctx.meta_drives["d_store"], payload)

    def test_decode_selects_best_pair_by_detuning_and_pol_penalty(self) -> None:
        """
        Decoder should pick the transition pair with lowest penalized score.

        Here:
        - omega_L is exactly resonant with G_X2 (detuning=0),
          but G_X2 is weakly coupled to H polarization (penalized).
        - omega_L is slightly detuned from G_X1, but strongly coupled to H.
        We choose parameters so G_X1 wins after penalty.
        """
        omega_ref = {
            TransitionPair.G_X1: 1.01e12,  # slight detuning
            TransitionPair.G_X2: 1.00e12,  # perfect detuning
        }
        derived = _DerivedStub(omega_ref_by_pair=omega_ref)

        policy = DecodePolicy(
            allow_multi=False,
            pol_gate_eps=1e-6,
            pol_penalty_power=1.0,
            pol_penalty_weight=10.0,  # make penalty matter
        )
        ctx = QDDriveDecodeContext(
            derived=derived, policy=policy
        ).with_solver_grid(
            tlist=np.linspace(0.0, 10.0, 101),
            time_unit_s=1.0e-12,
        )

        payload = _FakeDrivePayload(
            omega0_rad_s=1.00e12,
            pol=np.array([1.0 + 0.0j, 0.0 + 0.0j]),
        )
        spec = DriveSpec(drive_id="d1", payload=payload)

        dec = QDDriveDecoder()
        out = dec.decode([spec], ctx=ctx)

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].drive_id, "d1")
        self.assertEqual(out[0].transition_key, TransitionPair.G_X1)

        self.assertIn("coupling_mag", out[0].meta)
        self.assertIn("detuning_score_rad_s", out[0].meta)
        self.assertIn("score", out[0].meta)

    def test_decode_respects_preferred_kind_filter(self) -> None:
        """
        If payload.preferred_kind is set, decoder should prefer matching kinds
        when possible.
        """
        omega_ref = {
            TransitionPair.G_X1: 1.0e12,  # 1ph
            # 2ph family (order=2 in default registry)
            TransitionPair.G_XX: 2.0e12,
        }
        derived = _DerivedStub(omega_ref_by_pair=omega_ref)

        ctx = QDDriveDecodeContext(derived=derived).with_solver_grid(
            tlist=np.linspace(0.0, 10.0, 51),
            time_unit_s=1.0e-12,
        )

        payload = _FakeDrivePayload(
            omega0_rad_s=1.0e12,
            preferred_kind="2ph",
        )
        spec = DriveSpec(drive_id="d_kind", payload=payload)

        dec = QDDriveDecoder()
        out = dec.decode([spec], ctx=ctx)

        # Because preferred_kind="2ph", target set should include G_XX (2ph).
        # Even if the score is worse, filtering happens first and should keep 2ph
        # candidates if any exist.
        self.assertGreaterEqual(len(out), 1)
        for rd in out:
            self.assertEqual(rd.meta.get("kind"), "2ph")

    def test_decode_allow_multi_can_return_multiple_targets(self) -> None:
        """
        When policy.allow_multi=True, decoder can return multiple pairs whose
        score is within best + k_bandwidth*sigma_omega.
        """
        omega_ref = {
            TransitionPair.G_X1: 1.00e12,
            TransitionPair.G_X2: 1.00e12 + 1.0e8,  # very close
        }
        derived = _DerivedStub(omega_ref_by_pair=omega_ref)

        policy = DecodePolicy(
            allow_multi=True,
            k_bandwidth=100.0,  # permissive
            sample_points=3,
            pol_penalty_weight=0.0,  # ignore polarization so both are close
        )
        ctx = QDDriveDecodeContext(
            derived=derived, policy=policy
        ).with_solver_grid(
            tlist=np.linspace(0.0, 10.0, 101),
            time_unit_s=1.0e-12,
        )

        payload = _FakeDrivePayload(omega0_rad_s=1.00e12)
        spec = DriveSpec(drive_id="d_multi", payload=payload)

        dec = QDDriveDecoder()
        out = dec.decode([spec], ctx=ctx)

        self.assertEqual(
            {rd.transition_key for rd in out},
            {TransitionPair.G_X1, TransitionPair.G_X2},
        )


if __name__ == "__main__":
    unittest.main()
