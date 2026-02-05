from __future__ import annotations

import unittest
from typing import Any

import numpy as np
from smef.core.ir.ops import OpExprKind
from smef.core.ir.terms import TermKind

from bec.quantum_dot.enums import QDState, Transition, TransitionPair
from bec.quantum_dot.smef.drives.emitter.detuning_term import (
    build_detuning_h_term,
)
from bec.quantum_dot.smef.drives.emitter.drive_term import build_drive_h_term
from bec.quantum_dot.smef.drives.emitter.eid_terms import (
    build_eid_c_term_phenom,
)


class TestEmitterComponents(unittest.TestCase):
    def test_build_drive_term_coeff_and_shape(self) -> None:
        """
        Tests that the drive term:
          - is TermKind.H
          - uses ArrayCoeff with correct values
          - builds a SUM of two embedded transition operators with 0.5 prefactor
        """
        qd_index = 0
        omega = np.array([1.0 + 0.0j, 2.0 + 1.0j], dtype=complex)
        meta: dict[str, Any] = {"kind": "1ph"}

        term = build_drive_h_term(
            qd_index=qd_index,
            drive_id="d0",
            pair=TransitionPair.G_X1,
            fwd=Transition.G_X1,
            bwd=Transition.X1_G,
            omega_solver=omega,
            meta=meta,
        )

        self.assertEqual(term.kind, TermKind.H)
        y = term.coeff.eval(np.array([0.0, 1.0], dtype=float))
        np.testing.assert_allclose(y, omega)

        self.assertEqual(term.op.kind, OpExprKind.SCALE)
        self.assertAlmostEqual(float(term.op.scalar.real), 0.5)
        self.assertEqual(term.op.args[0].kind, OpExprKind.SUM)
        self.assertEqual(len(term.op.args[0].args), 2)

    def test_build_detuning_term_coeff(self) -> None:
        """
        Tests that detuning term:
          coeff_solver = -(0.5 * detuning_rad_s) * time_unit_s
        and TermKind.H.
        """
        qd_index = 0
        det = np.array([10.0, -20.0], dtype=float)
        time_unit_s = 2.0e-12

        term = build_detuning_h_term(
            qd_index=qd_index,
            drive_id="d1",
            pair=TransitionPair.G_X2,
            src=QDState.G,
            dst=QDState.X2,
            detuning_rad_s=det,
            time_unit_s=time_unit_s,
            meta={"kind": "1ph"},
        )

        self.assertEqual(term.kind, TermKind.H)

        expected = (-(0.5 * det) * time_unit_s).astype(complex)
        y = term.coeff.eval(np.array([0.0, 1.0], dtype=float))
        np.testing.assert_allclose(y, expected)

    def test_build_eid_term_none_when_disabled(self) -> None:
        """
        Tests that EID returns None when scale <= 0.
        """
        t = build_eid_c_term_phenom(
            qd_index=0,
            drive_id="d2",
            pair=TransitionPair.G_X1,
            dst_proj_state=QDState.X1,
            omega_solver=np.array([1.0 + 0.0j], dtype=complex),
            eid_scale=0.0,
            meta={},
        )
        self.assertIsNone(t)

    def test_build_eid_term_coeff(self) -> None:
        """
        Tests that EID term:
          gamma = scale * |Omega|^2
          coeff = sqrt(gamma)
        and TermKind.C.
        """
        omega = np.array([1.0 + 0.0j, 0.0 + 2.0j], dtype=complex)
        scale = 3.0

        term = build_eid_c_term_phenom(
            qd_index=0,
            drive_id="d3",
            pair=TransitionPair.G_X1,
            dst_proj_state=QDState.X1,
            omega_solver=omega,
            eid_scale=scale,
            meta={"kind": "1ph"},
        )

        self.assertIsNotNone(term)
        assert term is not None
        self.assertEqual(term.kind, TermKind.C)

        gamma = scale * (np.abs(omega) ** 2)
        expected = np.sqrt(gamma).astype(complex)
        y = term.coeff.eval(np.array([0.0, 1.0], dtype=float))
        np.testing.assert_allclose(y, expected)


if __name__ == "__main__":
    unittest.main()
