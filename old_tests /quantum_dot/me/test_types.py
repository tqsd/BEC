import unittest
import numpy as np
from qutip import Qobj

from bec.quantum_dot.me.types import (
    HamiltonianTerm,
    HamiltonianTermKind,
    CollapseTerm,
    CollapseTermKind,
    h_terms_to_qutip,
    c_terms_to_qutip,
)
from bec.quantum_dot.me.coeffs import ConstCoeff, FuncCoeff


class TestTypes(unittest.TestCase):
    def setUp(self):
        # simple 2x2 operators
        self.A = Qobj(np.array([[0, 1], [1, 0]], dtype=complex))  # sigma_x
        self.B = Qobj(np.array([[1, 0], [0, -1]], dtype=complex))  # sigma_z

    def test_h_terms_to_qutip_static(self):
        terms = [
            HamiltonianTerm(
                kind=HamiltonianTermKind.STATIC,
                op=self.A,
                coeff=None,
                label="A",
            ),
        ]
        out = h_terms_to_qutip(terms)
        self.assertEqual(len(out), 1)
        self.assertTrue(isinstance(out[0], Qobj))

    def test_h_terms_to_qutip_time_dep(self):
        coeff = FuncCoeff(lambda t, args=None: 2.0 * t)
        terms = [
            HamiltonianTerm(
                kind=HamiltonianTermKind.DRIVE,
                op=self.A,
                coeff=coeff,
                label="A(t)",
            ),
        ]
        out = h_terms_to_qutip(terms)
        self.assertEqual(len(out), 1)
        self.assertTrue(isinstance(out[0], list))
        self.assertEqual(out[0][0], self.A)
        self.assertTrue(callable(out[0][1]))

    def test_c_terms_to_qutip_const_coeff_optimized(self):
        c = CollapseTerm(
            kind=CollapseTermKind.RADIATIVE,
            op=self.B,
            coeff=ConstCoeff(3.0),
            label="3B",
        )
        out = c_terms_to_qutip([c])
        self.assertEqual(len(out), 1)
        self.assertTrue(isinstance(out[0], Qobj))
        # should be multiplied into operator
        diff = (out[0] - (3.0 * self.B)).norm()
        self.assertLess(diff, 1e-12)

    def test_c_terms_to_qutip_func_coeff_list_form(self):
        coeff = FuncCoeff(lambda t, args=None: 1.0 + t)
        c = CollapseTerm(
            kind=CollapseTermKind.PHENOMENOLOGICAL,
            op=self.B,
            coeff=coeff,
            label="B(t)",
        )
        out = c_terms_to_qutip([c])
        self.assertEqual(len(out), 1)
        self.assertTrue(isinstance(out[0], list))
        self.assertEqual(out[0][0], self.B)
        self.assertTrue(callable(out[0][1]))


if __name__ == "__main__":
    unittest.main()
