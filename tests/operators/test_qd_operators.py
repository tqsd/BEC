import unittest
import numpy as np
from bec.operators.qd_operators import QDState, transition_operator


class TestQDOperators(unittest.TestCase):

    def test_transition_shape(self):
        """Transition operator should be a 4x4 complex matrix."""
        op = transition_operator(QDState.XX, QDState.X1)
        self.assertEqual(op.shape, (4, 4))
        self.assertEqual(op.dtype, np.complex128)

    def test_transition_operator_single_element(self):
        """Check that transition operator has a single 1.0 element at correct position."""
        op = transition_operator(QDState.XX, QDState.X1)
        expected = np.zeros((4, 4), dtype=complex)
        expected[QDState.X1.value, QDState.XX.value] = 1.0
        np.testing.assert_array_equal(op, expected)

    def test_transition_operator_identity(self):
        """Transition from a state to itself should be a projector."""
        for state in QDState:
            op = transition_operator(state, state)
            expected = np.zeros((4, 4), dtype=complex)
            expected[state.value, state.value] = 1.0
            np.testing.assert_array_equal(op, expected)

    def test_hermitian_conjugate(self):
        """Check that conjugate of transition |to><from| is |from><to|."""
        op = transition_operator(QDState.X2, QDState.XX)
        op_dag = op.conj().T
        expected = transition_operator(QDState.XX, QDState.X2)
        np.testing.assert_array_equal(op_dag, expected)

    def test_invalid_enum(self):
        """Ensure error is raised for invalid enum input."""
        with self.assertRaises(ValueError):
            _ = QDState(
                "invalid"
            )  # This will fail because QDState doesn't have 'invalid'


if __name__ == "__main__":
    unittest.main()
