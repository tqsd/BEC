import unittest
import numpy as np

from bec.operators.fock_operators import (
    Pol,
    Ladder,
    ladder_operator,
    rotated_ladder_operator,
)


def top_projector(d: int) -> np.ndarray:
    """|d-1><d-1| in a d-dim space."""
    e = np.zeros((d, 1))
    e[-1, 0] = 1.0
    return e @ e.T.conj()


class RotatedLadderTests(unittest.TestCase):

    def angles(self):
        return [(0.0, 0.0), (0.3, 0.1), (1.1, 2.0)]

    def test_single_mode_ladder_basics(self):
        """a† is adjoint of a; truncated commutator = I - d |d-1><d-1|."""
        for d in (2, 3, 5):
            a = ladder_operator(d, Ladder.A)
            adag = ladder_operator(d, Ladder.A_DAG)
            np.testing.assert_allclose(adag, a.conj().T, atol=1e-12, rtol=1e-12)

            I = np.eye(d)
            Ptop = top_projector(d)
            expected = I - d * Ptop

            comm = a @ adag - adag @ a
            np.testing.assert_allclose(comm, expected, atol=1e-12, rtol=1e-12)

    def test_rotated_commutation_relation_truncated(self):
        for d in (2, 3, 4):
            I = np.eye(d, dtype=complex)
            a = ladder_operator(d, Ladder.A)

            # Base H/V ops in H⊗V
            aH = np.kron(a, I)
            aHdag = aH.conj().T
            aV = np.kron(I, a)
            aVdag = aV.conj().T

            # Truncated-space commutators for H and V
            comm_H = aH @ aHdag - aHdag @ aH
            comm_V = aV @ aVdag - aVdag @ aV

            for pol in (Pol.PLUS, Pol.MINUS):
                for theta, phi in self.angles():
                    c, s = np.cos(theta), np.sin(theta)

                    # Rotated operators
                    a_rot = rotated_ladder_operator(
                        d, theta, phi, pol, Ladder.A
                    )
                    a_rotdag = rotated_ladder_operator(
                        d, theta, phi, pol, Ladder.A_DAG
                    )
                    comm = a_rot @ a_rotdag - a_rotdag @ a_rot

                    # Expected linear combination (cross-terms vanish)
                    if pol is Pol.PLUS:
                        expected = (c**2) * comm_H + (s**2) * comm_V
                    else:  # Pol.MINUS
                        expected = (s**2) * comm_H + (c**2) * comm_V

                    np.testing.assert_allclose(
                        comm, expected, atol=1e-11, rtol=1e-11
                    )

                    # (Optional) explicitly check cross-terms vanish numerically
                    cross = (
                        c
                        * s
                        * (aH @ aVdag - aVdag @ aH + aV @ aHdag - aHdag @ aV)
                    )
                    np.testing.assert_allclose(cross, 0, atol=1e-12, rtol=1e-12)

    def test_invalid_operator_rejected(self):
        with self.assertRaises(ValueError):
            _ = rotated_ladder_operator(3, 0.2, 0.1, Pol.PLUS, Ladder.N)


if __name__ == "__main__":
    unittest.main()
