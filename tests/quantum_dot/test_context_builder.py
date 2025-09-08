import unittest
import numpy as np

from bec.quantum_dot.context_builder import QDContextBuilder
from types import SimpleNamespace


class FakeModeProvider:
    def __init__(self, n_modes: int):
        # Only the length matters for QDContextBuilder; it iterates over .modes
        self.modes = [SimpleNamespace(label=f"m{i}") for i in range(n_modes)]


class TestQDContextBuilder(unittest.TestCase):
    def setUp(self):
        # Two light modes; mode 0 truncated to {0,1} per pol; mode 1 to {0,1,2}
        self.mode_provider = FakeModeProvider(n_modes=2)
        # dims layout: [QD=4, m0(+), m0(-), m1(+), m1(-)]
        self.dims = [4, 2, 2, 3, 3]

    def test_build_provides_expected_keys_and_shapes(self):
        builder = QDContextBuilder(self.mode_provider, theta=0.3, phi=1.1)
        ctx = builder.build()

        # --- QD operator keys exist ---
        qd_keys = {
            "s_XX_G",
            "s_XX_X1",
            "s_XX_X2",
            "s_X1_G",
            "s_X2_G",
            "s_G_X1",
            "s_G_X2",
            "s_G_XX",
            "s_X1_XX",
            "s_X2_XX",
            "s_X1_X1",
            "s_X1_X2",
            "s_X2_X1",
            "s_X2_X2",
            "s_XX_XX",
            "idq",
        }
        self.assertTrue(qd_keys.issubset(set(ctx.keys())))

        # QD pieces are 4x4
        for k in qd_keys:
            arr = np.array(ctx[k]([]))  # these ignore the dims arg
            self.assertEqual(arr.shape, (4, 4))

        # --- Per-mode operator keys exist ---
        # For 2 modes, we expect a0+/-, a1+/- etc
        for i, (d_plus, d_minus) in enumerate(
            [(self.dims[1], self.dims[2]), (self.dims[3], self.dims[4])]
        ):
            base = {
                f"a{i}+",
                f"a{i}+_dag",
                f"a{i}-",
                f"a{i}-_dag",
                f"n{i}+",
                f"n{i}-",
                f"if{i}",
                f"vac{i}",
            }
            self.assertTrue(base.issubset(set(ctx.keys())))

            # Shapes: rotated ladders/vacuum live on the 2-pol subspace of that mode.
            # Our builder uses a single 'dim' for both pols; ensure dims[+]==dims[-].
            self.assertEqual(d_plus, d_minus)
            subdim = d_plus * d_minus  # expected two-pol subspace dimension

            Aplus = np.array(ctx[f"a{i}+"](self.dims))
            Aplus_dag = np.array(ctx[f"a{i}+_dag"](self.dims))
            Aminus = np.array(ctx[f"a{i}-"](self.dims))
            Aminus_dag = np.array(ctx[f"a{i}-_dag"](self.dims))
            Nplus = np.array(ctx[f"n{i}+"](self.dims))
            Nminus = np.array(ctx[f"n{i}-"](self.dims))
            Ifull = np.array(ctx[f"if{i}"](self.dims))
            Pvac = np.array(ctx[f"vac{i}"](self.dims))

            for M in (
                Aplus,
                Aplus_dag,
                Aminus,
                Aminus_dag,
                Nplus,
                Nminus,
                Pvac,
            ):
                self.assertEqual(M.shape, (subdim, subdim))
            self.assertEqual(Ifull.shape, (subdim, subdim))

            # a† is the adjoint of a
            np.testing.assert_allclose(
                Aplus_dag, Aplus.conj().T, atol=1e-12, rtol=1e-12
            )
            np.testing.assert_allclose(
                Aminus_dag, Aminus.conj().T, atol=1e-12, rtol=1e-12
            )

            # number operators are a†a
            np.testing.assert_allclose(
                Nplus, Aplus_dag @ Aplus, atol=1e-12, rtol=1e-12
            )
            np.testing.assert_allclose(
                Nminus, Aminus_dag @ Aminus, atol=1e-12, rtol=1e-12
            )

            # identity and vacuum are Hermitian
            np.testing.assert_allclose(
                Ifull, Ifull.conj().T, atol=1e-12, rtol=1e-12
            )
            np.testing.assert_allclose(
                Pvac, Pvac.conj().T, atol=1e-12, rtol=1e-12
            )

    def test_theta_phi_affect_rotated_operators(self):
        # Same dims/modes, different (theta, phi) should change ladder matrices
        ctx1 = QDContextBuilder(self.mode_provider, theta=0.0, phi=0.0).build()
        ctx2 = QDContextBuilder(self.mode_provider, theta=0.6, phi=1.0).build()

        A1 = np.array(ctx1["a0+"](self.dims))
        A2 = np.array(ctx2["a0+"](self.dims))

        # They should differ for nontrivial rotation
        self.assertFalse(np.allclose(A1, A2))

        # Shapes still OK
        self.assertEqual(A1.shape, (self.dims[1] * self.dims[2],) * 2)
        self.assertEqual(A2.shape, (self.dims[1] * self.dims[2],) * 2)


if __name__ == "__main__":
    unittest.main()
