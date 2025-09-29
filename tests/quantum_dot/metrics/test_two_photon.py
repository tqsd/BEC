import unittest
import numpy as np
from qutip import basis, tensor, qeye, Qobj

from bec.quantum_dot.metrics.two_photon import TwoPhotonProjector


def proj_on_factor(dims, f_idx, n):
    """|n><n| on factor f_idx, identity elsewhere (Qobj on full photonic space)."""
    ops = []
    for i, d in enumerate(dims):
        if i == f_idx:
            ket = basis(d, n)
            ops.append((ket * ket.dag()).to("csr"))
        else:
            ops.append(qeye(d).to("csr"))
    return tensor(ops).to("csr")


class FakePhotonicRegistry:
    def __init__(self, dims_phot, early_factors, late_factors):
        self.dims_phot = list(dims_phot)
        self.early_factors = list(early_factors)
        self.late_factors = list(late_factors)
        Dp = int(np.prod(dims_phot))
        self.Dp = Dp
        self.I_phot = Qobj(np.eye(Dp), dims=[dims_phot, dims_phot]).to("csr")
        # projectors per factor
        self.proj0_by_factor = {
            f: proj_on_factor(dims_phot, f, 0) for f in range(len(dims_phot))
        }
        self.proj1_by_factor = {
            f: proj_on_factor(dims_phot, f, 1) for f in range(len(dims_phot))
        }


def ket(dims, occ):
    return tensor([basis(d, n) for d, n in zip(dims, occ)]).to("csr")


def rho_pure(dims, occ):
    k = ket(dims, occ)
    return (k * k.dag()).full()


class TwoPhotonProjectorTests(unittest.TestCase):
    def setUp(self):
        # Two modes with +/- each -> 4 factors; early=(0,1), late=(2,3)
        self.dims = [2, 2, 2, 2]
        self.early = [0, 1]
        self.late = [2, 3]
        self.reg = FakePhotonicRegistry(self.dims, self.early, self.late)
        self.tp = TwoPhotonProjector(self.reg)

    def test_proj_exactly_one_on_early(self):
        P = self.tp._proj_exactly_one(self.early)

        # |10..> and |01..> -> expectation 1
        rho_10xx = rho_pure(self.dims, [1, 0, 0, 0])
        rho_01xx = rho_pure(self.dims, [0, 1, 0, 0])
        self.assertAlmostEqual(
            float((P * Qobj(rho_10xx, dims=[self.dims, self.dims])).tr().real),
            1.0,
            places=12,
        )
        self.assertAlmostEqual(
            float((P * Qobj(rho_01xx, dims=[self.dims, self.dims])).tr().real),
            1.0,
            places=12,
        )

        # |00..> and |11..> -> expectation 0
        rho_00xx = rho_pure(self.dims, [0, 0, 0, 0])
        rho_11xx = rho_pure(self.dims, [1, 1, 0, 0])
        self.assertAlmostEqual(
            float((P * Qobj(rho_00xx, dims=[self.dims, self.dims])).tr().real),
            0.0,
            places=12,
        )
        self.assertAlmostEqual(
            float((P * Qobj(rho_11xx, dims=[self.dims, self.dims])).tr().real),
            0.0,
            places=12,
        )

    def test_projector_requires_one_each_branch(self):
        P = self.tp.projector()

        # Valid: |1,0,1,0> and |0,1,0,1> -> expectation 1
        rho_ok1 = rho_pure(self.dims, [1, 0, 1, 0])
        rho_ok2 = rho_pure(self.dims, [0, 1, 0, 1])
        self.assertAlmostEqual(
            float((P * Qobj(rho_ok1, dims=[self.dims, self.dims])).tr().real),
            1.0,
            places=12,
        )
        self.assertAlmostEqual(
            float((P * Qobj(rho_ok2, dims=[self.dims, self.dims])).tr().real),
            1.0,
            places=12,
        )

        # Invalid: both photons in early or both in late
        rho_bad_e = rho_pure(self.dims, [1, 1, 0, 0])
        rho_bad_l = rho_pure(self.dims, [0, 0, 1, 1])
        self.assertAlmostEqual(
            float((P * Qobj(rho_bad_e, dims=[self.dims, self.dims])).tr().real),
            0.0,
            places=12,
        )
        self.assertAlmostEqual(
            float((P * Qobj(rho_bad_l, dims=[self.dims, self.dims])).tr().real),
            0.0,
            places=12,
        )

    def test_postselect_returns_normalized_state_and_prob(self):
        # rho = 0.3|1,0,1,0><...| + 0.2|0,1,0,1><...| + 0.5|0,0,0,0><...|
        rho = (
            0.3 * rho_pure(self.dims, [1, 0, 1, 0])
            + 0.2 * rho_pure(self.dims, [0, 1, 0, 1])
            + 0.5 * rho_pure(self.dims, [0, 0, 0, 0])
        )
        R2, p2 = self.tp.postselect(rho)
        self.assertAlmostEqual(p2, 0.5, places=12)

        # Expected postselected state: normalized mixture 0.6|A><A| + 0.4|B><B|
        A = rho_pure(self.dims, [1, 0, 1, 0])
        B = rho_pure(self.dims, [0, 1, 0, 1])
        R_expected = 0.6 * A + 0.4 * B
        np.testing.assert_allclose(R2, R_expected, atol=1e-12, rtol=0)

        # Trace is 1 after postselection
        self.assertAlmostEqual(float(np.trace(R2).real), 1.0, places=12)

    def test_postselect_zero_probability_returns_zero_matrix(self):
        # State with two photons both in early -> outside the target subspace
        rho_bad = rho_pure(self.dims, [1, 1, 0, 0])
        R2, p2 = self.tp.postselect(rho_bad)
        self.assertEqual(p2, 0.0)
        self.assertEqual(R2.shape, (self.reg.Dp, self.reg.Dp))
        self.assertTrue(np.allclose(R2, 0.0))

    def test_postselect_accepts_qobj_input(self):
        rho = Qobj(
            rho_pure(self.dims, [1, 0, 1, 0]), dims=[self.dims, self.dims]
        ).to("csr")
        R2, p2 = self.tp.postselect(rho)
        self.assertAlmostEqual(p2, 1.0, places=12)
        # pure state stays pure in the selected subspace
        self.assertAlmostEqual(float(np.trace(R2 @ R2).real), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
