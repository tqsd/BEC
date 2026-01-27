import unittest
from types import SimpleNamespace
import numpy as np
from qutip import Qobj, basis, tensor

from unittest.mock import patch

from bec.quantum_dot.metrics.decompose import PopulationDecomposer


def proj_on_factor(dims, factor, n):
    """
    Projector onto |n> on a given factor, tensored with identity on the others.
    """
    D = int(np.prod(dims))
    # Build |n><n| on the factor and Identity elsewhere via tensor
    proj_kets = []
    for i, d in enumerate(dims):
        if i == factor:
            ket = basis(d, n)
            proj_kets.append(ket * ket.dag())
        else:
            proj_kets.append(Qobj(np.eye(d), dims=[[d], [d]]))
    P = tensor(proj_kets)
    return P.to("csr")


def vacuum_projector(dims):
    """
    Projector onto the global vacuum |0,...,0>.
    """
    kets = [basis(d, 0) for d in dims]
    ket0 = tensor(kets)
    return (ket0 * ket0.dag()).to("csr")


def one_one_projector(dims):
    """
    For two factors with dims [2,2], projector onto |1,1>.
    """
    assert len(dims) == 2 and dims[0] == 2 and dims[1] == 2
    ket11 = tensor([basis(2, 1), basis(2, 1)])
    return (ket11 * ket11.dag()).to("csr")


class FakePhotonicRegistry:
    """
    Minimal registry supplying fields used by PopulationDecomposer.
    """

    def __init__(self, dims):
        self.dims_phot = dims[:]
        D = int(np.prod(dims))
        self.I_phot = Qobj(np.eye(D), dims=[dims, dims]).to("csr")
        # Per-factor P0 and P1 (factor-local projectors extended to full space)
        self.proj0_by_factor = {
            i: proj_on_factor(dims, i, 0) for i in range(len(dims))
        }
        self.proj1_by_factor = {
            i: proj_on_factor(dims, i, 1) for i in range(len(dims))
        }


class PopulationDecomposerTests(unittest.TestCase):
    def setUp(self):
        self.dims = [2, 2]  # two photonic factors truncated to {0,1}
        self.reg = FakePhotonicRegistry(self.dims)
        self.dec = PopulationDecomposer(self.reg)

    def test_Pi0_all_is_global_vacuum_projector(self):
        P0_all = self.dec._Pi0_all()
        # Compare with explicit vacuum projector
        P0_expected = vacuum_projector(self.dims)
        np.testing.assert_allclose(
            P0_all.full(), P0_expected.full(), atol=1e-12, rtol=1e-12
        )

    def test_Pi1_total_picks_exactly_one_photon_anywhere(self):
        P1 = self.dec._Pi1_total()

        # Expectation on |10>
        ket10 = tensor([basis(2, 1), basis(2, 0)])
        rho10 = ket10 * ket10.dag()
        val10 = float((P1 * rho10).tr().real)
        self.assertAlmostEqual(val10, 1.0, places=12)

        # Expectation on |01>
        ket01 = tensor([basis(2, 0), basis(2, 1)])
        rho01 = ket01 * ket01.dag()
        val01 = float((P1 * rho01).tr().real)
        self.assertAlmostEqual(val01, 1.0, places=12)

        # On |00> and |11> it should be zero
        ket00 = tensor([basis(2, 0), basis(2, 0)])
        rho00 = ket00 * ket00.dag()
        self.assertAlmostEqual(float((P1 * rho00).tr().real), 0.0, places=12)

        ket11 = tensor([basis(2, 1), basis(2, 1)])
        rho11 = ket11 * ket11.dag()
        self.assertAlmostEqual(float((P1 * rho11).tr().real), 0.0, places=12)

    @patch("bec.quantum_dot.metrics.decompose.ensure_rho")
    def test_p0_p1_p2_exact_multi_on_known_mixture(self, ensure_rho_mock):
        """
        Build rho = 0.3|00><00| + 0.2|10><10| + 0.2|01><01| + 0.3|11><11|.
        Expect:
          p0 = 0.3
          p1_total = 0.4
          p2_exact = 0.3
          p_multiphoton = 0.0
        """
        ket00 = tensor([basis(2, 0), basis(2, 0)])
        ket10 = tensor([basis(2, 1), basis(2, 0)])
        ket01 = tensor([basis(2, 0), basis(2, 1)])
        ket11 = tensor([basis(2, 1), basis(2, 1)])

        rho = (
            0.3 * (ket00 * ket00.dag())
            + 0.2 * (ket10 * ket10.dag())
            + 0.2 * (ket01 * ket01.dag())
            + 0.3 * (ket11 * ket11.dag())
        ).full()

        # Bypass external normalization details; return the same rho
        ensure_rho_mock.return_value = rho

        P2 = one_one_projector(self.dims)  # |11><11|
        out = self.dec.p0_p1_p2_exact_multi(rho, P2)

        self.assertAlmostEqual(out["p0"], 0.3, places=12)
        self.assertAlmostEqual(out["p1_total"], 0.4, places=12)
        self.assertAlmostEqual(out["p2_exact"], 0.3, places=12)
        self.assertAlmostEqual(out["p_multiphoton"], 0.0, places=12)

    @patch("bec.quantum_dot.metrics.decompose.ensure_rho")
    def test_probabilities_sum_to_one_within_tolerance(self, ensure_rho_mock):
        """
        For a normalized state supported entirely on {vac, 1-photon, 2-photon},
        p0 + p1_total + p2_exact + p_multiphoton should be 1 up to numeric error.
        """
        # Use a random diagonal state over the computational basis with sum 1
        rng = np.random.default_rng(123)
        w = rng.random(4)
        w = w / w.sum()

        basis_kets = [
            tensor([basis(2, 0), basis(2, 0)]),
            tensor([basis(2, 1), basis(2, 0)]),
            tensor([basis(2, 0), basis(2, 1)]),
            tensor([basis(2, 1), basis(2, 1)]),
        ]
        rho = sum(
            float(wi) * (k * k.dag()) for wi, k in zip(w, basis_kets)
        ).full()
        ensure_rho_mock.return_value = rho

        P2 = one_one_projector(self.dims)
        out = self.dec.p0_p1_p2_exact_multi(rho, P2)
        total = (
            out["p0"] + out["p1_total"] + out["p2_exact"] + out["p_multiphoton"]
        )
        self.assertAlmostEqual(total, 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
