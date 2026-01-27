import unittest
from types import SimpleNamespace
import numpy as np
from qutip import Qobj, basis, qeye, tensor
from unittest.mock import patch

from bec.quantum_dot.metrics.mode_registry import build_registry


def number_projector_for_factor(dims_phot, f_idx):
    """
    Return Qobj for |1><1| on factor f_idx, identity elsewhere (photonic-only).
    """
    ops = []
    for i, d in enumerate(dims_phot):
        if i == f_idx:
            ket1 = basis(d, 1)
            ops.append((ket1 * ket1.dag()).to("csr"))
        else:
            ops.append(qeye(d).to("csr"))
    return tensor(ops).to("csr")


class FakeObservableProvider:
    """
    Produces a photonic-only dictionary of number operators matching the keys used
    in build_registry: "N+[label]" and "N-[label]".
    """

    def __init__(self, dims_phot, labels, offset=0):
        self._dims_phot = list(dims_phot)
        self._labels = list(labels)
        self._offset = int(offset)

    def light_mode_projectors(self, dims_full, include_qd=False):
        # We ignore dims_full/include_qd and return photonic-only ops (as required).
        assert include_qd is False
        ops = {}
        for i, lab in enumerate(self._labels):
            f_plus = self._offset + 2 * i
            f_minus = self._offset + 2 * i + 1
            ops[f"N+[{lab}]"] = number_projector_for_factor(
                self._dims_phot, f_plus
            )
            ops[f"N-[{lab}]"] = number_projector_for_factor(
                self._dims_phot, f_minus
            )
        return ops


class FakeMode:
    def __init__(self, label):
        self.label = label


# A PhotonicRegistry echo-class that just stores provided kwargs as attributes
class EchoRegistry:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# ---------- tests ----------


class BuildRegistryTests(unittest.TestCase):
    def setUp(self):
        # Two modes, each with two pol factors => 4 photonic factors total
        self.dims_phot = [2, 2, 2, 2]
        self.offset = 0
        # Early factors are mode 0 (+,-) -> 0,1 ; Late are mode 1 (+,-) -> 2,3
        self.early = [0, 1]
        self.late = [2, 3]
        self.labels = ["m0", "m1"]

        # Fake providers
        self.mode_provider = SimpleNamespace(
            intrinsic=[FakeMode("m0"), FakeMode("m1")]
        )
        self.observable_provider = FakeObservableProvider(
            self.dims_phot, self.labels, offset=self.offset
        )

        # Dummy QD object (only used by the patched infer function)
        self.qd = SimpleNamespace()

    @patch(
        "bec.quantum_dot.metrics.mode_registry.PhotonicRegistry", EchoRegistry
    )
    @patch(
        "bec.quantum_dot.metrics.mode_registry.infer_index_sets_from_registry"
    )
    def test_basic_fields_and_labels(self, infer_mock):
        # Arrange infer function to return our controlled indices/dims
        infer_mock.return_value = (
            list(self.early),
            list(self.late),
            [],  # plus set (unused)
            [],  # minus set (unused)
            list(self.dims_phot),
            self.offset,
        )

        reg = build_registry(
            self.qd, self.mode_provider, self.observable_provider
        )

        # dims/factors passthrough
        self.assertEqual(reg.dims_phot, self.dims_phot)
        self.assertEqual(reg.early_factors, self.early)
        self.assertEqual(reg.late_factors, self.late)
        self.assertEqual(reg.offset, self.offset)

        # labels pulled from mode_provider.intrinsic
        self.assertEqual(reg.labels_by_mode_index, self.labels)

        # I_phot shape/dims
        Dp = int(np.prod(self.dims_phot))
        self.assertIsInstance(reg.I_phot, Qobj)
        self.assertEqual(reg.I_phot.shape, (Dp, Dp))
        self.assertEqual(reg.I_phot.dims, [self.dims_phot, self.dims_phot])

    @patch(
        "bec.quantum_dot.metrics.mode_registry.PhotonicRegistry", EchoRegistry
    )
    @patch(
        "bec.quantum_dot.metrics.mode_registry.infer_index_sets_from_registry"
    )
    def test_number_op_by_factor_matches_observables(self, infer_mock):
        infer_mock.return_value = (
            list(self.early),
            list(self.late),
            [],
            [],
            list(self.dims_phot),
            self.offset,
        )

        reg = build_registry(
            self.qd, self.mode_provider, self.observable_provider
        )

        # Factor -> expected key mapping (offset=0 -> even: "+", odd: "-")
        expected_map = {
            0: "N+[m0]",
            1: "N-[m0]",
            2: "N+[m1]",
            3: "N-[m1]",
        }

        # Rebuild the same ops the provider returns to assert equality
        provider_ops = self.observable_provider.light_mode_projectors(
            [4] + self.dims_phot, include_qd=False
        )

        for f_idx, key in expected_map.items():
            self.assertIn(f_idx, reg.number_op_by_factor)
            got = reg.number_op_by_factor[f_idx]
            exp = provider_ops[key]
            self.assertIsInstance(got, Qobj)
            self.assertEqual(got.dims, [self.dims_phot, self.dims_phot])
            np.testing.assert_allclose(
                got.full(), exp.full(), atol=1e-12, rtol=0
            )

    @patch(
        "bec.quantum_dot.metrics.mode_registry.PhotonicRegistry", EchoRegistry
    )
    @patch(
        "bec.quantum_dot.metrics.mode_registry.infer_index_sets_from_registry"
    )
    def test_proj0_proj1_by_factor_act_locally(self, infer_mock):
        infer_mock.return_value = (
            list(self.early),
            list(self.late),
            [],
            [],
            list(self.dims_phot),
            self.offset,
        )

        reg = build_registry(
            self.qd, self.mode_provider, self.observable_provider
        )

        # Test a few factors
        for f_idx in (0, 1, 2, 3):
            P0 = reg.proj0_by_factor[f_idx]
            P1 = reg.proj1_by_factor[f_idx]

            # State |1> on the tested factor, |0> elsewhere -> P1 expectation 1, P0 expectation 0
            occ = [0] * len(self.dims_phot)
            occ[f_idx] = 1
            ket_10 = tensor(
                [basis(d, n) for d, n in zip(self.dims_phot, occ)]
            ).to("csr")
            rho_10 = (ket_10 * ket_10.dag()).to("csr")

            v1 = float((P1 * rho_10).tr().real)
            v0 = float((P0 * rho_10).tr().real)
            self.assertAlmostEqual(v1, 1.0, places=12)
            self.assertAlmostEqual(v0, 0.0, places=12)

            # State |0> on the tested factor, |0> elsewhere -> P0 expectation 1, P1 expectation 0
            occ0 = [0] * len(self.dims_phot)
            ket_00 = tensor(
                [basis(d, n) for d, n in zip(self.dims_phot, occ0)]
            ).to("csr")
            rho_00 = (ket_00 * ket_00.dag()).to("csr")

            v0_ok = float((P0 * rho_00).tr().real)
            v1_zero = float((P1 * rho_00).tr().real)
            self.assertAlmostEqual(v0_ok, 1.0, places=12)
            self.assertAlmostEqual(v1_zero, 0.0, places=12)

    @patch(
        "bec.quantum_dot.metrics.mode_registry.PhotonicRegistry", EchoRegistry
    )
    @patch(
        "bec.quantum_dot.metrics.mode_registry.infer_index_sets_from_registry"
    )
    def test_uses_modes_when_intrinsic_missing(self, infer_mock):
        """
        If `mode_provider.intrinsic` is None, build_registry should fall back to `mode_provider.modes`.
        """
        infer_mock.return_value = (
            list(self.early),
            list(self.late),
            [],
            [],
            list(self.dims_phot),
            self.offset,
        )
        # mode_provider with intrinsic = None, modes populated
        mp = SimpleNamespace(
            intrinsic=None, modes=[FakeMode("A"), FakeMode("B")]
        )
        obs = FakeObservableProvider(
            self.dims_phot, ["A", "B"], offset=self.offset
        )

        reg = build_registry(self.qd, mp, obs)
        self.assertEqual(reg.labels_by_mode_index, ["A", "B"])


if __name__ == "__main__":
    unittest.main()
