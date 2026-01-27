import unittest
import numpy as np
from dataclasses import dataclass
from unittest.mock import patch

from qutip import Qobj

from bec.quantum_dot.me.observables_builder import ObservablesBuilder


# -------------------------
# Helpers / fakes
# -------------------------

LEVELS = ["G", "X1", "X2", "XX"]
IDX = {name: i for i, name in enumerate(LEVELS)}


def basis_ket(i: int, N: int = 4) -> np.ndarray:
    v = np.zeros((N, 1), dtype=complex)
    v[i, 0] = 1.0
    return v


def dot_op_matrix(bra: str, ket: str) -> np.ndarray:
    """|bra><ket| on the 4-dim dot space."""
    i = IDX[bra]
    j = IDX[ket]
    return basis_ket(i) @ basis_ket(j).conj().T


def kron3(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    return np.kron(np.kron(A, B), C)


@dataclass
class FakeMode:
    label: str


class FakeModeProvider:
    def __init__(self, labels):
        self.modes = [FakeMode(label=l) for l in labels]


class FakeKronPad:
    def pad(self, local_op, subsystem_label, idx):
        # dot embedding tag
        if (
            isinstance(local_op, np.ndarray)
            and subsystem_label == "i"
            and idx == -1
        ):
            return ("DOT_OP", local_op)

        # For mode ops, the "subsystem_label" argument is actually the kind ("n+", "n-", "vac", "i")
        layout = local_op  # e.g. "idq"
        kind = subsystem_label  # e.g. "n+"
        return ("MODE_OP", kind, layout, idx)


def make_fake_ctx():
    """
    ctx keys s_<BRA>_<KET> return dot operator matrices.
    Called like ctx[key]([]).
    """
    ctx = {}
    for bra in LEVELS:
        for ket in LEVELS:
            mat = dot_op_matrix(bra, ket)
            ctx[f"s_{bra}_{ket}"] = lambda _ignored, m=mat: m
    return ctx


def make_mode_ops_full(dims_full):
    """
    Build full-space matrices for a system with:
      dims_full = [4, 2, 2]  (dot, plus, minus)
    """
    d_qd, d_p, d_m = dims_full
    assert (d_qd, d_p, d_m) == (4, 2, 2)

    I_qd = np.eye(4, dtype=complex)
    I2 = np.eye(2, dtype=complex)

    # number operator on 0/1 truncated space: n = |1><1|
    n = np.array([[0, 0], [0, 1]], dtype=complex)

    # vacuum projector |0><0|
    vac = np.array([[1, 0], [0, 0]], dtype=complex)

    Np = kron3(I_qd, n, I2)
    Nm = kron3(I_qd, I2, n)
    I_full = kron3(I_qd, I2, I2)
    Pvac = kron3(I_qd, vac, vac)

    return {
        "n+": Np,
        "n-": Nm,
        "i": I_full,
        "vac": Pvac,
    }


def interpreter_stub_factory(dims_full):
    mode_ops = make_mode_ops_full(dims_full)

    def _interp(expr, ctx, dims):
        # dot embedding: op ⊗ I ⊗ I
        if isinstance(expr, tuple) and len(expr) == 2 and expr[0] == "DOT_OP":
            dot = expr[1]
            d_qd, d_p, d_m = dims_full
            I_p = np.eye(d_p, dtype=complex)
            I_m = np.eye(d_m, dtype=complex)
            return kron3(dot, I_p, I_m)

        if isinstance(expr, tuple) and len(expr) == 4 and expr[0] == "MODE_OP":
            _, kind, layout, idx = expr
            if kind not in mode_ops:
                raise KeyError(f"Unknown mode op kind '{kind}'")
            return mode_ops[kind]

        raise TypeError(f"Unexpected expr type: {type(expr)} / {expr}")

    return _interp


# -------------------------
# Tests
# -------------------------


class TestObservablesBuilder(unittest.TestCase):
    def setUp(self):
        # One photonic "mode" with two polarizations (+ and -) each truncated to 0/1 => dim=2
        self.dims_full = [4, 2, 2]
        self.dims_phot = [2, 2]

        self.ctx = make_fake_ctx()
        self.kron = FakeKronPad()
        self.modes = FakeModeProvider(labels=["L0"])

        self.builder = ObservablesBuilder(
            context=self.ctx, kron=self.kron, mode_provider=self.modes
        )

        self._interp = interpreter_stub_factory(self.dims_full)

    def test_build_includes_qd_projectors(self):
        with (
            patch(
                "bec.quantum_dot.me.base_builder.interpreter",
                side_effect=self._interp,
            ),
            patch(
                "bec.quantum_dot.me.observables_builder.interpreter",
                side_effect=self._interp,
            ),
        ):
            obs = self.builder.build(self.dims_full, include_qd=True)

        self.assertIn("P_G", obs.qd)
        self.assertIn("P_X1", obs.qd)
        self.assertIn("P_X2", obs.qd)
        self.assertIn("P_XX", obs.qd)

        for name, P in obs.qd.items():
            self.assertIsInstance(P, Qobj)
            self.assertEqual(P.dims, [self.dims_full, self.dims_full])
            self.assertTrue(P.isherm)
            self.assertLess((P * P - P).norm(), 1e-12)

    def test_build_includes_mode_keys(self):
        with (
            patch(
                "bec.quantum_dot.me.base_builder.interpreter",
                side_effect=self._interp,
            ),
            patch(
                "bec.quantum_dot.me.observables_builder.interpreter",
                side_effect=self._interp,
            ),
        ):
            obs = self.builder.build(self.dims_full, include_qd=True)

        # Check a representative set of keys for the one mode "L0"
        expected_keys = [
            "N[L0]",
            "N+[L0]",
            "N-[L0]",
            "Pvac[L0]",
            "P10[L0]",
            "P01[L0]",
            "P11[L0]",
            "S0[L0]",
            "S1[L0]",
        ]
        for k in expected_keys:
            self.assertIn(k, obs.modes)
            self.assertIsInstance(obs.modes[k], Qobj)
            self.assertEqual(
                obs.modes[k].dims, [self.dims_full, self.dims_full]
            )

    def test_include_qd_false_returns_photonic_only_dims(self):
        with (
            patch(
                "bec.quantum_dot.me.base_builder.interpreter",
                side_effect=self._interp,
            ),
            patch(
                "bec.quantum_dot.me.observables_builder.interpreter",
                side_effect=self._interp,
            ),
        ):
            obs = self.builder.build(self.dims_full, include_qd=False)

        # modes should be photonic-only
        for k, Op in obs.modes.items():
            self.assertEqual(Op.dims, [self.dims_phot, self.dims_phot])

        # QD projectors are still computed in this builder design; if you later decide
        # to omit qd in include_qd=False, adjust this assertion accordingly.
        for k, Op in obs.qd.items():
            self.assertEqual(Op.dims, [self.dims_full, self.dims_full])


if __name__ == "__main__":
    unittest.main()
