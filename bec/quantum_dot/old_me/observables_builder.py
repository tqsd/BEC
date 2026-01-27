from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from photon_weave.extra import interpreter

from bec.quantum_dot.kron_pad_utility import KronPad
from bec.quantum_dot.protocols import ModeProvider

from .base_builder import BaseBuilder
from .types import Observables


class ObservablesBuilder(BaseBuilder):
    def __init__(
        self,
        context: Dict[str, Any],
        kron: KronPad,
        mode_provider: ModeProvider,
    ):
        super().__init__(context=context, kron=kron)
        self._modes = mode_provider

    def build(self, dims: List[int], *, include_qd: bool = True) -> Observables:
        qd = self._build_qd_projectors(dims)
        modes = self._build_mode_observables(dims, include_qd=include_qd)
        return Observables(qd=qd, modes=modes, extra=None)

    def _build_qd_projectors(self, dims: List[int]) -> Dict[str, np.ndarray]:
        return {
            "P_G": self.op("G", "G", dims),
            "P_X1": self.op("X1", "X1", dims),
            "P_X2": self.op("X2", "X2", dims),
            "P_XX": self.op("XX", "XX", dims),
        }

    def _build_mode_observables(
        self, dims: List[int], *, include_qd: bool = True
    ) -> Dict[str, np.ndarray]:
        ops: Dict[str, np.ndarray] = {}

        for i, m in enumerate(self._modes.modes):
            label = getattr(m, "label", f"mode_{i}")

            # these expressions depend on your KronPad DSL; keep as-is
            n_plus_expr = self._kron.pad("idq", "n+", i)
            n_minus_expr = self._kron.pad("idq", "n-", i)
            vac_expr = self._kron.pad("idq", "vac", i)
            I_expr = self._kron.pad("idq", "i", i)

            Np = self._eval(n_plus_expr, dims)
            Nm = self._eval(n_minus_expr, dims)
            Pvac = self._eval(vac_expr, dims)
            I = self._eval(I_expr, dims)

            # assuming number operators are projectors onto |1> in your truncated basis:
            P1p, P0p = Np, I - Np
            P1m, P0m = Nm, I - Nm

            ops[f"N[{label}]"] = Np + Nm
            ops[f"N+[{label}]"] = Np
            ops[f"N-[{label}]"] = Nm

            ops[f"Pvac[{label}]"] = Pvac
            ops[f"P10[{label}]"] = P1p @ P0m
            ops[f"P01[{label}]"] = P0p @ P1m
            ops[f"P11[{label}]"] = P1p @ P1m

            ops[f"S0[{label}]"] = Np + Nm
            ops[f"S1[{label}]"] = Np - Nm

        return ops
