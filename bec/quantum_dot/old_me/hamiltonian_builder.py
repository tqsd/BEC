from __future__ import annotations

from typing import Any, Callable, Dict, List

from scipy.constants import e as _e, hbar as _hbar

from bec.params.energy_levels import EnergyLevels

from .base_builder import BaseBuilder
from .types import HamiltonianTerm, HamiltonianTermKind


class HamiltonianBuilder(BaseBuilder):
    def __init__(
        self,
        context: Dict[str, Any],
        kron,
        energy_levels: EnergyLevels,
        pm_map: Callable[[int], str],
    ):
        super().__init__(context=context, kron=kron)
        self._EL = energy_levels
        self._pm = pm_map

    def build_catalog(
        self, dims: List[int], time_unit_s: float
    ) -> List[HamiltonianTerm]:
        terms: List[HamiltonianTerm] = []
        terms += self._build_static_terms(dims, time_unit_s)
        terms += self._build_detuning_terms(dims)
        terms += self._build_coherence_terms(dims)
        return terms

    def _build_static_terms(
        self, dims: List[int], time_unit_s: float
    ) -> List[HamiltonianTerm]:
        fss = getattr(self._EL, "fss", 0.0)
        delta_prime = getattr(self._EL, "delta_prime", 0.0)

        Delta = fss * _e / _hbar * time_unit_s
        Delta_p = delta_prime * _e / _hbar * time_unit_s

        proj_X1 = self._require_ctx("s_X1_X1")([])
        proj_X2 = self._require_ctx("s_X2_X2")([])
        X1X2 = self._require_ctx("s_X1_X2")([])
        X2X1 = self._require_ctx("s_X2_X1")([])

        Hloc = (Delta / 2) * (proj_X1 - proj_X2) + (Delta_p / 2) * (X1X2 + X2X1)
        H_expr = self._pad_dot(Hloc)
        H = self._eval(H_expr, dims)

        return [
            HamiltonianTerm(
                kind=HamiltonianTermKind.STATIC,
                op=H,
                coeff=None,
                label="fss",
                meta={
                    "type": "fss",
                    "subspace": "X",
                    "params": {"fss_eV": fss, "delta_prime_eV": delta_prime},
                },
            )
        ]

    def _build_detuning_terms(self, dims: List[int]) -> List[HamiltonianTerm]:
        out: List[HamiltonianTerm] = []
        for level in self.DOT_LEVELS:
            if level == "G":
                continue
            P = self.op(level, level, dims)
            out.append(
                HamiltonianTerm(
                    kind=HamiltonianTermKind.DETUNING,
                    op=P,
                    coeff=None,
                    label=f"proj_{level}",
                    meta={
                        "type": "projector",
                        "level": level,
                        "bra": level,
                        "ket": level,
                    },
                )
            )
        return out

    def _build_coherence_terms(self, dims: List[int]) -> List[HamiltonianTerm]:
        out: List[HamiltonianTerm] = []
        for bra in self.DOT_LEVELS:
            for ket in self.DOT_LEVELS:
                if bra == ket:
                    continue
                out.append(
                    HamiltonianTerm(
                        kind=HamiltonianTermKind.DRIVE,
                        op=self.op(bra, ket, dims),
                        coeff=None,
                        label=f"coh_{bra}_{ket}",
                        meta={
                            "type": "coherence",
                            "bra": bra,
                            "ket": ket,
                            "transition": (ket, bra),
                        },
                    )
                )
        return out
