from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
from bec.quantum_dot.protocols import ModeProvider
import numpy as np

from bec.operators.qd_operators import QDState
from bec.params.transitions import Transition, TransitionType

from .base_builder import BaseBuilder
from .types import CollapseTerm, CollapseTermKind


class CollapseBuilder(BaseBuilder):
    def __init__(
        self,
        context: Dict[str, Any],
        mode_provider: ModeProvider,
        kron,
        pm_map: Optional[Callable[[int], str]] = None,
    ):
        super().__init__(context=context, kron=kron)
        self._modes = mode_provider
        self._pm = pm_map

    def build_catalog(self, dims: List[int]) -> List[CollapseTerm]:
        terms: List[CollapseTerm] = []
        terms += self._build_radiative_branch_terms(dims)
        terms += self._build_phonon_dephasing_terms(dims)
        return terms

    def _emit_channel(
        self, initial: QDState, final: QDState
    ) -> tuple[int, str, str]:
        """
        Returns (mode_pos, pol, merge_key)
        """

        # Map QDState pair -> Transition enum used by ModeRegistry
        pair_to_transition: Dict[tuple[QDState, QDState], Transition] = {
            (QDState.XX, QDState.X1): Transition.X1_XX,
            (QDState.XX, QDState.X2): Transition.X2_XX,
            (QDState.X1, QDState.G): Transition.G_X1,
            (QDState.X2, QDState.G): Transition.G_X2,
        }

        tr = pair_to_transition.get((initial, final))
        if tr is None:
            raise KeyError(f"No Transition mapping for {initial}->{final}")

        mode_pos, _mode = self._modes.by_transition_and_source(
            tr, TransitionType.INTERNAL
        )

        # Stable per-channel index for pm_map:
        # (choose a fixed order; MUST match your physical convention)
        order: List[tuple[QDState, QDState]] = [
            (QDState.X1, QDState.G),
            (QDState.X2, QDState.G),
            (QDState.XX, QDState.X1),
            (QDState.XX, QDState.X2),
        ]
        idx = order.index((initial, final))

        if self._pm is not None:
            pol = str(self._pm(idx))
            if pol not in {"+", "-"}:
                raise ValueError(f"pm_map must return '+' or '-', got {pol!r}")
        else:
            # fallback: a sane default (you can adjust)
            pol = "+" if final in (QDState.X2,) else "-"

        # merge_key:
        # - branches are mergeable only if they land in SAME optical mode (same mode_pos)
        # so mode_pos must be part of the key.
        if initial == QDState.XX:
            group = "XX->X"
        else:
            group = "X->G"
        merge_key = f"{group}:{mode_pos}"

        return int(mode_pos), pol, merge_key

    def emit_branch_op(
        self,
        *,
        final: QDState,
        initial: QDState,
        mode_pos: int,
        pol: str,  # "+" or "-"
        dims: List[int],
    ) -> np.ndarray:
        fock_op = "a+_dag" if pol == "+" else "a-_dag"
        fock_op = "i"
        expr = self._kron.pad(
            f"s_{initial.name}_{final.name}",
            fock_op,
            mode_pos,
        )
        return self._eval(expr, dims)

    def _build_radiative_branch_terms(
        self, dims: List[int]
    ) -> List[CollapseTerm]:
        channels = [
            (QDState.G, QDState.X1),
            (QDState.G, QDState.X2),
            (QDState.X1, QDState.XX),
            (QDState.X2, QDState.XX),
        ]

        out: List[CollapseTerm] = []
        for final, initial in channels:
            mode_pos, pol, merge_key = self._emit_channel(initial, final)

            op = self.emit_branch_op(
                final=final,
                initial=initial,
                mode_pos=mode_pos,
                pol=pol,
                dims=dims,
            )

            # rate_key naming should match RatesBundle keys used in composer
            rate_key = {
                (QDState.XX, QDState.X1): "gamma_xx_x1",
                (QDState.XX, QDState.X2): "gamma_xx_x2",
                (QDState.X1, QDState.G): "gamma_x1_g",
                (QDState.X2, QDState.G): "gamma_x2_g",
            }[(initial, final)]

            out.append(
                CollapseTerm(
                    kind=CollapseTermKind.RADIATIVE,
                    op=op,
                    coeff=None,  # composer applies sqrt(gamma)
                    label=f"L_{initial.name}_to_{final.name}_{pol}",
                    meta={
                        "type": "radiative",
                        "initial": initial,
                        "final": final,
                        "bra": final,
                        "ket": initial,
                        "pol": pol,
                        "mode_pos": int(mode_pos),
                        "merge_key": merge_key,
                        "rate_key": rate_key,
                        # Optional: branch id so composer can attach phase to the right one
                        "branch": final.name,  # "X1" or "X2" etc.
                    },
                )
            )

        return out

    def _build_phonon_dephasing_terms(
        self, dims: List[int]
    ) -> List[CollapseTerm]:
        out: List[CollapseTerm] = []
        for level in ["X1", "X2", "XX"]:
            out.append(
                CollapseTerm(
                    kind=CollapseTermKind.PHONON,
                    op=self.op(level, level, dims),
                    coeff=None,
                    label=f"Lphi_{level}",
                    meta={
                        "type": "pure_dephasing",
                        "level": level,
                        "bra": level,
                        "ket": level,
                    },
                )
            )
        return out
