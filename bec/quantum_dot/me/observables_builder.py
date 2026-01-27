from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from bec.quantum_dot.enums import QDState
from bec.quantum_dot.ir.ops import (
    EmbeddedKron,
    FockOpKind,
    FockOpRef,
    OpExpr,
    OpExprKind,
    QDOpRef,
)
from bec.quantum_dot.mode_registry import ModeRegistry

from bec.quantum_dot.me.types import Term, TermKind


def PRIM_QD(key: str, *, pretty: Optional[str] = None) -> OpExpr:
    """
    Primitive operator acting only on the QD (fock=None).
    key is resolved via symbol table (e.g. "s_X1_G", "idq").
    """
    return OpExpr(
        kind=OpExprKind.PRIMITIVE,
        primitive=EmbeddedKron(qd=QDOpRef(key=key), fock=None, pretty=pretty),
        pretty=pretty,
    )


def PRIM_FOCK(
    *,
    channel_key: Any,  # ChannelKey
    kind: FockOpKind,
    label: Optional[str] = None,
    pretty: Optional[str] = None,
) -> OpExpr:
    """
    Primitive operator acting on one photonic channel, with QD identity.
    """
    return OpExpr(
        kind=OpExprKind.PRIMITIVE,
        primitive=EmbeddedKron(
            qd=QDOpRef(key="idq"),
            fock=FockOpRef(key=channel_key, kind=kind, label=label),
            pretty=pretty,
        ),
        pretty=pretty,
    )


def SUM(*xs: OpExpr, pretty: Optional[str] = None) -> OpExpr:
    xs2 = [x for x in xs if x is not None]
    if len(xs2) == 1:
        return xs2[0]
    return OpExpr(kind=OpExprKind.SUM, args=tuple(xs2), pretty=pretty)


@dataclass(frozen=True)
class ObservablesCatalog:
    qd: List[Term]
    channels: List[Term]
    extra: List[Term]

    @property
    def all_terms(self) -> List[Term]:
        return [*self.qd, *self.channels, *self.extra]


class ObservablesBuilder:
    """
    IR-only builder for observables.

    Produces Term(kind=E) with OpExpr primitives:
      - QD projectors: P_G, P_X1, P_X2, P_XX
      - per-channel photon number: N[label]
      - per-channel vacuum projector: Pvac[label]

    Notes:
      - Your ModeRegistry currently stores 'channels' (polarization already resolved into ChannelKey).
      - So we report per-channel observables. Later you can add grouping to combine H/V into one physical mode.
    """

    DOT_LEVELS = (QDState.G, QDState.X1, QDState.X2, QDState.XX)

    def __init__(self, *, modes: ModeRegistry):
        self._modes = modes

    def build_catalog(
        self,
        *,
        include_qd: bool = True,
        include_numbers: bool = True,
        include_vacuum: bool = True,
    ) -> ObservablesCatalog:
        qd_terms: List[Term] = []
        ch_terms: List[Term] = []
        extra: List[Term] = []

        if include_qd:
            qd_terms.extend(self._build_qd_projectors())

        if include_numbers:
            ch_terms.extend(self._build_channel_numbers())

        if include_vacuum:
            ch_terms.extend(self._build_channel_vacuum())

        return ObservablesCatalog(qd=qd_terms, channels=ch_terms, extra=extra)

    def _build_qd_projectors(self) -> List[Term]:
        out: List[Term] = []
        for st in self.DOT_LEVELS:
            key = f"s_{st.name}_{st.name}"
            out.append(
                Term(
                    kind=TermKind.E,
                    label=f"P_{st.name}",
                    op=PRIM_QD(key, pretty=f"|{st.name}><{st.name}|"),
                    coeff=None,
                    meta={"type": "qd_projector", "state": st.name},
                    pretty=f"|{st.name}><{st.name}|",
                )
            )
        return out

    def _channel_label(self, i: int) -> str:
        ch = self._modes.channels[i]
        lbl = getattr(ch, "label", None)
        if isinstance(lbl, str) and lbl:
            return lbl
        # fallback: derive from ChannelKey if it has something printable
        k = getattr(ch, "key", None)
        if k is not None:
            return str(k)
        return f"ch_{i}"

    def _build_channel_numbers(self) -> List[Term]:
        out: List[Term] = []
        for i, ch in enumerate(self._modes.channels):
            label = self._channel_label(i)
            out.append(
                Term(
                    kind=TermKind.E,
                    label=f"N[{label}]",
                    op=PRIM_FOCK(
                        channel_key=ch.key,
                        kind=FockOpKind.N,
                        label=label,
                        pretty=f"N({label})",
                    ),
                    coeff=None,
                    meta={
                        "type": "channel_number",
                        "channel_index": i,
                        "channel_key": str(ch.key),
                    },
                    pretty=f"N({label})",
                )
            )
        return out

    def _build_channel_vacuum(self) -> List[Term]:
        out: List[Term] = []
        for i, ch in enumerate(self._modes.channels):
            label = self._channel_label(i)
            out.append(
                Term(
                    kind=TermKind.E,
                    label=f"Pvac[{label}]",
                    op=PRIM_FOCK(
                        channel_key=ch.key,
                        kind=FockOpKind.VAC,
                        label=label,
                        pretty=f"Pvac({label})",
                    ),
                    coeff=None,
                    meta={
                        "type": "channel_vacuum",
                        "channel_index": i,
                        "channel_key": str(ch.key),
                    },
                    pretty=f"Pvac({label})",
                )
            )
        return out
