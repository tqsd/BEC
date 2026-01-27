from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Any


from bec.quantum_dot.enums import (
    QDState,
    TransitionPair,
)  # canonical basis states

from bec.quantum_dot.ir.ops import (
    OpExpr,
    OpExprKind,
    EmbeddedKron,
    QDOpRef,
)
from bec.quantum_dot.me.types import (
    Term,
    TermKind,
    CoeffRef,
    CoeffRefKind,
)
from bec.quantum_dot.parameters.energy_structure import (
    EnergyStructure,
)
from bec.quantum_dot.transitions import (
    DEFAULT_REGISTRY as DEFAULT_TRANSITION_REGISTRY,
    TransitionRegistry,
)
from bec.quantum_dot.units import as_eV
from bec.units import Q, energy_to_rad_s, magnitude


# -----------------------------
# Small IR helpers (local)
# -----------------------------


def PRIM_QD(key: str) -> OpExpr:
    """Primitive operator: QD-only operator (fock=None)."""
    return OpExpr(
        kind=OpExprKind.PRIMITIVE,
        primitive=EmbeddedKron(qd=QDOpRef(key=key), fock=None),
    )


def SUM(*xs: OpExpr, pretty: Optional[str] = None) -> OpExpr:
    xs2 = [x for x in xs if x is not None]
    if len(xs2) == 1:
        return xs2[0]
    return OpExpr(kind=OpExprKind.SUM, args=tuple(xs2), pretty=pretty)


def SCALE(c: Any, x: OpExpr, pretty: Optional[str] = None) -> OpExpr:
    # Keep SCALE nodes explicit so emit/materialize can turn them into ("smult", c, ...)
    return OpExpr(kind=OpExprKind.SCALE, scalar=c, args=(x,), pretty=pretty)


# -----------------------------
# Builder
# -----------------------------
@dataclass(frozen=True)
class HamiltonianCatalog:
    static_terms: List[Term]
    detuning_basis: List[Term]
    coherence_basis: List[Term]

    @property
    def all_terms(self) -> List[Term]:
        return [*self.static_terms, *self.detuning_basis, *self.coherence_basis]


class HamiltonianBuilder:
    """
    Build Hamiltonian terms as typed IR (Term + OpExpr).

    Emits:
      - static exciton mixing/FSS term (numeric constant in simulation time units)
      - optional basis projectors/coherences (coeff=None) for later composition
    """

    DOT_LEVELS = (QDState.G, QDState.X1, QDState.X2, QDState.XX)

    def __init__(
        self,
        energy_structure: EnergyStructure,
        exciton_mixing: Any,
        transitions: TransitionRegistry = DEFAULT_TRANSITION_REGISTRY,
    ):
        self._transitions = transitions
        self._ES = energy_structure
        self._mix = exciton_mixing

    def build_catalog(self) -> HamiltonianCatalog:
        static_terms = self._build_static_terms()
        detuning_basis = self._build_detuning_basis()
        coherence_basis = self._build_coherence_basis()
        return HamiltonianCatalog(
            static_terms=static_terms,
            detuning_basis=detuning_basis,
            coherence_basis=coherence_basis,
        )

    @property
    def transitions(self) -> TransitionRegistry:
        return self._transitions

    # ---------- Static term: FSS / exciton mixing ----------
    def _build_static_terms(self) -> List[Term]:
        # Keep as quantities (eV)

        fss_q = self._ES.fss  # QuantityLike [eV]
        delta_prime_q = getattr(self._mix, "delta_prime", 0.0)
        delta_prime_q = (
            as_eV(delta_prime_q) if delta_prime_q is not None else as_eV(0.0)
        )

        Delta_q = energy_to_rad_s(fss_q)  # rad/s quantity
        Delta_p_q = energy_to_rad_s(delta_prime_q)

        op = SUM(
            SCALE(+0.5 * Delta_q, PRIM_QD("s_X1_X1")),
            SCALE(-0.5 * Delta_q, PRIM_QD("s_X2_X2")),
            SCALE(+0.5 * Delta_p_q, PRIM_QD("s_X1_X2")),
            SCALE(+0.5 * Delta_p_q, PRIM_QD("s_X2_X1")),
        )
        return [
            Term(
                kind=TermKind.H,
                label="H_fss",
                op=op,
                coeff=None,
                meta={
                    "type": "static",
                    "subspace": "exciton",
                    "params": {
                        "fss_eV": float(magnitude(fss_q, "eV")),
                        "delta_prime_eV": float(magnitude(delta_prime_q, "eV")),
                        "Delta_rad_s": float(magnitude(Delta_q, "rad/s")),
                        "Delta_p_rad_s": float(magnitude(Delta_p_q, "rad/s")),
                    },
                },
            )
        ]

    # ---------- Basis terms ----------

    def _build_detuning_basis(self) -> List[Term]:
        out: List[Term] = []
        for st in self.DOT_LEVELS:
            if st == QDState.G:
                continue
            key = f"s_{st.name}_{st.name}"
            out.append(
                Term(
                    kind=TermKind.H,
                    label=f"proj_{st.name}",
                    op=PRIM_QD(key),
                    coeff=None,
                    meta={"type": "projector", "level": st.name},
                    pretty=f"|{st.name}><{st.name}|",
                )
            )
        return out

    def _build_coherence_basis(self) -> List[Term]:
        out: List[Term] = []
        reg = self._transitions

        for bra in self.DOT_LEVELS:
            for ket in self.DOT_LEVELS:
                if bra == ket:
                    continue

                tr = reg.from_states(bra, ket)  # directed Transition or None

                meta = {
                    "type": "coherence",
                    "bra": bra.name,
                    "ket": ket.name,
                }

                if tr is not None:
                    tp = reg.as_pair(tr)
                    spec = reg.spec(tp)

                    # Optionally: only keep coherences that are drive-allowed
                    if not spec.drive_allowed:
                        continue

                    meta.update(
                        {
                            "transition": tr.value,  # "G_X1"
                            "transition_id": tr.name,  # "G_X1"
                            "transition_pair": tp.value,  # "G<->X1"
                            "transition_pair_id": tp.name,  # "G_X1"
                            "order": spec.order,
                            "kind": spec.kind.value,
                            "drive_allowed": spec.drive_allowed,
                            "decay_allowed": spec.decay_allowed,
                        }
                    )
                else:
                    # This catches e.g. X1<->X2 (not in registry endpoints)
                    meta["internal"] = True

                key = f"s_{bra.name}_{ket.name}"
                out.append(
                    Term(
                        kind=TermKind.H,
                        label=f"coh_{bra.name}_{ket.name}",
                        op=PRIM_QD(key),
                        coeff=None,
                        meta=meta,
                        pretty=f"|{bra.name}><{ket.name}|",
                    )
                )
        return out
