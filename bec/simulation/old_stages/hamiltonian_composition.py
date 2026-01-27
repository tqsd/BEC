from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np


from bec.params.transitions import Transition
from bec.operators.qd_operators import QDState
from bec.quantum_dot.dot import QuantumDot
from bec.simulation.protocols import HamiltonianComposer
from bec.simulation.types import DriveCoefficients, ResolvedDrive
from bec.quantum_dot.me.types import HamiltonianTerm, HamiltonianTermKind

# coeff utilities (your project types)
from bec.quantum_dot.me.coeffs import as_coeff, scale, CoeffExpr


# -----------------------------------------------------------------------------
# Transition mapping
# -----------------------------------------------------------------------------


def _transition_to_bra_ket(tr: Transition) -> Tuple[QDState, QDState]:
    """
    Map a transition identifier to the "raising" coherence operator |bra><ket|.

    Convention:
      - For 1ph transitions: bra is excited, ket is lower.
      - For 2ph virtual transition G_XX: bra is XX, ket is G.

    Drive Hamiltonian is typically:
        H_drive(t) = Omega(t) * (A + A.dag())
    with A = |bra><ket|, or a coherent sum of such terms.
    """
    if tr == Transition.G_X1:
        return (QDState.X1, QDState.G)
    if tr == Transition.G_X2:
        return (QDState.X2, QDState.G)
    if tr == Transition.X1_XX:
        return (QDState.XX, QDState.X1)
    if tr == Transition.X2_XX:
        return (QDState.XX, QDState.X2)
    if tr == Transition.G_X:
        # Degenerate shorthand: treat as X2 for operator lookup (avoid relying on this if you use coherent decomposition)
        return (QDState.X2, QDState.G)
    if tr == Transition.X_XX:
        # Degenerate shorthand: treat as XX<-X1 for operator lookup (avoid relying on this if you use coherent decomposition)
        return (QDState.XX, QDState.X1)
    if tr == Transition.G_XX:
        return (QDState.XX, QDState.G)

    raise ValueError(f"Unknown transition: {tr!r}")


def _transition_to_detuned_level(tr: Transition) -> QDState:
    """
    Choose which level projector gets the detuning term for this transition.

    In rotating-frame + RWA style models, a common convention is detuning on
    the "upper" (bra) state projector.

    For 2ph G_XX, detuning goes on |XX><XX|.
    """
    bra, _ket = _transition_to_bra_ket(tr)
    return bra


# -----------------------------------------------------------------------------
# Policy
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class HamiltonianCompositionPolicy:
    """
    include_detuning_terms:
        Emit detuning projector terms with coefficient from ResolvedDrive.detuning.

    hermitian_drive:
        If True, drive term operator is (A + A.dag()).
        If False, drive term operator is A only.

    split_components:
        If True, emit one DRIVE HamiltonianTerm per ResolvedDrive component.
        This is the recommended setting for your "drive decoder splits into components"
        pipeline design.
    """

    include_detuning_terms: bool = True
    hermitian_drive: bool = True
    split_components: bool = True
    build_drive_coeff_from_physical: bool = True


# -----------------------------------------------------------------------------
# Composer
# -----------------------------------------------------------------------------


class DefaultHamiltonianComposer(HamiltonianComposer):
    """
    Compose HamiltonianTerm objects from:
      - QD Hamiltonian catalog (operators only)
      - decoded drives (ResolvedDrive)

    IMPORTANT design choice (per your pipeline):
      - chirp / time-dependent laser frequency is represented via rd.detuning
        (projector DETUNING terms), not via a phase in the drive coefficient.
      - polarization only scales DRIVE coefficients via the component weights
        in ResolvedDrive.components.

    Therefore:
      DRIVE coeff(t) ~ omega0 * envelope(t_phys) * component_weight
      DETUNING coeff(t) ~ rd.detuning(t_solver)  (already includes chirp via delta_omega)
    """

    def __init__(self, policy: Optional[HamiltonianCompositionPolicy] = None):
        self.policy = policy or HamiltonianCompositionPolicy()

    def compose(
        self,
        *,
        qd: QuantumDot,
        dims: List[int],
        resolved: Tuple[ResolvedDrive, ...],
        drive_coeffs: Dict[str, DriveCoefficients],
        time_unit_s: float,
        tlist: Optional[np.ndarray] = None,
    ) -> List[HamiltonianTerm]:
        catalog = qd.h_builder.build_catalog(
            dims=dims, time_unit_s=time_unit_s)

        static_terms = [
            t for t in catalog if t.kind == HamiltonianTermKind.STATIC
        ]
        detuning_catalog = [
            t for t in catalog if t.kind == HamiltonianTermKind.DETUNING
        ]
        drive_catalog = [
            t for t in catalog if t.kind == HamiltonianTermKind.DRIVE
        ]

        proj_by_level = self._index_projectors(detuning_catalog)
        coh_by_pair = self._index_coherences(drive_catalog)

        out: List[HamiltonianTerm] = []
        out.extend(static_terms)

        # Precompute drive base coeff expr per physical drive_id (so repeated components share one envelope callable)
        for rd in resolved:
            dc = drive_coeffs[rd.drive_id]
            # ---- DRIVE TERMS ----
            drive_terms = self._emit_drive_terms(
                rd=rd,
                coh_by_pair=coh_by_pair,
                omega_by_tr=dc.omega_by_transition,
                omega_2ph=dc.omega_2ph,
            )

            out.extend(drive_terms)

            # ---- DETUNING TERMS ----
            if self.policy.include_detuning_terms:
                out.extend(
                    self._emit_detuning_terms(
                        rd=rd, proj_by_level=proj_by_level
                    )
                )

        return out

    # -------------------------------------------------------------------------
    # Coeff construction
    # -------------------------------------------------------------------------

    def _drive_base_coeff(
        self, rd: ResolvedDrive, *, time_unit_s: float
    ) -> CoeffExpr:
        """
        Base coefficient for a physical coherent drive, *without* polarization component weights.

        Convention here:
          Omega_base(t) = omega0 * envelope(t_phys)
        with t_phys = time_unit_s * t_solver.

        Notes:
          - This is amplitude only (real, typically). If you later want a complex carrier phase,
            add it here, but then DO NOT also represent it via detuning terms.
        """
        drv = rd.physical
        env = drv.envelope
        omega0 = float(getattr(drv, "omega0", 1.0))
        s = float(time_unit_s)

        def omega_base(
            t_solver: float, args: Optional[Dict[str, Any]] = None
        ) -> complex:
            t_phys = s * float(t_solver)
            return complex(0.5 * s * omega0 * float(env(t_phys)))

        return as_coeff(omega_base)

    def _detuning_coeff(self, rd: ResolvedDrive) -> Optional[CoeffExpr]:
        """
        Normalize rd.detuning into a CoeffExpr (callable).
        rd.detuning is already in solver units; keep it that way.
        """
        if rd.detuning is None:
            return None
        return as_coeff(rd.detuning)

    # -------------------------------------------------------------------------
    # DRIVE: splitting into per-component terms
    # -------------------------------------------------------------------------

    def _drive_components(
        self, rd: ResolvedDrive
    ) -> Tuple[Tuple[Transition, complex], ...]:
        """
        Return canonical (transition, complex weight) tuples.

        Handles:
        - 1ph single: transition set, components empty -> [(transition, 1)]
        - 1ph weighted: components provided -> return as-is
        - 1ph_coherent: transition None, components has multiple -> return as-is
        - 2ph: transition == G_XX, components empty -> [(G_XX, 1)]
            (we still represent the "addressed transition" here; coefficient comes from omega_2ph)
        """
        comps = tuple(getattr(rd, "components", ()) or ())
        if comps:
            return comps

        tr = getattr(rd, "transition", None)
        if tr is not None:
            return ((tr, 1.0 + 0j),)

        # Backward compatibility: some code may only store kind in meta
        kind = getattr(rd, "kind", None)
        if kind is None:
            meta = getattr(rd, "meta", {}) or {}
            kind = meta.get("kind", None)

        if str(kind) == "2ph":
            # If a 2ph drive was emitted without transition, assume G_XX
            return ((Transition.G_XX, 1.0 + 0j),)

        raise ValueError(
            "ResolvedDrive has neither transition nor components.")

    def _emit_drive_terms(
        self,
        *,
        rd: ResolvedDrive,
        coh_by_pair: Mapping[Tuple[QDState, QDState], np.ndarray],
        omega_by_tr: Mapping[Transition, CoeffExpr],
        omega_2ph: Optional[CoeffExpr],
    ) -> List[HamiltonianTerm]:
        """
        Emit DRIVE Hamiltonian terms using per-transition coefficients from the
        drive-strength stage.

        Contract:
        - omega_by_tr[tr] is the (possibly complex) Omega_solver(t) coefficient for
        that transition/component.
        - omega_2ph is used only when rd.kind == "2ph".
        - Apply Omega/2 convention here.
        """

        # Special case: 2-photon effective drive uses omega_2ph
        if getattr(rd, "kind", "1ph") == "2ph":
            if omega_2ph is None:
                raise KeyError(
                    f"Missing omega_2ph for 2ph drive_id={rd.drive_id}."
                )

            tr2 = rd.transition
            if tr2 is None and rd.components:
                tr2 = rd.components[0][0]

            if tr2 != Transition.G_XX:
                raise ValueError(
                    f"2ph drive_id={
                        rd.drive_id} expected Transition.G_XX, got {tr2!r}"
                )

            A = self._coherence_for_transition(Transition.G_XX, coh_by_pair)
            Hop = (A + A.conj().T) if self.policy.hermitian_drive else A
            coeff = scale(omega_2ph, 0.5)

            return [
                HamiltonianTerm(
                    kind=HamiltonianTermKind.DRIVE,
                    op=Hop,
                    coeff=coeff,
                    label=f"drive_{rd.drive_id}_2ph",
                    meta={
                        "type": "drive_2ph",
                        "drive_id": rd.drive_id,
                        "transition": str(Transition.G_XX),
                        "candidates": [str(t) for t in rd.candidates],
                        **dict(rd.meta),
                    },
                )
            ]

        if not self.policy.split_components:
            raise NotImplementedError(
                "split_components=False is not supported with per-transition coefficients yet. "
                "Use split_components=True."
            )

        comps = self._drive_components(rd)

        out: List[HamiltonianTerm] = []
        for tr, w in comps:
            A = self._coherence_for_transition(tr, coh_by_pair)
            Hop = (A + A.conj().T) if self.policy.hermitian_drive else A

            if tr not in omega_by_tr:
                raise KeyError(
                    f"Missing drive coefficient for transition {tr} "
                    f"(drive_id={rd.drive_id}). Available: {
                        list(omega_by_tr.keys())}"
                )

            coeff = scale(omega_by_tr[tr], 0.5)

            out.append(
                HamiltonianTerm(
                    kind=HamiltonianTermKind.DRIVE,
                    op=Hop,
                    coeff=coeff,
                    label=f"drive_{rd.drive_id}_{tr}",
                    meta={
                        "type": "drive_component",
                        "drive_id": rd.drive_id,
                        "transition": str(tr),
                        "component_weight": complex(w),
                        "all_components": [
                            (str(t), complex(c)) for (t, c) in comps
                        ],
                        "candidates": [str(t) for t in rd.candidates],
                        **dict(rd.meta),
                    },
                )
            )

        return out

    def _coherence_for_transition(
        self,
        tr: Transition,
        coh_by_pair: Mapping[Tuple[QDState, QDState], np.ndarray],
    ) -> np.ndarray:
        bra, ket = _transition_to_bra_ket(tr)
        A = coh_by_pair.get((bra, ket))
        if A is None:
            raise KeyError(
                f"Missing coherence operator for (bra={bra}, ket={ket}) "
                f"required by transition {tr}."
            )
        return A

    def _build_drive_operator_sum(
        self,
        comps: Tuple[Tuple[Transition, complex], ...],
        coh_by_pair: Mapping[Tuple[QDState, QDState], np.ndarray],
    ) -> np.ndarray:
        """
        A = sum_i c_i * |bra_i><ket_i|
        """
        A_sum: Optional[np.ndarray] = None
        for tr, c in comps:
            A = self._coherence_for_transition(tr, coh_by_pair)
            term = complex(c) * A
            A_sum = term if A_sum is None else (A_sum + term)
        assert A_sum is not None
        return A_sum

    # -------------------------------------------------------------------------
    # DETUNING: per-drive projectors
    # -------------------------------------------------------------------------

    def _emit_detuning_terms(
        self,
        *,
        rd: ResolvedDrive,
        proj_by_level: Mapping[QDState, np.ndarray],
    ) -> List[HamiltonianTerm]:
        """
        Emit DETUNING projector term(s) for this drive.

        For split components, we still typically want detuning on the relevant upper levels.
        We deduplicate projectors (e.g., if multiple components detune the same upper level).
        """
        det = self._detuning_coeff(rd)
        if det is None:
            return []

        comps = self._drive_components(rd)

        levels: List[QDState] = []
        for tr, _w in comps:
            lv = _transition_to_detuned_level(tr)
            levels.append(lv)

        # Deduplicate while preserving order
        seen: set[QDState] = set()
        levels_unique: List[QDState] = []
        for lv in levels:
            if lv not in seen:
                seen.add(lv)
                levels_unique.append(lv)

        out: List[HamiltonianTerm] = []
        for lv in levels_unique:
            P = proj_by_level.get(lv)
            if P is None:
                raise KeyError(
                    f"Missing projector term for level {
                        lv} in Hamiltonian catalog."
                )

            out.append(
                HamiltonianTerm(
                    kind=HamiltonianTermKind.DETUNING,
                    op=P,
                    coeff=det,
                    label=f"detuning_{rd.drive_id}_{lv}",
                    meta={
                        "type": "detuning",
                        "drive_id": rd.drive_id,
                        "level": str(lv),
                        # For debugging:
                        "components": [
                            (str(t), complex(c)) for (t, c) in comps
                        ],
                    },
                )
            )

        return out

    # -------------------------------------------------------------------------
    # Catalog indexing
    # -------------------------------------------------------------------------

    def _index_projectors(
        self, detuning_terms: List[HamiltonianTerm]
    ) -> Dict[QDState, np.ndarray]:
        """
        Index DETUNING catalog terms by QD level.

        Expects builder meta:
            {"type": "projector", "level": <QDState or str>, ...}
        """
        out: Dict[QDState, np.ndarray] = {}
        for t in detuning_terms:
            meta = dict(t.meta or {})
            lvl = meta.get("level", None)
            if lvl is None:
                continue

            if isinstance(lvl, QDState):
                out[lvl] = t.op
                continue

            # If stored as string, attempt normalization
            try:
                out[QDState[str(lvl)]] = t.op  # type: ignore[index]
            except Exception:
                # If QDState doesn't support string indexing, map manually here.
                continue
        return out

    def _index_coherences(
        self, drive_terms: List[HamiltonianTerm]
    ) -> Dict[Tuple[QDState, QDState], np.ndarray]:
        """
        Index DRIVE catalog terms by (bra, ket) for |bra><ket|.

        Expects builder meta:
            {"type": "coherence", "bra": <QDState or str>, "ket": <QDState or str>, ...}
        """
        out: Dict[Tuple[QDState, QDState], np.ndarray] = {}
        for t in drive_terms:
            meta = dict(t.meta or {})
            bra = meta.get("bra", None)
            ket = meta.get("ket", None)
            if bra is None or ket is None:
                continue

            if not isinstance(bra, QDState) or not isinstance(ket, QDState):
                try:
                    bra = QDState[str(bra)]  # type: ignore[index]
                    ket = QDState[str(ket)]  # type: ignore[index]
                except Exception:
                    continue

            out[(bra, ket)] = t.op
        return out
