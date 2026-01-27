# bec/simulation/stages/hamiltonian_composition_photonweave.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

import numpy as np

# Your existing types
# IR terms from HamiltonianBuilder
from bec.quantum_dot.me.types import Term, TermKind
from bec.quantum_dot.ir.ops import (
    OpExpr,
    OpExprKind,
    EmbeddedKron,
    QDOpRef,
    FockOpRef,
)  # IR ops

# the coeff module you showed
from bec.quantum_dot.me.coeff import ConstCoeff, CallableCoeff, CoeffExpr
from bec.simulation.types import ResolvedDrive  # the resolved drive you showed


Args = Mapping[str, Any]


# -----------------------------------------------------------------------------
# Drive coefficients contract (minimal)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DriveCoefficients:
    """
    Coefficients are expected to already be in solver-time units:
      coeff(t_solver, args) -> complex

    omega_by_transition:
      maps TransitionKey (string) -> CoeffExpr (Omega(t) in solver units)
    omega_2ph:
      optional, used for 2ph drives if you prefer a single coefficient channel
    """

    omega_by_transition: Mapping[str, CoeffExpr] = field(default_factory=dict)
    omega_2ph: Optional[CoeffExpr] = None


# -----------------------------------------------------------------------------
# Output HamiltonianTerm (numeric op + coeff)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class HamiltonianTermOut:
    label: str
    op: np.ndarray  # numeric matrix (dimensionless basis operator)
    # None only if you truly mean "factor 1.0"
    coeff: Optional[CoeffExpr] = None
    meta: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Materialization context
# -----------------------------------------------------------------------------


class OpMaterializeContext(Protocol):
    """
    The composer must not know QuantumDot internals. It only asks for operators by key.

    You can implement this with PhotonWeave interpret, or with direct cached matrices.
    The required invariant:
      resolve_symbol(key, dims) returns the FULL embedded operator matrix of shape (D, D),
      where D = prod(dims).
    """

    def resolve_symbol(self, key: str, dims: Sequence[int]) -> np.ndarray: ...


# -----------------------------------------------------------------------------
# Policy
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class HamiltonianCompositionPolicy:
    # Emit detuning projector terms driven by ResolvedDrive.detuning.
    include_detuning_terms: bool = True

    # Drive term operator is (A + A^dag) if True, else A only.
    hermitian_drive: bool = True

    # Emit one DRIVE term per component.
    split_components: bool = True

    # Multiply drive coefficients by the component complex weight from decoder.
    # If your drive-strength stage already bakes the weight in, set this False.
    apply_component_weight: bool = True

    # Apply the standard RWA convention: H_drive = (Omega/2) * (A + A^dag)
    use_omega_over_2: bool = True


# -----------------------------------------------------------------------------
# Small helpers: coeff normalization and scaling
# -----------------------------------------------------------------------------


def _as_coeffexpr_from_detuning(
    det: Callable[[float], float] | float,
) -> CoeffExpr:
    if callable(det):

        def fn(t: float, args: Args) -> complex:
            return complex(det(float(t)))

        return CallableCoeff(fn=fn)
    return ConstCoeff(value=complex(det))


def _scale_coeff(c: CoeffExpr, s: complex) -> CoeffExpr:
    if isinstance(c, ConstCoeff):
        return ConstCoeff(value=c.as_complex() * complex(s))

    def fn(t: float, args: Args) -> complex:
        return complex(s) * complex(c(t, args))

    return CallableCoeff(fn=fn)


def _maybe_scale_omega(
    c: CoeffExpr, policy: HamiltonianCompositionPolicy
) -> CoeffExpr:
    if policy.use_omega_over_2:
        return _scale_coeff(c, 0.5 + 0.0j)
    return c


# -----------------------------------------------------------------------------
# TransitionKey parsing
# -----------------------------------------------------------------------------


def _split_transition_key(tr_key: str) -> Tuple[str, str]:
    """
    TransitionKey expected like:
      "G_X1", "G_X2", "X1_XX", "X2_XX", "G_XX"
    Returns (lower, upper) in the sense lower->upper.
    """
    parts = tr_key.split("_")
    if len(parts) != 2:
        raise ValueError(f"TransitionKey must look like 'A_B', got {tr_key!r}")
    return parts[0], parts[1]


def _upper_level_from_transition_key(tr_key: str) -> str:
    _low, up = _split_transition_key(tr_key)
    return up


def _coherence_symbol_for_transition(tr_key: str) -> str:
    """
    HamiltonianBuilder coherence basis uses keys like:
      s_{bra}_{ket} where bra is "upper" and ket is "lower" for raising coherence.
    For a transition key "G_X1" we want |X1><G| -> "s_X1_G".
    """
    low, up = _split_transition_key(tr_key)
    return f"s_{up}_{low}"


def _projector_symbol_for_level(level: str) -> str:
    """
    HamiltonianBuilder detuning basis uses:
      s_{st}_{st}
    for st in {X1, X2, XX} (no G projector in basis by default).
    """
    return f"s_{level}_{level}"


# -----------------------------------------------------------------------------
# OpExpr materialization (IR -> numeric matrix)
# -----------------------------------------------------------------------------


def _materialize_opexpr(
    expr: OpExpr, ctx: OpMaterializeContext, dims: Sequence[int]
) -> np.ndarray:
    """
    Materialize your OpExpr WITHOUT needing PhotonWeave's interpret format.
    We only require ctx.resolve_symbol(symbol, dims) for primitives.

    This supports:
      PRIMITIVE: EmbeddedKron(qd=QDOpRef(key=...)) -> resolve_symbol(key)
      SUM: sum child matrices
      PROD: matrix product of child matrices
      SCALE: scalar * child matrix (scalar must be numeric)
    """
    if expr.kind == OpExprKind.PRIMITIVE:
        prim = expr.primitive
        if prim is None:
            raise ValueError(
                "OpExprKind.PRIMITIVE requires primitive=EmbeddedKron"
            )

        # For now we support QDOpRef.key and QDOpRef.mat. If you also use FockOpRef in primitives,
        # you should encode those into a unique symbol and support it in ctx.resolve_symbol.
        if prim.qd.key is not None:
            return np.asarray(
                ctx.resolve_symbol(prim.qd.key, dims), dtype=complex
            )

        if prim.qd.mat is not None:
            # If you embed raw matrices, ctx must know how to embed them.
            # Easiest: require ctx to expose a symbol for this path; here we disallow it to avoid silent misuse.
            raise ValueError(
                "QDOpRef.mat is not supported by this materializer; provide a key instead."
            )

        raise ValueError("Invalid QDOpRef: requires key or mat")

    if expr.kind == OpExprKind.SUM:
        if not expr.args:
            raise ValueError("SUM requires args")
        acc = np.zeros((int(np.prod(dims)), int(np.prod(dims))), dtype=complex)
        for a in expr.args:
            acc = acc + _materialize_opexpr(a, ctx, dims)
        return acc

    if expr.kind == OpExprKind.PROD:
        if not expr.args:
            raise ValueError("PROD requires args")
        acc = _materialize_opexpr(expr.args[0], ctx, dims)
        for a in expr.args[1:]:
            acc = acc @ _materialize_opexpr(a, ctx, dims)
        return acc

    if expr.kind == OpExprKind.SCALE:
        if not expr.args or len(expr.args) != 1:
            raise ValueError("SCALE requires exactly one arg")
        s = expr.scalar
        if s is None:
            raise ValueError("SCALE requires scalar")
        # Scalar must be numeric at this stage (already converted to solver units upstream if it had units).
        return complex(s) * _materialize_opexpr(expr.args[0], ctx, dims)

    raise ValueError(f"Unsupported OpExprKind: {expr.kind!r}")


# -----------------------------------------------------------------------------
# Composer
# -----------------------------------------------------------------------------


class DefaultHamiltonianComposer:
    """
    Compose concrete HamiltonianTermOut from:
      - catalog: IR HamiltonianBuilder terms (static + basis)
      - resolved: ResolvedDrive (already coupled/selected)
      - drive_coeffs: coefficients in solver units

    Convention:
      - Drive operator is (A + A^dag) unless policy.hermitian_drive is False.
      - Coefficient uses Omega/2 if policy.use_omega_over_2 is True.
      - Detuning is represented via projector terms using ResolvedDrive.detuning (solver units).
      - This stage does not decide coupling; it only assembles what ResolvedDrive says.
    """

    def __init__(self, policy: Optional[HamiltonianCompositionPolicy] = None):
        self.policy = policy or HamiltonianCompositionPolicy()

    def compose(
        self,
        *,
        catalog: Iterable[Term],
        ctx: OpMaterializeContext,
        dims: Sequence[int],
        resolved: Sequence[ResolvedDrive],
        drive_coeffs: Mapping[str, DriveCoefficients],
    ) -> List[HamiltonianTermOut]:
        # Split catalog terms by meta/type and build indices.
        static_terms: List[Term] = []
        projector_terms: Dict[str, Term] = {}  # level -> Term
        # transition_key -> Term  (by meta) AND fallback by symbol
        coherence_terms: Dict[str, Term] = {}

        for t in catalog:
            if t.kind != TermKind.H:
                continue

            meta = dict(getattr(t, "meta", {}) or {})
            ttype = meta.get("type", None)

            if ttype == "static":
                static_terms.append(t)
                continue

            if ttype == "projector":
                lvl = meta.get("level", None)
                if isinstance(lvl, str) and lvl:
                    projector_terms[lvl] = t
                continue

            if ttype == "coherence":
                # Prefer transition_id (like "G_X1") when present; otherwise we can still use symbol keys.
                tr_id = meta.get("transition_id", None) or meta.get(
                    "transition", None
                )
                if isinstance(tr_id, str) and tr_id:
                    coherence_terms[tr_id] = t
                continue

        out: List[HamiltonianTermOut] = []

        # 1) Static terms: materialize op; if they contain SCALE nodes with numeric scalars, they are already included in op.
        #    For simplicity we keep coeff=None here. If you want scalars in coeff channel, refactor SCALE extraction.
        for t in static_terms:
            op = _materialize_opexpr(t.op, ctx, dims)
            out.append(
                HamiltonianTermOut(
                    label=t.label,
                    op=op,
                    coeff=None,
                    meta=dict(getattr(t, "meta", {}) or {}),
                )
            )

        # 2) Drive + detuning terms from resolved drives.
        for rd in resolved:
            if rd.kind == "unresolved":
                # Policy choice: skip unresolved or raise
                raise ValueError(
                    f"ResolvedDrive {
                        rd.drive_id!r} has kind='unresolved'"
                )

            dc = drive_coeffs.get(rd.drive_id, None)
            if dc is None:
                raise KeyError(
                    f"Missing DriveCoefficients for drive_id={
                        rd.drive_id!r}"
                )

            # Determine components
            comps: Tuple[Tuple[str, complex], ...]
            if rd.components:
                comps = tuple((str(k), complex(w)) for (k, w) in rd.components)
            elif rd.transition is not None:
                comps = ((str(rd.transition), 1.0 + 0.0j),)
            else:
                # A 2ph drive could choose to not set transition; fall back to candidates if needed.
                raise ValueError(
                    f"ResolvedDrive {
                        rd.drive_id!r} has no components and no transition"
                )

            # DRIVE terms
            if self.policy.split_components:
                out.extend(
                    self._emit_drive_terms_split(
                        ctx, dims, rd, comps, dc, coherence_terms
                    )
                )
            else:
                out.extend(
                    self._emit_drive_terms_summed(
                        ctx, dims, rd, comps, dc, coherence_terms
                    )
                )

            # DETUNING terms
            if self.policy.include_detuning_terms and rd.detuning is not None:
                out.extend(
                    self._emit_detuning_terms(
                        ctx, dims, rd, comps, projector_terms
                    )
                )

        return out

    # -------------------------------------------------------------------------
    # DRIVE emission
    # -------------------------------------------------------------------------

    def _emit_drive_terms_split(
        self,
        ctx: OpMaterializeContext,
        dims: Sequence[int],
        rd: ResolvedDrive,
        comps: Tuple[Tuple[str, complex], ...],
        dc: DriveCoefficients,
        coherence_terms: Mapping[str, Term],
    ) -> List[HamiltonianTermOut]:
        out: List[HamiltonianTermOut] = []

        for tr_key, w in comps:
            # Build coherence operator A = |upper><lower|
            sym = _coherence_symbol_for_transition(tr_key)

            # Use basis catalog when available (ensures it is drive-allowed), else resolve symbol directly.
            if tr_key in coherence_terms:
                A = _materialize_opexpr(coherence_terms[tr_key].op, ctx, dims)
            else:
                A = np.asarray(ctx.resolve_symbol(sym, dims), dtype=complex)

            Hop = (A + A.conj().T) if self.policy.hermitian_drive else A

            # Choose coefficient channel
            if rd.kind == "2ph":
                base = dc.omega_2ph
                if base is None:
                    raise KeyError(
                        f"Missing omega_2ph for 2ph drive_id={
                            rd.drive_id!r}"
                    )
            else:
                base = dc.omega_by_transition.get(tr_key, None)
                if base is None:
                    raise KeyError(
                        f"Missing omega coefficient for transition {tr_key!r} "
                        f"(drive_id={rd.drive_id!r}). Available: {
                            list(dc.omega_by_transition.keys())}"
                    )

            coeff = _maybe_scale_omega(base, self.policy)

            if self.policy.apply_component_weight:
                coeff = _scale_coeff(coeff, w)

            out.append(
                HamiltonianTermOut(
                    label=f"drive_{rd.drive_id}_{tr_key}",
                    op=Hop,
                    coeff=coeff,
                    meta={
                        "type": "drive_component",
                        "drive_id": rd.drive_id,
                        "kind": rd.kind,
                        "transition": tr_key,
                        "component_weight": complex(w),
                        "candidates": tuple(str(x) for x in rd.candidates),
                        **dict(rd.meta or {}),
                    },
                )
            )

        return out

    def _emit_drive_terms_summed(
        self,
        ctx: OpMaterializeContext,
        dims: Sequence[int],
        rd: ResolvedDrive,
        comps: Tuple[Tuple[str, complex], ...],
        dc: DriveCoefficients,
        coherence_terms: Mapping[str, Term],
    ) -> List[HamiltonianTermOut]:
        # Sum A = sum_i w_i * |upper_i><lower_i|
        A_sum: Optional[np.ndarray] = None
        for tr_key, w in comps:
            sym = _coherence_symbol_for_transition(tr_key)
            if tr_key in coherence_terms:
                A = _materialize_opexpr(coherence_terms[tr_key].op, ctx, dims)
            else:
                A = np.asarray(ctx.resolve_symbol(sym, dims), dtype=complex)

            term = complex(w) * A
            A_sum = term if A_sum is None else (A_sum + term)

        assert A_sum is not None
        Hop = (A_sum + A_sum.conj().T) if self.policy.hermitian_drive else A_sum

        # Coefficient in summed mode must be defined carefully.
        # If you sum operators with weights, the natural choice is to use a single envelope coefficient.
        # We pick:
        #   - omega_2ph for 2ph
        #   - otherwise, the coefficient of the first component transition
        tr0 = comps[0][0]
        if rd.kind == "2ph":
            base = dc.omega_2ph
            if base is None:
                raise KeyError(
                    f"Missing omega_2ph for 2ph drive_id={
                        rd.drive_id!r}"
                )
        else:
            base = dc.omega_by_transition.get(tr0, None)
            if base is None:
                raise KeyError(
                    f"Missing omega coefficient for transition {
                        tr0!r} (drive_id={rd.drive_id!r})"
                )

        coeff = _maybe_scale_omega(base, self.policy)

        return [
            HamiltonianTermOut(
                label=f"drive_{rd.drive_id}_sum",
                op=Hop,
                coeff=coeff,
                meta={
                    "type": "drive_sum",
                    "drive_id": rd.drive_id,
                    "kind": rd.kind,
                    "components": tuple((k, complex(w)) for (k, w) in comps),
                    "candidates": tuple(str(x) for x in rd.candidates),
                    **dict(rd.meta or {}),
                },
            )
        ]

    # -------------------------------------------------------------------------
    # DETUNING emission
    # -------------------------------------------------------------------------

    def _emit_detuning_terms(
        self,
        ctx: OpMaterializeContext,
        dims: Sequence[int],
        rd: ResolvedDrive,
        comps: Tuple[Tuple[str, complex], ...],
        projector_terms: Mapping[str, Term],
    ) -> List[HamiltonianTermOut]:
        # Normalize detuning into CoeffExpr (solver units already)
        det_coeff = _as_coeffexpr_from_detuning(
            rd.detuning
        )  # type: ignore[arg-type]

        # Determine which upper-level projectors to include (deduplicate, preserve order)
        seen: set[str] = set()
        levels: List[str] = []
        for tr_key, _w in comps:
            up = _upper_level_from_transition_key(tr_key)
            if up not in seen:
                seen.add(up)
                levels.append(up)

        out: List[HamiltonianTermOut] = []
        for lvl in levels:
            sym = _projector_symbol_for_level(lvl)

            if lvl in projector_terms:
                P = _materialize_opexpr(projector_terms[lvl].op, ctx, dims)
            else:
                # Fallback to symbol resolution, if you provide it in ctx
                P = np.asarray(ctx.resolve_symbol(sym, dims), dtype=complex)

            out.append(
                HamiltonianTermOut(
                    label=f"detuning_{rd.drive_id}_{lvl}",
                    op=P,
                    coeff=det_coeff,
                    meta={
                        "type": "detuning",
                        "drive_id": rd.drive_id,
                        "level": lvl,
                        "kind": rd.kind,
                        "components": tuple(
                            (k, complex(w)) for (k, w) in comps
                        ),
                        **dict(rd.meta or {}),
                    },
                )
            )

        return out
