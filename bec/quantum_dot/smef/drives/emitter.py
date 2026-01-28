from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from smef.core.drives.protocols import (
    DriveDecodeContextProto,
    DriveTermEmitterProto,
)
from smef.core.drives.types import (
    DriveCoefficients,
    DriveTermBundle,
    ResolvedDrive,
)
from smef.core.ir.coeffs import CallableCoeff
from smef.core.ir.ops import EmbeddedKron, LocalSymbolOp, OpExpr
from smef.core.ir.terms import Term, TermKind

from bec.quantum_dot.enums import Transition, TransitionPair
from bec.quantum_dot.smef.drives.context import QDDriveDecodeContext


def _symbol_for_transition(tr: Transition) -> str:
    # Convention used in your examples: Transition.SRC_DST corresponds to operator |DST><SRC|
    # Symbol key in materializer is "t_DST_SRC"
    src, dst = tr.value.split("_", 1)
    return f"t_{dst}_{src}"


@dataclass
class QDDriveTermEmitter(DriveTermEmitterProto):
    """
    Emit Hamiltonian drive terms from (ResolvedDrive, coeffs).

    Builds H_drive = 0.5 * (|high><low| + |low><high|) with time-dependent coeff Omega(t).
    """

    def emit_drive_terms(
        self,
        resolved: Sequence[ResolvedDrive],
        coeffs: DriveCoefficients,
        *,
        decode_ctx: Optional[DriveDecodeContextProto] = None,
    ) -> DriveTermBundle:
        if decode_ctx is None or not isinstance(
            decode_ctx, QDDriveDecodeContext
        ):
            raise TypeError("QDDriveTermEmitter expects QDDriveDecodeContext")

        derived = decode_ctx.derived
        qd_index = 0  # QDModes keeps qd at index 0; keep consistent

        h_terms = []

        for rd in resolved:
            pair = rd.transition_key
            if not isinstance(pair, TransitionPair):
                continue

            # Only emit coherent drive terms for drive-allowed families
            if not derived.t_registry.spec(pair).drive_allowed:
                continue

            fwd, bwd = derived.t_registry.directed(
                pair
            )  # fwd: low->high, bwd: high->low
            sym_up = _symbol_for_transition(fwd)
            sym_dn = _symbol_for_transition(bwd)

            op_up = OpExpr.atom(
                EmbeddedKron(
                    indices=(qd_index,), locals=(LocalSymbolOp(sym_up),)
                )
            )
            op_dn = OpExpr.atom(
                EmbeddedKron(
                    indices=(qd_index,), locals=(LocalSymbolOp(sym_dn),)
                )
            )
            op = OpExpr.scale(0.5 + 0.0j, OpExpr.summation((op_up, op_dn)))

            key = (rd.drive_id, pair)
            if key not in coeffs.coeffs:
                raise KeyError(
                    f"Missing coeffs for drive_id={
                               rd.drive_id}, pair={pair}"
                )

            y = coeffs.coeffs[key]

            # Wrap sampled array into a callable coeff
            def _make_fn(arr):
                def _fn(t):
                    # t is the solver grid passed by SMEF; we assume it matches coeffs.tlist
                    # This is fine for now; later we can add interpolation.
                    return arr

                return _fn

            h_terms.append(
                Term(
                    kind=TermKind.H,
                    op=op,
                    coeff=CallableCoeff(_make_fn(y)),
                    label=f"H_drive_{rd.drive_id}_{pair.value}",
                    meta=dict(rd.meta),
                )
            )

        return DriveTermBundle(h_terms=tuple(h_terms))
