from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from smef.core.drives.protocols import (
    DriveDecodeContextProto,
    DriveTermEmitterProto,
)
from smef.core.drives.types import (
    DriveCoefficients,
    DriveTermBundle,
    ResolvedDrive,
)
from smef.core.ir.ops import EmbeddedKron, LocalSymbolOp, OpExpr
from smef.core.ir.terms import Term, TermKind

from bec.quantum_dot.enums import QDState, TransitionPair
from bec.quantum_dot.smef.drives.context import QDDriveDecodeContext


class _ArrayCoeff:
    def __init__(self, values: np.ndarray):
        self._values = np.asarray(values, dtype=complex).reshape(-1)

    def eval(self, tlist: np.ndarray) -> np.ndarray:
        tlist = np.asarray(tlist, dtype=float).reshape(-1)
        if tlist.size != self._values.size:
            raise ValueError(
                "Coeff length mismatch: len(tlist)=%d vs len(values)=%d"
                % (tlist.size, self._values.size)
            )
        return self._values


def _qd_op(qd_index: int, symbol: str) -> OpExpr:
    return OpExpr.atom(
        EmbeddedKron(indices=(qd_index,), locals=(LocalSymbolOp(symbol),))
    )


def _proj_symbol(st: QDState) -> str:
    if st is QDState.G:
        return "proj_G"
    if st is QDState.X1:
        return "proj_X1"
    if st is QDState.X2:
        return "proj_X2"
    if st is QDState.XX:
        return "proj_XX"
    raise KeyError(st)


def _payload_from_ctx(drive_id: Any, ctx: QDDriveDecodeContext) -> Any:
    if drive_id in ctx.meta_drives:
        return ctx.meta_drives[drive_id]
    raise KeyError("Missing drive payload for drive_id=%s" % (drive_id,))


def _sample_omega_L_rad_s(
    payload: Any, t_phys_s: np.ndarray
) -> Optional[np.ndarray]:
    fn = getattr(payload, "omega_L_rad_s", None)
    if not callable(fn):
        return None
    out = np.empty(t_phys_s.size, dtype=float)
    for i in range(t_phys_s.size):
        w = fn(float(t_phys_s[i]))
        if w is None:
            return None
        out[i] = float(w)
    return out


@dataclass
class QDDriveTermEmitter(DriveTermEmitterProto):
    """
    Emits:
      - coherent drive Hamiltonian term: 0.5*(sigma_up + sigma_down) with coeff Omega(t)
      - detuning term in rotating frame:
          H_det(t) = -Delta(t) * P_high
        where Delta(t) depends on kind:
          - 1ph: Delta(t) = omega_L(t) - omega_ref(pair)
          - 2ph: Delta(t) = 2*omega_L(t) - omega_ref(pair)   (omega_L is per-photon)
        All expressed in solver units by multiplying rad/s with time_unit_s.
    """

    def emit_drive_terms(
        self,
        resolved: Sequence[ResolvedDrive],
        coeffs: DriveCoefficients,
        *,
        decode_ctx: Optional[DriveDecodeContextProto] = None,
    ) -> DriveTermBundle:
        if not isinstance(decode_ctx, QDDriveDecodeContext):
            raise TypeError("QDDriveTermEmitter expects QDDriveDecodeContext")

        derived = decode_ctx.derived
        qd_index = 0  # fixed by QDModes

        tlist_solver = np.asarray(coeffs.tlist, dtype=float).reshape(-1)

        # Must be injected by SMEF via with_solver_grid
        time_unit_s = getattr(decode_ctx, "time_unit_s", None)
        if time_unit_s is None:
            raise ValueError(
                "QDDriveDecodeContext.time_unit_s missing. "
                "Ensure SMEF calls decode_ctx.with_solver_grid(tlist=..., time_unit_s=...)."
            )
        time_unit_s_f = float(time_unit_s)
        t_phys_s = time_unit_s_f * tlist_solver

        h_terms: list[Term] = []

        for rd in resolved:
            pair = rd.transition_key
            if not isinstance(pair, TransitionPair):
                continue
            if not derived.t_registry.spec(pair).drive_allowed:
                continue

            # ---- coherent drive term ----
            fwd, bwd = derived.t_registry.directed(
                pair)  # low->high, high->low

            sym_up = "t_" + str(fwd.value)
            sym_dn = "t_" + str(bwd.value)

            op_up = _qd_op(qd_index, sym_up)
            op_dn = _qd_op(qd_index, sym_dn)
            op_drive = OpExpr.scale(
                0.5 + 0.0j, OpExpr.summation((op_up, op_dn))
            )

            key = (rd.drive_id, pair)
            if key not in coeffs.coeffs:
                raise KeyError(
                    "Missing coeffs for drive_id=%s, pair=%s"
                    % (rd.drive_id, pair)
                )
            omega_solver = np.asarray(
                coeffs.coeffs[key], dtype=complex
            ).reshape(-1)

            if omega_solver.size != tlist_solver.size:
                raise ValueError(
                    "Omega length mismatch: len(tlist)=%d vs len(Omega)=%d"
                    % (tlist_solver.size, omega_solver.size)
                )

            h_terms.append(
                Term(
                    kind=TermKind.H,
                    op=op_drive,
                    coeff=_ArrayCoeff(omega_solver),
                    label="H_drive_%s_%s" % (rd.drive_id, str(pair.value)),
                    meta=dict(rd.meta),
                )
            )

            # ---- detuning term (optional; requires carrier) ----
            payload = _payload_from_ctx(rd.drive_id, decode_ctx)
            omega_L = _sample_omega_L_rad_s(payload, t_phys_s)
            if omega_L is None:
                continue

            # Reference transition frequency (rad/s). Prefer decoder-provided value.
            omega_ref = rd.meta.get("omega_ref_rad_s")
            if omega_ref is None:
                omega_ref = float(derived.omega_ref_rad_s(pair))
            omega_ref_f = float(omega_ref)

            kind = rd.meta.get("kind", "1ph")
            mult = 2.0 if kind == "2ph" else 1.0

            detuning_rad_s = (mult * omega_L) - omega_ref_f
            detuning_solver = (-detuning_rad_s * time_unit_s_f).astype(complex)

            # projector onto the "high" state of the driven pair
            src, dst = derived.t_registry.endpoints(fwd)  # src(low)->dst(high)
            P_high = _qd_op(qd_index, _proj_symbol(dst))

            h_terms.append(
                Term(
                    kind=TermKind.H,
                    op=P_high,
                    coeff=_ArrayCoeff(detuning_solver),
                    label="H_det_%s_%s" % (rd.drive_id, str(pair.value)),
                    meta={
                        **dict(rd.meta),
                        "omega_ref_rad_s": omega_ref_f,
                        "detuning_mult": mult,
                    },
                )
            )

        return DriveTermBundle(h_terms=tuple(h_terms))
