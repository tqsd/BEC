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
from smef.core.units import hbar as _hbar, kB as _kB, Q

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


def _coth(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    ax = np.abs(x)
    small = ax < 1e-6
    out[small] = (1.0 / x[small]) + (x[small] / 3.0)
    out[~small] = 1.0 / np.tanh(x[~small])
    return out


def _polaron_eid_gamma_phys_1_s(
    *,
    omega_rabi_rad_s: np.ndarray,
    s2: float,
    alpha_s2: float,
    omega_c_rad_s: float,
    temperature_K: float,
) -> np.ndarray:
    Omega = np.asarray(omega_rabi_rad_s)
    Omega2 = np.abs(Omega) ** 2

    if alpha_s2 <= 0.0 or omega_c_rad_s <= 0.0 or s2 <= 0.0:
        return np.zeros_like(Omega2, dtype=float)

    if temperature_K <= 0.0:
        coth_fac = 1.0
    else:
        eta_q = (_hbar * Q(float(omega_c_rad_s), "rad/s")) / (
            2.0 * _kB * Q(float(temperature_K), "K")
        )
        eta = float(eta_q.to_base_units().magnitude)
        coth_fac = float(_coth(np.asarray([eta], dtype=float))[0])

    wc = abs(float(omega_c_rad_s))
    wc2 = wc * wc
    wc3 = wc2 * wc

    sat = Omega2 / (wc2 + Omega2 + 1e-30)

    gamma = float(s2) * float(alpha_s2) * wc3 * float(coth_fac) * sat
    gamma = np.asarray(gamma, dtype=float)
    gamma[~np.isfinite(gamma)] = 0.0
    gamma[gamma < 0.0] = 0.0
    return gamma


def _qd_op(qd_index: int, symbol: str) -> OpExpr:
    return OpExpr.atom(
        EmbeddedKron(indices=(qd_index,), locals=(LocalSymbolOp(symbol),))
    )


def _abs2(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=complex).reshape(-1)
    re = np.asarray(np.real(z), dtype=float)
    im = np.asarray(np.imag(z), dtype=float)
    return (re * re) + (im * im)


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


def _eid_scale_from_derived(derived: Any) -> float:
    """
    Drive-dependent phonon dephasing "scale" for the simple EID model:
        gamma_eid_solver(t) = scale * |Omega_solver(t)|^2

    Put this on DerivedQD (via a mixin) as:
        derived.gamma_phi_eid_scale -> float

    If missing, defaults to 0.0 (disabled).
    """
    return float(getattr(derived, "gamma_phi_eid_scale", 0.0) or 0.0)


@dataclass
class QDDriveTermEmitter(DriveTermEmitterProto):
    """
    Emits per resolved drive (TransitionPair):

    1) Coherent drive term:
         H_drive(t) = 0.5 * Omega(t) * (sigma_up + sigma_down)

       In SMEF IR form: op = 0.5*(t_fwd + t_bwd), coeff = Omega_solver(t)

    2) Optional detuning term (if payload provides omega_L_rad_s):
         H_det(t) = -(0.5 * Delta(t)) * (P_high - P_low)
       Delta(t) = mult*omega_L(t) - omega_ref(pair)
       mult = 1 for 1ph, 2 for 2ph (omega_L is per-photon in the 2ph case)

    3) Optional drive-dependent phonon dephasing (EID) if scale>0:
         L_eid(t) = sqrt(gamma_eid(t)) * P_high
         gamma_eid_solver(t) = scale * |Omega_solver(t)|^2

    Note: coefficients are arrays on solver tlist (unitless).
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

        time_unit_s = getattr(decode_ctx, "time_unit_s", None)
        if time_unit_s is None:
            raise ValueError(
                "QDDriveDecodeContext.time_unit_s missing. "
                "Ensure SMEF calls decode_ctx.with_solver_grid(tlist=..., time_unit_s=...)."
            )
        time_unit_s_f = float(time_unit_s)
        t_phys_s = time_unit_s_f * tlist_solver

        eid_scale = _eid_scale_from_derived(derived)

        h_terms: list[Term] = []
        c_terms: list[Term] = []

        for rd in resolved:
            pair = rd.transition_key
            if not isinstance(pair, TransitionPair):
                continue
            if not derived.t_registry.spec(pair).drive_allowed:
                continue

            # Directed transitions: low->high and high->low
            fwd, bwd = derived.t_registry.directed(pair)

            # ---- Omega(t) from strength model ----
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

            # ---- coherent drive term ----
            sym_up = "t_" + str(fwd.value)
            sym_dn = "t_" + str(bwd.value)
            op_up = _qd_op(qd_index, sym_up)
            op_dn = _qd_op(qd_index, sym_dn)
            op_drive = OpExpr.scale(
                0.5 + 0.0j, OpExpr.summation((op_up, op_dn))
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

            # ---- EID collapse term (optional) ----
            # Priority:
            # 1) phenomenological eid_scale if set (backwards compatible)
            # 2) polaron-based EID if enabled in derived.phonon_outputs.polaron_eid
            src, dst = derived.t_registry.endpoints(fwd)  # low->high
            P_high = _qd_op(qd_index, _proj_symbol(dst))

            used_eid = False

            if eid_scale > 0.0:
                gamma_solver = float(eid_scale) * _abs2(omega_solver)
                sqrt_gamma = np.sqrt(np.maximum(gamma_solver, 0.0)).astype(
                    complex
                )

                c_terms.append(
                    Term(
                        kind=TermKind.C,
                        op=P_high,
                        coeff=_ArrayCoeff(sqrt_gamma),
                        label="L_eid_%s_%s" % (rd.drive_id, str(pair.value)),
                        meta={
                            **dict(rd.meta),
                            "kind": "phonon_eid",
                            "model": "phenomenological",
                            "scale": float(eid_scale),
                        },
                    )
                )
                used_eid = True

            if not used_eid:
                po = getattr(derived, "phonon_outputs", None)
                pol = (
                    getattr(po, "polaron_eid", None) if po is not None else None
                )
                if pol is not None and bool(getattr(pol, "enabled", False)):
                    # Convert Omega from solver units back to physical rad/s:
                    # omega_solver(t) = Omega_phys(t) * time_unit_s
                    # => Omega_phys(t) = omega_solver(t) / time_unit_s
                    # Omega_phys = (np.real(omega_solver).astype(float)) / float(
                    #     time_unit_s_f
                    # )

                    Omega_phys = np.abs(omega_solver) / float(time_unit_s_f)

                    # transition coupling strength
                    # use derived.polaron_B(tr) exists, but we want s2 = (phi_i - phi_j)^2
                    # If you add derived.phonon_model.s2_for_transition later, use it.
                    # For now: infer via phonon params on qd.
                    qd_ph = getattr(
                        getattr(derived, "qd", None), "phonons", None
                    )
                    if qd_ph is None:
                        s2 = 1.0
                    else:

                        def phi_for_state(st):
                            if st is QDState.G:
                                return float(getattr(qd_ph, "phi_G", 0.0))
                            if st in (QDState.X1, QDState.X2):
                                return float(getattr(qd_ph, "phi_X", 0.0))
                            if st is QDState.XX:
                                return float(getattr(qd_ph, "phi_XX", 0.0))
                            return 0.0

                        phi_i = phi_for_state(src)
                        phi_j = phi_for_state(dst)
                        dphi = phi_i - phi_j
                        s2 = float(dphi * dphi)

                    gamma_phys_1_s = _polaron_eid_gamma_phys_1_s(
                        omega_rabi_rad_s=Omega_phys,
                        s2=s2,
                        alpha_s2=float(getattr(pol, "alpha_s2", 0.0)),
                        omega_c_rad_s=float(getattr(pol, "omega_c_rad_s", 0.0)),
                        temperature_K=float(getattr(pol, "temperature_K", 0.0)),
                    )

                    # Convert physical rate (1/s) to solver rate (1/solver_time_unit):
                    gamma_solver = gamma_phys_1_s * float(time_unit_s_f)
                    sqrt_gamma = np.sqrt(np.maximum(gamma_solver, 0.0)).astype(
                        complex
                    )

                    c_terms.append(
                        Term(
                            kind=TermKind.C,
                            op=P_high,
                            coeff=_ArrayCoeff(sqrt_gamma),
                            label="L_eid_%s_%s"
                            % (rd.drive_id, str(pair.value)),
                            meta={
                                **dict(rd.meta),
                                "kind": "phonon_eid",
                                "model": "polaron_simple",
                                "s2": float(s2),
                                "alpha_s2": float(
                                    getattr(pol, "alpha_s2", 0.0)
                                ),
                                "omega_c_rad_s": float(
                                    getattr(pol, "omega_c_rad_s", 0.0)
                                ),
                                "temperature_K": float(
                                    getattr(pol, "temperature_K", 0.0)
                                ),
                            },
                        )
                    )
            # ---- detuning term (optional; requires omega_L) ----
            payload = _payload_from_ctx(rd.drive_id, decode_ctx)
            omega_L = _sample_omega_L_rad_s(payload, t_phys_s)
            if omega_L is None:
                continue

            omega_ref = rd.meta.get("omega_ref_rad_s")
            if omega_ref is None:
                omega_ref = float(derived.omega_ref_rad_s(pair))
            omega_ref_f = float(omega_ref)

            kind = rd.meta.get("kind", "1ph")
            mult = 2.0 if kind == "2ph" else 1.0

            detuning_rad_s = (mult * omega_L) - omega_ref_f

            detuning_solver = (-(0.5 * detuning_rad_s) * time_unit_s_f).astype(
                complex
            )

            src, dst = derived.t_registry.endpoints(fwd)
            P_high = _qd_op(qd_index, _proj_symbol(dst))
            P_low = _qd_op(qd_index, _proj_symbol(src))
            op_det = OpExpr.summation(
                (P_high, OpExpr.scale(-1.0 + 0.0j, P_low))
            )

            h_terms.append(
                Term(
                    kind=TermKind.H,
                    op=op_det,
                    coeff=_ArrayCoeff(detuning_solver),
                    label="H_det_%s_%s" % (rd.drive_id, str(pair.value)),
                    meta={
                        **dict(rd.meta),
                        "omega_ref_rad_s": omega_ref_f,
                        "detuning_mult": mult,
                    },
                )
            )

        return DriveTermBundle(h_terms=tuple(h_terms), c_terms=tuple(c_terms))
