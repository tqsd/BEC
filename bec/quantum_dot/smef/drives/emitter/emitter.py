from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

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

from bec.quantum_dot.enums import QDState, TransitionPair
from bec.quantum_dot.smef.drives.context import QDDriveDecodeContext
from bec.quantum_dot.smef.drives.emitter.drive_term import build_drive_h_term
from bec.quantum_dot.smef.drives.emitter.eid_terms import (
    build_eid_c_term_phenom,
    build_eid_c_term_polaron,
)
from bec.quantum_dot.smef.drives.emitter.frame_solver import (
    FrameConstraint,
    solve_state_energies_ls,
)
from bec.quantum_dot.smef.drives.emitter.frame_terms import build_frame_h_terms
from bec.quantum_dot.smef.drives.emitter.polaron_scattering_terms import (
    build_polaron_scattering_c_terms,
)
from bec.quantum_dot.smef.drives.emitter.sampling import (
    payload_from_ctx,
    sample_omega_L_rad_s,
)


def _eid_scale_from_derived(derived: Any) -> float:
    val = getattr(derived, "gamma_phi_eid_scale", 0.0)
    try:
        return float(val or 0.0)
    except Exception:
        return 0.0


def _polaron_rates_from_derived(derived: Any) -> Any:
    po = getattr(derived, "phonon_outputs", None)
    if po is None:
        return None
    return getattr(po, "polaron_rates", None)


@dataclass
class QDDriveTermEmitter(DriveTermEmitterProto):
    """
    Emits:
      - H_drive per resolved (drive_id, pair)
      - H_frame (rotating frame diagonal) once per emit call, constructed from all detuning constraints
      - L_eid per resolved (drive_id, pair) (edge-based), unchanged
    """

    qd_index: int = 0  # fixed by QDModes layout

    def emit_drive_terms(
        self,
        resolved: Sequence[ResolvedDrive],
        coeffs: DriveCoefficients,
        *,
        decode_ctx: DriveDecodeContextProto | None = None,
    ) -> DriveTermBundle:
        if not isinstance(decode_ctx, QDDriveDecodeContext):
            raise TypeError("QDDriveTermEmitter expects QDDriveDecodeContext")

        derived = decode_ctx.derived
        qd_index = int(self.qd_index)

        tlist_solver = np.asarray(coeffs.tlist, dtype=float).reshape(-1)

        time_unit_s = getattr(decode_ctx, "time_unit_s", None)
        if time_unit_s is None:
            raise ValueError(
                "QDDriveDecodeContext.time_unit_s missing. "
                "Ensure SMEF calls decode_ctx.with_solver_grid(tlist=..., time_unit_s=...)."
            )

        s = float(time_unit_s)
        if s <= 0.0:
            raise ValueError("time_unit_s must be > 0")

        t_phys_s = s * tlist_solver

        # eid_scale = _eid_scale_from_derived(derived)
        eid_scale = float(getattr(derived, "eid_calibration", 1.0))
        eid_enabled = bool(getattr(derived, "eid_enabled", False))
        polaron_rates = _polaron_rates_from_derived(derived)

        h_terms: list[Any] = []
        c_terms: list[Any] = []

        frame_constraints: list[FrameConstraint] = []

        for rd in resolved:
            pair = rd.transition_key
            if not isinstance(pair, TransitionPair):
                continue
            if not derived.t_registry.spec(pair).drive_allowed:
                continue

            fwd, bwd = derived.t_registry.directed(pair)
            src, dst = derived.t_registry.endpoints(fwd)

            key = (rd.drive_id, pair)
            if key not in coeffs.coeffs:
                raise KeyError(
                    "Missing coeffs for drive_id=%s, pair=%s"
                    % (str(rd.drive_id), str(pair))
                )

            omega_solver = np.asarray(
                coeffs.coeffs[key], dtype=complex
            ).reshape(-1)
            if omega_solver.size != tlist_solver.size:
                raise ValueError(
                    "Omega length mismatch: len(tlist)=%d vs len(Omega)=%d"
                    % (tlist_solver.size, omega_solver.size)
                )

            # 1) coherent drive term (unchanged)
            h_terms.append(
                build_drive_h_term(
                    qd_index=qd_index,
                    drive_id=rd.drive_id,
                    pair=pair,
                    fwd=fwd,
                    bwd=bwd,
                    omega_solver=omega_solver,
                    meta=dict(rd.meta),
                )
            )

            # 2) detuning: DO NOT emit per-edge Hamiltonian.
            # Instead collect constraint eps_dst - eps_src = -Delta_edge(t)
            detuning_rad_s = None
            payload = payload_from_ctx(rd.drive_id, decode_ctx)
            omega_L = sample_omega_L_rad_s(payload, t_phys_s)

            if omega_L is not None:
                omega_ref = rd.meta.get("omega_ref_rad_s")
                if omega_ref is None:
                    omega_ref = float(derived.omega_ref_rad_s(pair))
                omega_ref_f = float(omega_ref)

                kind = rd.meta.get("kind", "1ph")
                mult = 2.0 if str(kind) == "2ph" else 1.0

                detuning_rad_s = (mult * omega_L) - omega_ref_f

                frame_constraints.append(
                    FrameConstraint(
                        src=src,
                        dst=dst,
                        rhs_rad_s=(-1.0)
                        * np.asarray(detuning_rad_s, dtype=float).reshape(-1),
                        meta={
                            **dict(rd.meta),
                            "drive_id": rd.drive_id,
                            "pair": (
                                pair.value
                                if hasattr(pair, "value")
                                else str(pair)
                            ),
                            "omega_ref_rad_s": float(omega_ref_f),
                            "detuning_mult": float(mult),
                        },
                    )
                )

            # Only emit EID if explicitly enabled

            print("eid_enabled attr:", getattr(derived, "eid_enabled", None))
            print("has phonon_outputs:", hasattr(derived, "phonon_outputs"))
            print("DETUNING RAD S", detuning_rad_s)
            print("DETUNIGN SCALE", eid_scale)
            if hasattr(derived, "phonon_outputs"):
                po = derived.phonon_outputs
                print(
                    "po.eid.enabled:",
                    getattr(getattr(po, "eid", None), "enabled", None),
                )
                print(
                    "po.polaron_rates is None:",
                    getattr(po, "polaron_rates", None) is None,
                )
            if (
                eid_enabled
                and (detuning_rad_s is not None)
                and (eid_scale > 0.0)
            ):
                print("ADDING EID C TERM POLARON")
                eid_term = build_eid_c_term_polaron(
                    qd_index=qd_index,
                    drive_id=rd.drive_id,
                    pair=pair,
                    dst_proj_state=dst,
                    src_proj_state=src,
                    omega_solver=omega_solver,
                    detuning_rad_s=detuning_rad_s,
                    time_unit_s=s,
                    polaron_rates=polaron_rates,
                    scale=eid_scale,
                    meta=dict(rd.meta),
                )

                # optional fallback: only when enabled
                if eid_term is None:
                    eid_term = build_eid_c_term_phenom(
                        qd_index=qd_index,
                        drive_id=rd.drive_id,
                        pair=pair,
                        src_proj_state=src,
                        dst_proj_state=dst,
                        omega_solver=omega_solver,
                        eid_scale=eid_scale,
                        meta=dict(rd.meta),
                    )

                if eid_term is not None:
                    c_terms.append(eid_term)
                    eid_term = None

            if detuning_rad_s is not None:
                sc_terms = build_polaron_scattering_c_terms(
                    qd_index=qd_index,
                    drive_id=rd.drive_id,
                    pair=pair,
                    dst_state=dst,
                    src_state=src,
                    omega_solver=omega_solver,
                    detuning_rad_s=detuning_rad_s,
                    time_unit_s=s,
                    polaron_rates=polaron_rates,
                    scale=eid_scale,  # consider a separate knob later
                    meta=dict(rd.meta),
                    b_polaron=1.0,  # IMPORTANT: omega already contains B
                    Nt=4096,
                )
                c_terms.extend(sc_terms)

        # After collecting all constraints, emit ONE consistent rotating-frame diagonal Hamiltonian.
        if frame_constraints:
            states = (QDState.G, QDState.X1, QDState.X2, QDState.XX)
            eps_by_state = solve_state_energies_ls(
                states=states,
                constraints=frame_constraints,
                gauge_state=QDState.G,
            )

            h_terms.extend(
                build_frame_h_terms(
                    qd_index=qd_index,
                    eps_rad_s_by_state=eps_by_state,
                    time_unit_s=s,
                    label_prefix="H_frame",
                    meta={
                        "source": "rotating_frame",
                        "n_constraints": len(frame_constraints),
                    },
                )
            )

        return DriveTermBundle(h_terms=tuple(h_terms), c_terms=tuple(c_terms))
