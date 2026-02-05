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

from bec.quantum_dot.enums import TransitionPair
from bec.quantum_dot.smef.drives.context import QDDriveDecodeContext
from bec.quantum_dot.smef.drives.emitter.detuning_term import (
    build_detuning_h_term,
)
from bec.quantum_dot.smef.drives.emitter.drive_term import build_drive_h_term
from bec.quantum_dot.smef.drives.emitter.eid_terms import (
    build_eid_c_term_phenom,
    build_eid_c_term_polaron,
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
    """
    Returns derived.phonon_outputs.polaron_rates if present, else None.
    Kept permissive so derived implementations can evolve without breaking.
    """
    po = getattr(derived, "phonon_outputs", None)
    if po is None:
        return None
    return getattr(po, "polaron_rates", None)


@dataclass
class QDDriveTermEmitter(DriveTermEmitterProto):
    """
    Glue emitter that assembles terms using the small builder modules.

    Emits (per ResolvedDrive / TransitionPair):
      - H_drive always
      - H_det optionally (if payload.omega_L_rad_s(t) exists)
      - L_eid optionally:
          * prefer polaron-shaped EID if detuning exists and polaron_rates enabled
          * else fall back to phenomenological EID if eid_scale > 0
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

        eid_scale = _eid_scale_from_derived(derived)
        polaron_rates = _polaron_rates_from_derived(derived)

        h_terms: list[Any] = []
        c_terms: list[Any] = []

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

            # 1) coherent drive term
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

            # 2) optional detuning term
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

                h_terms.append(
                    build_detuning_h_term(
                        qd_index=qd_index,
                        drive_id=rd.drive_id,
                        pair=pair,
                        src=src,
                        dst=dst,
                        detuning_rad_s=detuning_rad_s,
                        time_unit_s=s,
                        meta={
                            **dict(rd.meta),
                            "omega_ref_rad_s": omega_ref_f,
                            "detuning_mult": float(mult),
                        },
                    )
                )

            # 3) optional EID collapse (prefer polaron-shaped if possible)
            eid_term = None

            if detuning_rad_s is not None:
                eid_term = build_eid_c_term_polaron(
                    qd_index=qd_index,
                    drive_id=rd.drive_id,
                    pair=pair,
                    dst_proj_state=dst,
                    omega_solver=omega_solver,
                    detuning_rad_s=detuning_rad_s,
                    time_unit_s=s,
                    polaron_rates=polaron_rates,
                    scale=eid_scale,
                    meta=dict(rd.meta),
                )

            if eid_term is None:
                eid_term = build_eid_c_term_phenom(
                    qd_index=qd_index,
                    drive_id=rd.drive_id,
                    pair=pair,
                    dst_proj_state=dst,
                    omega_solver=omega_solver,
                    eid_scale=eid_scale,
                    meta=dict(rd.meta),
                )

            if eid_term is not None:
                c_terms.append(eid_term)

        return DriveTermBundle(h_terms=tuple(h_terms), c_terms=tuple(c_terms))
