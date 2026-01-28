from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from smef.core.drives.protocols import (
    DriveDecoderProto,
    DriveDecodeContextProto,
)
from smef.core.drives.types import DriveSpec, ResolvedDrive

from bec.quantum_dot.enums import TransitionPair
from bec.quantum_dot.smef.drives.context import (
    QDDriveDecodeContext,
    DecodePolicy,
)


def _estimate_sigma_omega_fallback(
    tlist: np.ndarray, time_unit_s: float
) -> float:
    t_phys = float(time_unit_s) * np.asarray(tlist, dtype=float)
    tspan = float(np.max(t_phys) - np.min(t_phys))
    if tspan <= 0.0:
        return 0.0
    return 2.0 / tspan


def _laser_omega_samples(
    payload: Any,
    *,
    tlist_solver: np.ndarray,
    time_unit_s: float,
    n: int,
) -> np.ndarray:
    tlist_solver = np.asarray(tlist_solver, dtype=float)
    if tlist_solver.size == 0:
        raise ValueError("tlist is empty")

    n = int(max(1, n))
    if n == 1:
        idxs = [int(tlist_solver.size // 2)]
    else:
        idxs = np.linspace(0, tlist_solver.size - 1, n, dtype=int).tolist()

    s = float(time_unit_s)
    out: List[float] = []
    for i in idxs:
        t_phys = s * float(tlist_solver[i])
        w = (
            payload.omega_L_rad_s(t_phys)
            if hasattr(payload, "omega_L_rad_s")
            else None
        )
        if w is None:
            raise ValueError(
                "Drive carrier is None: cannot decode without omega_L."
            )
        out.append(float(w))
    return np.asarray(out, dtype=float)


def _coupling_mag(
    derived: Any, pair: TransitionPair, E: Optional[np.ndarray]
) -> float:
    if E is None:
        return 1.0
    fwd, _ = derived.t_registry.directed(pair)
    try:
        c = derived.drive_projection(fwd, E)
    except Exception:
        return 1.0
    return float(abs(c))


def _penalized_score(
    detuning_min: float,
    *,
    coupling_mag: float,
    sigma_omega: float,
    pol_gate_eps: float,
    pol_penalty_power: float,
    pol_penalty_weight: float,
) -> float:
    if coupling_mag >= 1.0:
        return float(detuning_min)

    eps = float(pol_gate_eps)
    c_eff = max(float(coupling_mag), eps)

    p = float(pol_penalty_power)
    w = float(pol_penalty_weight)

    penalty = w * float(sigma_omega) * ((1.0 / (c_eff**p)) - 1.0)
    return float(detuning_min) + float(penalty)


class QDDriveDecoder(DriveDecoderProto):
    def decode(
        self,
        specs: Sequence[DriveSpec],
        *,
        ctx: Optional[DriveDecodeContextProto] = None,
    ) -> Sequence[ResolvedDrive]:
        if ctx is None or not isinstance(ctx, QDDriveDecodeContext):
            raise TypeError("QDDriveDecoder requires QDDriveDecodeContext")

        out: List[ResolvedDrive] = []
        for spec in specs:
            out.extend(self._decode_one(spec, ctx=ctx))

        for spec in specs:
            if spec.drive_id is None:
                raise ValueError(
                    "DriveSpec.drive_id must be set for QD pipeline"
                )
            ctx.meta_drives[spec.drive_id] = spec.payload
        return tuple(out)

    def _decode_one(
        self, spec: DriveSpec, *, ctx: QDDriveDecodeContext
    ) -> List[ResolvedDrive]:
        drive = spec.payload  # <-- THIS IS THE KEY CHANGE (no meta_drives)
        drive_id = (
            spec.drive_id
            if spec.drive_id is not None
            else getattr(drive, "label", None) or f"drive_{id(drive)}"
        )

        # Use ctx.derived transitions to score candidates
        derived = ctx.derived
        targets = derived.drive_targets()

        # polarization vector (dimensionless HV)
        E = getattr(drive, "effective_pol", None)
        E = E() if callable(E) else None

        # get a carrier omega for scoring (physical rad/s float)
        # (keep this simple for now)
        wL = None
        if hasattr(drive, "omega_L_rad_s"):
            wL = drive.omega_L_rad_s(0.0)  # you can improve this later

        # If no carrier, you can still allow preferred_kind filtering etc.
        # For now: just pick best by omega_ref if possible
        scored = []
        for pair in targets:
            kind = derived.drive_kind(pair)  # "1ph" or "2ph"
            omega_ref = derived.omega_ref_rad_s(pair)

            if wL is None:
                score = float("inf")
            else:
                if kind == "1ph":
                    score = abs(float(wL) - float(omega_ref))
                else:
                    score = abs(2.0 * float(wL) - float(omega_ref))

            scored.append((score, pair, kind, omega_ref))

        scored.sort(key=lambda x: float(x[0]))
        best = scored[0]
        _, pair, kind, omega_ref = best

        return [
            ResolvedDrive(
                drive_id=drive_id,
                transition_key=pair,
                carrier_omega_rad_s=float(omega_ref),
                t_s=None,
                envelope=None,
                meta={
                    "kind": kind,
                    "preferred_kind": getattr(drive, "preferred_kind", None),
                },
            )
        ]
