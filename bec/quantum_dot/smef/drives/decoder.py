from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    # crude: "a few cycles over the window"
    return 2.0 / tspan


def _laser_omega_samples(
    payload: Any,
    *,
    tlist_solver: np.ndarray,
    time_unit_s: float,
    n: int,
) -> np.ndarray:
    tlist_solver = np.asarray(tlist_solver, dtype=float).reshape(-1)
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
        fn = getattr(payload, "omega_L_rad_s", None)
        w = fn(t_phys) if callable(fn) else None
        if w is None:
            raise ValueError(
                "Drive carrier is None: cannot decode without omega_L_rad_s(t)."
            )
        out.append(float(w))
    return np.asarray(out, dtype=float)


def _effective_pol(payload: Any) -> Optional[np.ndarray]:
    fn = getattr(payload, "effective_pol", None)
    if callable(fn):
        v = fn()
        if v is None:
            return None
        return np.asarray(v, dtype=complex).reshape(2)
    return None


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
    # smaller coupling -> increase effective "distance"
    if coupling_mag >= 1.0:
        return float(detuning_min)

    eps = float(pol_gate_eps)
    c_eff = max(float(coupling_mag), eps)

    p = float(pol_penalty_power)
    w = float(pol_penalty_weight)

    penalty = w * float(sigma_omega) * ((1.0 / (c_eff**p)) - 1.0)
    return float(detuning_min) + float(penalty)


def _detuning_for_pair(
    kind: str, omega_L_rad_s: float, omega_ref_rad_s: float
) -> float:
    # "distance to resonance" in rad/s used only for scoring/selection.
    # For 2ph: compare 2 * omega_L to omega_ref (G<->XX energy gap).
    if kind == "2ph":
        return abs(2.0 * float(omega_L_rad_s) - float(omega_ref_rad_s))
    return abs(float(omega_L_rad_s) - float(omega_ref_rad_s))


# ------------------------
# Decoder
# ------------------------


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
            # store payload first (so later stages can always find it)
            if spec.drive_id is None:
                raise ValueError(
                    "DriveSpec.drive_id must be set for QD pipeline"
                )
            ctx.meta_drives[spec.drive_id] = spec.payload

            out.extend(self._decode_one(spec, ctx=ctx))

        return tuple(out)

    def _decode_one(
        self, spec: DriveSpec, *, ctx: QDDriveDecodeContext
    ) -> List[ResolvedDrive]:
        drive = spec.payload
        if spec.drive_id is None:
            raise ValueError("DriveSpec.drive_id must be set for QD pipeline")
        drive_id = spec.drive_id

        derived = ctx.derived
        policy = ctx.policy

        # candidate transition families
        targets = list(derived.drive_targets())
        if not targets:
            raise ValueError(
                "No drive targets available from derived.drive_targets()."
            )

        # optional user hint
        preferred_kind = getattr(drive, "preferred_kind", None)
        if preferred_kind in ("1ph", "2ph"):
            targets = [
                p for p in targets if derived.drive_kind(p) == preferred_kind
            ] or targets

        # polarization for penalty
        E = _effective_pol(drive)

        # require solver grid for chirp-safe decode
        if ctx.tlist_solver is None or ctx.time_unit_s is None:
            raise ValueError(
                "QDDriveDecodeContext is missing solver grid. "
                "Ensure SMEF calls ctx.with_solver_grid(tlist=..., time_unit_s=...)."
            )
        tlist_solver = np.asarray(ctx.tlist_solver, dtype=float).reshape(-1)
        time_unit_s = float(ctx.time_unit_s)

        # estimate drive carrier omega_L(t) samples
        w_samples = _laser_omega_samples(
            drive,
            tlist_solver=tlist_solver,
            time_unit_s=time_unit_s,
            n=int(policy.sample_points),
        )
        wL_med = float(np.median(w_samples))
        detuning_min_over_time = float(
            np.min(np.abs(w_samples - wL_med))
        )  # small helper; not essential

        # bandwidth estimate
        sigma_omega = (
            float(ctx.bandwidth_sigma_omega_rad_s)
            if ctx.bandwidth_sigma_omega_rad_s is not None
            else 0.0
        )
        if sigma_omega <= 0.0:
            sigma_omega = _estimate_sigma_omega_fallback(
                tlist_solver, time_unit_s
            )
        # score all candidates
        scored: List[Tuple[float, TransitionPair, str, float, float, float]] = (
            []
        )
        for pair in targets:
            kind = derived.drive_kind(pair)  # "1ph" or "2ph"
            omega_ref = float(derived.omega_ref_rad_s(pair))

            # detuning vs resonance (use median carrier for scoring)
            d = _detuning_for_pair(kind, wL_med, omega_ref)

            # polarization overlap penalty
            c_mag = _coupling_mag(derived, pair, E)
            if float(policy.min_coupling_mag) > 0.0 and c_mag < float(
                policy.min_coupling_mag
            ):
                continue

            score = _penalized_score(
                d,
                coupling_mag=c_mag,
                sigma_omega=sigma_omega,
                pol_gate_eps=float(policy.pol_gate_eps),
                pol_penalty_power=float(policy.pol_penalty_power),
                pol_penalty_weight=float(policy.pol_penalty_weight),
            )

            scored.append(
                (float(score), pair, kind, omega_ref, float(c_mag), float(d))
            )

        if not scored:
            raise ValueError(
                "All drive targets were filtered out (coupling gate or missing data)."
            )

        scored.sort(key=lambda x: float(x[0]))
        best_score = float(scored[0][0])

        # optionally select multiple close-by targets
        chosen = [scored[0]]
        if bool(policy.allow_multi):
            thr = best_score + float(policy.k_bandwidth) * float(sigma_omega)
            for item in scored[1:]:
                if float(item[0]) <= thr:
                    chosen.append(item)

        out: List[ResolvedDrive] = []
        for score, pair, kind, omega_ref, c_mag, d0 in chosen:
            out.append(
                ResolvedDrive(
                    drive_id=drive_id,
                    transition_key=pair,
                    # store a representative carrier omega (not omega_ref)
                    carrier_omega_rad_s=float(wL_med),
                    t_s=None,
                    envelope=None,
                    meta={
                        "kind": kind,
                        "preferred_kind": preferred_kind,
                        "omega_ref_rad_s": float(omega_ref),
                        "omega_L_median_rad_s": float(wL_med),
                        "sigma_omega_rad_s": float(sigma_omega),
                        "coupling_mag": float(c_mag),
                        "score": float(score),
                        "detuning_score_rad_s": float(d0),
                    },
                )
            )

        return out
