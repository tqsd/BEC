from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from bec.simulation.drive_decode.protocols import (
    DriveDecodeContext,
    TransitionKey,
)
from bec.simulation.types import ResolvedDrive
from bec.units import magnitude


@dataclass(frozen=True)
class DecodePolicy:
    allow_multi: bool = True
    k_bandwidth: float = 3.0
    pol_gate_eps: float = 1e-6
    pol_penalty_power: float = 1.0
    pol_penalty_weight: float = 1.0
    sample_points: int = 5

    # preference
    prefer_gate: float = 1.0
    prefer_margin: float = 1.0


def _laser_omega_samples(
    drive: Any, *, tlist_solver: np.ndarray, time_unit_s: float, n: int
) -> np.ndarray:
    tlist_solver = np.asarray(tlist_solver, dtype=float)
    if tlist_solver.size == 0:
        raise ValueError("tlist is empty")

    n = int(max(1, n))
    if n == 1:
        idxs = [tlist_solver.size // 2]
    else:
        idxs = np.linspace(0, tlist_solver.size - 1, n, dtype=int).tolist()

    s = float(time_unit_s)
    out: List[float] = []
    for i in idxs:
        t_phys = s * float(tlist_solver[i])
        w = drive.omega_L_phys(t_phys)
        if w is None:
            raise ValueError(
                "Drive carrier is None: cannot decode without omega_L."
            )
        out.append(float(magnitude(w, "rad/s")))
    return np.asarray(out, dtype=float)


def _estimate_sigma_omega_fallback(
    *, tlist_solver: np.ndarray, time_unit_s: float
) -> float:
    t_phys = float(time_unit_s) * np.asarray(tlist_solver, dtype=float)
    tspan = float(np.max(t_phys) - np.min(t_phys))
    if tspan <= 0.0:
        return 0.0
    return 2.0 / tspan


def _coupling_mag(
    ctx: DriveDecodeContext, tr: TransitionKey, E: Optional[np.ndarray]
) -> float:
    if ctx.pol is None or E is None:
        base = 1.0
    else:
        base = float(abs(ctx.pol.coupling_weight(tr, E)))

    # optional phonon renormalization
    if ctx.phonons is not None:
        B = float(ctx.phonons.polaron_B())
        base *= max(0.0, min(1.0, B))
    return base


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


def _detuning_coeff_1ph(
    drive: Any, omega_tr_rad_s: float, *, time_unit_s: float
):
    s = float(time_unit_s)

    def delta(t_solver: float) -> float:
        t_phys = s * float(t_solver)
        wL = drive.omega_L_phys(t_phys)
        if wL is None:
            raise ValueError("Drive carrier is None: cannot compute detuning.")
        return (float(wL) - float(omega_tr_rad_s)) * s

    return delta


def _detuning_coeff_2ph(
    drive: Any, omega_gxx_rad_s: float, *, time_unit_s: float
):
    s = float(time_unit_s)

    def delta(t_solver: float) -> float:
        t_phys = s * float(t_solver)
        wL = drive.omega_L_phys(t_phys)
        if wL is None:
            raise ValueError("Drive carrier is None: cannot compute detuning.")
        return (2.0 * float(wL) - float(omega_gxx_rad_s)) * s

    return delta


class DefaultDriveDecoder:
    def __init__(self, *, policy: Optional[DecodePolicy] = None):
        self.policy = policy or DecodePolicy()

    def decode(
        self,
        *,
        ctx: DriveDecodeContext,
        drives: Sequence[Any],
        tlist: np.ndarray,
        time_unit_s: float,
    ) -> Tuple[ResolvedDrive, ...]:
        tlist = np.asarray(tlist, dtype=float)
        time_unit_s = float(time_unit_s)

        out: List[ResolvedDrive] = []
        for drv in drives:
            out.extend(
                self._decode_one(
                    ctx=ctx, drv=drv, tlist=tlist, time_unit_s=time_unit_s
                )
            )
        return tuple(out)

    def _decode_one(
        self,
        *,
        ctx: DriveDecodeContext,
        drv: Any,
        tlist: np.ndarray,
        time_unit_s: float,
    ) -> List[ResolvedDrive]:
        E = getattr(drv, "effective_pol", None)
        E = E() if callable(E) else None  # ClassicalFieldDrive.effective_pol()

        wL = _laser_omega_samples(
            drv,
            tlist_solver=tlist,
            time_unit_s=time_unit_s,
            n=self.policy.sample_points,
        )

        if ctx.bandwidth is not None:
            sigma_omega = float(
                ctx.bandwidth.sigma_omega_rad_s(
                    drive=drv, tlist_solver=tlist, time_unit_s=time_unit_s
                )
            )
        else:
            sigma_omega = _estimate_sigma_omega_fallback(
                tlist_solver=tlist, time_unit_s=time_unit_s
            )

        thresh = float(self.policy.k_bandwidth) * float(sigma_omega)

        scores: List[Dict[str, Any]] = []
        for tr in ctx.transitions.transitions():
            kind = str(ctx.transitions.kind(tr))
            omega_ref = float(ctx.transitions.omega_ref_rad_s(tr))

            if kind == "1ph":
                dmin = float(np.min(np.abs(wL - omega_ref)))
            elif kind == "2ph":
                dmin = float(np.min(np.abs(2.0 * wL - omega_ref)))
            else:
                continue

            cmag = _coupling_mag(ctx, tr, E)
            score = _penalized_score(
                dmin,
                coupling_mag=cmag,
                sigma_omega=sigma_omega,
                pol_gate_eps=self.policy.pol_gate_eps,
                pol_penalty_power=self.policy.pol_penalty_power,
                pol_penalty_weight=self.policy.pol_penalty_weight,
            )

            scores.append(
                dict(
                    kind=kind,
                    transition=tr,
                    omega_ref=omega_ref,
                    dmin=dmin,
                    score=score,
                    cmag=cmag,
                )
            )

        within = [s for s in scores if float(s["score"]) <= float(thresh)]
        if not within:
            within = [min(scores, key=lambda s: float(s["score"]))]

        if not self.policy.allow_multi:
            within = [min(within, key=lambda s: float(s["score"]))]

        # --- soft preference: drv.preferred_kind ---
        pref = getattr(drv, "preferred_kind", None)
        used_pref = False
        if pref in ("1ph", "2ph"):
            pref_within = [s for s in within if s["kind"] == pref]
            if not pref_within:
                # allow preference to "pull in" candidates outside within,
                # but only if they are plausibly within prefer_gate*thresh
                pref_within = [
                    s
                    for s in scores
                    if s["kind"] == pref
                    and float(s["score"])
                    <= float(self.policy.prefer_gate) * float(thresh)
                ]
            if pref_within:
                within = pref_within
                used_pref = True

        # hard gate for 1ph if polarization is essentially dark
        if E is not None and ctx.pol is not None:
            eps = float(self.policy.pol_gate_eps)
            gated = [
                s
                for s in within
                if not (s["kind"] == "1ph" and float(s["cmag"]) < eps)
            ]
            within = gated if gated else within

        drive_id = getattr(drv, "label", None) or f"drive_{id(drv)}"
        candidates = tuple(dict.fromkeys([s["transition"] for s in scores]))

        out: List[ResolvedDrive] = []
        for entry in within:
            kind = entry["kind"]
            tr = entry["transition"]
            omega_ref = float(entry["omega_ref"])

            if kind == "1ph":
                det = _detuning_coeff_1ph(
                    drv, omega_ref, time_unit_s=time_unit_s
                )
                comps = ()
                if E is not None and ctx.pol is not None:
                    comps = ((tr, complex(ctx.pol.coupling_weight(tr, E))),)

                out.append(
                    ResolvedDrive(
                        drive_id=drive_id,
                        physical=drv,
                        kind="1ph",
                        components=comps,
                        transition=tr,
                        detuning=det,
                        candidates=candidates,
                        meta={
                            "min_detuning_phys_rad_s": float(entry["dmin"]),
                            "score_phys_rad_s": float(entry["score"]),
                            "sigma_omega_phys_rad_s": float(sigma_omega),
                            "threshold_phys_rad_s": float(thresh),
                            "pol_coupling_mag": float(entry["cmag"]),
                            "omegaL_samples_phys_rad_s": wL.tolist(),
                            "preferred_kind": getattr(
                                drv, "preferred_kind", None
                            ),
                            "used_preference": bool(used_pref),
                        },
                    )
                )
            elif kind == "2ph":
                det = _detuning_coeff_2ph(
                    drv, omega_ref, time_unit_s=time_unit_s
                )
                out.append(
                    ResolvedDrive(
                        drive_id=drive_id,
                        physical=drv,
                        kind="2ph",
                        components=(),
                        transition=tr,
                        detuning=det,
                        candidates=candidates,
                        meta={
                            "min_detuning_phys_rad_s": float(entry["dmin"]),
                            "score_phys_rad_s": float(entry["score"]),
                            "sigma_omega_phys_rad_s": float(sigma_omega),
                            "threshold_phys_rad_s": float(thresh),
                            "omegaL_samples_phys_rad_s": wL.tolist(),
                        },
                    )
                )

        return out
