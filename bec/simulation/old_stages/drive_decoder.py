from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from scipy.constants import hbar as _hbar, e as _e

from bec.light.classical import ClassicalCoherentDrive
from bec.params.transitions import Transition
from bec.quantum_dot.dot import QuantumDot
from bec.simulation.protocols import DriveDecoder
from bec.simulation.types import ResolvedDrive


# Linear polarization basis in QD frame
DIPOLE_VECTORS = {
    Transition.G_X1: np.array([1.0, 0.0], dtype=complex),  # H
    Transition.G_X2: np.array([0.0, 1.0], dtype=complex),  # V
    Transition.X1_XX: np.array([1.0, 0.0], dtype=complex),  # H
    Transition.X2_XX: np.array([0.0, 1.0], dtype=complex),  # V
}


def _coupling_weight_qd_frame(
    tr: Transition,
    E: np.ndarray,
) -> complex:
    d = DIPOLE_VECTORS.get(tr)
    if d is None:
        # No polarization rule defined for this transition
        return 1.0 + 0j
    # scalar coupling ~ d^* . E  (vdot does conjugate on first arg)
    return complex(np.vdot(d, E))


def _ev_to_omega(e_ev: float) -> float:
    # E [eV] -> J -> rad/s
    return float(e_ev) * _e / _hbar


def _estimate_sigma_omega(
    qd: QuantumDot, tlist_solver: np.ndarray, time_unit_s: float
) -> float:
    # Preferred: from EnergyLevels.pulse_sigma_t_s
    sigma_t = getattr(qd.energy_levels, "pulse_sigma_t_s", None)
    if sigma_t is not None and float(sigma_t) > 0.0:
        return 1.0 / float(sigma_t)

    # Fallback: from simulation window (very rough)
    t_phys = time_unit_s * np.asarray(tlist_solver, dtype=float)
    tspan = float(np.max(t_phys) - np.min(t_phys))
    if tspan <= 0.0:
        return 0.0
    return 2.0 / tspan


def _laser_omega_phys(
    drv: ClassicalCoherentDrive, t_phys: float
) -> Optional[float]:
    w0 = drv.laser_omega0
    if w0 is None:
        return None

    dw = drv.delta_omega
    w = float(w0)

    if callable(dw):
        w += float(dw(float(t_phys)))
    else:
        w += float(dw)

    return w


def _detuning_coeff_1ph(
    drv: ClassicalCoherentDrive,
    omega_tr: float,
    *,
    time_unit_s: float,
) -> Callable[[float], float]:
    s = float(time_unit_s)

    def delta(t_solver: float) -> float:
        t_phys = s * float(t_solver)
        wL = _laser_omega_phys(drv, t_phys)
        if wL is None:
            raise ValueError("laser_omega0 is None, cannot compute detuning.")
        # physical detuning [rad/s] -> solver detuning [rad/solver_unit]
        return (float(wL) - float(omega_tr)) * s

    return delta


def _detuning_coeff_2ph(
    drv: ClassicalCoherentDrive,
    omega_g_xx: float,
    *,
    time_unit_s: float,
) -> Callable[[float], float]:
    s = float(time_unit_s)

    def delta(t_solver: float) -> float:
        t_phys = s * float(t_solver)
        wL = _laser_omega_phys(drv, t_phys)
        if wL is None:
            raise ValueError("laser_omega0 is None, cannot compute detuning.")
        return (2.0 * float(wL) - float(omega_g_xx)) * s

    return delta


def _coupling_mag(tr: Transition, E: Optional[np.ndarray]) -> float:
    if E is None:
        return 1.0
    c = _coupling_weight_qd_frame(tr, E)
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
    """
    detuning_min: physical rad/s
    coupling_mag: |d^* . E| in QD frame (dimensionless)
    """
    # No polarization info -> no penalty
    if coupling_mag >= 1.0:
        return float(detuning_min)

    eps = float(pol_gate_eps)
    c_eff = max(float(coupling_mag), eps)

    p = float(pol_penalty_power)
    w = float(pol_penalty_weight)

    # penalty = w * sigma_omega * (1/c^p - 1)
    penalty = w * float(sigma_omega) * ((1.0 / (c_eff**p)) - 1.0)
    return float(detuning_min) + float(penalty)


@dataclass(frozen=True)
class DecodePolicy:
    """
    Decoder behavior.

    allow_multi:
        If True, emit multiple ResolvedDrive entries for a single physical
        drive when mutliple transitions are within threshold.
        If False, choose the best match (minimum error).

    k_bandwidth:
        Threshold factor in units of sigma_omega.

    default_kind:
        If the drive does not specify order/kind chose this
    """

    allow_multi: bool = True
    k_bandwidth: float = 3.0
    default_kind: Literal["1ph", "2ph"] = "1ph"
    # polarization handling
    pol_gate_eps: float = 1e-6  # "dark" if |c| < eps
    pol_penalty_power: float = 1.0  # 1.0 or 2.0 are common
    pol_penalty_weight: float = 1.0  # how strongly to penalize


class DefaultDriveDecoder(DriveDecoder):
    def __init__(self, policy: Optional[DecodePolicy] = None):
        self.policy = policy or DecodePolicy()

    def decode(
        self,
        qd: QuantumDot,
        drives: Sequence[ClassicalCoherentDrive],
        *,
        time_unit_s: float,
        tlist: np.ndarray,
    ) -> Tuple[ResolvedDrive, ...]:
        out: List[ResolvedDrive] = []
        for drv in drives:
            out.extend(
                self._decode_one(
                    qd,
                    drv,
                    tlist=np.asarray(tlist, dtype=float),
                    time_unit_s=float(time_unit_s),
                )
            )
        return tuple(out)

    def _decode_one(
        self,
        qd: QuantumDot,
        drv: ClassicalCoherentDrive,
        *,
        tlist: np.ndarray,
        time_unit_s: float,
    ) -> List[ResolvedDrive]:
        el = qd.energy_levels
        E = drv.effective_polarization()

        wL = self._laser_omega_grid(drv, tlist=tlist, time_unit_s=time_unit_s)

        sigma_omega = _estimate_sigma_omega(qd, tlist, time_unit_s)
        thresh = float(self.policy.k_bandwidth) * float(sigma_omega)

        one_ph = self._score_1ph(el, wL=wL, E=E, sigma_omega=sigma_omega)
        two_ph = (
            self._score_2ph(el, wL=wL) if hasattr(Transition, "G_XX") else []
        )

        all_scored = one_ph + two_ph

        selected = self._select_candidates(all_scored, thresh=thresh)
        selected = self._apply_pol_gate_if_possible(selected, E=E)
        selected = self._maybe_merge_exciton_coherently(selected, E=E)

        candidates = self._collect_candidates_from_scores(all_scored)

        drive_id = drv.label or f"drive_{id(drv)}"

        return [
            self._emit_resolved(
                drv=drv,
                drive_id=drive_id,
                entry=entry,
                time_unit_s=time_unit_s,
                sigma_omega=sigma_omega,
                thresh=thresh,
                E=E,
                candidates=candidates,
            )
            for entry in selected
        ]

    # ----------------- small helpers (reduce cyclomatic complexity) -----------------

    def _collect_candidates_from_scores(
        self, scores: List[Dict[str, Any]]
    ) -> Tuple[Transition, ...]:
        cands: List[Transition] = []
        for s in scores:
            tr = s.get("transition")
            if isinstance(tr, Transition):
                cands.append(tr)
        return tuple(dict.fromkeys(cands))  # keep order, remove duplicates

    def _laser_omega_grid(
        self,
        drv: ClassicalCoherentDrive,
        *,
        tlist: np.ndarray,
        time_unit_s: float,
    ) -> np.ndarray:
        t_phys = float(time_unit_s) * np.asarray(tlist, dtype=float)
        w_vals: List[float] = []
        for tp in t_phys:
            w = _laser_omega_phys(drv, float(tp))
            if w is None:
                raise ValueError(
                    "Cannot decode drive: drv.laser_omega0 is None."
                )
            w_vals.append(float(w))
        return np.asarray(w_vals, dtype=float)

    def _score_1ph(
        self,
        el,
        *,
        wL: np.ndarray,
        E: Optional[np.ndarray],
        sigma_omega: float,
    ) -> List[Dict[str, Any]]:
        trans_1ph: List[Tuple[Transition, float]] = [
            (Transition.G_X1, _ev_to_omega(el.X1)),
            (Transition.G_X2, _ev_to_omega(el.X2)),
            (Transition.X1_XX, _ev_to_omega(el.XX - el.X1)),
            (Transition.X2_XX, _ev_to_omega(el.XX - el.X2)),
        ]

        out: List[Dict[str, Any]] = []
        for tr, wtr in trans_1ph:
            dmin = float(np.min(np.abs(wL - float(wtr))))
            cmag = _coupling_mag(tr, E)
            score = _penalized_score(
                dmin,
                coupling_mag=cmag,
                sigma_omega=sigma_omega,
                pol_gate_eps=self.policy.pol_gate_eps,
                pol_penalty_power=self.policy.pol_penalty_power,
                pol_penalty_weight=self.policy.pol_penalty_weight,
            )
            out.append(
                {
                    "kind": "1ph",
                    "transition": tr,
                    "omega_ref": float(wtr),
                    "dmin": float(dmin),  # raw physical detuning
                    "score": float(score),  # used for selection
                    "cmag": float(cmag),
                }
            )
        return out

    def _score_2ph(self, el, *, wL: np.ndarray) -> List[Dict[str, Any]]:
        omega_g_xx = _ev_to_omega(el.XX)

        # for diagnostics + TPE effective coupling (field mapping)
        omega_gx1 = _ev_to_omega(el.X1)
        omega_gx2 = _ev_to_omega(el.X2)

        # choose a representative laser frequency for "delta_gx*_phys"
        # simplest: center of time grid
        wL_mid = float(wL[len(wL) // 2])

        delta_gx1 = wL_mid - float(omega_gx1)  # rad/s
        delta_gx2 = wL_mid - float(omega_gx2)  # rad/s

        dmin2 = float(np.min(np.abs(2.0 * wL - float(omega_g_xx))))

        return [
            {
                "kind": "2ph",
                "transition": Transition.G_XX,
                "omega_ref": float(omega_g_xx),
                "dmin": float(dmin2),
                "score": float(dmin2),
                "cmag": 1.0,
                "delta_gx1_phys": float(delta_gx1),
                "delta_gx2_phys": float(delta_gx2),
            }
        ]

    def _select_candidates(
        self, scores: List[Dict[str, Any]], *, thresh: float
    ) -> List[Dict[str, Any]]:
        # "within threshold" computed on SCORE (so polarization influences eligibility)
        within = [s for s in scores if float(s["score"]) <= float(thresh)]
        if not within:
            within = [min(scores, key=lambda s: float(s["score"]))]

        if not self.policy.allow_multi:
            within = [min(within, key=lambda s: float(s["score"]))]

        return within

    def _apply_pol_gate_if_possible(
        self, selected: List[Dict[str, Any]], *, E: Optional[np.ndarray]
    ) -> List[Dict[str, Any]]:
        if E is None:
            return selected

        eps = float(self.policy.pol_gate_eps)

        # gate only 1ph by coupling magnitude
        gated = [
            s
            for s in selected
            if not (s["kind"] == "1ph" and float(s.get("cmag", 1.0)) < eps)
        ]

        # only apply gating if it doesn't delete everything
        return gated if gated else selected

    def _maybe_merge_exciton_coherently(
        self, selected: List[Dict[str, Any]], *, E: Optional[np.ndarray]
    ) -> List[Dict[str, Any]]:
        if E is None:
            return selected

        gx = [
            s
            for s in selected
            if s["kind"] == "1ph"
            and s.get("transition") in (Transition.G_X1, Transition.G_X2)
        ]
        if len(gx) != 2:
            return selected

        # Remove the two exciton entries
        remaining = [s for s in selected if s not in gx]

        # Build weights
        c1 = _coupling_weight_qd_frame(Transition.G_X1, E)
        c2 = _coupling_weight_qd_frame(Transition.G_X2, E)

        # Per-transition omega refs (phys rad/s)
        omega_gx1 = float(
            next(
                s["omega_ref"] for s in gx if s["transition"] == Transition.G_X1
            )
        )
        omega_gx2 = float(
            next(
                s["omega_ref"] for s in gx if s["transition"] == Transition.G_X2
            )
        )
        omega_map = {Transition.G_X1: omega_gx1, Transition.G_X2: omega_gx2}

        # Diagnostics: coherent selection should summarize both transitions
        dmin_min = float(min(float(s["dmin"]) for s in gx))
        score_min = float(min(float(s["score"]) for s in gx))
        cmag_min = float(min(float(s.get("cmag", 1.0)) for s in gx))

        # Keep one reference omega_ref ONLY for the existing detuning() signature.
        # (This is "reference-only" if FSS != 0; composer must use omega_components.)
        ref = min(gx, key=lambda s: float(s["score"]))
        omega_ref = float(ref["omega_ref"])

        remaining.append(
            {
                "kind": "1ph_coherent",
                "transition": None,
                "omega_ref": omega_ref,
                "dmin": dmin_min,
                "score": score_min,
                "cmag": cmag_min,
                "components": (
                    (Transition.G_X1, c1),
                    (Transition.G_X2, c2),
                ),
                "omega_components": omega_map,
            }
        )
        return remaining

    def _emit_resolved(
        self,
        *,
        drv: ClassicalCoherentDrive,
        drive_id: str,
        entry: Dict[str, Any],
        time_unit_s: float,
        sigma_omega: float,
        thresh: float,
        E: Optional[np.ndarray],
        candidates: Tuple[Transition, ...],
    ) -> ResolvedDrive:
        kind = str(entry["kind"])
        omega_ref = float(entry["omega_ref"])
        tr = entry.get("transition")

        if kind in ("1ph", "1ph_coherent"):
            det = _detuning_coeff_1ph(drv, omega_ref, time_unit_s=time_unit_s)
        elif kind == "2ph":
            det = _detuning_coeff_2ph(drv, omega_ref, time_unit_s=time_unit_s)
        else:
            raise ValueError(f"Unknown kind {kind!r}")

        if kind == "1ph_coherent":
            components = tuple(entry.get("components") or ())
            transition = None
        elif kind == "2ph":
            components = ()
            transition = tr
        else:
            # kind == "1ph"
            if E is None:
                components = ()
                transition = tr
            else:
                c = _coupling_weight_qd_frame(tr, E)
                components = ((tr, c),)
                transition = tr

        meta: Dict[str, Any] = {
            "kind": kind,
            "min_detuning_phys_rad_s": float(entry.get("dmin", 0.0)),
            "score_phys_rad_s": float(
                entry.get("score", entry.get("dmin", 0.0))
            ),
            "sigma_omega_phys_rad_s": float(sigma_omega),
            "threshold_phys_rad_s": float(thresh),
            "pol_coupling_mag": float(entry.get("cmag", 1.0)),
        }

        if kind == "1ph_coherent":
            omega_components = entry.get("omega_components", {})
            meta["omega_components_phys_rad_s"] = {
                str(k): float(v) for k, v in omega_components.items()
            }
            # Critical: the single detuning() callable is only a reference for coherent drives.
            # The Hamiltonian composer must compute per-component detunings using omega_components.
            meta["detuning_is_reference_only"] = True
        if kind == "2ph":
            if "delta_gx1_phys" in entry:
                meta["delta_gx1_phys"] = float(entry["delta_gx1_phys"])
            if "delta_gx2_phys" in entry:
                meta["delta_gx2_phys"] = float(entry["delta_gx2_phys"])
        return ResolvedDrive(
            drive_id=drive_id,
            physical=drv,
            kind="2ph" if kind == "2ph" else "1ph",
            components=components,
            transition=transition,
            detuning=det,
            candidates=candidates,
            meta=meta,
        )
