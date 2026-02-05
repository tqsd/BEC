from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np

try:
    # Only used if we need to convert eV -> rad/s via hbar.
    import scipy.constants as _const
except Exception:  # pragma: no cover
    _const = None


@dataclass(frozen=True)
class OverlapCalculator:
    """
    Compute Lorentzian spectral overlap and HOM visibility for the QD cascade.

    Inputs are decay rates (in 1/s) for the early and late emission,
    plus the splitting delta_rad_s = |FSS|/hbar in rad/s.

    The effective pair linewidth used in the overlap formula depends on `which`:
    - "early": 0.5 * (gamma_XX_X1 + gamma_XX_X2)
    - "late":  0.5 * (gamma_X1_G + gamma_X2_G)
    - "avg":   0.5 * ("early" + "late") implemented as per-pair average

    All quantities are treated as rates in 1/s.
    """

    gamma_XX_X1: float
    gamma_XX_X2: float
    gamma_X1_G: float
    gamma_X2_G: float
    delta_rad_s: float  # |FSS|/hbar

    def _pair_linewidth(self, which: str) -> float:
        if which == "early":
            g1, g2 = self.gamma_XX_X1, self.gamma_XX_X2
            return 0.5 * (g1 + g2)
        if which == "late":
            g1, g2 = self.gamma_X1_G, self.gamma_X2_G
            return 0.5 * (g1 + g2)

        # "avg": average the two pair linewidths
        early = 0.5 * (self.gamma_XX_X1 + self.gamma_XX_X2)
        late = 0.5 * (self.gamma_X1_G + self.gamma_X2_G)
        return 0.5 * (early + late)

    def overlap(self, which: str) -> float:
        gamma = float(self._pair_linewidth(str(which)))
        if not math.isfinite(gamma) or gamma <= 0.0:
            return 0.0
        delta = float(self.delta_rad_s)
        if not math.isfinite(delta):
            return float("nan")
        return float(gamma / math.hypot(gamma, delta))

    def hom(self, which: str) -> float:
        lam = self.overlap(which)
        if not math.isfinite(lam):
            return float("nan")
        return float(lam * lam)


@dataclass(frozen=True)
class OverlapInputs:
    gamma_xx_x1: float
    gamma_xx_x2: float
    gamma_x1_g: float
    gamma_x2_g: float
    delta_rad_s: float


@dataclass(frozen=True)
class OverlapMetrics:
    overlap_abs_avg: float
    overlap_abs_early: float
    overlap_abs_late: float
    hom_vis_avg: float
    hom_vis_early: float
    hom_vis_late: float

    # optional provenance/debug
    gamma_xx_x1: float
    gamma_xx_x2: float
    gamma_x1_g: float
    gamma_x2_g: float
    delta_rad_s: float


def _to_float_rate_1_s(x: Any) -> float:
    """
    Convert a QuantityLike (smef.core.units.Q or pint-like) to float in 1/s.
    Falls back to float(x) if no .to() exists.
    """
    try:
        return float(x.to("1/s").magnitude)
    except Exception:
        return float(x)


def _energy_to_delta_rad_s(energy: Any) -> float:
    """
    Best-effort extraction of delta = |FSS|/hbar in rad/s.

    We try common attribute names that might already be rad/s, otherwise
    we assume energy.fss is an energy (eV) and convert via hbar.
    """
    if energy is None:
        return float("nan")

    # Already angular frequency?
    for name in (
        "delta_rad_s",
        "delta_fss_rad_s",
        "fss_rad_s",
        "exciton_split_rad_s",
    ):
        if hasattr(energy, name):
            v = getattr(energy, name)
            try:
                return float(v.to("rad/s").magnitude)
            except Exception:
                try:
                    return float(v)
                except Exception:
                    pass

    # Fall back to energy.fss as energy in eV (or QuantityLike with .to("eV"))
    if not hasattr(energy, "fss"):
        return float("nan")

    fss = getattr(energy, "fss")

    try:
        fss_eV = float(fss.to("eV").magnitude)
    except Exception:
        try:
            fss_eV = float(fss)
        except Exception:
            return float("nan")

    if _const is None:
        # scipy not available; cannot convert eV -> rad/s robustly
        return float("nan")

    return float(abs(fss_eV) * _const.e / _const.hbar)


def _get_rate_value(rates: Mapping[Any, Any], key: Any) -> Optional[float]:
    if key in rates:
        return _to_float_rate_1_s(rates[key])
    return None


def _find_key_by_substrings(
    rates: Mapping[Any, Any], *subs: str
) -> Optional[Any]:
    """
    Heuristic fallback: find a key whose string representation contains all substrings.
    """
    want = [str(s) for s in subs]
    for k in rates.keys():
        s = str(k)
        ok = True
        for w in want:
            if w not in s:
                ok = False
                break
        if ok:
            return k
    return None


def extract_overlap_inputs(qd: Any) -> Optional[OverlapInputs]:
    """
    Extract overlap inputs from your QuantumDot.

    Uses:
      - qd.rates: Mapping[RateKey, QuantityLike]
      - qd.energy: EnergyStructure, expected to hold FSS

    For rates, we try:
      1) direct attribute lookups on RateKey enum members by common names
      2) string matching over the keys in qd.rates

    If any required value is missing, returns None.
    """
    rates: Mapping[Any, Any] = getattr(qd, "rates", {}) or {}

    # Try RateKey enum if available
    RateKey = None
    try:
        from bec.quantum_dot.transitions import RateKey as _RateKey  # type: ignore

        RateKey = _RateKey
    except Exception:
        RateKey = None

    def try_ratekey_attr(name: str) -> Optional[Any]:
        if RateKey is None:
            return None
        if hasattr(RateKey, name):
            return getattr(RateKey, name)
        return None

    # Candidate names to try on RateKey
    candidate_sets = [
        ("XX_X1", "XX_X2", "X1_G", "X2_G"),
        ("XX_TO_X1", "XX_TO_X2", "X1_TO_G", "X2_TO_G"),
        ("GAMMA_XX_X1", "GAMMA_XX_X2", "GAMMA_X1_G", "GAMMA_X2_G"),
    ]

    for a, b, c, d in candidate_sets:
        ka = try_ratekey_attr(a)
        kb = try_ratekey_attr(b)
        kc = try_ratekey_attr(c)
        kd = try_ratekey_attr(d)
        if None in (ka, kb, kc, kd):
            continue

        va = _get_rate_value(rates, ka)
        vb = _get_rate_value(rates, kb)
        vc = _get_rate_value(rates, kc)
        vd = _get_rate_value(rates, kd)
        if None in (va, vb, vc, vd):
            continue

        delta = _energy_to_delta_rad_s(getattr(qd, "energy", None))
        return OverlapInputs(
            gamma_xx_x1=float(va),
            gamma_xx_x2=float(vb),
            gamma_x1_g=float(vc),
            gamma_x2_g=float(vd),
            delta_rad_s=float(delta),
        )

    # Fallback: string matching on keys
    # We look for keys containing these tokens; adapt if your RateKey string repr differs.
    k_xx_x1 = _find_key_by_substrings(rates, "XX", "X1")
    k_xx_x2 = _find_key_by_substrings(rates, "XX", "X2")
    k_x1_g = _find_key_by_substrings(rates, "X1", "G")
    k_x2_g = _find_key_by_substrings(rates, "X2", "G")

    if None in (k_xx_x1, k_xx_x2, k_x1_g, k_x2_g):
        return None

    va = _get_rate_value(rates, k_xx_x1)
    vb = _get_rate_value(rates, k_xx_x2)
    vc = _get_rate_value(rates, k_x1_g)
    vd = _get_rate_value(rates, k_x2_g)
    if None in (va, vb, vc, vd):
        return None

    delta = _energy_to_delta_rad_s(getattr(qd, "energy", None))
    return OverlapInputs(
        gamma_xx_x1=float(va),
        gamma_xx_x2=float(vb),
        gamma_x1_g=float(vc),
        gamma_x2_g=float(vd),
        delta_rad_s=float(delta),
    )


def compute_overlap_metrics(qd: Any) -> OverlapMetrics:
    """
    Compute overlap/HOM metrics from a QuantumDot instance.

    If extraction fails, returns NaNs for overlap/HOM quantities.
    """
    ov = extract_overlap_inputs(qd)
    if ov is None:
        nan = float("nan")
        return OverlapMetrics(
            overlap_abs_avg=nan,
            overlap_abs_early=nan,
            overlap_abs_late=nan,
            hom_vis_avg=nan,
            hom_vis_early=nan,
            hom_vis_late=nan,
            gamma_xx_x1=nan,
            gamma_xx_x2=nan,
            gamma_x1_g=nan,
            gamma_x2_g=nan,
            delta_rad_s=nan,
        )

    oc = OverlapCalculator(
        gamma_XX_X1=float(ov.gamma_xx_x1),
        gamma_XX_X2=float(ov.gamma_xx_x2),
        gamma_X1_G=float(ov.gamma_x1_g),
        gamma_X2_G=float(ov.gamma_x2_g),
        delta_rad_s=float(ov.delta_rad_s),
    )

    lam_avg = oc.overlap("avg")
    lam_early = oc.overlap("early")
    lam_late = oc.overlap("late")

    return OverlapMetrics(
        overlap_abs_avg=float(lam_avg),
        overlap_abs_early=float(lam_early),
        overlap_abs_late=float(lam_late),
        hom_vis_avg=float(oc.hom("avg")),
        hom_vis_early=float(oc.hom("early")),
        hom_vis_late=float(oc.hom("late")),
        gamma_xx_x1=float(ov.gamma_xx_x1),
        gamma_xx_x2=float(ov.gamma_xx_x2),
        gamma_x1_g=float(ov.gamma_x1_g),
        gamma_x2_g=float(ov.gamma_x2_g),
        delta_rad_s=float(ov.delta_rad_s),
    )


def overlap_metrics_as_dict(qd: Any) -> dict[str, float]:
    """
    Convenience helper for wiring into meta dicts.
    """
    m = compute_overlap_metrics(qd)
    return {
        "overlap_abs_avg": float(m.overlap_abs_avg),
        "overlap_abs_early": float(m.overlap_abs_early),
        "overlap_abs_late": float(m.overlap_abs_late),
        "hom_vis_avg": float(m.hom_vis_avg),
        "hom_vis_early": float(m.hom_vis_early),
        "hom_vis_late": float(m.hom_vis_late),
    }
