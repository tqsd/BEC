from __future__ import annotations
from typing import Any, Dict, List, Mapping, Optional, Sequence
import math
import numpy as np

from qsi.state import State, StateProp

# Treat ASCII '-' and Unicode '−' as the same "minus/polarization-"
_POL_MINUS_ALIASES = ("-", "−")
_POL_PLUS_ALIASES = ("+",)


def _resolve_pol_symbol(sym: str, pol_map: Dict[str, str]) -> str:
    """Map '+' or '-'/'−' (and any aliases present) to a polarization label."""
    if sym in _POL_PLUS_ALIASES:
        return pol_map.get("+", pol_map.get("plus", "H"))
    if sym in _POL_MINUS_ALIASES:
        # Try a few common keys so both '-' and '−' work
        return (
            pol_map.get("-", None)
            or pol_map.get("−", None)
            or pol_map.get("minus", None)
            or "V"
        )
    return pol_map.get(sym, "H")


def _bandwidth_from_diagnostics_for_label(
    diagnostics: Mapping[str, Any],
    label: str,
    default_bandwidth_Hz: float,
) -> float:
    """
    Try several diagnostics fields to get a per-mode bandwidth in Hz.
    Fallback to default_bandwidth_Hz when unavailable.
    """
    # 1) Direct per-mode bandwidths in Hz
    bw_hz_map = diagnostics.get("bandwidths_Hz")
    if isinstance(bw_hz_map, dict) and label in bw_hz_map:
        try:
            return float(bw_hz_map[label])
        except Exception:
            pass

    # 2) Per-mode bandwidths in rad/s (convert to Hz)
    bw_rad_map = diagnostics.get("bandwidths_rad_s")
    if isinstance(bw_rad_map, dict) and label in bw_rad_map:
        try:
            return float(bw_rad_map[label]) / (2.0 * math.pi)
        except Exception:
            pass

    # 3) Derive from radiative rates if available (rad/s → Hz)
    #    Map rate keys to labels
    rates = diagnostics.get("rates_rad_s")
    if isinstance(rates, dict):

        def _maybe(k: str) -> Optional[float]:
            try:
                return float(rates[k])
            except Exception:
                return None

        # fss != 0 labels
        if label == "X1_XX":
            r = _maybe("L_XX_X1")
            if r is not None:
                return r / (2.0 * math.pi)
        if label == "X2_XX":
            r = _maybe("L_XX_X2")
            if r is not None:
                return r / (2.0 * math.pi)
        if label == "G_X1":
            r = _maybe("L_X1_G")
            if r is not None:
                return r / (2.0 * math.pi)
        if label == "G_X2":
            r = _maybe("L_X2_G")
            if r is not None:
                return r / (2.0 * math.pi)

        # fss == 0 aggregated labels
        if label == "G_X":
            r1, r2 = _maybe("L_X1_G"), _maybe("L_X2_G")
            if r1 is not None and r2 is not None:
                return 0.5 * (r1 + r2) / (2.0 * math.pi)
            if r1 is not None:
                return r1 / (2.0 * math.pi)
            if r2 is not None:
                return r2 / (2.0 * math.pi)

        if label == "X_XX":
            r1, r2 = _maybe("L_XX_X1"), _maybe("L_XX_X2")
            if r1 is not None and r2 is not None:
                return 0.5 * (r1 + r2) / (2.0 * math.pi)
            if r1 is not None:
                return r1 / (2.0 * math.pi)
            if r2 is not None:
                return r2 / (2.0 * math.pi)

    # 4) Fallback
    return float(default_bandwidth_Hz)


def stateprops_from_qd_diagnostics(
    diagnostics: Mapping[str, Any],
    *,
    trunc_per_pol: int,
    default_bandwidth_Hz: float = 1.0,  # used when diagnostics doesn't provide one
    pol_map: Optional[Dict[str, str]] = None,
    wavelength_override_nm: Optional[Dict[str, float]] = None,
) -> List[StateProp]:
    """
    Build light-mode StateProps in the *QD registry order*:
        for each diagnostics['labels'] entry → two factors ('+', '−').

    The function will use bandwidths from `diagnostics` when present:
        - diagnostics['bandwidths_Hz'][label]              (Hz)
        - diagnostics['bandwidths_rad_s'][label]           (rad/s → Hz)
        - diagnostics['rates_rad_s'] (maps L_* to labels)  (rad/s → Hz)
      Otherwise falls back to `default_bandwidth_Hz`.

    Args
    ----
    diagnostics : dict from qd.diagnostics.mode_layout_summary(...).
                  Must include 'labels' and 'central_frequencies'.
    trunc_per_pol : per-pol Fock truncation (e.g., 2).
    default_bandwidth_Hz : fallback bandwidth (Hz) when diagnostics has none.
    pol_map : mapping for '+' and '-'/'−' → 'H'/'V' or 'R'/'L', etc.
              Default: {"+":"H","-":"V","−":"V"}.
    wavelength_override_nm : optional {label: λ_nm} to override inferred λ.

    Returns
    -------
    list[StateProp] ordered as [label0 '+', label0 '−', label1 '+', label1 '−', ...].
    """
    if pol_map is None:
        pol_map = {"+": "H", "-": "V", "−": "V"}  # robust default

    labels = list(diagnostics.get("labels", []))
    cf = diagnostics.get("central_frequencies", {}) or {}

    props: List[StateProp] = []
    for label in labels:
        # wavelength per mode used for both pols
        if wavelength_override_nm and label in wavelength_override_nm:
            lam_nm = float(wavelength_override_nm[label])
        else:
            lam_m = float(cf.get(label, {}).get("lambda_m", 0.0))
            lam_nm = (
                0.0 if (not np.isfinite(lam_m) or lam_m <= 0.0) else 1e9 * lam_m
            )

        # bandwidth per mode (Hz), if present in diagnostics
        bw_hz = _bandwidth_from_diagnostics_for_label(
            diagnostics, label, default_bandwidth_Hz
        )

        # '+' then '−' to match the QD factor order
        for pol_sym in ("+", "−"):
            pol = _resolve_pol_symbol(pol_sym, pol_map)
            props.append(
                StateProp(
                    state_type="light",
                    truncation=int(trunc_per_pol),
                    wavelength=float(lam_nm),
                    polarization=pol,
                    bandwidth=float(bw_hz),
                )
            )
    return props


def apply_prepare_from_scalar(
    kraus_ops: Sequence[np.ndarray],  # list of NumPy arrays
    layout: List[StateProp],
    *,
    normalize: bool = True,
) -> State:
    """
    Construct a new State with the given factor layout and set:
        ρ_out = Σ_i K_i K_i^†
    for a prepare-from-scalar channel (each K_i : C → H_out).

    Accepts K_i shapes:
      - (N_out,)      → reshaped to (N_out, 1)
      - (N_out, 1)    → used as-is
    """
    if not layout:
        raise ValueError("layout must contain at least one StateProp.")
    if not kraus_ops:
        raise ValueError("kraus_ops must be non-empty.")

    # Build container by joining factors (we'll overwrite .state below)
    out = State(layout[0])
    for p in layout[1:]:
        out.join(State(p))
    N_out = out.dimensions

    rho = np.zeros((N_out, N_out), dtype=complex)
    for K in kraus_ops:
        K = np.asarray(K, dtype=complex)
        if K.ndim == 1:
            K = K.reshape(N_out, 1)
        if K.shape != (N_out, 1):
            raise ValueError(
                f"Kraus op has shape {K.shape}, expected ({N_out}, 1) or ({
                    N_out},)."
            )
        rho += K @ K.conj().T

    if normalize:
        tr = float(np.trace(rho).real)
        if tr > 0.0:
            rho /= tr

    out.state = rho
    return out
