from __future__ import annotations
from typing import Any, Dict, Literal, Mapping

import math
import numpy as np
from scipy.constants import e, hbar, c, pi
from qutip import Qobj

from bec.params.transitions import Transition
from bec.quantum_dot.helpers import infer_index_sets_from_registry
from bec.quantum_dot.protocols import (
    DiagnosticsProvider,
    ModeProvider,
    ObservableProvider,
)


_E_OVER_HBAR = e / hbar
_TWO_PI = 2.0 * pi


class Diagnostics(DiagnosticsProvider):
    """Diagnostics for a Quantum-Dot (QD) configuration.

    Provides read-only analysis methods for impD energy levels, radiative rates,
    overlaps, central frequencies, and mode layout summaries.

    Attributes:
        _EL: Energy level object. Must provide at least `fss` (in eV).
        _g: Dictionary of radiative rates (simulation units).
        _s: Conversion factor for simulation time unit (seconds per solver unit).
        _modes: Mode provider, exposing intrinsic modes.
        _observable_provider: Provides number projectors for photonic modes.
        qd: Quantum dot object used for registry lookups.

    Args:
        energy_levels: Object with `fss` in eV, and optionally X1, X2, XX,
            exciton, biexciton.
        gammas: Radiative rates in simulation units. Expected keys:
            "L_XX_X1", "L_XX_X2", "L_X1_G", "L_X2_G".
        mode_provider: Provides access to mode information.
        observable_provider: Provides light mode projectors.
        qd: Quantum dot instance.
    """

    def __init__(
        self,
        energy_levels: Any,
        gammas: Mapping[str, float],
        mode_provider: ModeProvider,
        observable_provider: ObservableProvider,
        qd: Any,
    ) -> None:
        self._EL = energy_levels
        self._g = dict(gammas)
        self._modes = mode_provider
        self._observable_provider = observable_provider
        self.qd = qd

        required = {"L_XX_X1", "L_XX_X2", "L_X1_G", "L_X2_G"}
        missing = required.difference(self._g)
        if missing:
            for k in missing:
                self._g.setdefault(k, 0.0)

    # ---------------- internals ----------------

    def _delta_rad_per_s(self) -> float:
        """Compute splitting frequency.

        Returns:
            Splitting frequency Delta = |FSS| / hbar in rad/s.
        """
        FSS_eV = float(getattr(self._EL, "fss", 0.0))
        return abs(FSS_eV) * _E_OVER_HBAR

    def _pair_linewidth(self, which: Literal["early", "late", "avg"]) -> float:
        """Compute effective linewidth.

        Args:
            which: Either "early", "late", or "avg".

        Returns:
            Effective linewidth gamma_eff (1/s).
        """
        g, s = self._g, 1.0
        if s <= 0.0:
            return 0.0
        if which == "late":
            g1, g2 = g["L_X1_G"] / s, g["L_X2_G"] / s
        elif which == "early":
            g1, g2 = g["L_XX_X1"] / s, g["L_XX_X2"] / s
        else:
            g1 = 0.5 * ((g["L_X1_G"] + g["L_XX_X1"]) / s)
            g2 = 0.5 * ((g["L_X2_G"] + g["L_XX_X2"]) / s)
        return 0.5 * (g1 + g2)

    def _ensure_dense_normalized(
        self, rho: Qobj | np.ndarray, dims: list[int]
    ) -> np.ndarray:
        """Return a dense, trace-1 matrix matching `dims`."""
        R = rho.full() if isinstance(rho, Qobj) else np.asarray(rho)
        D = int(np.prod(dims))
        if R.shape != (D, D):
            raise ValueError(
                f"rho shape {R.shape} is incompatible with dims {
                    dims} (product {D})."
            )
        tr = np.trace(R).real
        if tr <= 0.0:
            raise ValueError("rho has non-positive trace.")
        return (R / tr).astype(complex)

    def _pt_on_subsystems(
        self, R: np.ndarray, dims: list[int], sys_indices: list[int]
    ) -> np.ndarray:
        """Partial transpose on subsystems in `sys_indices`."""
        M = len(dims)
        tens = R.reshape(dims + dims)  # ket: 0..M-1, bra: M..2M-1
        for k in sys_indices:
            tens = np.swapaxes(tens, k, k + M)
        return tens.reshape(R.shape)

    def _log_negativity_early_late(
        self,
        rho_phot: Qobj | np.ndarray,
        dims_phot: list[int],
        early_idxs: list[int],
        late_idxs: list[int],
    ) -> tuple[float, float]:
        """Compute log-negativity and its error for the (early)|(late) cut.

        Returns:
            Tuple (E_N, err_log_neg) with
            E_N = log2(|| rho^{T_late} ||_1),
            err_log_neg = max(0, 1 - E_N).
        """
        R = self._ensure_dense_normalized(rho_phot, dims_phot)
        Rpt = self._pt_on_subsystems(R, dims_phot, late_idxs)
        s = np.linalg.svd(Rpt, compute_uv=False)
        trace_norm = float(np.sum(s).real)
        E_N = float(np.log2(trace_norm))
        err = float(max(0.0, 1.0 - E_N))
        return E_N, err

    # ---------------- API ----------------

    def effective_overlap(
        self, which: Literal["early", "late", "avg"] = "late"
    ) -> float:
        """Compute overlap under fine-structure splitting.

        The overlap is defined as:

            |<psi_H | psi_V>| = gamma / sqrt(gamma^2 + Delta^2)

        Args:
            which: Which photon pair to use ("early", "late", or "avg").

        Returns:
            Overlap in [0, 1]. If gamma <= 0, returns 0.
        """
        gamma = self._pair_linewidth(which)
        if gamma <= 0.0:
            return 0.0
        delta = self._delta_rad_per_s()
        denom = math.hypot(gamma, delta)
        return float(max(0.0, min(1.0, gamma / denom)))

    def _transition_energy_pair_eV(self, tr: Transition) -> tuple[float, float]:
        """Return initial and final energies for a transition."""
        EL = self._EL
        G = 0.0
        fss = float(getattr(EL, "fss", 0.0))
        exciton = float(getattr(EL, "exciton", 0.0))
        X1 = float(getattr(EL, "X1", exciton + 0.5 * fss))
        X2 = float(getattr(EL, "X2", exciton - 0.5 * fss))
        X = float(getattr(EL, "exciton", 0.5 * (X1 + X2)))
        XX = float(getattr(EL, "XX", getattr(EL, "biexciton", 0.0)))

        n = getattr(tr, "name", "")
        if n in ("G_X1", "X1_G"):
            return (X1, G)
        if n in ("G_X2", "X2_G"):
            return (X2, G)
        if n in ("X1_XX", "XX_X1"):
            return (XX, X1)
        if n in ("X2_XX", "XX_X2"):
            return (XX, X2)
        if n in ("G_X", "X_G"):
            return (X, G)
        if n in ("X_XX", "XX_X"):
            return (XX, X)
        return (0.0, 0.0)

    def _central_frequency_from_transition(
        self, tr: Transition
    ) -> dict[str, float]:
        """Compute central frequency for a transition.

        Returns:
            dict with keys:
                - omega_rad_s: Angular frequency (rad/s)
                - freq_Hz: Frequency in Hz
                - lambda_m: Wavelength in meters (inf if zero)
        """
        Ei_eV, Ef_eV = self._transition_energy_pair_eV(tr)
        dE_eV = abs(Ei_eV - Ef_eV)
        if dE_eV <= 0.0:
            return {
                "omega_rad_s": 0.0,
                "freq_Hz": 0.0,
                "lambda_m": float("inf"),
            }
        omega = (dE_eV * e) / hbar
        freq = omega / _TWO_PI
        lam = c / freq if freq > 0.0 else float("inf")
        return {
            "omega_rad_s": float(omega),
            "freq_Hz": float(freq),
            "lambda_m": float(lam),
        }

    def central_frequencies_by_mode(self) -> dict[str, dict[str, float]]:
        """Return central frequencies for each intrinsic mode."""
        modes = getattr(self._modes, "intrinsic", None)
        if modes is None:
            modes = self._modes.modes  # type: ignore[attr-defined]
        out: dict[str, dict[str, float]] = {}
        for i, m in enumerate(modes):
            label = getattr(m, "label", f"mode_{i}")
            tr = getattr(m, "transition", None)
            out[label] = (
                self._central_frequency_from_transition(tr)
                if tr is not None
                else {
                    "omega_rad_s": 0.0,
                    "freq_Hz": 0.0,
                    "lambda_m": float("inf"),
                }
            )
        return out

    def _label_and_pol_for_factor(
        self, factor_idx: int, offset: int
    ) -> tuple[str, str]:
        """Map a photonic factor index to (mode_label, '+'|'-')."""
        if factor_idx < offset:
            raise ValueError(
                f"Invalid factor index {
                    factor_idx} for offset {offset}."
            )
        k = factor_idx - offset
        i_mode, rem = divmod(k, 2)
        is_plus = rem == 0
        try:
            label = getattr(
                self._modes.modes[i_mode], "label", f"mode_{i_mode}"
            )
        except IndexError as exc:
            raise IndexError(
                f"Factor {factor_idx} maps to mode {
                    i_mode}, which is out of range of the registry."
            ) from exc
        return label, ("+" if is_plus else "-")

    def _photon_number_metrics(
        self, rho_phot: Qobj | np.ndarray
    ) -> tuple[Dict[str, float], Dict[str, float]]:
        """Photon number expectations and errors grouped for summary.

        Returns:
            (values, errors) where:
                values = {"N_early": float, "N_late": float}
                errors = {"early": |1 - N_early|, "late": |1 - N_late|}
        """
        early_facts, late_facts, _p, _m, dims_phot, offset = (
            infer_index_sets_from_registry(self.qd, rho_has_qd=False)
        )
        dims_full = [4] + list(dims_phot)
        ops_phot = self._observable_provider.light_mode_projectors(
            dims_full, include_qd=False
        )

        Dp = int(np.prod(dims_phot))
        N_early = Qobj(
            np.zeros((Dp, Dp), complex), dims=[dims_phot, dims_phot]
        ).to("csr")
        N_late = Qobj(
            np.zeros((Dp, Dp), complex), dims=[dims_phot, dims_phot]
        ).to("csr")

        def add_factor(N_accum: Qobj, f_idx: int) -> Qobj:
            lab, pol = self._label_and_pol_for_factor(f_idx, offset=offset)
            key = f"N{pol}[{lab}]"
            return (N_accum + ops_phot[key]).to("csr")

        for fi in early_facts:
            N_early = add_factor(N_early, fi)
        for fj in late_facts:
            N_late = add_factor(N_late, fj)

        rhoA = (
            rho_phot.full()
            if isinstance(rho_phot, Qobj)
            else np.asarray(rho_phot)
        )
        tr = float(np.trace(rhoA).real)
        if tr <= 0.0:
            values = {"N_early": 0.0, "N_late": 0.0}
            errors = {"early": 1.0, "late": 1.0}
            return values, errors

        rhoQ = Qobj(rhoA / tr, dims=[dims_phot, dims_phot]).to("csr")
        N_e = float((N_early * rhoQ).tr().real)
        N_l = float((N_late * rhoQ).tr().real)
        values = {"N_early": N_e, "N_late": N_l}
        errors = {"early": abs(1.0 - N_e), "late": abs(1.0 - N_l)}
        return values, errors

    def mode_layout_summary(
        self, *, rho_phot: Qobj | np.ndarray | None = None
    ) -> Dict[str, Any]:
        """Summarize mode layout and diagnostics.

        Returns:
            dict with:
                - labels, overlaps, fss, central_frequencies
                - if rho_phot is provided:
                    - photon_numbers: {"N_early", "N_late"}
                    - entanglement: {"log_negativity": E_N}
                    - errors:
                        - photon_numbers: {"early", "late"}
                        - entanglement: {"log_negativity": err_log_neg}
        """
        modes = getattr(self._modes, "intrinsic", None)
        if modes is None:
            modes = self._modes.modes  # type: ignore[attr-defined]
        labels = [getattr(m, "label", f"mode_{i}") for i, m in enumerate(modes)]

        summary: Dict[str, Any] = {
            "num_intrinsic_modes": len(modes),
            "labels": labels,
            "overlap_abs_late": self.effective_overlap("late"),
            "overlap_abs_early": self.effective_overlap("early"),
            "overlap_abs_avg": self.effective_overlap("avg"),
            "fss_eV": float(getattr(self._EL, "fss", 0.0)),
            "central_frequencies": self.central_frequencies_by_mode(),
        }

        if rho_phot is not None:
            # infer early/late/dims for LN
            early_facts, late_facts, _p, _m, dims_phot, _offset = (
                infer_index_sets_from_registry(self.qd, rho_has_qd=False)
            )
            # photon numbers and their errors
            pn_values, pn_errors = self._photon_number_metrics(rho_phot)
            # log-negativity and its error
            E_N, err_ln = self._log_negativity_early_late(
                rho_phot, list(dims_phot), list(early_facts), list(late_facts)
            )

            summary.update(
                {
                    "photon_numbers": pn_values,
                    "entanglement": {"log_negativity": E_N},
                    "errors": {
                        "photon_numbers": pn_errors,
                        "entanglement": {"log_negativity": err_ln},
                    },
                }
            )

        return summary
