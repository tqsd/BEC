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

from bec.quantum_dot.metrics.mode_registry import build_registry
from bec.quantum_dot.metrics import (
    PhotonCounter,
    TwoPhotonProjector,
    EntanglementCalculator,
    OverlapCalculator,
    ensure_rho,
    purity,
    BellAnalyzer,
    PopulationDecomposer,
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
    def _mode_label_to_rate_per_s(self, label: str) -> float:
        """
        Map a mode label to an effective ratiavive rate (1/s)

        Handles both fine-structure-resolved labels and combined labels.

        Parameters
        ----------
        label: str
            Mode label used by the registry

        Returns
        -------
        float
            Radiative rate in 1/s. Returns 0.0 if mapping is unknown.
        """
        g = self._g  # assumed in 1/s
        # fine-structure resolved labels
        if label == "X1_XX":
            return float(g.get("L_XX_X1", 0.0))
        if label == "X2_XX":
            return float(g.get("L_XX_X2", 0.0))
        if label == "G_X1":
            return float(g.get("L_X1_G", 0.0))
        if label == "G_X2":
            return float(g.get("L_X2_G", 0.0))

        # aggregated labels (fss == 0 case)
        if label == "G_X":
            r1 = float(g.get("L_X1_G", 0.0))
            r2 = float(g.get("L_X2_G", 0.0))
            return 0.5 * (r1 + r2) if (r1 > 0.0 and r2 > 0.0) else (r1 or r2)
        if label == "X_XX":
            r1 = float(g.get("L_XX_X1", 0.0))
            r2 = float(g.get("L_XX_X2", 0.0))
            return 0.5 * (r1 + r2) if (r1 > 0.0 and r2 > 0.0) else (r1 or r2)

        # unknown label
        return 0.0

    def _mode_transitions_by_label(self) -> dict[str, str]:
        """
        Build a map from mode labels to transition names.

        Uses `self._modes.intrinsic` if present, otherwise uses
        `self._modes.modes`. Each label reports the `.tranistion.name`
        or `UNKNOWN` if missing.

        Returns
        -------
        dict[str, str]
            Mapping {label: transition_name}
        """
        modes = getattr(self._modes, "intrinsic", None) or self._modes.modes
        out: dict[str, str] = {}
        for i, m in enumerate(modes):
            label = getattr(m, "label", f"mode_{i}")
            tr = getattr(m, "transition", None)
            out[label] = getattr(tr, "name", "UNKNOWN")
        return out

    def _delta_rad_per_s(self) -> float:
        """
        Compute the fine-structure splitting frequency in rad/s.

        Uses the `fss` parameters in eV and converts using e/hbar

        Returns
        -------
        float
            Delta = |fss| * e/hbar (rad/s)
        """
        FSS_eV = float(getattr(self._EL, "fss", 0.0))
        return abs(FSS_eV) * _E_OVER_HBAR

    def _pair_linewidth(self, which: Literal["early", "late", "avg"]) -> float:
        """
        Effective linewidth for a photon pair, in Hz.

        Parameters
        ----------
        which: {"early", "late", "avg"}
          - "early": uses XX->X rtaes (L_XX_X1, L_XX_X2)
          - "late": uses X->G rtaes (L_X1_G, L_X2_G)
          - "avg": average of the rates

        Returns
        -------
        float
            0.5 * (gamma1+gamma2) for the chosen pair. Returns 0.0 if
            inputs are not available
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
        """
        Returns a dense, trace-1 density matrix matching `dims`.

        Validates shape so that it corresponds to the prod(dims)
        and normalizes by trace.

        Parameters
        ----------
        rho: qutip.Qobj or np.ndarray
            Density matrix in QuTiP or numpy
        dims: list[int]
            Local dimensions (full Hilbert space)

        Returns
        -------
        np.ndarray
            Complex array (square matrix) with unit trace.

        Raises
        ------
        VauleError
            If shape is incompatible or the trace is non-positive
        """
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
        """
        Partial trnspose on the specified subsystems.

        Parameters
        ----------
        R: numpy.ndarray
            Density matrix (D,D) as `numpy.ndarray`
        dimes: list[int]
            Local dimensions
        sys_indices: list[int]
            Indices of subsystems to transpose

        Returns
        -------
        np.ndarray
            Matrix after partial transpose on the requested subsystem
        """
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
        """
        Log-negativity for the early|late partition.

        Performs a partial transpose on the "late" indices, computes
        the trace via singular values, and returns log2(norm).
        An auxiliary error measure `max(0,1-E_N)` is also returned.

        Parameters
        ----------
        rho_phot: qutip.Qobj or numpy.ndarray
            Photonic density matrix (QD traced out)
        dims_phot: list[int]
            Local photonic dimensions (no QD in this cut).
        early_idxs: list [int]
            Indices of early modes
        late_idxs: list[int]
            Indices of late modes

        Returns
        -------
        tuple[float, float]
            (E_N, err), (log_negativity, deviation from 1)
        """
        R = self._ensure_dense_normalized(rho_phot, dims_phot)
        Rpt = self._pt_on_subsystems(R, dims_phot, late_idxs)
        s = np.linalg.svd(Rpt, compute_uv=False)
        trace_norm = float(np.sum(s).real)
        E_N = float(np.log2(trace_norm))
        err = float(max(0.0, 1.0 - E_N))
        return E_N, err

    def effective_overlap(
        self, which: Literal["early", "late", "avg"] = "late"
    ) -> float:
        """
        Overlap <psi_H|psi_V>

        Uses gamma (effective linewidth) for the chosen pair and
        Delta=FSS/habr.
        Returns gamma/sqrd(gamma^2+Delta^2), clamped to [0,1]

        Parameters
        ----------
        which: {"early", "late", "avg"}, optional
            Which pari linewidth to use, default "late"

        Returns
        -------
        float
            Overlap in [0,1]. Returns 0.0
        """
        gamma = self._pair_linewidth(which)
        if gamma <= 0.0:
            return 0.0
        delta = self._delta_rad_per_s()
        denom = math.hypot(gamma, delta)
        return float(max(0.0, min(1.0, gamma / denom)))

    def _transition_energy_pair_eV(self, tr: Transition) -> tuple[float, float]:
        """
        Initial and final energies (eV) for a given transition label.

        Parameters:
        -----------
        tr: Transition
            Transition enum with `.name`

        Returns
        -------
        tuple[float, float]
            (E_i, E_f) in eV
        """
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
        """
        Central frequency for a transition

        Computes angular frequency, frequency and wavelength.

        Parameters
        ----------
        tr: Transition
            Transition object

        Returns
        -------
        dict[str, float]
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
        """
        Central frequencies for each mode

        Returns
        -------
        dict[str, dict[str, float]]
        """
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
        """
        Map a photonic factor index to (mode_label, polarization).

        PLUS first, then MINUS.
        `offset` is subtracted before mapping.

        Parameters
        ----------
        factor_idx: int
            Global factor index (including offset)
        offset: int
            Number of non-photonic or earlyer factors to skip

        Returns
        -------
        tuple[str, str]
            (label, "+") or (label, "-")

        Raises
        ------
        ValueError
            If `factor_idx<offset`.
        IndexError
            If the derived mode index is out of range of the registry
        """
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
        """
        Photon-number expectations for early and late emissions with simple
        error.

        Builds per-emission number operators by summing the projectors
        from observable_provider, normalizes `rho_phot`, and evaluates
        `N_early`, `N_late`. Errors are |1-N| for each emission.

        Parameters
        ----------
        rho_pho: qutip.Qobj or numpy.ndarray
            Density matrix with QD traced out

        Returns
        -------
        tuple[dict[str, float], dict[str, float]]
            (values, errors)
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
        """
        Summary of the mode layout, rates, overlaps, and optional state metrics
        if the `rho_phot` is provided

        Metrics
        -------
        - `num_intrinsic_modes`, `labels`, `mode_transitions`
        - `central_frequencies` (omega, f, lambda) per mode
        - `rates_per_s` and derived bandwidths (rad/s, Hz)
        - `overlap_abs_{early, late, avg}` and `HOM_visibility` for each

        If `rho_phot` given:
        - `population_breakdown`
        - `photon_numebrs` for emissions
        - `entanglement` characteristics (log_negativity, conditional
           log_negativity on 2-photon)
        - `purity` (unconditional, contidional)
        - `probabilities` (p_two_photon)
        - `bel-component`

        Parameters
        ----------
        rho_phot: qutip.Qobj or np.ndarray, optional
            Density matrix with QD traced out (only photonic part remaining)

        Returns
        -------
        dict[str, Any]
            A structured dictionary of scalar metrics and per-mode summaries.
        """
        reg = build_registry(self.qd, self._modes, self._observable_provider)

        oc = OverlapCalculator(
            gamma_XX_X1=float(self._g.get("L_XX_X1", 0.0)),
            gamma_XX_X2=float(self._g.get("L_XX_X2", 0.0)),
            gamma_X1_G=float(self._g.get("L_X1_G", 0.0)),
            gamma_X2_G=float(self._g.get("L_X2_G", 0.0)),
            delta_rad_s=self._delta_rad_per_s(),
        )

        summary: Dict[str, Any] = {
            "num_intrinsic_modes": len(reg.labels_by_mode_index),
            "labels": reg.labels_by_mode_index,
            "overlap_abs_early": oc.overlap("early"),
            "overlap_abs_late": oc.overlap("late"),
            "overlap_abs_avg": oc.overlap("avg"),
            "HOM_visibility": {
                "early": oc.hom("early"),
                "late": oc.hom("late"),
                "avg": oc.hom("avg"),
            },
            "fss_eV": float(getattr(self._EL, "fss", 0.0)),
            "central_frequencies": self.central_frequencies_by_mode(),
            "rates_per_s": {
                "L_XX_X1": float(self._g.get("L_XX_X1", 0.0)),
                "L_XX_X2": float(self._g.get("L_XX_X2", 0.0)),
                "L_X1_G": float(self._g.get("L_X1_G", 0.0)),
                "L_X2_G": float(self._g.get("L_X2_G", 0.0)),
            },
            "bandwidths_rad_s": {
                lab: self._mode_label_to_rate_per_s(lab)
                for lab in reg.labels_by_mode_index
            },
            "bandwidths_Hz": {
                lab: self._mode_label_to_rate_per_s(lab) / (2.0 * np.pi)
                for lab in reg.labels_by_mode_index
            },
            "mode_transitions": self._mode_transitions_by_label(),
        }

        if rho_phot is not None:
            pc = PhotonCounter(reg)
            ent = EntanglementCalculator(reg)
            tp = TwoPhotonProjector(reg)
            P2 = tp.projector()
            pop = PopulationDecomposer(reg).p0_p1_p2_exact_multi(rho_phot, P2)
            pn = pc.counts(rho_phot)
            E_N = ent.log_neg_early_late(rho_phot)

            R = ensure_rho(rho_phot, reg.dims_phot)
            purity_uncond = purity(R)

            R2, p2 = tp.postselect(rho_phot)
            if p2 > 0.0:
                E_N_cond = ent.log_neg_early_late(R2)
                purity_cond = purity(R2)
            else:
                E_N_cond = purity_cond = 0.0
            bell = {}
            if p2 > 0.0:
                bell = BellAnalyzer(reg).analyze(R2)

            summary.update(
                {
                    "population_breakdown": pop,
                    "bell_component": bell,
                }
            )
            summary.update(
                {
                    "photon_numbers": pn,
                    "entanglement": {
                        "log_negativity": E_N,
                        "log_negativity_conditional": E_N_cond,
                    },
                    "purity": {
                        "unconditional": purity_uncond,
                        "conditional_two_photon": purity_cond,
                    },
                    "probabilities": {"p_two_photon": p2},
                }
            )
        return summary
