from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from bec.metrics.bell import fidelity_to_bell
from bec.metrics.decompose import PhotonDecomposition, decompose_photons
from bec.metrics.entanglement import log_negativity
from bec.metrics.linops import partial_trace
from bec.metrics.mode_registry import MetricGroups, default_groups, indices_of
from bec.metrics.photon_count import expected_n_group, expected_n_per_mode
from bec.metrics.two_photon import TwoPhotonResult, two_photon_postselect


@dataclass(frozen=True)
class StateSanity:
    trace: float
    hermitian_error: float
    min_eig: float


@dataclass(frozen=True)
class PhotonCounts:
    n_gx_total: float
    n_xx_total: float
    n_gx_h: float
    n_gx_v: float
    n_xx_h: float
    n_xx_v: float


@dataclass(frozen=True)
class QDMetrics:
    sanity: StateSanity
    qd_pop: Mapping[str, float]

    photons_all: PhotonDecomposition
    photons_gx: PhotonDecomposition
    photons_xx: PhotonDecomposition

    counts: PhotonCounts

    two_photon: TwoPhotonResult
    bell_fidelity_phi_plus: float
    log_negativity_pol: float

    meta: Mapping[str, Any] = field(default_factory=dict)

    def to_text(self, *, precision: int = 6) -> str:
        """
        Return a plain-text, human-readable report of all metrics.
        Safe for printing, logging, and tests (ASCII only).
        """
        p = int(precision)

        def f(x: float) -> str:
            return f"{x:.{p}g}"

        lines: list[str] = []

        lines.append("=" * 78)
        lines.append("QUANTUM DOT METRICS REPORT")
        lines.append("=" * 78)

        # --- Sanity ---
        lines.append("")
        lines.append("STATE SANITY")
        lines.append("-" * 78)
        lines.append(f"Trace               : {f(self.sanity.trace)}")
        lines.append(f"Hermiticity error   : {f(self.sanity.hermitian_error)}")
        lines.append(f"Min eigenvalue      : {f(self.sanity.min_eig)}")

        # --- QD populations ---
        lines.append("")
        lines.append("QD POPULATIONS (final state)")
        lines.append("-" * 78)
        for k in ("G", "X1", "X2", "XX"):
            if k in self.qd_pop:
                lines.append(f"{k:>3} : {f(self.qd_pop[k])}")

        # --- Photon number decomposition ---
        def block_decomp(title: str, d: PhotonDecomposition) -> None:
            lines.append("")
            lines.append(title)
            lines.append("-" * 78)
            lines.append(f"p0        : {f(d.p0)}")
            lines.append(f"p1_total  : {f(d.p1_total)}")
            lines.append(f"p2_exact  : {f(d.p2_exact)}")

        block_decomp("PHOTON NUMBER DECOMPOSITION (GX + XX)", self.photons_all)
        block_decomp("PHOTON NUMBER DECOMPOSITION (GX only)", self.photons_gx)
        block_decomp("PHOTON NUMBER DECOMPOSITION (XX only)", self.photons_xx)

        # --- Photon counts ---
        lines.append("")
        lines.append("PHOTON NUMBER EXPECTATION VALUES")
        lines.append("-" * 78)
        lines.append(f"<n_GX_total> : {f(self.counts.n_gx_total)}")
        lines.append(f"<n_XX_total> : {f(self.counts.n_xx_total)}")
        lines.append(f"<n_GX_H>     : {f(self.counts.n_gx_h)}")
        lines.append(f"<n_GX_V>     : {f(self.counts.n_gx_v)}")
        lines.append(f"<n_XX_H>     : {f(self.counts.n_xx_h)}")
        lines.append(f"<n_XX_V>     : {f(self.counts.n_xx_v)}")

        # --- Two-photon metrics ---
        lines.append("")
        lines.append("TWO-PHOTON POSTSELECTION (early=XX, late=GX)")
        lines.append("-" * 78)
        lines.append(f"P(n_early=1, n_late=1) : {f(self.two_photon.p11)}")

        if self.two_photon.p11 > 0.0:
            lines.append(
                f"Bell fidelity (phi+)  : {f(self.bell_fidelity_phi_plus)}"
            )
            lines.append(
                f"Log negativity       : {
                    f(self.log_negativity_pol)}"
            )
        else:
            lines.append("Bell fidelity (phi+)  : n/a (p11 = 0)")
            lines.append("Log negativity       : n/a (p11 = 0)")

        # --- Meta ---
        if self.meta:
            lines.append("")
            lines.append("META")
            lines.append("-" * 78)
            for k in sorted(self.meta.keys()):
                lines.append(f"{k} : {self.meta[k]}")

        lines.append("")
        lines.append("=" * 78)

        return "\n".join(lines)


def _rho_final_from_result(res: Any) -> np.ndarray:
    st = getattr(res, "states", None)
    if st is None:
        raise ValueError(
            "MESolveResult.states is None; cannot compute end metrics"
        )

    # Case A: states is a single ndarray (D,D)
    if isinstance(st, np.ndarray) and st.ndim == 2:
        return np.asarray(st, dtype=complex)

    # Case B: states is a sequence; take last
    try:
        last = st[-1]
    except Exception as e:
        raise TypeError(
            "Unsupported states container type for MESolveResult.states"
        ) from e

    return np.asarray(last, dtype=complex)


def _state_sanity(rho: np.ndarray, *, eps: float = 1e-12) -> StateSanity:
    rho = np.asarray(rho, dtype=complex)
    tr = float(np.real_if_close(np.trace(rho)))
    herm_err = float(np.linalg.norm(rho - rho.conjugate().T, ord="fro"))

    # min eigenvalue (can be slightly negative numerically)
    try:
        w = np.linalg.eigvalsh((rho + rho.conjugate().T) * 0.5)
        min_e = float(np.min(np.real_if_close(w)))
    except Exception:
        min_e = float("nan")

    # Optional: clip tiny negatives in reporting is a choice; here we just report.
    return StateSanity(trace=tr, hermitian_error=herm_err, min_eig=min_e)


def _qd_populations(
    rho_full: np.ndarray, dims: Sequence[int], qd_index: int = 0
) -> dict[str, float]:
    # Reduce to QD only and take diagonal in basis (G, X1, X2, XX)
    rho_qd = partial_trace(rho_full, dims, keep=[int(qd_index)])
    diag = np.real_if_close(np.diag(rho_qd)).astype(float).reshape(-1)
    # QD dim is fixed 4 in your model
    labels = ["G", "X1", "X2", "XX"]
    out: dict[str, float] = {}
    for i, lab in enumerate(labels):
        out[lab] = float(diag[i]) if i < len(diag) else 0.0
    return out


@dataclass(frozen=True)
class QDDiagnostics:
    """
    Master metrics facade.

    Usage:
        diag = QDDiagnostics()
        m = diag.compute(qd, res, units=units)

    Notes:
    - Convention: early = XX band, late = GX band.
    - Assumes modes are QDModes ordering: qd, GX_H, GX_V, XX_H, XX_V
    """

    groups: Optional[MetricGroups] = None
    bell_target: str = "phi_plus"

    def compute(
        self, qd: Any, res: Any, *, units: Optional[Any] = None
    ) -> QDMetrics:
        # Get modes/dims from compile_bundle (units needed by your API)
        if units is None:
            raise ValueError(
                "units must be provided to compile_bundle for metrics"
            )

        bundle = qd.compile_bundle(units=units)
        modes = bundle.modes
        dims = list(modes.dims())

        rho_final = _rho_final_from_result(res)

        sanity = _state_sanity(rho_final)
        qd_pop = _qd_populations(rho_final, dims, qd_index=modes.index_of("qd"))

        # Build groups
        groups = self.groups or default_groups(modes)

        # Indices
        gx_idx = indices_of(modes, groups.gx)  # GX_H, GX_V
        xx_idx = indices_of(modes, groups.xx)  # XX_H, XX_V
        all_ph_idx = gx_idx + xx_idx

        # Photon decompositions on selected mode sets
        photons_all = decompose_photons(
            rho_final, dims, keep_mode_indices=all_ph_idx
        )
        photons_gx = decompose_photons(
            rho_final, dims, keep_mode_indices=gx_idx
        )
        photons_xx = decompose_photons(
            rho_final, dims, keep_mode_indices=xx_idx
        )

        # Photon counts (expectation of number operators)
        n_gx_h = expected_n_per_mode(rho_final, dims, gx_idx[0])
        n_gx_v = expected_n_per_mode(rho_final, dims, gx_idx[1])
        n_xx_h = expected_n_per_mode(rho_final, dims, xx_idx[0])
        n_xx_v = expected_n_per_mode(rho_final, dims, xx_idx[1])

        counts = PhotonCounts(
            n_gx_total=float(n_gx_h + n_gx_v),
            n_xx_total=float(n_xx_h + n_xx_v),
            n_gx_h=float(n_gx_h),
            n_gx_v=float(n_gx_v),
            n_xx_h=float(n_xx_h),
            n_xx_v=float(n_xx_v),
        )

        # Two-photon postselection and polarization metrics
        # Convention: early=XX, late=GX
        tp = two_photon_postselect(
            rho_final,
            dims,
            early_indices=xx_idx,
            late_indices=gx_idx,
        )

        if tp.p11 > 0.0:
            bell_f = fidelity_to_bell(tp.rho_pol, which=self.bell_target)
            ln = log_negativity(tp.rho_pol, dims=(2, 2), sys=0)
        else:
            bell_f = 0.0
            ln = 0.0

        meta = dict(getattr(res, "meta", {}) or {})
        meta.update(
            {
                "dims": tuple(dims),
                "channels": tuple(getattr(modes, "channels", []) or []),
                "bell_target": self.bell_target,
                "early_band": "XX",
                "late_band": "GX",
            }
        )

        return QDMetrics(
            sanity=sanity,
            qd_pop=qd_pop,
            photons_all=photons_all,
            photons_gx=photons_gx,
            photons_xx=photons_xx,
            counts=counts,
            two_photon=tp,
            bell_fidelity_phi_plus=float(bell_f),
            log_negativity_pol=float(ln),
            meta=meta,
        )
