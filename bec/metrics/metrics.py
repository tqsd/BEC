from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bec.metrics.bell import fidelity_to_bell, bell_component_from_rho_pol
from bec.metrics.decompose import PhotonDecomposition, decompose_photons
from bec.metrics.entanglement import log_negativity
from bec.metrics.linops import partial_trace
from bec.metrics.mode_registry import MetricGroups, default_groups, indices_of
from bec.metrics.overlap import overlap_metrics_as_dict
from bec.metrics.photon_count import expected_n_per_mode
from bec.metrics.two_photon import TwoPhotonResult, two_photon_postselect


def _fmt_num(x: float, precision: int) -> str:
    return f"{x:.{int(precision)}g}"


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _delta_rel(a: float, b: float) -> tuple[float, float]:
    """
    Return (delta, rel), where:
      delta = b - a
      rel = (b - a) / max(abs(a), eps)
    """
    eps = 1e-15
    da = b - a
    denom = max(abs(a), eps)
    return da, da / denom


def _fmt_triplet(a: float, b: float, precision: int) -> str:
    d, r = _delta_rel(a, b)
    a_s = _fmt_num(a, precision)
    b_s = _fmt_num(b, precision)
    d_s = _fmt_num(d, precision)
    r_s = _fmt_num(100.0 * r, precision)
    return f"{a_s} -> {b_s}   d={d_s}   rel={r_s}%"


def _is_finite(x: float) -> bool:
    return math.isfinite(float(x))


def _log_negativity_photons_early_late(
    rho_full: np.ndarray,
    dims: Sequence[int],
    early_indices: Sequence[int],
    late_indices: Sequence[int],
    *,
    sys: int = 0,
) -> float:
    """
    Log-negativity of the full photonic state across the bipartition:
      early band (XX_H, XX_V) | late band (GX_H, GX_V)

    This is computed on rho_ph = Tr_QD(rho_full) restricted to the four optical modes.
    """
    dims = [int(d) for d in dims]
    e = [int(i) for i in early_indices]
    l = [int(i) for i in late_indices]
    keep = e + l  # order: early then late

    rho_opt = partial_trace(rho_full, dims, keep=keep)

    de = int(dims[e[0]])
    if int(dims[e[1]]) != de:
        raise ValueError("early H/V dims mismatch")

    dl = int(dims[l[0]])
    if int(dims[l[1]]) != dl:
        raise ValueError("late H/V dims mismatch")

    d_early = de * de
    d_late = dl * dl

    # rho_opt is on (eH,eV,lH,lV) i.e. dims (de,de,dl,dl)
    # reshape to (early_pair, late_pair) so log_negativity can treat it as bipartite
    rho_bip = np.asarray(rho_opt, dtype=complex).reshape(
        (d_early * d_late, d_early * d_late)
    )

    return float(log_negativity(rho_bip, dims=(d_early, d_late), sys=int(sys)))


def _purity(rho: np.ndarray) -> float:
    rho = np.asarray(rho, dtype=complex)
    try:
        val = np.trace(rho @ rho)
        val = np.real_if_close(val)
        return float(np.real(val))
    except Exception:
        return float("nan")


def _purity_reduced(
    rho_full: np.ndarray, dims: Sequence[int], keep: Sequence[int]
) -> float:
    rho_red = partial_trace(rho_full, dims, keep=[int(i) for i in keep])
    return _purity(rho_red)


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
    log_negativity_uncond: float
    log_negativity_pol: float

    purity_full: float
    purity_photons_all: float
    purity_qd: float
    purity_pol_post: float

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

        lines.append("")
        lines.append("PURITY")
        lines.append("-" * 78)
        lines.append(f"Purity (full)        : {f(self.purity_full)}")
        lines.append(f"Purity (QD reduced)  : {f(self.purity_qd)}")
        lines.append(f"Purity (photons GX+XX): {f(self.purity_photons_all)}")
        if self.two_photon.p11 > 0.0:
            lines.append(f"Purity (pol, post)   : {f(self.purity_pol_post)}")
        else:
            lines.append("Purity (pol, post)   : n/a (p11 = 0)")

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

    def compare(
        self,
        other: QDMetrics,
        *,
        name_self: str = "A",
        name_other: str = "B",
        precision: int = 6,
        include_meta: bool = False,
    ) -> str:
        """
        Compare two QDMetrics objects.

        Returns a plain-text report (ASCII only) suitable for printing/logging.

        Convention:
          - We show A -> B plus delta and relative change (relative to A).
          - For quantities that can be 0 (p11 etc), relative change is still computed
            with a tiny epsilon denominator.
        """
        p = int(precision)
        lines: list[str] = []

        lines.append("=" * 78)
        lines.append("QDMETRICS COMPARISON REPORT")
        lines.append("=" * 78)
        lines.append(f"{name_self} -> {name_other}")
        lines.append("")

        # --- Sanity
        lines.append("STATE SANITY")
        lines.append("-" * 78)
        lines.append(
            f"Trace             : {_fmt_triplet(
                self.sanity.trace, other.sanity.trace, p)}"
        )
        lines.append(
            f"Hermiticity error : {_fmt_triplet(
                self.sanity.hermitian_error, other.sanity.hermitian_error, p)}"
        )
        lines.append(
            f"Min eigenvalue    : {_fmt_triplet(
                self.sanity.min_eig, other.sanity.min_eig, p)}"
        )

        # --- QD populations
        lines.append("")
        lines.append("QD POPULATIONS (final state)")
        lines.append("-" * 78)
        for k in ("G", "X1", "X2", "XX"):
            a = _safe_float(self.qd_pop.get(k, 0.0))
            b = _safe_float(other.qd_pop.get(k, 0.0))
            lines.append(f"{k:>3} : {_fmt_triplet(a, b, p)}")

        # --- Photon decompositions helper
        def block_decomp(
            title: str, da: PhotonDecomposition, db: PhotonDecomposition
        ) -> None:
            lines.append("")
            lines.append(title)
            lines.append("-" * 78)
            lines.append(
                f"p0       : {_fmt_triplet(
                    _safe_float(da.p0), _safe_float(db.p0), p)}"
            )
            lines.append(
                f"p1_total : {_fmt_triplet(_safe_float(
                    da.p1_total), _safe_float(db.p1_total), p)}"
            )
            lines.append(
                f"p2_exact : {_fmt_triplet(_safe_float(
                    da.p2_exact), _safe_float(db.p2_exact), p)}"
            )

        block_decomp(
            "PHOTON NUMBER DECOMPOSITION (GX + XX)",
            self.photons_all,
            other.photons_all,
        )
        block_decomp(
            "PHOTON NUMBER DECOMPOSITION (GX only)",
            self.photons_gx,
            other.photons_gx,
        )
        block_decomp(
            "PHOTON NUMBER DECOMPOSITION (XX only)",
            self.photons_xx,
            other.photons_xx,
        )

        # --- Counts
        lines.append("")
        lines.append("PHOTON NUMBER EXPECTATION VALUES")
        lines.append("-" * 78)
        lines.append(
            f"<n_GX_total> : {_fmt_triplet(
                self.counts.n_gx_total, other.counts.n_gx_total, p)}"
        )
        lines.append(
            f"<n_XX_total> : {_fmt_triplet(
                self.counts.n_xx_total, other.counts.n_xx_total, p)}"
        )
        lines.append(
            f"<n_GX_H>     : {_fmt_triplet(
                self.counts.n_gx_h, other.counts.n_gx_h, p)}"
        )
        lines.append(
            f"<n_GX_V>     : {_fmt_triplet(
                self.counts.n_gx_v, other.counts.n_gx_v, p)}"
        )
        lines.append(
            f"<n_XX_H>     : {_fmt_triplet(
                self.counts.n_xx_h, other.counts.n_xx_h, p)}"
        )
        lines.append(
            f"<n_XX_V>     : {_fmt_triplet(
                self.counts.n_xx_v, other.counts.n_xx_v, p)}"
        )

        lines.append(f"Purity (full)        : {(self.purity_full)}")
        lines.append(f"Purity (QD reduced)  : {(self.purity_qd)}")
        lines.append(f"Purity (photons GX+XX): {(self.purity_photons_all)}")
        # --- Two-photon and entanglement
        lines.append("")
        lines.append("TWO-PHOTON POSTSELECTION (early=XX, late=GX)")
        lines.append("-" * 78)

        a_p11 = _safe_float(self.two_photon.p11)
        b_p11 = _safe_float(other.two_photon.p11)
        lines.append(
            f"P(n_early=1, n_late=1) : {
                _fmt_triplet(a_p11, b_p11, p)}"
        )

        # Only report fidelity/negativity meaningfully if p11 is nonzero; still show numbers
        a_bf = _safe_float(self.bell_fidelity_phi_plus)
        b_bf = _safe_float(other.bell_fidelity_phi_plus)
        a_ln = _safe_float(self.log_negativity_pol)
        b_ln = _safe_float(other.log_negativity_pol)

        lines.append(f"Purity (pol, post)   : {(self.purity_pol_post)}")
        if a_p11 > 0.0 or b_p11 > 0.0:
            lines.append(
                f"Bell fidelity (phi+)  : {_fmt_triplet(a_bf, b_bf, p)}"
            )
            lines.append(
                f"Log negativity        : {
                    _fmt_triplet(a_ln, b_ln, p)}"
            )
        else:
            lines.append("Bell fidelity (phi+)  : n/a (both p11 = 0)")
            lines.append("Log negativity        : n/a (both p11 = 0)")

        # --- Optional meta
        if include_meta:
            lines.append("")
            lines.append("META (selected keys)")
            lines.append("-" * 78)
            # show common keys only
            keys = sorted(
                set(self.meta.keys()).intersection(set(other.meta.keys()))
            )
            for k in keys:
                va = self.meta.get(k)
                vb = other.meta.get(k)
                if va == vb:
                    lines.append(f"{k} : {va}")
                else:
                    lines.append(f"{k} : {va} -> {vb}")

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

    groups: MetricGroups | None = None
    bell_target: str = "phi_plus"

    def compute(
        self, qd: Any, res: Any, *, units: Any | None = None
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

        purity_full = _purity(rho_final)
        purity_qd = _purity_reduced(
            rho_final, dims, keep=[modes.index_of("qd")]
        )
        purity_photons_all = _purity_reduced(rho_final, dims, keep=all_ph_idx)

        ln_phot = _log_negativity_photons_early_late(
            rho_final,
            dims,
            early_indices=xx_idx,
            late_indices=gx_idx,
            sys=0,
        )

        # Two-photon postselection and polarization metrics
        # Convention: early=XX, late=GX
        tp = two_photon_postselect(
            rho_final,
            dims,
            early_indices=xx_idx,
            late_indices=gx_idx,
        )

        bell_component = None

        if tp.p11 > 0.0:
            bell_f = fidelity_to_bell(tp.rho_pol, which=self.bell_target)
            ln = log_negativity(tp.rho_pol, dims=(2, 2), sys=0)
            purity_pol_post = _purity(tp.rho_pol)
            bc = bell_component_from_rho_pol(tp.rho_pol)
            bell_component = {
                "weights": {
                    # names chosen to match your old interface
                    "p_pp": bc.p_hh,
                    "p_pm": bc.p_hv,
                    "p_mp": bc.p_vh,
                    "p_mm": bc.p_vv,
                    "parallel": bc.parallel,
                    "cross": bc.cross,
                },
                "coherence_cross": {
                    "abs": bc.coh_phi_abs,
                    "phase_rad": bc.phi_phase_rad,
                    "phase_deg": bc.phi_phase_deg,
                },
            }
        else:
            bell_f = 0.0
            ln = 0.0
            purity_pol_post = float("nan")
            purity_pol_post = float("nan")
            bell_component = {
                "weights": {
                    "p_pp": 0.0,
                    "p_pm": 0.0,
                    "p_mp": 0.0,
                    "p_mm": 0.0,
                    "parallel": 0.0,
                    "cross": 0.0,
                },
                "coherence_cross": {
                    "abs": 0.0,
                    "phase_rad": float("nan"),
                    "phase_deg": float("nan"),
                },
            }

        meta = dict(getattr(res, "meta", {}) or {})
        meta.update(
            {
                "dims": tuple(dims),
                "channels": tuple(getattr(modes, "channels", []) or []),
                "bell_target": self.bell_target,
                "early_band": "XX",
                "late_band": "GX",
                "bell_component": bell_component,
            }
        )
        meta.update(overlap_metrics_as_dict(qd))

        return QDMetrics(
            sanity=sanity,
            qd_pop=qd_pop,
            photons_all=photons_all,
            photons_gx=photons_gx,
            photons_xx=photons_xx,
            counts=counts,
            log_negativity_uncond=float(ln_phot),
            two_photon=tp,
            bell_fidelity_phi_plus=float(bell_f),
            log_negativity_pol=float(ln),
            purity_full=float(purity_full),
            purity_qd=float(purity_qd),
            purity_photons_all=float(purity_photons_all),
            purity_pol_post=float(purity_pol_post),
            meta=meta,
        )
