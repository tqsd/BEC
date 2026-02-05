from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from matplotlib import ticker as mticker
from matplotlib.axes import Axes

from .labels import ax_label
from .styles import PlotStyle
from .traces import DriveSeries, QDTraces
from .twin_registry import clear_twin_axes, register_twin_axes

_TWO_PI = 2.0 * np.pi


def _as_1d(x) -> np.ndarray:
    return np.asarray(x).reshape(-1)


def _sci(ax: Axes, *, right: bool = False) -> None:
    fmt = mticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(fmt)
    if right:
        ax.yaxis.set_offset_position("right")


def _tag(h, group: str) -> None:
    try:
        h.set_gid("legend:%s" % group)
    except Exception:
        pass


def _strip_parenthesized_nm(s: str) -> str:
    out = str(s).strip()
    while True:
        i = out.find("(")
        if i < 0:
            break
        j = out.find(")", i + 1)
        if j < 0:
            break
        inside = out[i + 1 : j].strip().lower()
        if inside.endswith("nm"):
            out = (out[:i] + out[j + 1 :]).strip()
            continue
        break
    return out


def _drive_base_label(d: DriveSeries) -> str:
    lab = d.label if isinstance(d.label, str) else ""
    lab = lab.strip() if lab else "drive"
    return _strip_parenthesized_nm(lab)


def _output_transition_tex(k: str) -> str:
    if "X1G" in k or "X1_G" in k or "X1G" in k.replace("_", ""):
        return r"$\mathrm{X}_1 \leftrightarrow \mathrm{G}$"
    if "X2G" in k or "X2_G" in k or "X2G" in k.replace("_", ""):
        return r"$\mathrm{X}_2 \leftrightarrow \mathrm{G}$"
    if "XX_X1" in k or "XX_X1" in k.replace("-", "_"):
        return r"$\mathrm{XX} \leftrightarrow \mathrm{X}_1$"
    if "XX_X2" in k or "XX_X2" in k.replace("-", "_"):
        return r"$\mathrm{XX} \leftrightarrow \mathrm{X}_2$"

    if k.startswith("n_GX_"):
        return r"$\mathrm{X} \leftrightarrow \mathrm{G}$"
    if k.startswith("n_XX_"):
        return r"$\mathrm{XX} \leftrightarrow \mathrm{X}_1$"

    return r"$%s$" % str(k)


def _output_pol_tex(k: str) -> str:
    if k.endswith("_H"):
        return r"$(H)$"
    if k.endswith("_V"):
        return r"$(V)$"
    return ""


def output_label_pretty(k: str) -> str:
    tr_tex = _output_transition_tex(k)
    pol = _output_pol_tex(k)
    if pol:
        return tr_tex + ", " + pol
    return tr_tex


def _is_effectively_zero(arr: np.ndarray, *, atol: float, rtol: float) -> bool:
    a = np.asarray(arr, dtype=float).reshape(-1)
    if a.size == 0:
        return True
    m = float(np.nanmax(np.abs(a)))
    if not np.isfinite(m):
        return False
    thr = float(atol) + float(rtol) * max(1.0, m)
    return m <= thr


def _rad_s_to_GHz(dw_rad_s: np.ndarray) -> np.ndarray:
    return np.asarray(dw_rad_s, dtype=float) / (_TWO_PI * 1e9)


def _markers_from_spacing_x(
    t: np.ndarray, *, every_x: float, offset_x: float = 0.0
) -> np.ndarray:
    """
    Return explicit marker indices such that markers are spaced by `every_x`
    in the x units of `t` (the same `t` you pass to ax.plot).
    """
    tt = np.asarray(t, dtype=float).reshape(-1)
    n = int(tt.size)
    if n == 0:
        return np.array([], dtype=int)

    ex = float(every_x)
    if not np.isfinite(ex) or ex <= 0.0:
        return np.array([], dtype=int)

    x0 = float(tt[0]) + float(offset_x)
    x1 = float(tt[-1])

    # include both ends
    xs = np.arange(x0, x1 + 0.5 * ex, ex, dtype=float)
    idx = np.searchsorted(tt, xs, side="left")
    idx = np.unique(np.clip(idx, 0, n - 1))
    return idx.astype(int)


def _maybe_marker_kwargs(
    style: PlotStyle,
    *,
    t: np.ndarray,
    want_markers: bool,
    color: str | None,
) -> dict:
    """
    Marker control:
      1) If style.marker_every_x is set (>0), use physical spacing in x units.
      2) Else fall back to style.dotted_markevery as "every N points".
    """
    if not want_markers:
        return {}

    c = color if color is not None else "C0"

    every_x = float(getattr(style, "marker_every_x", 0.0) or 0.0)
    offset_x = float(getattr(style, "marker_offset_x", 0.0) or 0.0)

    if every_x > 0.0:
        me = _markers_from_spacing_x(t, every_x=every_x, offset_x=offset_x)
        # If something went wrong, fall back to integer markevery.
        if me.size > 0:
            return dict(
                marker=getattr(style, "dotted_marker", "o"),
                markevery=me,
                markersize=float(getattr(style, "dotted_markersize", 5.5)),
                markerfacecolor=c,
                markeredgecolor=c,
                markeredgewidth=0.0,
            )

    # fallback: points-based
    mp = int(getattr(style, "dotted_markevery", 12))
    mp = 1 if mp <= 0 else mp
    return dict(
        marker=getattr(style, "dotted_marker", "o"),
        markevery=mp,
        markersize=float(getattr(style, "dotted_markersize", 5.5)),
        markerfacecolor=c,
        markeredgecolor=c,
        markeredgewidth=0.0,
    )


def _limit_yticks(ax: Axes, *, nbins: int = 3) -> None:
    """
    Useful for narrow subplot columns: fewer tick labels.
    """
    try:
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=nbins))
    except Exception:
        pass


def draw_drive_panel(
    ax: Axes,
    *,
    t: np.ndarray,
    tr: QDTraces,
    style: PlotStyle,
    show_right_axis: bool = True,
) -> None:
    ax.grid(True, alpha=style.grid_alpha)
    clear_twin_axes(ax)

    drives: Sequence[DriveSeries] = getattr(tr, "drives", ()) or ()
    if not drives:
        ax.text(
            0.5,
            0.5,
            "No drive",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_ylabel(ax_label("Envelope", "E(t)", r"\volt\per\meter"))
        return

    palette = getattr(style, "palette_drives", None) or ["C0", "C1", "C2", "C3", "C4"]
    uniq = list(dict.fromkeys([d.label for d in drives]))
    color_of = {lab: palette[i % len(palette)] for i, lab in enumerate(uniq)}

    any_env = False
    for d in drives:
        if d.E_env_V_m is None:
            continue
        any_env = True

        base = color_of.get(d.label, "C0")
        lab = _drive_base_label(d)

        h = ax.plot(
            t,
            _as_1d(d.E_env_V_m).astype(float),
            lw=1.8,
            ls="-",
            color=base,
            label=lab,
        )[0]
        _tag(h, "drives")

    ax.set_ylabel(ax_label("Envelope", "E(t)", r"\volt\per\meter"))
    _sci(ax, right=False)
    _limit_yticks(ax, nbins=int(getattr(style, "yticks_nbins_drive", 3) or 3))

    detuning_atol_rad_s = float(getattr(style, "detuning_atol_rad_s", 1e8))
    detuning_rtol = float(getattr(style, "detuning_rtol", 0.0))
    use_GHz = bool(getattr(style, "detuning_use_GHz", True))

    detuning_curves: list[tuple[DriveSeries, np.ndarray]] = []
    for d in drives:
        dw = getattr(d, "delta_omega_rad_s", None)
        if dw is None:
            continue
        arr = _as_1d(dw).astype(float)
        if arr.shape[0] != t.shape[0]:
            continue
        if _is_effectively_zero(arr, atol=detuning_atol_rad_s, rtol=detuning_rtol):
            continue
        detuning_curves.append((d, arr))

    if show_right_axis and detuning_curves:
        ax2 = ax.twinx()
        register_twin_axes(ax, ax2)
        ax2.grid(False)

        ymax = 0.0
        for d, dw in detuning_curves:
            base = color_of.get(d.label, "C0")
            lab = _drive_base_label(d)

            y = _rad_s_to_GHz(dw) if use_GHz else dw
            if y.size:
                ymax = max(ymax, float(np.nanmax(np.abs(y))))

            h2 = ax2.plot(
                t,
                y,
                lw=1.4,
                ls=":",
                color=base,
                label="%s: chirp" % lab,
            )[0]
            _tag(h2, "chirp")

        if use_GHz:
            ax2.set_ylabel(ax_label("Chirp", r"\Delta f(t)", r"\giga\hertz"))
            _limit_yticks(ax2, nbins=int(getattr(style, "yticks_nbins_detuning", 3) or 3))
        else:
            ax2.set_ylabel(ax_label("Chirp", r"\Delta\omega(t)", r"rad\per\second"))
            _sci(ax2, right=True)
            _limit_yticks(ax2, nbins=int(getattr(style, "yticks_nbins_detuning", 3) or 3))

        if ymax > 0.0 and np.isfinite(ymax):
            ax2.set_ylim(-1.05 * ymax, 1.05 * ymax)

    if not any_env:
        ax.text(
            0.5,
            0.5,
            "No envelope",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )


def draw_pops_panel(
    ax: Axes,
    *,
    t: np.ndarray,
    tr: QDTraces,
    style: PlotStyle,
) -> None:
    ax.grid(True, alpha=style.grid_alpha)
    clear_twin_axes(ax)

    pops: Mapping[str, np.ndarray] = getattr(tr, "pops", {}) or {}
    if not pops:
        ax.text(
            0.5,
            0.5,
            "No populations",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_ylabel(ax_label("QD pop", None, r"1"))
        return

    order = list(style.qd_labels) if style.qd_labels else ["pop_G", "pop_X1", "pop_X2", "pop_XX"]
    keys = [k for k in order if k in pops] + [k for k in pops if k not in order]

    # Decide which population curves get markers:
    # - default: only those whose linestyle is a tuple (your existing behavior)
    # - override via style.pop_markers="all" or style.pop_markers=["pop_X2", ...]
    pop_markers = getattr(style, "pop_markers", None)

    for k in keys:
        y = _as_1d(pops[k])
        if y.shape[0] != t.shape[0]:
            raise ValueError("%s length %d != t length %d" % (k, int(y.shape[0]), int(t.shape[0])))

        yr = np.real(y).astype(float)

        color = style.qd_colors.get(k, None)
        ls = style.qd_styles.get(k, "-")
        lw = float(style.qd_widths.get(k, 1.4))
        alpha = float(style.qd_alphas.get(k, 0.9))
        lab = style.qd_labels_tex.get(k, k)

        if pop_markers == "all":
            want_markers = True
        elif isinstance(pop_markers, (list, tuple, set)):
            want_markers = k in set(pop_markers)
        else:
            want_markers = isinstance(ls, tuple)

        h = ax.plot(
            t,
            yr,
            label=str(lab),
            color=color,
            ls=ls,
            lw=lw,
            alpha=alpha,
            **_maybe_marker_kwargs(style, t=t, want_markers=want_markers, color=color),
        )[0]
        _tag(h, "pops")

    ax.set_ylabel(ax_label("QD pop", None, r"1"))
    ax.set_ylim(-0.05, 1.05)
    _limit_yticks(ax, nbins=int(getattr(style, "yticks_nbins_pops", 4) or 4))


def draw_outputs_panel(
    ax: Axes,
    *,
    t: np.ndarray,
    tr: QDTraces,
    style: PlotStyle,
) -> None:
    ax.grid(True, alpha=style.grid_alpha)
    clear_twin_axes(ax)

    outputs: Mapping[str, np.ndarray] = getattr(tr, "outputs", {}) or {}
    if not outputs:
        ax.text(
            0.5,
            0.5,
            "No outputs",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_ylabel(ax_label("Output", r"\langle n \rangle", r"1"))
        return

    order = ["n_GX_H", "n_GX_V", "n_XX_H", "n_XX_V"]
    keys = [k for k in order if k in outputs] + [k for k in outputs if k not in order]

    palette = style.palette_outputs if style.palette_outputs else ["C0", "C1", "C2", "C3"]
    color_map = {k: palette[i % len(palette)] for i, k in enumerate(keys)}

    # If you want true marker dots and not "dash-dots", keep linestyle simple
    ls_H = "-"
    ls_V = "--"

    ymax = 0.0
    for k in keys:
        y = _as_1d(outputs[k])
        if y.shape[0] != t.shape[0]:
            raise ValueError("%s length %d != t length %d" % (k, int(y.shape[0]), int(t.shape[0])))

        yr = np.real(y).astype(float)
        if yr.size:
            ymax = max(ymax, float(np.nanmax(yr)))

        if k.endswith("_H"):
            ls = ls_H
            lw = 2.0
            want_markers = bool(getattr(style, "output_markers_H", True))
        elif k.endswith("_V"):
            ls = ls_V
            lw = 1.4
            want_markers = bool(getattr(style, "output_markers_V", False))
        else:
            ls = "-"
            lw = 1.4
            want_markers = False

        color = color_map.get(k, "C0")
        h = ax.plot(
            t,
            yr,
            label=output_label_pretty(k),
            color=color,
            ls=ls,
            lw=lw,
            **_maybe_marker_kwargs(style, t=t, want_markers=want_markers, color=color),
        )[0]
        _tag(h, "outputs")

    ax.set_ylabel(ax_label("Output", r"\langle n \rangle", r"1"))
    ax.set_ylim(-0.05, max(0.5, 1.10 * ymax))
    _limit_yticks(ax, nbins=int(getattr(style, "yticks_nbins_outputs", 4) or 4))


def draw_coupling_panel(
    ax: Axes,
    *,
    t: np.ndarray,
    tr: QDTraces,
    style: PlotStyle,
    mode: str = "abs",
    max_traces: int = 6,
    show_right_axis: bool = True,
) -> None:
    ax.grid(True, alpha=style.grid_alpha)
    clear_twin_axes(ax)
    ax.text(
        0.5,
        0.5,
        "Detuning is shown in the Drive panel",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    ax.set_ylabel(ax_label("Detuning", None, r"1"))
