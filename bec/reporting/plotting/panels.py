from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib import ticker as mticker

from .labels import ax_label
from .styles import PlotStyle
from .traces import DriveSeries


def _sci(ax: Axes, *, right: bool = False) -> None:
    fmt = mticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(fmt)
    if right:
        ax.yaxis.set_offset_position("right")


def _legend(ax: Axes, style: PlotStyle, *args, **kwargs):
    leg = ax.legend(
        *args,
        frameon=True,
        facecolor=getattr(style, "legend_facecolor", "white"),
        edgecolor=getattr(style, "legend_edgecolor", "black"),
        framealpha=float(getattr(style, "legend_framealpha", 1.0)),
        fancybox=False,
        borderpad=0.4,
        **kwargs,
    )
    if leg is not None:
        frame = leg.get_frame()
        frame.set_linewidth(float(getattr(style, "legend_linewidth", 0.9)))
        frame.set_edgecolor(getattr(style, "legend_edgecolor", "black"))
        frame.set_facecolor(getattr(style, "legend_facecolor", "white"))
        frame.set_alpha(float(getattr(style, "legend_framealpha", 1.0)))

        # Put legend above everything, including twin axes artists
        leg.set_zorder(10_000)
    return leg


def _maybe_marker_kwargs(style: PlotStyle, *, want_markers: bool, color):
    if not want_markers:
        return {}
    return dict(
        marker=getattr(style, "dotted_marker", "o"),
        markevery=int(getattr(style, "dotted_markevery", 12)),
        markersize=float(getattr(style, "dotted_markersize", 5.5)),
        markerfacecolor=color,
        markeredgecolor=color,
        markeredgewidth=0.0,
    )


def draw_drive_panel(
    ax: Axes,
    *,
    t: np.ndarray,
    drives: Sequence[DriveSeries],
    style: PlotStyle,
) -> None:
    ax.grid(True, alpha=style.grid_alpha)

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

    fig = ax.figure
    ax2 = ax.twinx()

    # Do NOT zorder-hack axes for legends. It can hide the right axis.
    ax2.patch.set_alpha(0.0)  # transparent face, keeps things clean

    palette = getattr(style, "palette_drives", None) or [
        "C0",
        "C1",
        "C2",
        "C3",
        "C4",
    ]
    uniq = list(dict.fromkeys([d.label for d in drives]))
    color_of = {lab: palette[i % len(palette)] for i, lab in enumerate(uniq)}

    ls_env = "-"
    ls_omega = (0, (6.0, 2.5))
    ls_delta = (0, (5.0, 2.0, 1.5, 2.0))
    ls_Omega = (0, (1.2, 5.0))

    def marker_kwargs(want: bool, color: str):
        if not want:
            return {}
        return dict(
            marker=getattr(style, "dotted_marker", "o"),
            markevery=int(getattr(style, "dotted_markevery", 12)),
            markersize=float(getattr(style, "dotted_markersize", 5.5)),
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.0,
        )

    left_handles = []
    left_labels = []
    right_handles = []
    right_labels = []

    for d in drives:
        base = color_of.get(d.label, "C0")

        if d.E_env_V_m is not None:
            h = ax.plot(
                t,
                np.asarray(d.E_env_V_m, dtype=float),
                lw=1.8,
                ls=ls_env,
                color=base,
            )[0]
            left_handles.append(h)
            left_labels.append(rf"{d.label}: $E$")

        if d.omega_L_rad_s is not None:
            h = ax2.plot(
                t,
                np.asarray(d.omega_L_rad_s, dtype=float),
                lw=1.2,
                ls=ls_omega,
                color=base,
                alpha=0.9,
                **marker_kwargs(True, base),
            )[0]
            right_handles.append(h)
            right_labels.append(rf"{d.label}: $\omega_L$")

        if d.delta_omega_rad_s is not None:
            h = ax2.plot(
                t,
                np.asarray(d.delta_omega_rad_s, dtype=float),
                lw=1.2,
                ls=ls_delta,
                color=base,
                alpha=0.9,
            )[0]
            right_handles.append(h)
            right_labels.append(rf"{d.label}: $\Delta\omega$")

        if d.Omega_rad_s is not None:
            h = ax2.plot(
                t,
                np.asarray(d.Omega_rad_s, dtype=float),
                lw=1.3,
                ls=ls_Omega,
                color=base,
                alpha=0.95,
                **marker_kwargs(True, base),
            )[0]
            right_handles.append(h)
            right_labels.append(rf"{d.label}: $\Omega$")

    ax.set_ylabel(ax_label("Envelope", "E(t)", r"\volt\per\meter"))
    ax2.set_ylabel(ax_label("", r"\omega(t)", r"rad\per\second"))

    # ---- robust figure-legend management ----
    # Remove previously created legends (important if you replot in same figure).
    old = getattr(fig, "_smef_drive_legends", None)
    if old:
        for lg in old:
            try:
                lg.remove()
            except Exception:
                pass
    fig._smef_drive_legends = []

    def style_legend(leg):
        fr = leg.get_frame()
        fr.set_facecolor(getattr(style, "legend_facecolor", "white"))
        fr.set_edgecolor(getattr(style, "legend_edgecolor", "black"))
        fr.set_alpha(float(getattr(style, "legend_framealpha", 1.0)))
        fr.set_linewidth(float(getattr(style, "legend_linewidth", 0.9)))
        leg.set_zorder(10000)

    # Place legends in FIGURE coordinates, not axes coords (more stable with layout engines).
    # You can tune these anchors once and they will be consistent.
    if left_handles:
        leg1 = fig.legend(
            left_handles,
            left_labels,
            loc="upper left",
            bbox_to_anchor=(0.1, 0.95),
            frameon=True,
            fancybox=False,
        )
        style_legend(leg1)
        fig._smef_drive_legends.append(leg1)

    if right_handles:
        leg2 = fig.legend(
            right_handles,
            right_labels,
            loc="upper right",
            bbox_to_anchor=(0.92, 0.95),
            frameon=True,
            fancybox=False,
        )
        style_legend(leg2)
        fig._smef_drive_legends.append(leg2)


def draw_pops_panel(
    ax: Axes,
    *,
    t: np.ndarray,
    pops: Mapping[str, np.ndarray],
    style: PlotStyle,
) -> None:
    ax.grid(True, alpha=style.grid_alpha)

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

    order = (
        list(style.qd_labels)
        if style.qd_labels
        else ["pop_G", "pop_X1", "pop_X2", "pop_XX"]
    )
    keys = [k for k in order if k in pops] + [k for k in pops if k not in order]

    for k in keys:
        y = np.asarray(pops[k]).reshape(-1)
        if y.shape[0] != t.shape[0]:
            raise ValueError(
                f"{k} length {y.shape[0]} != t length {t.shape[0]}"
            )
        yr = np.real(y).astype(float)

        color = style.qd_colors.get(k, None)
        ls = style.qd_styles.get(k, "-")
        lw = float(style.qd_widths.get(k, 1.4))
        alpha = float(style.qd_alphas.get(k, 0.9))
        lab = style.qd_labels_tex.get(k, k)

        want_markers = isinstance(ls, tuple)
        ax.plot(
            t,
            yr,
            label=lab,
            color=color,
            ls=ls,
            lw=lw,
            alpha=alpha,
            **_maybe_marker_kwargs(
                style, want_markers=want_markers, color=color
            ),
        )

    ax.set_ylabel(ax_label("QD pop", None, r"1"))
    ax.set_ylim(-0.05, 1.05)
    _legend(ax, style, loc="center right")


def draw_outputs_panel(
    ax: Axes,
    *,
    t: np.ndarray,
    outputs: Mapping[str, np.ndarray],
    style: PlotStyle,
) -> None:
    ax.grid(True, alpha=style.grid_alpha)

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
    keys = [k for k in order if k in outputs] + [
        k for k in outputs if k not in order
    ]

    palette = (
        style.palette_outputs
        if style.palette_outputs
        else ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
    )
    color_map = {k: palette[i % len(palette)] for i, k in enumerate(keys)}

    # More spaced dots for H by default (since you asked for “spaced more apart”)
    ls_H = (0, (1.2, 5.0))
    ls_V = "--"

    ymax = 0.0
    for k in keys:
        y = np.asarray(outputs[k]).reshape(-1)
        if y.shape[0] != t.shape[0]:
            raise ValueError(
                f"{k} length {y.shape[0]} != t length {t.shape[0]}"
            )
        yr = np.real(y).astype(float)
        ymax = max(ymax, float(np.nanmax(yr)) if yr.size else 0.0)

        if k.endswith("_H"):
            ls = ls_H
            lw = 2.0
            want_markers = True
        elif k.endswith("_V"):
            ls = ls_V
            lw = 1.4
            want_markers = False
        else:
            ls = "-"
            lw = 1.4
            want_markers = False

        color = color_map[k]
        ax.plot(
            t,
            yr,
            label=rf"$\langle n_{{{k[2:]}}}\rangle$",
            color=color,
            ls=ls,
            lw=lw,
            **_maybe_marker_kwargs(
                style, want_markers=want_markers, color=color
            ),
        )

    ax.set_ylabel(ax_label("Output", r"\langle n \rangle", r"1"))
    ax.set_ylim(-0.05, max(0.5, 1.10 * ymax))
    _legend(ax, style, loc="upper left")


def draw_coupling_panel(
    ax: Axes,
    *,
    t: np.ndarray,
    coherences: Mapping[str, np.ndarray],
    drives: Optional[Sequence[DriveSeries]] = None,
    style: Optional[PlotStyle] = None,
    mode: str = "abs",
    max_traces: int = 6,
) -> None:
    style = style or PlotStyle()
    ax.grid(True, alpha=style.grid_alpha)

    if not coherences:
        ax.text(
            0.5,
            0.5,
            "No coherences found",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_ylabel(ax_label("Coupling", r"\chi(t)", r"1"))
        return

    keys = sorted([k for k in coherences.keys() if isinstance(k, str)])[
        :max_traces
    ]

    handles = []
    labels = []

    for k in keys:
        y = np.asarray(coherences[k]).reshape(-1)
        if y.shape[0] != t.shape[0]:
            raise ValueError(
                f"coherence {k} length {y.shape[0]} != t length {t.shape[0]}"
            )

        nice = k[len("coh_") :] if k.startswith("coh_") else k

        if mode == "abs":
            h = ax.plot(t, np.abs(y), lw=1.4)[0]
            handles.append(h)
            labels.append(rf"$|{nice}|$")
        elif mode == "reim":
            h1 = ax.plot(t, np.real(y), lw=1.2)[0]
            h2 = ax.plot(t, np.imag(y), lw=1.2, ls="--")[0]
            handles.extend([h1, h2])
            labels.extend([rf"$\Re({nice})$", rf"$\Im({nice})$"])
        else:
            raise ValueError("mode must be 'abs' or 'reim'")

    ax.set_ylabel(ax_label("Coupling", r"\chi(t)", r"1"))
    _sci(ax, right=False)

    omega = None
    if drives:
        for d in drives:
            if d.Omega_rad_s is not None:
                omega = np.asarray(d.Omega_rad_s, dtype=float).reshape(-1)
                break

    if omega is not None and omega.shape[0] == t.shape[0]:
        ax2 = ax.twinx()
        ax2.plot(t, omega, lw=1.1, ls=":", alpha=0.9)
        ax2.set_ylabel(ax_label("", r"\Omega(t)", r"rad\per\second"))
        _sci(ax2, right=True)

    if handles:
        _legend(
            ax,
            style,
            handles,
            labels,
            loc="upper right",
        )
