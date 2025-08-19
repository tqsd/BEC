import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from matplotlib import ticker as mticker
import matplotlib as mpl

# 'butt' | 'round' | 'projecting'
mpl.rcParams["lines.dash_capstyle"] = "round"
mpl.rcParams["lines.dash_joinstyle"] = "round"  # 'miter' | 'round' | 'bevel'
mpl.rcParams["text.usetex"] = True

# QD colors / labels
QD_COLORS = {
    r"$|G\rangle$": "#0072B2",  # blue
    r"$|X_1\rangle$": "#D55E00",  # vermillion (orange-red)
    r"$|X_2\rangle$": "#104911",  # reddish-purple  ← replaces green
    r"$|XX\rangle$": "#222222",  # dark gray/black
}
QD_LABELS = [
    r"$|G\rangle$",
    r"$|X_1\rangle$",
    r"$|X_2\rangle$",
    r"$|XX\rangle$",
]
QD_STYLES = {
    r"$|G\rangle$": "-",
    r"$|X_1\rangle$": (0, (2.0, 2.0)),
    r"$|X_2\rangle$": (0, (0.01, 2.0)),
    r"$|XX\rangle$": "-",
}

QD_WIDTHS = {
    r"$|G\rangle$": 1,
    r"$|X_1\rangle$": 1.5,
    r"$|X_2\rangle$": 2,
    r"$|XX\rangle$": 1,
}

QD_ALPHAS = {
    r"$|G\rangle$": 0.75,
    r"$|X_1\rangle$": 1,
    r"$|X_2\rangle$": 1,
    r"$|XX\rangle$": 0.75,
}


def _build_color_map(labels, offset=0):
    uniq = list(dict.fromkeys(labels))
    return {lbl: f"C{(i + offset) % 10}" for i, lbl in enumerate(uniq)}


def latex_mode_label(lbl: str) -> str:
    s = lbl
    s = s.replace("X1", r"X_1").replace("X2", r"X_2")
    s = s.replace("<->", r"\leftrightarrow{}")
    s = s.replace("&", r"~\&~")
    return f"${s}$"


def _omega_and_area(plotter):
    if plotter.classical_2g is None:
        return None, None
    Ωf = plotter.classical_2g.qutip_coeff()
    t = plotter.tlist
    Ω = np.array([Ωf(tt, {}) for tt in t])
    A = np.concatenate([[0.0], 0.5 * np.cumsum(np.diff(t) * (Ω[1:] + Ω[:-1]))])
    return Ω, A


def plot_qd_comparison(
    plotters,
    titles=None,
    show_top=True,
    filename=None,
    figsize=(9.2, 6.0),
):
    """Compare up to 3 QDPlotter runs side-by-side with a shared legend."""
    assert 1 <= len(plotters) <= 3

    # 1) Gather all traces
    datas = [p.compute_traces(show_top=show_top) for p in plotters]

    # 2) Decide input/output color maps
    top_sharey = all(d.classical for d in datas) or all(
        not d.classical for d in datas
    )
    all_flying = [lbl for d in datas for lbl in d.flying_labels]
    all_intrin = [lbl for d in datas for lbl in d.intrinsic_labels]
    fly_color = _build_color_map(all_flying, offset=0)
    out_color = _build_color_map(all_intrin, offset=4)  # different palette

    # 3) Precompute global Ω and ∫Ω dt limits (only over classical columns)
    Ωmax, Amax = 0.0, 0.0
    have_classical = False
    for p in plotters:
        Ω, A = _omega_and_area(p)
        if Ω is not None:
            have_classical = True
            if np.size(Ω):
                Ωmax = max(Ωmax, float(np.nanmax(Ω)))
            if np.size(A):
                Amax = max(Amax, float(np.nanmax(A)))
    # Avoid zeros if nothing classical
    if not have_classical:
        Ωmax, Amax = 1.0, 1.0

    # 4) Layout
    ncols = len(datas)
    nrows = 3 if show_top else 2
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    fig.set_constrained_layout_pads(
        w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02
    )
    gs = gridspec.GridSpec(
        nrows=nrows,
        ncols=ncols + 1,
        width_ratios=[1] * ncols + [0.48],
        figure=fig,
    )

    top_axes, mid_axes, bot_axes = [], [], []
    master_x = None

    for j, (p, d) in enumerate(zip(plotters, datas)):
        # Create axes
        if show_top:
            ax_top = fig.add_subplot(
                gs[0, j],
                sharey=(top_axes[0] if (top_axes and top_sharey) else None),
                sharex=master_x,
            )
            top_axes.append(ax_top)
            if master_x is None:
                master_x = ax_top
        mid_row = 1 if show_top else 0
        ax_mid = fig.add_subplot(
            gs[mid_row, j],
            sharey=(mid_axes[0] if mid_axes else None),
            sharex=master_x,
        )
        mid_axes.append(ax_mid)
        ax_bot = fig.add_subplot(
            gs[(2 if show_top else 1), j],
            sharey=(bot_axes[0] if bot_axes else None),
            sharex=master_x,
        )
        bot_axes.append(ax_bot)

        # ---------- TOP ----------
        if show_top:
            if p.classical_2g is not None:
                t = p.tlist
                Ω, A = _omega_and_area(p)

                ax = ax_top
                ax2 = ax.twinx()

                # Left axis: Ω(t) shared limits
                ax.plot(t, Ω, lw=1.6)
                ax.set_ylim(0, 1.05 * max(Ωmax, 1e-12))
                if j == 0:
                    ax.set_ylabel(r"$\Omega(t)$")
                else:
                    # hide duplicate y ticks/labels on other columns
                    ax.tick_params(axis="y", labelleft=False)

                # Right axis: ∫Ω dt shared limits, ticks only on last column
                ax2.plot(t, A, ls="--", lw=1.4, alpha=0.9)
                ax2.set_ylim(0, 1.05 * max(Amax, 1e-12))
                if j == (ncols - 1):
                    ax2.set_ylabel(r"$\int^{t}\Omega(t')\,dt'$")
                    ax2.yaxis.set_major_locator(mticker.MaxNLocator(4))
                    ax2.tick_params(axis="y", right=True, labelright=True)
                else:
                    ax2.tick_params(axis="y", right=False, labelright=False)
                    ax2.spines["right"].set_visible(False)

                ax.grid(True, alpha=0.3)

            else:
                # quantum inputs (unchanged, but hide duplicate y labels)
                for k, lbl in enumerate(d.flying_labels):
                    print(f"Plotting {lbl}, {fly_color[lbl]}")
                    if k < len(d.fly_V):
                        ax_top.plot(
                            d.t,
                            d.fly_V[k],
                            color=fly_color[lbl],
                            ls="--",
                            lw=1,
                            alpha=1,
                        )
                    if k < len(d.fly_H):
                        ax_top.plot(
                            d.t,
                            d.fly_H[k],
                            color=fly_color[lbl],
                            lw=2,
                            linestyle=(0, (0.01, 4.0)),
                            alpha=1,
                        )
                if j == 0:
                    ax_top.set_ylabel(r"$\langle N\rangle$ (in)")
                else:
                    ax_top.tick_params(axis="y", labelleft=False)
                ax_top.grid(True, alpha=0.3)

        # ----- MID row -----
        for lab, y in zip(QD_LABELS, d.qd):
            ax_mid.plot(
                d.t,
                y,
                color=QD_COLORS[lab],
                lw=QD_WIDTHS[lab],
                ls=QD_STYLES[lab],
                alpha=QD_ALPHAS[lab],
            )
        if j == 0:
            ax_mid.set_ylabel("QD")
        else:
            ax_mid.tick_params(axis="y", labelleft=False)
        ax_mid.grid(True, alpha=0.3)

        # ----- BOT row -----
        for k, lbl in enumerate(d.intrinsic_labels):
            if k < len(d.out_H):
                ax_bot.plot(
                    d.t,
                    d.out_H[k],
                    color=out_color[lbl],
                    linestyle=(0, (0.01, 4.0)),
                    lw=2,
                )

            if k < len(d.out_V):
                ax_bot.plot(
                    d.t,
                    d.out_V[k],
                    color=out_color[lbl],
                    ls="--",
                    lw=1,
                    alpha=0.7,
                )
        if j == 0:
            ax_bot.set_ylabel(r"$\langle N\rangle$ (out)")
        else:
            ax_bot.tick_params(axis="y", labelleft=False)

        ax_bot.set_xlabel(r"Time ($ns$)")
        ax_bot.grid(True, alpha=0.3)

        # hide x-tick labels except on bottom row (global xlabel already set)
        if show_top:
            ax_top.tick_params(axis="x", labelbottom=False)
        ax_mid.tick_params(axis="x", labelbottom=False)

    # ---------- Shared legend on the right ----------
    legend_ax = fig.add_subplot(gs[:, -1])
    legend_ax.axis("off")

    qd_handles = [
        Line2D(
            [0],
            [0],
            color=QD_COLORS[l],
            lw=QD_WIDTHS[l],
            ls=QD_STYLES[l],
            label=l,
        )
        for l in QD_LABELS
    ]
    pol_handles = [
        Line2D(
            [0], [0], color="k", lw=2.0, linestyle=(0, (0.01, 4.0)), label="H"
        ),
        Line2D([0], [0], color="k", lw=1.0, ls="--", label="V"),
    ]
    out_label_handles = [
        Line2D(
            [0], [0], color=out_color[lbl], lw=2.0, label=latex_mode_label(lbl)
        )
        for lbl in dict.fromkeys(all_intrin)
    ]
    outputs_header = Line2D([], [], color="none", label="Outputs (colors)")

    legend_ax.legend(
        handles=qd_handles + pol_handles + [outputs_header] + out_label_handles,
        loc="center",
        frameon=False,
        handlelength=2.4,
    )

    # Tidy ticks
    for ax in (
        (top_axes + mid_axes + bot_axes) if show_top else (mid_axes + bot_axes)
    ):
        ax.locator_params(axis="x", nbins=5)
        ax.locator_params(axis="y", nbins=4)

    if filename:
        if filename.lower().endswith((".png", ".pdf")):
            fig.savefig(
                filename,
                dpi=300 if filename.endswith(".png") else None,
                bbox_inches="tight",
            )
        else:
            fig.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")
            fig.savefig(f"{filename}.pdf", bbox_inches="tight")
    return fig
