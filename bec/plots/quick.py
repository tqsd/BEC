from typing import Optional
from .plotter import QDPlotGrid, PlotConfig
from .styles import default_theme


def plot_traces(
    traces,
    title: Optional[str] = None,
    save: Optional[str] = None,
    show_top: bool = True,
    figsize=(9.2, 6.0),
):
    """
    Minimal helper to plot a single QDTraces with default styling.

    Parameters
    ----------
    traces : QDTraces-like
    title : str, optional
    save : str, optional
        If provided, saves to this path (png/pdf).
    show_top : bool
        Whether to include the top panel.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    cfg = PlotConfig(
        show_top=show_top,
        figsize=figsize,
        titles=[title] if title else None,
        filename=save,
    )
    grid = QDPlotGrid(theme=default_theme(), cfg=cfg)
    return grid.render([traces])
