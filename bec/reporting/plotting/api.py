from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, Tuple

from .extract import extract_qd_traces
from .grid import PlotConfig, plot_qd_run, plot_qd_runs_grid
from .styles import PlotStyle
from .traces import QDTraces


def plot_run(
    res: Any,
    *,
    units: Any,
    drive: Optional[Any] = None,
    qd: Optional[Any] = None,
    window_s: Optional[Tuple[float, float]] = None,
    cfg: Optional[PlotConfig] = None,
    style: Optional[PlotStyle] = None,
):
    tr = extract_qd_traces(
        res, units=units, drive=drive, qd=qd, window_s=window_s
    )
    return plot_qd_run(tr, cfg=cfg, style=style)


def plot_runs(
    results: Sequence[Any],
    *,
    units: Any,
    drives: Optional[Sequence[Optional[Any]]] = None,
    qds: Optional[Sequence[Optional[Any]]] = None,
    windows_s: Optional[Sequence[Optional[Tuple[float, float]]]] = None,
    cfg: Optional[PlotConfig] = None,
    style: Optional[PlotStyle] = None,
) -> List[Any]:
    """
    Plot multiple runs in a grid (columns are runs, rows are panels).

    - results: sequence of solver results (engine.run outputs)
    - drives/qds/windows_s: optional per-run metadata; if omitted, None is used
      for that run
    - returns: list of matplotlib Figures (one per chunk, depending on cfg.ncols)
    """
    n = len(results)
    if n == 0:
        raise ValueError("results must be non-empty")

    if drives is None:
        drives = [None] * n
    if qds is None:
        qds = [None] * n
    if windows_s is None:
        windows_s = [None] * n

    if len(drives) != n:
        raise ValueError("drives length must match results length")
    if len(qds) != n:
        raise ValueError("qds length must match results length")
    if len(windows_s) != n:
        raise ValueError("windows_s length must match results length")

    traces: List[QDTraces] = []
    for res, drive, qd, window_s in zip(results, drives, qds, windows_s):
        tr = extract_qd_traces(
            res, units=units, drive=drive, qd=qd, window_s=window_s
        )
        traces.append(tr)

    return plot_qd_runs_grid(traces, cfg=cfg, style=style)
