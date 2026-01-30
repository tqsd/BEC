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
    drives: Sequence[Optional[Any]] = [],
    qd: Optional[Any] = None,
    window_s: Optional[Tuple[float, float]] = None,
    cfg: Optional[PlotConfig] = None,
    style: Optional[PlotStyle] = None,
):
    tr = extract_qd_traces(
        res, units=units, drives=drives, qd=qd, window_s=window_s
    )
    return plot_qd_run(tr, cfg=cfg, style=style)


def plot_runs(
    results: Sequence[Any],
    *,
    units: Any,
    drives_list: Optional[Sequence[Optional[Sequence[Optional[Any]]]]] = None,
    qds: Optional[Sequence[Optional[Any]]] = None,
    windows_s: Optional[Sequence[Optional[Tuple[float, float]]]] = None,
    cfg: Optional[PlotConfig] = None,
    style: Optional[PlotStyle] = None,
) -> List[Any]:
    n = len(results)
    if n == 0:
        raise ValueError("results must be non-empty")

    if drives_list is None:
        drives_list = [None] * n
    if qds is None:
        qds = [None] * n
    if windows_s is None:
        windows_s = [None] * n

    if len(drives_list) != n:
        raise ValueError("drives_list length must match results length")

    traces: List[QDTraces] = []
    for res, drives, qd, window_s in zip(results, drives_list, qds, windows_s):
        tr = extract_qd_traces(
            res, units=units, drives=drives, qd=qd, window_s=window_s
        )
        traces.append(tr)

    return plot_qd_runs_grid(traces, cfg=cfg, style=style)
