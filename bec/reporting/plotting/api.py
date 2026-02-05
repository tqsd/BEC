from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .extract import extract_qd_traces
from .grid import PlotConfig, plot_qd_run, plot_qd_runs_grid
from .styles import PlotStyle
from .traces import QDTraces


def plot_run(
    res: Any,
    *,
    units: Any,
    drives: Sequence[Any | None] = [],
    qd: Any | None = None,
    window_s: tuple[float, float] | None = None,
    cfg: PlotConfig | None = None,
    style: PlotStyle | None = None,
):
    tr = extract_qd_traces(
        res, units=units, drives=drives, qd=qd, window_s=window_s
    )
    return plot_qd_run(tr, cfg=cfg, style=style)


def plot_runs(
    results: Sequence[Any],
    *,
    units: Any,
    drives_list: Sequence[Sequence[Any | None] | None] | None = None,
    qds: Sequence[Any | None] | None = None,
    windows_s: Sequence[tuple[float, float] | None] | None = None,
    cfg: PlotConfig | None = None,
    style: PlotStyle | None = None,
) -> list[Any]:
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

    traces: list[QDTraces] = []
    for res, drives, qd, window_s in zip(results, drives_list, qds, windows_s):
        tr = extract_qd_traces(
            res, units=units, drives=drives, qd=qd, window_s=window_s
        )
        traces.append(tr)

    return plot_qd_runs_grid(traces, cfg=cfg, style=style)
