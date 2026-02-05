from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .factories import get_scheme_factory
from .run import run_scenario_once
from .types import (
    SchemeKind,
    Sweep1DResult,
    Sweep2DResult,
    SweepAxis,
    SweepGridSpec,
    SweepSpec,
)


def run_sweep_1d(
    qd: Any,
    *,
    scheme: SchemeKind,
    cfg: Any,
    sweep: SweepSpec,
    base_kwargs: Mapping[str, Any],
) -> Sweep1DResult:
    make_drive = get_scheme_factory(scheme)

    values = np.asarray(sweep.values, dtype=float).reshape(-1)
    xx = np.empty(values.size, dtype=float)

    for i, v in enumerate(values):
        kwargs = dict(base_kwargs)

        if sweep.axis is SweepAxis.AMP_SCALE:
            kwargs["amp_scale"] = float(v)
        elif sweep.axis is SweepAxis.DETUNING_OFFSET_RAD_S:
            kwargs["detuning_offset_rad_s"] = float(v)
        else:
            raise ValueError(f"Unknown sweep axis: {sweep.axis}")

        specs, payloads = make_drive(qd, cfg=cfg, **kwargs)

        _, metrics, _, _, xx_peak = run_scenario_once(
            qd,
            specs=specs,
            cfg=cfg,
        )
        xx[i] = float(xx_peak)

    return Sweep1DResult(
        scheme=scheme,
        axis=sweep.axis,
        values=values,
        xx_final=xx,
        meta={"threshold_xx": float(sweep.threshold_xx)},
    )


def run_sweep_2d(
    qd: Any,
    *,
    scheme: SchemeKind,
    cfg: Any,
    grid: SweepGridSpec,
    base_kwargs: Mapping[str, Any],
) -> Sweep2DResult:
    make_drive = get_scheme_factory(scheme)

    xvals = np.asarray(grid.x_axis.values, dtype=float).reshape(-1)
    yvals = np.asarray(grid.y_axis.values, dtype=float).reshape(-1)

    Z = np.empty((yvals.size, xvals.size), dtype=float)

    for yi, y in enumerate(yvals):
        for xi, x in enumerate(xvals):
            kwargs = dict(base_kwargs)

            kwargs = _apply_axis_value(kwargs, grid.x_axis.axis, float(x))
            kwargs = _apply_axis_value(kwargs, grid.y_axis.axis, float(y))

            specs, payloads = make_drive(qd, cfg=cfg, **kwargs)
            _, metrics, _, _, xx_peak = run_scenario_once(
                qd,
                specs=specs,
                cfg=cfg,
            )
            Z[yi, xi] = float(xx_peak)

    return Sweep2DResult(
        scheme=scheme,
        x_axis=grid.x_axis.axis,
        y_axis=grid.y_axis.axis,
        x_values=xvals,
        y_values=yvals,
        xx_final=Z,
        meta={"threshold_xx": float(grid.threshold_xx)},
    )


def _apply_axis_value(
    kwargs: dict[str, Any], axis: SweepAxis, v: float
) -> dict[str, Any]:
    out = dict(kwargs)
    if axis is SweepAxis.AMP_SCALE:
        out["amp_scale"] = float(v)
    elif axis is SweepAxis.DETUNING_OFFSET_RAD_S:
        out["detuning_offset_rad_s"] = float(v)
    else:
        raise ValueError(f"Unknown sweep axis: {axis}")
    return out
