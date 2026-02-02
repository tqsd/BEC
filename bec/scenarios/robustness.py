from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .types import RobustnessSummary, SchemeKind, Sweep1DResult, Sweep2DResult


def pass_fraction(xx: np.ndarray, *, threshold: float) -> float:
    a = np.asarray(xx, dtype=float)
    return float(np.mean(a >= float(threshold)))


def auc_above_threshold(
    xx: np.ndarray, x: np.ndarray, *, threshold: float
) -> float:
    a = np.asarray(xx, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.maximum(a - float(threshold), 0.0)
    return float(np.trapezoid(y, x))


def robustness_summary_1d(
    res: Sweep1DResult, *, threshold: float
) -> Dict[str, RobustnessSummary]:
    pf = pass_fraction(res.xx_final, threshold=float(threshold))
    auc = auc_above_threshold(
        res.xx_final, res.values, threshold=float(threshold)
    )

    return {
        "pass_fraction": RobustnessSummary(
            scheme=res.scheme,
            metric="pass_fraction",
            value=float(pf),
            details={"axis": res.axis.value, "threshold": float(threshold)},
        ),
        "auc_above_threshold": RobustnessSummary(
            scheme=res.scheme,
            metric="auc_above_threshold",
            value=float(auc),
            details={"axis": res.axis.value, "threshold": float(threshold)},
        ),
    }


def robustness_summary_2d(
    res: Sweep2DResult, *, threshold: float
) -> Dict[str, RobustnessSummary]:
    Z = np.asarray(res.xx_final, dtype=float)
    pf = float(np.mean(Z >= float(threshold)))

    return {
        "area_fraction": RobustnessSummary(
            scheme=res.scheme,
            metric="area_fraction",
            value=float(pf),
            details={
                "x_axis": res.x_axis.value,
                "y_axis": res.y_axis.value,
                "threshold": float(threshold),
            },
        )
    }
