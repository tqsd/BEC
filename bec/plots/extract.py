from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple
import numpy as np

from bec.simulation.types import MEProblem, MESimulationResult
from .types import PlotColumn, PulseTrace


@dataclass(frozen=True)
class ObservableSpec:
    """
    Naming convention rules for pulling signals out of result.expect.

    You can keep this super simple at first, then refine when your observables expand.
    """

    qd_keys: Tuple[str, ...] = ("P_G", "P_X1", "P_X2", "P_XX")
    # Example output naming conventions you already use:
    # N[G_X], N+[G_X], N-[G_X], etc...
    # Here we assume you have keys like "N+[G_X]" for H and "N-[G_X]" for V (adjust!)
    out_H_prefix: str = "N+["
    out_V_prefix: str = "N-["
    out_suffix: str = "]"


def _get(result: MESimulationResult, key: str) -> Optional[np.ndarray]:
    arr = result.expect.get(key)
    if arr is None:
        return None
    return np.asarray(arr, dtype=float)


def _strip_brackets(k: str, prefix: str, suffix: str) -> Optional[str]:
    if not (k.startswith(prefix) and k.endswith(suffix)):
        return None
    return k[len(prefix): -len(suffix)]


def build_column(
    *,
    title: str,
    me: MEProblem,
    out: MESimulationResult,
    spec: ObservableSpec = ObservableSpec(),
    pulse: Optional[PulseTrace] = None,
) -> PlotColumn:
    # --- middle panel (QD) ---
    qd: Dict[str, np.ndarray] = {}
    for k in spec.qd_keys:
        v = _get(out, k)
        if v is not None:
            qd[k] = v

    # --- bottom panel (outputs H/V) ---
    outH: Dict[str, np.ndarray] = {}
    outV: Dict[str, np.ndarray] = {}

    for k, v in out.expect.items():
        lblH = _strip_brackets(k, spec.out_H_prefix, spec.out_suffix)
        if lblH is not None:
            outH[lblH] = np.asarray(v, dtype=float)
            continue
        lblV = _strip_brackets(k, spec.out_V_prefix, spec.out_suffix)
        if lblV is not None:
            outV[lblV] = np.asarray(v, dtype=float)

    return PlotColumn(
        title=title,
        t_solver=np.asarray(out.tlist, dtype=float),
        time_unit_s=float(me.time_unit_s),
        qd=qd,
        outputs_H=outH,
        outputs_V=outV,
        pulse=pulse,
        meta={"me_meta": dict(me.meta), "out_meta": dict(out.meta)},
    )
