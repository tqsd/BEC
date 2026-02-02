from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
import numpy as np

from bec.metrics import QDDiagnostics
from bec.scenarios.metrics import extract_xx_peak_from_expect

_SOLVE_OPTIONS = {
    "qutip_options": {
        "method": "bdf",
        "atol": 1e-10,
        "rtol": 1e-8,
        "nsteps": 200000,
        "max_step": 0.01,
        "progress_bar": "tqdm",
    }
}


def run_scenario_once(
    qd: Any,
    *,
    specs: Sequence[Any],
    cfg: Any,
    solve_options: Optional[Dict[str, Any]] = None,
    metric_window_sigma: Tuple[float, float] = (-2.0, 4.0),
    # Backward-compat: sweeps passes these but we don't need them here
    scheme: Any = None,
    x_params: Optional[Mapping[str, float]] = None,
    drives_for_plot: Optional[Sequence[Any]] = None,
) -> Tuple[Any, Any, np.ndarray, Any, float]:
    """
    Runs a single scenario and returns (res, metrics, tlist, units, xx_peak).
    """
    from smef.core.units import Q
    from smef.engine import SimulationEngine, UnitSystem
    from bec.quantum_dot.enums import QDState
    from bec.quantum_dot.smef.initial_state import rho0_qd_vacuum

    _ = scheme
    _ = x_params
    _ = drives_for_plot

    solve_options = (
        solve_options if solve_options is not None else _SOLVE_OPTIONS
    )

    time_unit_s = float(Q(1.0, "ns").to("s").magnitude)
    units = UnitSystem(time_unit_s=time_unit_s)
    engine = SimulationEngine(audit=bool(getattr(cfg, "audit", False)))

    tlist = np.linspace(0.0, float(cfg.t_end_ns), int(cfg.n_points))

    bundle = qd.compile_bundle(units=units)
    dims = bundle.modes.dims()
    rho0 = rho0_qd_vacuum(dims=dims, qd_state=QDState.G)

    res = engine.run(
        qd,
        tlist=tlist,
        time_unit_s=time_unit_s,
        rho0=rho0,
        drives=list(specs),
        solve_options=solve_options,
    )

    metrics = QDDiagnostics().compute(qd, res, units=units)

    xx_peak = extract_xx_peak_from_expect(
        tlist_solver=res.tlist,
        expect=res.expect,
        time_unit_s=units.time_unit_s,
        t0_ns=cfg.t0_ns,
        sigma_ns=cfg.sigma_ns,
        window=metric_window_sigma,
    )

    return res, metrics, tlist, units, xx_peak
