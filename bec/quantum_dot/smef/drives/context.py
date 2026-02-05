from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from smef.core.drives.protocols import DriveDecodeContextProto


@dataclass(frozen=True)
class DecodePolicy:
    allow_multi: bool = True
    k_bandwidth: float = 3.0
    sample_points: int = 5

    # polarization penalty/gating
    pol_gate_eps: float = 1e-6
    pol_penalty_power: float = 1.0
    pol_penalty_weight: float = 1.0

    # optional: hard gate if coupling too small
    min_coupling_mag: float = 0.0


@dataclass(frozen=True)
class QDDriveDecodeContext(DriveDecodeContextProto):
    derived: Any
    policy: DecodePolicy = DecodePolicy()
    bandwidth_sigma_omega_rad_s: float | None = None

    meta: Mapping[str, Any] = field(default_factory=dict)
    meta_drives: MutableMapping[Any, Any] = field(default_factory=dict)

    # NEW: solver grid injected by SMEF engine
    tlist_solver: np.ndarray | None = None
    time_unit_s: float | None = None

    def with_solver_grid(
        self, *, tlist: np.ndarray, time_unit_s: float
    ) -> QDDriveDecodeContext:
        return replace(
            self,
            tlist_solver=np.asarray(tlist, dtype=float).reshape(-1),
            time_unit_s=float(time_unit_s),
        )
