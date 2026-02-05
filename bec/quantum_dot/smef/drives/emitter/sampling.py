from __future__ import annotations

from typing import Any

import numpy as np

from bec.quantum_dot.smef.drives.context import QDDriveDecodeContext


def payload_from_ctx(drive_id: Any, ctx: QDDriveDecodeContext) -> Any:
    if drive_id in ctx.meta_drives:
        return ctx.meta_drives[drive_id]
    raise KeyError("Missing drive payload for drive_id=%s" % (drive_id,))


def sample_omega_L_rad_s(
    payload: Any, t_phys_s: np.ndarray
) -> np.ndarray | None:
    fn = getattr(payload, "omega_L_rad_s", None)
    if not callable(fn):
        return None

    t_phys_s = np.asarray(t_phys_s, dtype=float).reshape(-1)
    out = np.empty(t_phys_s.size, dtype=float)
    for i in range(t_phys_s.size):
        w = fn(float(t_phys_s[i]))
        if w is None:
            return None
        out[i] = float(w)
    return out
