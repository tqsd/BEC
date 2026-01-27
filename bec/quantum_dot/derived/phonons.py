from __future__ import annotations

from functools import cached_property
from typing import Optional


class PhononsMixin:
    @cached_property
    def phonon_params(self) -> Optional[object]:
        P = getattr(self.qd, "phonon_params", None)
        if P is not None:
            return P
        pm = getattr(self.qd, "phonon_model", None)
        return getattr(pm, "_P", None) if pm is not None else None

    @cached_property
    def phonon_outputs(self):
        pm = getattr(self.qd, "phonon_model", None)
        if pm is None or not hasattr(pm, "compute"):
            return None
        try:
            return pm.compute()
        except Exception:
            return None

    @cached_property
    def polaron_B(self) -> float:
        pm = getattr(self.qd, "phonon_model", None)
        if pm is None:
            return 1.0
        if hasattr(pm, "polaron_B"):
            return float(pm.polaron_B())
        outs = getattr(pm, "outputs", None)
        if outs is not None and hasattr(outs, "B"):
            return float(outs.B)
        return 1.0
