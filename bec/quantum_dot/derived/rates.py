from __future__ import annotations

from functools import cached_property
from typing import Mapping

from smef.core.units import QuantityLike, as_quantity


class RatesMixin:
    @cached_property
    def rates(self) -> Mapping[str, QuantityLike]:
        out: dict[str, QuantityLike] = {}

        dm = getattr(self.qd, "decay_model", None)
        if dm is not None and hasattr(dm, "compute"):
            try:
                r = dm.compute()
                for k, v in r.items():
                    key = k.value if hasattr(k, "value") else str(k)
                    out[key] = as_quantity(v, "1/s")
            except Exception:
                pass

        # NEW: merge rates produced by PhononsMixin -> phonon_outputs
        po = getattr(self, "phonon_outputs", None)
        if po is not None:
            try:
                r = getattr(po, "rates", None) or {}
                for k, v in r.items():
                    key = k.value if hasattr(k, "value") else str(k)
                    out[key] = as_quantity(v, "1/s")
            except Exception:
                pass

        # Keep this if you want, but it won't help unless you actually implement it on qd.phonon_model
        pm = getattr(self.qd, "phonon_model", None)
        if pm is not None and hasattr(pm, "compute_rates"):
            try:
                r = pm.compute_rates()
                for k, v in r.items():
                    key = k.value if hasattr(k, "value") else str(k)
                    out[key] = as_quantity(v, "1/s")
            except Exception:
                pass

        return out
