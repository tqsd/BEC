from __future__ import annotations

from functools import cached_property
from typing import Mapping

from bec.units import QuantityLike, as_quantity


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
