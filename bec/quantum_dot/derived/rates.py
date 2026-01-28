from __future__ import annotations

from functools import cached_property
from typing import Any, Mapping


class RatesMixin:
    @cached_property
    def rates(self) -> Mapping[Any, Any]:
        out: dict[Any, Any] = {}

        dm = getattr(self.qd, "decay_model", None)
        if dm is not None and hasattr(dm, "compute_q"):
            r = dm.compute_q()  # Dict[RateKey, QuantityLike]
            out.update(r)

        pm = getattr(self.qd, "phonon_model", None)
        if pm is not None and hasattr(pm, "compute_rates"):
            # could be Dict[RateKey, QuantityLike] or similar
            r2 = pm.compute_rates()
            out.update(r2)

        return out
