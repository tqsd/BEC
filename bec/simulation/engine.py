from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

from bec.simulation.protocols import SimulationAdapter
from bec.simulation.types import MEProblem, MESimulationResult


@dataclass
class SimulationEngine:
    adapter: SimulationAdapter

    def simulate(
        self,
        me: MEProblem,
        *,
        options: Optional[Dict[str, Any]] = None,
    ) -> MESimulationResult:
        out = self.adapter.simulate(me, options=options)
        # Engine can inject global meta / validation if you like:
        meta = dict(out.meta)
        meta.setdefault("backend", self.adapter.backend_name)
        return type(out)(
            tlist=out.tlist,
            expect=out.expect,
            states=out.states,
            final_state=out.final_state,
            meta=meta,
        )
