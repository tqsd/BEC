from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
from rich.console import Console

from bec.simulation.drive_decode import DefaultDriveDecoder
from bec.simulation.reporting.compile_report import (
    CompileReport,
    build_compile_panel,
)
from bec.simulation.types import (
    MEProblem,
    TruncationPolicy,
    build_dims,
    ModeRegistryView,
)

from bec.simulation.drive_decode.context_provider import DecodeContextProvider

QDStateLike = Union[str, int]


class MECompiler:
    def __init__(
        self,
        *,
        decoder: Optional[DefaultDriveDecoder] = None,
        truncation: Optional[TruncationPolicy] = None,
    ):
        self._decoder = decoder or DefaultDriveDecoder()
        self._trunc = truncation or TruncationPolicy(pol_dim=2)

    def compile(
        self,
        *,
        model: Any,
        provider: DecodeContextProvider,
        drives: Sequence[Any],
        tlist: np.ndarray,
        time_unit_s: float,
        rho0: Optional[Union[np.ndarray, QDStateLike]] = None,
        args: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
        qd_dim: int = 4,
        report: bool = False,
    ) -> MEProblem:
        tlist = np.asarray(tlist, dtype=float)
        time_unit_s = float(time_unit_s)
        if tlist.ndim != 1:
            raise ValueError(f"tlist must be 1D, got shape {tlist.shape}")
        if time_unit_s <= 0:
            raise ValueError(f"time_unit_s must be > 0, got {time_unit_s}")

        mode_registry: ModeRegistryView = provider.mode_registry(model)
        decode_ctx = provider.decode_ctx(model)

        dims = build_dims(mode_registry, self._trunc, qd_dim=qd_dim)
        D = int(np.prod(dims))

        resolved = self._decoder.decode(
            ctx=decode_ctx,
            drives=drives,
            tlist=tlist,
            time_unit_s=time_unit_s,
        )

        if report:
            derived = provider.derived_for_report(model)
            rep = CompileReport(
                derived=derived,
                drives=drives,
                resolved=resolved,
                tlist=tlist,
                time_unit_s=time_unit_s,
            )
            Console().print(build_compile_panel(rep))

        meta_out: Dict[str, Any] = dict(meta or {})
        meta_out.setdefault("num_modes", int(mode_registry.num_modes()))
        meta_out.setdefault("num_drives", len(drives))
        meta_out.setdefault(
            "truncation",
            {"pol_dim_default": getattr(self._trunc, "pol_dim", None)},
        )
        meta_out.setdefault("compiled_terms", {"H": 0, "C": 0, "E": 0})
        meta_out["resolved_drives"] = [
            {
                "drive_id": rd.drive_id,
                "kind": rd.kind,
                "transition": getattr(
                    rd.transition, "name", str(rd.transition)
                ),
                "candidates": [
                    getattr(x, "name", str(x)) for x in rd.candidates
                ],
                "meta": dict(rd.meta or {}),
            }
            for rd in resolved
        ]

        return MEProblem(
            model=model,
            tlist=tlist,
            time_unit_s=time_unit_s,
            dims=dims,
            rho0=rho0_arr,
            h_terms=tuple(),
            c_terms=tuple(),
            observables=None,
            args=dict(args or {}),
            meta=meta_out,
        )
