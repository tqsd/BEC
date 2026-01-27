from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

try:
    from rich.table import Table
    from rich.panel import Panel
    from rich.console import Group
    from rich import box

    _HAS_RICH = True
except Exception:
    _HAS_RICH = False


@dataclass(frozen=True)
class CompileReport:
    derived: Any  # DerivedQD
    drives: Sequence[Any]
    resolved: Sequence[Any]  # Sequence[ResolvedDrive]
    tlist: np.ndarray
    time_unit_s: float


def _drives_table(
    rep: CompileReport, *, use_unicode: bool = True, use_color: bool = True
):
    border = box.SIMPLE_HEAVY if use_unicode else box.ASCII
    t = Table(
        title="Drives → Decoded transitions", box=border, show_lines=False
    )

    t.add_column("drive_id", style="bold" if use_color else "")
    t.add_column("pref", justify="center")
    t.add_column("pol", justify="center")
    t.add_column("resolved", style="bold" if use_color else "")
    t.add_column("details", overflow="fold")

    # group resolved drives by drive_id
    by_id: dict[str, list[Any]] = {}
    for r in rep.resolved:
        by_id.setdefault(r.drive_id, []).append(r)

    for drv in rep.drives:
        drive_id = getattr(drv, "label", None) or f"drive_{id(drv)}"
        pref = getattr(drv, "preferred_kind", None) or "-"
        pol_state = getattr(drv, "pol_state", None)
        pol = "-"
        if pol_state is not None:
            # try to show basis + a compact vector
            try:
                E = drv.effective_pol()
                if E is not None:
                    E = np.asarray(E, dtype=complex).reshape(2)
                    pol = f"{getattr(pol_state, 'basis', 'pol')
                             } [{E[0]:.2g},{E[1]:.2g}]"
                else:
                    pol = str(getattr(pol_state, "basis", "pol"))
            except Exception:
                pol = str(getattr(pol_state, "basis", "pol"))

        rs = by_id.get(drive_id, [])
        if not rs:
            t.add_row(
                drive_id,
                str(pref),
                pol,
                "[red]UNRESOLVED[/red]" if use_color else "UNRESOLVED",
                "",
            )
            continue

        # one row per resolved entry (multi allowed)
        for j, r in enumerate(rs):
            tr = getattr(r.transition, "name", str(r.transition))
            resolved = f"{r.kind}:{tr}"

            dmin = r.meta.get("min_detuning_phys_rad_s", None)
            sig = r.meta.get("sigma_omega_phys_rad_s", None)
            cmag = r.meta.get("pol_coupling_mag", None)

            parts = []
            if dmin is not None:
                parts.append(f"Δmin={float(dmin):.3e} rad/s")
            if sig is not None:
                parts.append(f"σ={float(sig):.3e}")
            if cmag is not None and r.kind == "1ph":
                parts.append(f"|c|={float(cmag):.3g}")

            if getattr(r, "components", None):
                comps = []
                for tkey, c in r.components:
                    comps.append(
                        f"{getattr(tkey, 'name', str(tkey))}:{
                            complex(c):.2g}"
                    )
                parts.append("[" + ", ".join(comps) + "]")

            details = "  ".join(parts)

            # only print drive_id/pref/pol on the first line for that drive
            if j == 0:
                t.add_row(drive_id, str(pref), pol, resolved, details)
            else:
                t.add_row("", "", "", resolved, details)

    return t


def build_compile_panel(
    rep: CompileReport, *, use_unicode: bool = True, use_color: bool = True
):
    if not _HAS_RICH:
        # fallback
        lines = []
        lines.append("== Drives → Decoded transitions ==")
        for r in rep.resolved:
            lines.append(
                f"{r.drive_id}  {r.kind}  {
                    getattr(r.transition, 'name', r.transition)}"
            )
        return "\n".join(lines)

    body = Group(
        _drives_table(rep, use_unicode=use_unicode, use_color=use_color),
        # then reuse your QD report panel underneath:
        # build_panel(rep.derived, style=...)  <-- import from bec.quantum_dot.derived.report
    )
    return Panel(
        body,
        title="Compilation report",
        box=(box.ROUNDED if use_unicode else box.ASCII),
    )
