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


def build_compile_panel(
    rep: CompileReport, *, use_unicode: bool = True, use_color: bool = True
):
    if not _HAS_RICH:
        # fallback
        lines = []
        lines.append("== Drives → Decoded transitions (new) ==")
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


def _drives_table(
    rep: CompileReport, *, use_unicode: bool = True, use_color: bool = True
):
    border = box.SIMPLE_HEAVY if use_unicode else box.ASCII
    t = Table(
        title="Drives → Decoded transitions (new)",
        box=border,
        show_lines=False,
    )

    t.add_column("drive_id", style="bold" if use_color else "")
    t.add_column("pref", justify="center")
    t.add_column("pol", justify="center")
    t.add_column("resolved", style="bold" if use_color else "")
    t.add_column("details", overflow="fold")

    # group resolved drives by drive_id
    by_id: dict[str, list[Any]] = {}
    for r in rep.resolved:
        by_id.setdefault(getattr(r, "drive_id", ""), []).append(r)

    def _fmt_c(z: complex) -> str:
        # compact complex formatting for overlaps
        return f"{z.real:.2g}{z.imag:+.2g}j"

    def _fmt_pol(drv: Any) -> str:
        pol_state = getattr(drv, "pol_state", None)
        if pol_state is None:
            return "-"
        try:
            E = drv.effective_pol()
            if E is None:
                return str(getattr(pol_state, "basis", "pol"))
            E = np.asarray(E, dtype=complex).reshape(2)
            basis = getattr(pol_state, "basis", "pol")
            return f"{basis} [{E[0]:.2g},{E[1]:.2g}]"
        except Exception:
            return str(getattr(pol_state, "basis", "pol"))

    def _drive_id(drv: Any) -> str:
        return getattr(drv, "label", None) or f"drive_{id(drv)}"

    for drv in rep.drives:
        did = _drive_id(drv)
        pref = getattr(drv, "preferred_kind", None) or "-"
        pol = _fmt_pol(drv)

        rs = by_id.get(did, [])
        if not rs:
            t.add_row(
                did,
                str(pref),
                pol,
                "[red]UNRESOLVED[/red]" if use_color else "UNRESOLVED",
                "",
            )
            continue

        # stable ordering: 1ph before 2ph, then transition name
        def _sort_key(r: Any):
            kind = getattr(r, "kind", "")
            tr = getattr(
                getattr(r, "transition", None),
                "name",
                str(getattr(r, "transition", "")),
            )
            return (0 if kind == "1ph" else 1, tr)

        rs = sorted(rs, key=_sort_key)

        for j, r in enumerate(rs):
            tr_name = getattr(
                getattr(r, "transition", None),
                "name",
                str(getattr(r, "transition", "")),
            )
            resolved = f"{getattr(r, 'kind', '?')}:{tr_name}"

            meta = getattr(r, "meta", {}) or {}

            # --- decode metrics ---
            dmin = meta.get("min_detuning_phys_rad_s", None)
            sig = meta.get("sigma_omega_phys_rad_s", None)
            cmag = meta.get("pol_coupling_mag", None)

            is_chirped = meta.get("is_chirped", None)
            wswp = meta.get("omegaL_sweep_phys_rad_s", None)
            wmin = meta.get("omegaL_min_phys_rad_s", None)
            wmax = meta.get("omegaL_max_phys_rad_s", None)

            # --- build details string ---
            parts: list[str] = []

            if dmin is not None:
                parts.append(f"Δmin={float(dmin):.3e} rad/s")
            if sig is not None:
                parts.append(f"σ={float(sig):.3e}")

            # show chirp tag always if present
            if is_chirped is True:
                parts.append("chirp=Y")
            elif is_chirped is False:
                parts.append("chirp=N")

            # show sweep always if present (even if 0, which is informative)
            if wswp is not None:
                parts.append(f"Δω={float(wswp):.3e}")
            if wmin is not None and wmax is not None:
                parts.append(f"ωL∈[{float(wmin):.3e},{float(wmax):.3e}]")

            # polarization coupling (only makes sense for 1ph; still show if present)
            if cmag is not None and getattr(r, "kind", "") == "1ph":
                parts.append(f"|c|={float(cmag):.3g}")

            # components (if any)
            comps = getattr(r, "components", None) or ()
            if comps:
                comp_strs = []
                for tkey, c in comps:
                    name = getattr(tkey, "name", str(tkey))
                    try:
                        comp_strs.append(f"{name}:{_fmt_c(complex(c))}")
                    except Exception:
                        comp_strs.append(f"{name}:{c}")
                parts.append("[" + ", ".join(comp_strs) + "]")

            details = "  ".join(parts)

            # only print drive_id/pref/pol for first row per drive
            if j == 0:
                t.add_row(did, str(pref), pol, resolved, details)
            else:
                t.add_row("", "", "", resolved, details)

    return t
