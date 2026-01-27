from __future__ import annotations

from functools import cached_property
from typing import Any, Dict, List, Optional

from bec.units import magnitude


class ModesMixin:
    """
    Derived views over qd.mode_registry for reporting/debugging.

    Exposes:
      - mode_entries: list[dict] with idx/kind/pol/energy/wavelength/label
    """

    @cached_property
    def mode_entries(self) -> List[Dict[str, Any]]:
        mr = getattr(self.qd, "mode_registry", None)
        if mr is None:
            print("NONE")
            return []

        # support both naming schemes during refactor
        items = getattr(mr, "channels", None)
        if items is None:
            items = getattr(mr, "modes", None)
        if items is None:
            return []

        out: List[Dict[str, Any]] = []
        for i, ch in enumerate(items):
            # key may be stored as ch.key (LightChannel) or ch.channel_key (legacy LightMode)
            key = getattr(ch, "key", None) or getattr(ch, "channel_key", None)
            kind = getattr(key, "kind", None) if key is not None else None
            pol = getattr(key, "pol", None) if key is not None else None

            # optional unitful quantities
            E = getattr(ch, "energy", None)
            lam = getattr(ch, "wavelength", None)

            energy_eV: Optional[float] = None
            if E is not None:
                try:
                    energy_eV = float(magnitude(E, "eV"))
                except Exception:
                    pass

            wavelength_nm: Optional[float] = None
            if lam is not None:
                try:
                    wavelength_nm = float(magnitude(lam, "nm"))
                except Exception:
                    try:
                        wavelength_nm = float(magnitude(lam, "m")) * 1e9
                    except Exception:
                        pass

            out.append(
                {
                    "idx": i,
                    "kind": kind,
                    "pol": pol,
                    "energy_eV": energy_eV,
                    "wavelength_nm": wavelength_nm,
                    "label": getattr(ch, "label", None),
                }
            )
        return out
