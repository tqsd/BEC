from __future__ import annotations

from typing import Any, Dict, Type

from .base import SerializableEnvelope
from .gaussian import GaussianEnvelope
from .tabulated import TabulatedEnvelope
from .symbolic import SymbolicEnvelope


ENVELOPE_REGISTRY: Dict[str, Type[SerializableEnvelope]] = {
    "gaussian": GaussianEnvelope,
    "tabulated": TabulatedEnvelope,
    "symbolic": SymbolicEnvelope,
}


def envelope_to_json(env: SerializableEnvelope) -> Dict[str, Any]:
    return env.to_dict()


def envelope_from_json(data: Dict[str, Any]) -> SerializableEnvelope:
    t = data.get("type")
    if not isinstance(t, str):
        raise ValueError("Envelope JSON must contain a string 'type' field")
    cls = ENVELOPE_REGISTRY.get(t)
    if cls is None:
        known = ", ".join(sorted(ENVELOPE_REGISTRY.keys()))
        raise ValueError(f"Unknown envelope type {t!r}. Known: {known}")
    return cls.from_dict(data)
