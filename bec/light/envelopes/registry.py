from __future__ import annotations

from typing import Any, Dict, Type

from .base import SerializableEnvelopeU
from .gaussian import GaussianEnvelopeU
from .symbolic import SymbolicEnvelopeU
from .tabulated import TabulatedEnvelopeU


_REGISTRY: Dict[str, Type[SerializableEnvelopeU]] = {
    "gaussian": GaussianEnvelopeU,
    "symbolic": SymbolicEnvelopeU,
    "tabulated": TabulatedEnvelopeU,
}


def envelope_to_json(env: SerializableEnvelopeU) -> Dict[str, Any]:
    data = env.to_dict()
    if "type" not in data:
        raise ValueError("Envelope to_dict() must include a 'type' field")
    return data


def envelope_from_json(data: Dict[str, Any]) -> SerializableEnvelopeU:
    t = data.get("type")
    if not isinstance(t, str):
        raise TypeError("Envelope JSON must contain string field 'type'")

    cls = _REGISTRY.get(t)
    if cls is None:
        raise ValueError(f"Unknown envelope type: {t}")

    return cls.from_dict(data)
