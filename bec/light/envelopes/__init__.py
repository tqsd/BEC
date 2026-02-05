"""
Unitful temporal envelopes for classical optical fields.

This subpackage defines:
- unitful envelope interfaces (EnvelopeU)
- concrete envelope implementations (Gaussian, Symbolic, Tabulated)
- JSON serialization via a central registry
- a clean separation between unitful envelopes and compiled (unitless) forms

All envelopes:
- accept unitful time inputs (QuantityLike)
- return dimensionless float values
"""

from .base import (
    CompiledEnvelope,
    EnvelopeU,
    SerializableEnvelopeU,
    TimeBasisU,
)
from .gaussian import GaussianEnvelopeU
from .registry import (
    envelope_from_json,
    envelope_to_json,
)
from .symbolic import SymbolicEnvelopeU
from .tabulated import TabulatedEnvelopeU

__all__ = [
    # base protocols / helpers
    "EnvelopeU",
    "SerializableEnvelopeU",
    "CompiledEnvelope",
    "TimeBasisU",
    # concrete envelopes
    "GaussianEnvelopeU",
    "SymbolicEnvelopeU",
    "TabulatedEnvelopeU",
    # registry helpers
    "envelope_from_json",
    "envelope_to_json",
]
