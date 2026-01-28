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
    EnvelopeU,
    SerializableEnvelopeU,
    CompiledEnvelope,
    TimeBasisU,
)

from .gaussian import GaussianEnvelopeU
from .symbolic import SymbolicEnvelopeU
from .tabulated import TabulatedEnvelopeU

from .registry import (
    envelope_from_json,
    envelope_to_json,
)

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
