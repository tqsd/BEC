"""
Quantum-dot enum identifiers (no physics, no config, no registries).

This module defines stable IDs used across the codebase:
- QDState: internal dot states
- Transition: directed edges between states
- TransitionPair: undirected "physical" transition families (forward+backward)
- TransitionKind: classification labels used by TransitionSpec in the
  registry layer
"""

from __future__ import annotations

from enum import Enum


class QDState(str, Enum):
    """Internal quantum-dot basis states."""

    G = "G"
    X1 = "X1"
    X2 = "X2"
    XX = "XX"


class Transition(str, Enum):
    """
    Directed transitions (edges) between QD states.

    Naming convention: SRC_DST meaning SRC -> DST.
    Used for directed operators (e.g., |DST><SRC|), collapse terms, etc.
    """

    G_X1 = "G_X1"
    X1_G = "X1_G"

    G_X2 = "G_X2"
    X2_G = "X2_G"

    X1_XX = "X1_XX"
    XX_X1 = "XX_X1"

    X2_XX = "X2_XX"
    XX_X2 = "XX_X2"

    # Effective 2-photon excitation family (if enabled in registry/specs)
    G_XX = "G_XX"
    XX_G = "XX_G"


class TransitionPair(str, Enum):
    """
    Undirected transition families.

    Each pair corresponds to exactly two directed transitions in the registry:
    forward (low->high) and backward (high->low).
    """

    G_X1 = "G<->X1"
    G_X2 = "G<->X2"
    X1_XX = "X1<->XX"
    X2_XX = "X2<->XX"
    G_XX = "G<->XX"


class TransitionKind(str, Enum):
    """
    High-level classification labels for transition families.

    This is intentionally coarse; detailed behavior belongs in TransitionSpec
    (registry layer), potentially extended with selection rules, channels, etc.
    """

    DIPOLE_1PH = "dipole_1ph"
    EFFECTIVE_2PH = "effective_2ph"


class RateKey(str, Enum):
    RAD_XX_X1 = "RAD_XX_X1"
    RAD_XX_X2 = "RAD_XX_X2"
    RAD_X1_G = "RAD_X1_G"
    RAD_X2_G = "RAD_X2_G"
    PH_DEPH_X1 = "PH_DEPH_X1"
    PH_DEPH_X2 = "PH_DEPH_X2"
    PH_RELAX_X1_X2 = "PH_RELAX_X1_X2"
    PH_RELAX_X2_X1 = "PH_RELAX_X2_X1"
