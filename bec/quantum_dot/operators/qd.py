from __future__ import annotations

import numpy as np

# Import whichever QDState you intend to be canonical in operators.
# If you have two QDState definitions, pick ONE and standardize later.
from bec.quantum_dot.enums import QDState  # <- recommended canonical


# Define a fixed basis ordering for the 4-level dot.
_BASIS_ORDER = (QDState.G, QDState.X1, QDState.X2, QDState.XX)
_STATE_TO_INDEX = {s: i for i, s in enumerate(_BASIS_ORDER)}

# Also allow lookup by name/value strings to survive mixed-enum situations
_NAME_TO_INDEX = {
    getattr(s, "name", str(s)): i for i, s in enumerate(_BASIS_ORDER)
}
_VALUE_TO_INDEX = {
    getattr(s, "value", None): i for i, s in enumerate(_BASIS_ORDER)
}


def _state_index(state) -> int:
    """
    Convert a QDState-like object into a basis index 0..3.

    Supports:
      - canonical QDState members
      - enums with .name like "G", "X1", ...
      - enums/strings with .value == "G", ...
      - raw strings "G"/"X1"/...
      - raw ints 0..3 (if you ever use those)
    """
    if isinstance(state, int):
        if 0 <= state < 4:
            return state
        raise ValueError(f"Invalid integer QD state index: {state}")

    # exact canonical enum member
    if state in _STATE_TO_INDEX:
        return _STATE_TO_INDEX[state]

    # string directly
    if isinstance(state, str):
        if state in _NAME_TO_INDEX:
            return _NAME_TO_INDEX[state]
        if state in _VALUE_TO_INDEX:
            return _VALUE_TO_INDEX[state]
        raise ValueError(f"Unknown QDState string: {state}")

    # enum-like
    name = getattr(state, "name", None)
    if name in _NAME_TO_INDEX:
        return _NAME_TO_INDEX[name]

    val = getattr(state, "value", None)
    if val in _VALUE_TO_INDEX:
        return _VALUE_TO_INDEX[val]
    if isinstance(val, str) and val in _NAME_TO_INDEX:
        return _NAME_TO_INDEX[val]

    raise ValueError(f"Cannot map state {state!r} to 4-level basis index.")


def transition_operator(to_state, from_state) -> np.ndarray:
    """
    |to><from| in the fixed (G, X1, X2, XX) basis.
    """
    op = np.zeros((4, 4), dtype=np.complex128)
    i = _state_index(to_state)
    j = _state_index(from_state)
    op[i, j] = 1.0
    return op
