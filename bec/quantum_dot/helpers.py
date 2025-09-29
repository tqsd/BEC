from typing import Tuple, List, Optional
from bec.params.transitions import Transition, TransitionType


def infer_index_sets_from_registry(
    qd, *, rho_has_qd: bool = False, factor_dim: Optional[int] = None
) -> Tuple[List[int], List[int], List[int], List[int], List[int], int]:
    """
    Infer photonic factor indices for early/late emissions and polarizations
    from the mode registry, along with per-factor dimension list and offset.

    The function inspects `qd.modes.modes` and looks for intrinsic
    transitions to decide whether the redistry represents a
    4 mode layout or 2 mode layout (FSS dependent).

    indices are assigned por mode and polarizations as pairs
      `plus_index  = offset + 2 * i_mode`
      `minus_index = offset + 2 * i_mode + 1`

    where `i_mode` is the mode's index in the registry and `offset`
    is 1 if `rho_has_qd=True` (photonic parts start after the QD mode)
    otherwise `offset=0`.

    Parameters
    ----------
    qd: QuantumDot (here Any)
    rho_has_qd: bool, optional
    factor_dim: int, optional
        Per polarization Fock dimensions (usually 2)

    Returns
    -------
    tuple[list[int], list[int], list[int], list[int], list[int], int]
        A tuple with 6 elements
        - early_modes
        - late_modes
        - plus_modes
        - minus_modes
        - dims
        - offset

    Raises
    ------
    VauleError
        If the layout cannot be infered

    Notes
    -----
    - Factor ordering per mode is always (plus, minus)
    - The returned `dims` length equals `len(early_modes) + len(late_modes)`.
    """
    if factor_dim is None:
        # Try a few common locations/names
        ncut = (
            getattr(qd, "N_cut", None)
            or getattr(getattr(qd, "cavity_params", None), "N_cut", None)
            or getattr(getattr(qd, "modes", None), "N_cut", None)
        )
        if ncut is not None:
            try:
                factor_dim = int(ncut)
            except Exception:
                factor_dim = None
        if factor_dim is None:
            factor_dim = 2
    d = int(factor_dim)

    offset = 1 if rho_has_qd else 0

    def pm_indices(i_mode: int) -> Tuple[int, int]:
        i_plus = offset + 2 * i_mode
        i_minus = offset + 2 * i_mode + 1
        return i_plus, i_minus

    present = set()
    for i, m in enumerate(qd.modes.modes):
        tr = getattr(m, "transition", None)
        src = getattr(m, "source", None)
        if src == TransitionType.INTERNAL and isinstance(tr, Transition):
            present.add(tr)

    four_mode = all(
        t in present
        for t in (
            Transition.X1_XX,
            Transition.X2_XX,
            Transition.G_X1,
            Transition.G_X2,
        )
    )
    two_mode = all(t in present for t in (Transition.X_XX, Transition.G_X))

    if not (four_mode or two_mode):
        raise ValueError(
            "Could not infer 2-mode or 4-mode layout from registry. "
            f"Found intrinsic transitions: {
                sorted(present, key=lambda x: x.value)}"
        )

    if four_mode:
        i_e1, _ = qd.modes.by_transition_and_source(
            Transition.X1_XX, TransitionType.INTERNAL
        )
        i_e2, _ = qd.modes.by_transition_and_source(
            Transition.X2_XX, TransitionType.INTERNAL
        )
        i_l1, _ = qd.modes.by_transition_and_source(
            Transition.G_X1, TransitionType.INTERNAL
        )
        i_l2, _ = qd.modes.by_transition_and_source(
            Transition.G_X2, TransitionType.INTERNAL
        )
        e1p, e1m = pm_indices(i_e1)
        e2p, e2m = pm_indices(i_e2)
        l1p, l1m = pm_indices(i_l1)
        l2p, l2m = pm_indices(i_l2)
        early_modes = [e1p, e1m, e2p, e2m]
        late_modes = [l1p, l1m, l2p, l2m]
        plus_modes = [e1p, e2p, l1p, l2p]
        minus_modes = [e1m, e2m, l1m, l2m]
    else:
        i_e, _ = qd.modes.by_transition_and_source(
            Transition.X_XX, TransitionType.INTERNAL
        )
        i_l, _ = qd.modes.by_transition_and_source(
            Transition.G_X, TransitionType.INTERNAL
        )
        ep, em = pm_indices(i_e)
        lp, lm = pm_indices(i_l)
        early_modes = [ep, em]
        late_modes = [lp, lm]
        plus_modes = [ep, lp]
        minus_modes = [em, lm]

    num_factors = len(early_modes) + len(late_modes)
    dims = [d] * num_factors

    return early_modes, late_modes, plus_modes, minus_modes, dims, offset
