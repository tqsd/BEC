from typing import Tuple, List
from bec.params.transitions import Transition, TransitionType


def infer_index_sets_from_registry(
    qd, *, rho_has_qd: bool = False
) -> Tuple[List[int], List[int], List[int], List[int], List[int], int]:
    """
    Return index lists for early/late and +/- factors that match the
    photonic tail of your tensor layout.

    Works with either:
      - 4-mode layout: {X1_XX, X2_XX} emit early; {G_X1, G_X2} emit late
      - 2-mode layout: {X_XX} emits early; {G_X} emits late

    Returns:
        early_modes : list[int]    # factor indices of all +/- in early block
        late_modes  : list[int]    # factor indices of all +/- in late  block
        plus_modes  : list[int]    # factor indices of all '+' across early+late
        minus_modes : list[int]    # factor indices of all '−' across early+late
        dims        : list[int]    # dims list for these photonic factors (2 per factor)
        offset      : int          # 0 if rho is photonic-only; 1 if QD factor sits at head
    """
    # If the density matrix still contains the QD as the first factor, skip it.
    # (QD is 4-dim at dims[0] per QuantumDot docstring.)
    offset = 1 if rho_has_qd else 0

    # helper: for a given *mode index* i in ModeRegistry, return the two factor indices
    # ('+' then '−') in the global dims tail. KronPad pads operators in that order.
    def pm_indices(i_mode: int) -> Tuple[int, int]:
        i_plus = offset + 2 * i_mode
        i_minus = offset + 2 * i_mode + 1
        return i_plus, i_minus

    # Collect which intrinsic transitions are present
    present = set()
    # registry exposes a flat list of LightMode
    for i, m in enumerate(qd.modes.modes):
        tr = getattr(m, "transition", None)
        src = getattr(m, "source", None)
        if src == TransitionType.INTERNAL and isinstance(tr, Transition):
            present.add(tr)

    # Decide which layout we have
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

    # Build index lists
    if four_mode:
        # Early: XX→X1 and XX→X2  |  Late: X1→G and X2→G
        i_e1, _ = qd.modes.by_transition_and_source(
            Transition.X1_XX, TransitionType.INTERNAL
        )  #
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

        # 4 LightModes × 2 pol-factors each = 8 factors → dims=[2]*8
        dims = [2] * (len(early_modes) + len(late_modes))

    else:  # two_mode
        # Early: X↔XX  |  Late: G↔X
        i_e, _ = qd.modes.by_transition_and_source(
            Transition.X_XX, TransitionType.INTERNAL
        )  #
        i_l, _ = qd.modes.by_transition_and_source(
            Transition.G_X, TransitionType.INTERNAL
        )

        ep, em = pm_indices(i_e)
        lp, lm = pm_indices(i_l)

        early_modes = [ep, em]
        late_modes = [lp, lm]
        plus_modes = [ep, lp]
        minus_modes = [em, lm]

        # 2 LightModes × 2 pol-factors each = 4 factors → dims=[2]*4
        dims = [2] * (len(early_modes) + len(late_modes))

    return early_modes, late_modes, plus_modes, minus_modes, dims, offset
