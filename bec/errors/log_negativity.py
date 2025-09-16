import numpy as np
from qutip import Qobj


def _ensure_dense_qobj(rho, dims):
    """Return a normalized, dense density matrix for given local dimensions.

    Converts an input state ``rho`` (which may be a QuTiP ``Qobj`` or a numpy
    array) to a dense numpy array, checks shape compatibility with the provided
    local dimensions ``dims``, and returns a trace-1 complex matrix.

    Mathematics:
        Let ``D = prod(dims)``. This function enforces:
        - ``rho`` is represented as an array ``R`` in :math:`\\mathbb{C}^{D \\times D}`
        - :math:`\\mathrm{Tr}(R) > 0`
        - returns :math:`R / \\mathrm{Tr}(R)`

    Args:
        rho: Density operator as ``qutip.Qobj`` or array-like of shape (D, D).
        dims: Sequence of local dimensions, e.g., ``[2]*n`` for n qubits.

    Returns:
        np.ndarray: A dense, normalized complex matrix of shape (D, D).

    Raises:
        ValueError: If the shape is incompatible with ``dims`` or if the trace
            is non-positive.
    """
    if isinstance(rho, Qobj):
        R = rho.full()
    else:
        R = np.asarray(rho)

    D = int(np.prod(dims))
    if R.shape != (D, D):
        raise ValueError(
            f"rho shape {R.shape} is incompatible with dims {
                dims} (product {D})."
        )

    tr = np.trace(R).real
    if tr <= 0:
        raise ValueError("rho has non-positive trace.")

    return (R / tr).astype(complex)


def _pt_on_subsystems(R, dims, sys_indices):
    """Partial transpose on selected subsystems.

    Performs the partial transpose with respect to the subsystems listed in
    ``sys_indices``. The array is reshaped to a 2M-index tensor
    with ``M = len(dims)``: first M indices for kets, last M for bras. For each
    ``k`` in ``sys_indices``, it swaps the pair of axes ``(k, k+M)``, then
    reshapes back.

    Mathematics:
        Let :math:`R` be the density matrix on
        :math:`\\mathcal{H} = \\bigotimes_{i=0}^{M-1} \\mathcal{H}_i`.
        The partial transpose on a set :math:`S \\subseteq \\{0, \\ldots, M-1\\}` is:

        .. math::

            (|i_0 \\ldots i_{M-1}\\rangle\\langle j_0 \\ldots j_{M-1}|)^{T_S}
            = \\bigotimes_{k \\notin S} |i_k\\rangle\\langle j_k|
              \\otimes \\bigotimes_{k \\in S} |j_k\\rangle\\langle i_k|

        In tensor form with axes ``[0..M-1]`` for ket and ``[M..2M-1]`` for bra,
        this corresponds to swapping axes ``k`` and ``k+M`` for all ``k`` in ``S``.

    Args:
        R: Dense density matrix (np.ndarray) of shape (D, D).
        dims: Sequence of local dimensions; product must equal D.
        sys_indices: Iterable of subsystem indices on which to transpose.

    Returns:
        np.ndarray: The partially transposed matrix, same shape as R.

    Raises:
        ValueError: If reshaping using ``dims`` is not possible.
    """
    M = len(dims)
    tens = R.reshape(dims + dims)  # ket: 0..M-1, bra: M..2M-1
    for k in sys_indices:
        tens = np.swapaxes(tens, k, k + M)
    return tens.reshape(R.shape)


def log_negativity_early_late(rho, dims, early_idxs, late_idxs):
    """Log-negativity across the (early) vs (late) bipartition.

    Computes the log-negativity for a bipartition defined by indices
    ``early_idxs`` and ``late_idxs``, which must be disjoint and cover all
    subsystems.

    Mathematics:
        Given a bipartite state :math:`\\rho` on
        :math:`\\mathcal{H}_E \\otimes \\mathcal{H}_L` and the partial transpose
        with respect to L, denoted :math:`\\rho^{T_L}`, define the trace norm

        .. math::

            \\lVert A \\rVert_1 = \\mathrm{Tr}\\,\\sqrt{A^\\dagger A}

        Then the negativity and log-negativity are:

        .. math::

            \\mathcal{N}(\\rho) = \\tfrac{1}{2}(\\lVert \\rho^{T_L} \\rVert_1 - 1) \\\\
            E_{\\mathcal{N}}(\\rho) = \\log_2 \\lVert \\rho^{T_L} \\rVert_1

        Properties:
            - :math:`E_{\\mathcal{N}}(\\rho) \\geq 0`, with equality iff
              :math:`\\rho` is PPT (positive under partial transpose).
            - For a maximally entangled state of local dimension d:

              .. math:: E_{\\mathcal{N}}(\\rho) = \\log_2 d

              In particular, for a 2-qubit Bell state, :math:`E_{\\mathcal{N}} = 1`.

    Args:
        rho: Density operator as ``qutip.Qobj`` or array-like of shape (D, D).
        dims: Sequence of local dimensions (length M); product must be D.
        early_idxs: Iterable of subsystem indices forming the "early" side.
        late_idxs: Iterable of subsystem indices forming the "late" side.

    Returns:
        float: The log-negativity
        :math:`E_{\\mathcal{N}}(\\rho) = \\log_2 \\lVert \\rho^{T_L} \\rVert_1`.

    Raises:
        ValueError: If the indices do not form a disjoint cover of all
            subsystems, or if ``rho`` is invalid (shape/trace).
    """
    M = len(dims)
    all_idxs = sorted(set(list(early_idxs) + list(late_idxs)))
    if all_idxs != list(range(M)):
        raise ValueError(
            "early_idxs union late_idxs must equal all subsystem indices [0..M-1]."
        )
    if set(early_idxs) & set(late_idxs):
        raise ValueError("early_idxs and late_idxs must be disjoint.")

    R = _ensure_dense_qobj(rho, dims)
    Rpt = _pt_on_subsystems(R, dims, late_idxs)
    s = np.linalg.svd(Rpt, compute_uv=False)
    trace_norm = float(np.sum(s).real)
    return np.log2(trace_norm)
