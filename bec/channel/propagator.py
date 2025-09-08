import numpy as np
from qutip import Options
from typing import List, Optional
from qutip import propagator, Qobj


def superop_from_system(
    qd_sys,
    dims: List[int],
    tlist: np.ndarray,
    classical_2g=None,
    solver_opts: Optional[Options] = None,
) -> Qobj:
    """
    Returns the Liouville-space propagator S(t_final, 0) as a Qobj superoperator.
    """
    H_terms = qd_sys.build_hamiltonians(dims, classical_2g=classical_2g)
    c_ops = qd_sys.qutip_collapse_operators(dims)
    # QuTiP returns a list of propagators on the grid; pick the last one
    S_list = propagator(H_terms, tlist, c_ops=c_ops, options=solver_opts)
    S_final = S_list[-1] if isinstance(S_list, (list, tuple)) else S_list
    return S_final  # superoperator if c_ops given
