from typing import List, Optional, Any, Dict
from dataclasses import dataclass

from qutip import Qobj, mesolve
import numpy as np

from bec.simulation.protocols import SolverBackend


@dataclass
class MesolveOptions:
    """
    Configuration for QuTip mesolve.

    Parameters
    ----------
    nsteps: int
        Maximum internal ODE steps per point.
    rtol: float
        Relative tolerance for ODE integration.
    atol: float
        Absolute tolerance for ODE integration.
    progress_bar: str
        Progres bar implementation name for QuTiP
    store_final_state: bool
        if True, `mesolve` stores the final state on the result
    max_step: float
        Maximum internal step size (in solver time units).
    """

    nsteps: int = 10000
    rtol: float = 1e-9
    atol: float = 1e-11
    progress_bar: str = "tqdm"
    store_final_state: bool = False
    max_step: float = 1e-2


class QutipMesolveBackend(SolverBackend):
    """
    Thin adapter around QuTip mesolve with explicit options handling.

    Notes:
    ------
    - for QuTip >= 4.7, a plain dict can be passed as `options`.
    - for older QuTiP versions which still require `qutip.Options`
       a fallback import and construction is attempted.


    Attributes:
    -----------
    opts: MesolveOptions
        Stored options used for each solve call
    """

    def __init__(self, opts: Optional[MesolveOptions] = None):
        self.opts = opts or MesolveOptions()

    def _as_dict(self) -> Dict[str, Any]:
        return {
            "nsteps": self.opts.nsteps,
            "rtol": self.opts.rtol,
            "atol": self.opts.atol,
            "progress_bar": self.opts.progress_bar,
            "store_final_state": self.opts.store_final_state,
            "max_step": self.opts.max_step,
        }

    def solve(
        self,
        H: list,
        rho0: Qobj,
        tlist: np.ndarray,
        c_ops: list[Qobj] = [],
        e_ops: list[Qobj] = [],
    ):
        """
        Run QuTiP mesolve with stored options.

        Arguments:
        ----------
        H: Any
            Time-independent or time-dependent Hamiltonian accepted by mesolve
        rho0: Any
            Initial state (Qobj)
        tlist: np.ndarray
            Time grid in solver units
        c_ops: list
            List of collapse operators
        e_ops: list
            List of expectation-value operators

        Returns
        -------
        qutip.solver.Result
            Mesolve result object
        """
        opt_dict = self._as_dict()
        try:
            # QuTiP >= 4.7: pass a plain dict
            return mesolve(
                H, rho0, tlist, c_ops=c_ops, e_ops=e_ops, options=opt_dict
            )
        except TypeError:
            # Back-compat: some older versions still expect qutip.Options(...)
            from qutip import Options as _Options

            return mesolve(
                H,
                rho0,
                tlist,
                c_ops=c_ops,
                e_ops=e_ops,
                options=_Options(**opt_dict),
            )
