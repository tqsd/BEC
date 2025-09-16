from typing import Optional, Any, Dict
from dataclasses import dataclass

from qutip import mesolve

from bec.simulation.protocols import SolverBackend


@dataclass
class MesolveOptions:
    nsteps: int = 10000
    rtol: float = 1e-9
    atol: float = 1e-11
    progress_bar: str = "tqdm"
    store_final_state: bool = False
    max_step: float = 5e-3


class QutipMesolveBackend(SolverBackend):
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

    def solve(self, H, rho0, tlist, c_ops=None, e_ops=None):
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
