from dataclasses import dataclass

from qutip import Options, mesolve

from bec.simulation.protocols import SolverBackend


@dataclass
class MesolveOptions:
    nsteps: int = 10000
    rtol: float = 1e-9
    atol: float = 1e-11
    progress_bar: str = "tqdm"


class QutipMesolveBackend(SolverBackend):
    def __init__(self, opts: MesolveOptions | None = None):
        self.opts = opts or MesolveOptions()

    def solve(self, H, rho0, tlist, c_ops, e_ops):
        options = Options(
            nsteps=self.opts.nsteps,
            rtol=self.opts.rtol,
            atol=self.opts.atol,
            progress_bar=self.opts.progress_bar,
        )
        return mesolve(
            H, rho0, tlist, c_ops=c_ops, e_ops=e_ops, options=options
        )
