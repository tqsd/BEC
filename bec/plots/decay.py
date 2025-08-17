from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.cm import get_cmap
from qutip import basis, mesolve, Qobj
from bec.helpers import purcell_factor, vacuum_gamma
from bec.plots.base import BasePlot


@dataclass
class QDDecayConfig:
    label: str
    lambda_nm: float
    dipole_m_C: float
    Q: Optional[float] = None
    Veff_um3: Optional[float] = None
    n: float = 3.5
    fss: float = 0.0  # Fine structure splitting (in Hz)
    levels: List[float] = None  # Energy levels in Hz (G, X1, X2, XX)


class DecayPlot(BasePlot):
    def __init__(self, configs, output_path="~/.roam/plots/decay.png", dpi=300):
        super().__init__(output_path=output_path, dpi=dpi)
        self.configs = configs
        self.states = [basis(4, i) for i in range(4)]

    def _simulate(
        self, cfg: QDDecayConfig, tscale: float = 10.0, steps: int = 300
    ):
        gamma0 = vacuum_gamma(cfg.lambda_nm, cfg.dipole_m_C)
        gamma = gamma0
        if cfg.Q and cfg.Veff_um3:
            Fp = purcell_factor(
                cfg.lambda_nm, cfg.lambda_nm, cfg.Q, cfg.Veff_um3, cfg.n
            )
            gamma *= Fp

        g1 = g2 = gamma
        g3 = g4 = gamma / 2

        c_ops = [
            np.sqrt(g1) * self.states[1] * self.states[3].dag(),
            np.sqrt(g2) * self.states[2] * self.states[3].dag(),
            np.sqrt(g3) * self.states[0] * self.states[1].dag(),
            np.sqrt(g4) * self.states[0] * self.states[2].dag(),
        ]

        H = Qobj(np.zeros((4, 4)))
        if cfg.fss != 0.0:
            H += (cfg.fss / 2) * (
                self.states[1] * self.states[1].dag()
                - self.states[2] * self.states[2].dag()
            )
        if cfg.levels:
            for i, E in enumerate(cfg.levels):
                H += E * self.states[i] * self.states[i].dag()

        rho0 = self.states[3] * self.states[3].dag()
        tlist = np.linspace(0, tscale / gamma0, steps)
        result = mesolve(
            H, rho0, tlist, c_ops, e_ops=[s * s.dag() for s in self.states]
        )
        return tlist, result.expect

    def plot(self, path: Optional[Union[str, Path]] = None) -> Path:
        if path:
            self.output_path = Path(path).expanduser()
        return self.save()

    def _draw(self) -> Figure:

        fig, ax = plt.subplots(figsize=(10, 6))
        labels = ["G", "X1", "X2", "XX"]
        linestyles = ["-", "--", ":", "-."]
        cmap = get_cmap("tab10")

        for idx, cfg in enumerate(self.configs):
            tlist, expect = self._simulate(cfg)
            color = cmap(idx)
            for i, label in enumerate(labels):
                ax.plot(
                    tlist,
                    expect[i],
                    color=color,
                    linestyle=linestyles[i],
                    label=f"{cfg.label}: {label}",
                )

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Population")
        ax.set_title("Quantum Dot Biexciton Cascade Decay")
        ax.legend()
        ax.grid(True)
        return fig
