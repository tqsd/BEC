from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.cm import get_cmap

from qutip import Qobj
from bec.quantum_dot.base import QuantumDotSystemBase
from bec.quantum_dot.params import EnergyLevels, CavityParams, DipoleParams


@dataclass
class QuantumDotEnsembleConfig:
    label: str
    energy_levels: EnergyLevels
    cavity_params: Optional[CavityParams] = None
    dipole_params: Optional[DipoleParams] = None
    initial_state: Optional[Qobj] = None  # Defaults to |G><G|
    input_modes: Optional[dict] = None  # Photon input modes
    sim_params: dict = None  # e.g., {'Omega0': 1.0, 't0': 5.0, 'sigma': 1.0}


class QuantumDotEnsemble:
    def __init__(
        self,
        system: QuantumDotSystemBase,
        configs: List[QuantumDotEnsembleConfig],
        output_path="~/.roam/plots/qdot_ensemble.png",
        dpi=300,
    ):
        self.system = system
        self.configs = configs
        self.output_path = Path(output_path).expanduser()
        self.dpi = dpi

    def simulate(
        self,
        cfg: QuantumDotEnsembleConfig,
        tscale: float = 10.0,
        steps: int = 300,
    ):
        system = self.system(  # Directly call the class
            energy_levels=cfg.energy_levels,
            cavity_params=cfg.cavity_params,
            dipole_params=cfg.dipole_params,
        )

        rho0 = cfg.initial_state or system.get_initial_state(**cfg.sim_params)
        tlist = np.linspace(0, tscale, steps)
        result = system.simulate(tlist, **(cfg.sim_params or {}))

        return tlist, result.expect

    def plot(self, path: Optional[Union[str, Path]] = None) -> Path:
        if path:
            self.output_path = Path(path).expanduser()
        self._draw()
        plt.savefig(self.output_path, dpi=self.dpi)
        return self.output_path

    def _draw(self) -> Figure:
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = ["G", "X1", "X2", "XX"]
        linestyles = ["-", "--", ":", "-."]
        cmap = get_cmap("tab10")

        for idx, cfg in enumerate(self.configs):
            tlist, expect = self.simulate(cfg)
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
        ax.set_title("Quantum Dot Population Dynamics")
        ax.legend()
        ax.grid(True)
        return fig
