from typing import List, Optional, Tuple, Callable
import numpy as np
from scipy.constants import e as _e, hbar as _hbar
from math import erf  # if you prefer scipy.special.erf, that's fine too

from bec.light.classical import ClassicalTwoPhotonDrive
from bec.light.envelopes import GaussianEnvelope
from bec.operators.qd_operators import QDState
from bec.params.cavity_params import CavityParams
from bec.params.dipole_params import DipoleParams
from bec.params.energy_levels import EnergyLevels
from bec.params.phonon_params import PhononParams

from bec.plots.plotter import PlotConfig, QDPlotGrid
from bec.plots.styles import default_theme

from bec.quantum_dot.dot import QuantumDot
from bec.simulation.engine import SimulationConfig, SimulationEngine
from bec.simulation.qd_traces import QDTraces
from bec.simulation.scenarios import ClassicalDriveScenario
from bec.simulation.solvers import MesolveOptions, QutipMesolveBackend

# If you already put tanh_chirp into bec.light.detuning, import it:
from bec.light.detuning import tanh_chirp


# ---- Global dot settings ----
EXCITON = 1.300  # eV
BINDING = 3e-3  # eV

# We'll sweep FSS via a loop
FSS_DEFAULT = 5e-6  # eV

# ---- Global pulse settings ----
SIGMA = 5e-11
T_0 = 1e-9
OMEGA_0 = 1e10  # rad/s


# --- chirp helpers (rad/s) ---
def linear_chirp(t0: float, slope: float) -> Callable[[float], float]:
    # Δ(t) = slope * (t - t0)
    def det(t_phys: float) -> float:
        return float(slope * (t_phys - t0))

    return det


def erf_chirp(
    t0: float, Delta_max: float, tau: float
) -> Callable[[float], float]:
    # Δ(t) = Delta_max * erf((t - t0)/tau)
    def det(t_phys: float) -> float:
        return float(Delta_max * erf((t_phys - t0) / tau))

    return det


def make_arp_drive(
    *,
    env_sigma: float,
    detuning_fn: Callable[[float], float],
    label: str,
) -> ClassicalTwoPhotonDrive:
    # ARP doesn't target a pulse area, but you still need a coupling envelope.
    # Reuse your GaussianEnvelope(area=pi/Omega0) so amplitude scale remains familiar.
    pulse_area = np.pi / OMEGA_0
    env = GaussianEnvelope(t0=T_0, sigma=env_sigma, area=pulse_area)

    # set central laser frequency exactly on two-photon resonance
    w_xxg = (2 * EXCITON - BINDING) * _e / _hbar
    wL = 0.5 * w_xxg

    return ClassicalTwoPhotonDrive(
        envelope=env,
        omega0=OMEGA_0,
        detuning=detuning_fn,  # <-- chirped detuning
        label=label,
        laser_omega=wL,
    )


def make_dot(FSS: float, phonons: Optional[PhononParams] = None) -> QuantumDot:
    x1 = EXCITON + 0.5 * FSS
    x2 = EXCITON - 0.5 * FSS
    e_xx = (x1 + x2) - BINDING

    EL = EnergyLevels(
        biexciton=e_xx,
        exciton=EXCITON,
        fss=FSS,
        delta_prime=0,
    )
    CP = CavityParams(Q=5e4, Veff_um3=0.5, lambda_nm=930.0, n=3.4)
    DP = DipoleParams(dipole_moment_Cm=10.0 * 3.33564e-30)

    return QuantumDot(
        EL,
        cavity_params=CP,
        dipole_params=DP,
        time_unit_s=1e-9,
        initial_state=QDState.G,
        phonon_params=phonons,
    )


def run_once(
    drv: ClassicalTwoPhotonDrive, qd: QuantumDot
) -> Tuple[QDTraces, dict]:
    tlist = np.linspace(0.0, 2.0, 1001)
    cfg = SimulationConfig(tlist=tlist, trunc_per_pol=2, time_unit_s=1e-9)

    backend = QutipMesolveBackend(
        MesolveOptions(
            nsteps=20000,
            rtol=1e-7,
            atol=1e-9,
            progress_bar="tqdm",
            store_final_state=True,
            max_step=1e-1,
        )
    )

    scenario = ClassicalDriveScenario(drive=drv)
    engine = SimulationEngine(solver=backend)

    traces, rho_final, rho_phot_final = engine.run_with_state(
        qd, scenario, cfg)
    diag = qd.diagnostics.mode_layout_summary(rho_phot=rho_phot_final)
    return traces, diag


# reuse your extract_metrics / print_metrics_table / plot_row
def extract_metrics(diag: dict) -> dict:
    pn = diag.get("photon_numbers", {})
    ent = diag.get("entanglement", {})
    pur = diag.get("purity", {})
    bell = diag.get("bell_component", {}) or {}
    coh = bell.get("coherence_cross", {}) or {}
    return {
        "N_early": float(pn.get("N_early", 0.0)),
        "N_late": float(pn.get("N_late", 0.0)),
        "E_N": float(ent.get("log_negativity", 0.0)),
        "E_N_cond": float(ent.get("log_negativity_conditional", 0.0)),
        "Purity": float(pur.get("unconditional", 0.0)),
        "Lambda": float(diag.get("overlap_abs_avg", 0.0)),
        "Coh_abs": float(coh.get("abs", 0.0)),
        "Phase_rad": float(coh.get("phase_rad", float("nan"))),
    }


def print_metrics_table(label: str, cases: List[Tuple[str, dict]]):
    print("\n=== ", label, " ===")
    header = (
        f"{'case':<12} | {'N_e':>6} {'N_l':>6} | {'E_N':>6} {'E_Nc':>6} | "
        f"{'Pur':>6} | {'Lam':>6} | {'Coh':>6}"
    )
    print(header)
    print("-" * len(header))
    for name, diag in cases:
        m = extract_metrics(diag)
        print(
            f"{name:<12} | "
            f"{m['N_early']:6.3f} {m['N_late']:6.3f} | "
            f"{m['E_N']:6.3f} {m['E_N_cond']:6.3f} | "
            f"{m['Purity']:6.3f} | "
            f"{m['Lambda']:6.3f} | "
            f"{m['Coh_abs']:6.3f}"
        )


def plot_row(traces: List[QDTraces], titles: List[str], filename: str):
    cfg = PlotConfig(
        show_top=True, figsize=(10, 4.0), titles=titles, filename=filename
    )
    grid = QDPlotGrid(theme=default_theme(), cfg=cfg)
    grid.render(traces)


def main():
    # --- phonon cases ---
    pp_none = None

    pp_static = PhononParams(
        gamma_phi_Xp_1_s=1e9,
        gamma_phi_Xm_1_s=1e9,
        gamma_phi_XX_1_s=1e9,
        gamma_phi_eid_scale=0.0,
        enable_exciton_relaxation=False,
    )

    pp_eid = PhononParams(
        gamma_phi_Xp_1_s=0.0,
        gamma_phi_Xm_1_s=0.0,
        gamma_phi_XX_1_s=0.0,
        gamma_phi_eid_scale=1e-12,
        enable_exciton_relaxation=False,
    )

    phonon_cases = [
        ("EID", pp_eid),
        ("baseline", pp_none),
        (r"static $\varphi$", pp_static),
    ]

    # --- ARP detuning variants ---
    # Pick sweep parameters so |Δ| is big at the edges and crosses ~0 near t0.
    Delta_max = 8e10  # rad/s
    tau = 3e-11  # s  (controls how fast the sweep crosses resonance)

    detuning_variants = [
        ("linear", linear_chirp(t0=T_0, slope=Delta_max / tau), "ARP linear"),
        ("tanh", tanh_chirp(t0=T_0, Delta_max=Delta_max, tau=tau), "ARP tanh"),
        ("erf", erf_chirp(t0=T_0, Delta_max=Delta_max, tau=tau), "ARP erf"),
    ]

    # --- FSS choices ---
    fss_values = [
        (0.0, "FSS0"),
        (FSS_DEFAULT, "FSS5ueV"),
    ]

    for fss, fss_tag in fss_values:
        for det_key, det_fn, det_label in detuning_variants:
            drv = make_arp_drive(
                env_sigma=SIGMA, detuning_fn=det_fn, label=det_key
            )

            traces_row: List[QDTraces] = []
            titles_row: List[str] = []
            diags_for_table: List[Tuple[str, dict]] = []

            for name, pp in phonon_cases:
                qd = make_dot(FSS=fss, phonons=pp)
                tr, diag = run_once(drv, qd)
                traces_row.append(tr)
                titles_row.append(f"{det_label} — {name} ({fss_tag})")
                diags_for_table.append((name, diag))

            print_metrics_table(f"{det_label} ({fss_tag})", diags_for_table)
            plot_row(
                traces_row,
                titles_row,
                filename=f"arp_{det_key}_{fss_tag}.pdf",
            )


if __name__ == "__main__":
    main()
