from typing import List, Tuple
from scipy.constants import c as _c, e as _e, hbar as _hbar
import numpy as np
from bec.light.classical import ClassicalTwoPhotonDrive
from bec.light.envelopes import GaussianEnvelope
from bec.operators.qd_operators import QDState
from bec.params.cavity_params import CavityParams
from bec.params.dipole_params import DipoleParams
from bec.params.energy_levels import EnergyLevels
from bec.plots.plotter import PlotConfig, QDPlotGrid
from bec.plots.styles import default_theme
from bec.quantum_dot.dot import QuantumDot
from bec.simulation.engine import SimulationConfig, SimulationEngine
from bec.simulation.qd_traces import QDTraces
from bec.simulation.scenarios import ClassicalDriveScenario
from bec.simulation.solvers import MesolveOptions, QutipMesolveBackend


# GLOBAL DOT SETTIGNS
EXCITON = 1.300  # eV
BINDING = 3e-3  # eV
FSS = 5e-6
FSS_u_eV = int(round(FSS * 1e6))
FILE_NAME = f"fss_{FSS_u_eV}"

# GLOBAL PULSE SETTINGS
SIGMA = 5e-11
T_0 = 1e-9
OMEGA_0 = 1e10


def drive(
    area: float = np.pi, detuning: float = 0.0, sigma: float = SIGMA
) -> ClassicalTwoPhotonDrive:
    """
    Create a drive from given parameters and global settings
    """
    pulse_area = area / OMEGA_0
    env = GaussianEnvelope(t0=T_0, sigma=sigma, area=pulse_area)
    w_xxg = (2 * EXCITON - BINDING) * _e / _hbar
    wL = 0.5 * w_xxg + detuning
    drive = ClassicalTwoPhotonDrive(
        envelope=env, omega0=OMEGA_0, label="2g", laser_omega=wL
    )
    return drive


def dot() -> QuantumDot:

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
    )


def _run_once(drive, qd) -> Tuple[QDTraces, dict]:
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

    scenario = ClassicalDriveScenario(drive=drive)
    engine = SimulationEngine(solver=backend)

    # RUN SIMULATION
    traces, rho_final, rho_phot_final = engine.run_with_state(
        qd, scenario, cfg)

    diag = qd.diagnostics.mode_layout_summary(rho_phot=rho_phot_final)

    return traces, diag


def plot(traces: List[QDTraces], titles: List[str], file: str = "test.pdf"):
    cfg = PlotConfig(
        show_top=True, figsize=(10, 4.0), titles=titles, filename=file
    )
    grid = QDPlotGrid(theme=default_theme(), cfg=cfg)
    grid.render(traces)


def detune_to_wavelength_strings(detuning_rad_s: float) -> tuple[str, str, str]:
    # Exact two-photon resonance (biexciton ↔ ground)
    w_xxg = (2 * EXCITON - BINDING) * _e / _hbar
    omega0 = 0.5 * w_xxg  # resonant laser angular frequency
    omegaL = omega0 + detuning_rad_s

    # λ = 2π c / ω
    lam0 = 2 * np.pi * _c / omega0  # resonance wavelength (m)
    lamL = 2 * np.pi * _c / omegaL  # laser wavelength (m)
    dlam = lamL - lam0  # wavelength detuning (m)

    # Also give Δf
    df_Hz = detuning_rad_s / (2 * np.pi)

    lamL_nm = lamL * 1e9
    dlam_pm = dlam * 1e12
    df_GHz = df_Hz * 1e-9

    return (f"{lamL_nm:.2f} nm", f"{dlam_pm:+.1f} pm", f"{df_GHz:+.2f} GHz")


def detune_si_GHz(detuning_rad_s: float, digits: int = 2) -> str:
    df_GHz = detuning_rad_s / (2 * np.pi) * 1e-9
    return rf"\SI{{{df_GHz:+.{digits}f}}}{{\giga\hertz}}"


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
        # choose one: 'avg' | 'late' | 'early'
        "Lambda": float(diag.get("overlap_abs_avg", 0.0)),
        "p_two": float(diag.get("probabilities", {}).get("p_two_photon", 0.0)),
        "Lambda_late": float(diag.get("overlap_abs_late", 0.0)),
        "Lambda_early": float(diag.get("overlap_abs_early", 0.0)),
        # NEW: cross-coherence phase (conditional, on the postselected 2-photon subspace)
        "Phase_deg": float(coh.get("phase_deg", float("nan"))),
        "Phase_rad": float(coh.get("phase_rad", float("nan"))),
        "Coh_abs": float(
            coh.get("abs", 0.0)
        ),  # optional: magnitude of coherence
    }


def latex_row(label: str, diag: dict) -> str:
    m = extract_metrics(diag)
    # If phase is NaN (no 2-photon component), print '--'
    phase_str = "--" if np.isnan(m["Phase_rad"]) else f"{m['Phase_rad']:.1f}"
    return (
        f"{label} & "
        f"{m['N_early']:.3f} & {m['N_late']:.3f} & "
        f"{m['E_N']:.3f} & {m['E_N_cond']:.3f} & "
        f"{m['Purity']:.3f} & {m['Lambda']:.3f} & "
        f"{m['Coh_abs']:.3f} & {phase_str}\\\\"
    )


def run():
    drv_pi = drive(area=np.pi, detuning=0.0)
    drv_2pi = drive(area=5 * np.pi, detuning=0.0, sigma=3 * SIGMA)
    DETUNE = 2e10
    drv_pi_detuned = drive(area=1 * np.pi, detuning=DETUNE)

    tr1, diag1 = _run_once(drv_pi, dot())
    tr2, diag2 = _run_once(drv_2pi, dot())
    tr3, diag3 = _run_once(drv_pi_detuned, dot())
    print(latex_row(r"$\pi$-pulse", diag1))
    print(latex_row(r"$5\pi$-pulse", diag2))
    print(latex_row(r"detuned $\pi$-pulse", diag3))
    dGHz = detune_si_GHz(DETUNE)
    titles = [
        r"$\pi$-pulse, $\Delta_{2\gamma}=\SI{0}{\giga\hertz}$",
        r"$5\pi$-pulse, $\Delta_{2\gamma}=\SI{0}{\giga\hertz}$",
        rf"$\pi$-pulse, $\Delta_{{2\gamma}}={dGHz}$",
    ]
    tr1.to_csv(f"{FILE_NAME}_pi_pulse")
    tr1.to_csv(f"{FILE_NAME}_5pi_pulse")
    tr1.to_csv(f"{FILE_NAME}_detuned_pi_pulse")
    plot([tr1, tr2, tr3], titles, file=FILE_NAME)


if __name__ == "__main__":
    run()
