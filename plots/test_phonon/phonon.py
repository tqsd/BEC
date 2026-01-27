from typing import List, Optional, Tuple
import numpy as np
from scipy.constants import e as _e, hbar as _hbar, c as _c

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


# ---- Global dot settings ----
EXCITON = 1.300  # eV
BINDING = 3e-3  # eV
FSS = 5e-6  # eV

# ---- Global pulse settings ----
SIGMA = 5e-11
T_0 = 1e-9
OMEGA_0 = 1e10  # rad/s


def make_drive(
    area: float, detuning: float = 0.0, sigma: float = SIGMA
) -> ClassicalTwoPhotonDrive:
    pulse_area = area / OMEGA_0
    env = GaussianEnvelope(t0=T_0, sigma=sigma, area=pulse_area)

    w_xxg = (2 * EXCITON - BINDING) * _e / _hbar
    wL = 0.5 * w_xxg + detuning

    return ClassicalTwoPhotonDrive(
        envelope=env,
        omega0=OMEGA_0,
        label="2g",
        laser_omega=wL,
    )


def make_dot(phonons: Optional[PhononParams] = None) -> QuantumDot:
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

    # Omega_t is Omega_solver (rad / solver_time)
    Omega_solver = traces.omega
    Omega_phys = Omega_solver / traces.time_unit_s  # rad/s

    scale = 1e-12  # same as gamma_phi_eid_scale
    gamma_eid_phys = scale * (np.abs(Omega_phys) ** 2)  # 1/s
    print("max Omega_phys:", np.max(np.abs(Omega_phys)))
    print("max gamma_eid (1/s):", np.max(gamma_eid_phys))
    print(
        "max gamma_eid in solver units:",
        np.max(gamma_eid_phys) * traces.time_unit_s,
    )
    return traces, diag


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


def print_metrics_table(pulse_label: str, cases: List[Tuple[str, dict]]):
    print("\n=== ", pulse_label, " ===")
    header = f"{'case':<12} | {'N_e':>6} {'N_l':>6} | {'E_N':>6} {
        'E_Nc':>6} | {'Pur':>6} | {'Lam':>6} | {'Coh':>6}"
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
    # This makes a single figure with exactly 3 panels (your constraint).
    cfg = PlotConfig(
        show_top=True, figsize=(10, 4.0), titles=titles, filename=filename
    )
    grid = QDPlotGrid(theme=default_theme(), cfg=cfg)
    grid.render(traces)


def main():
    # --- Define three phonon “tests” ---
    # 1) no phonons
    pp_none = None

    # 2) static dephasing (choose something visible)
    pp_static = PhononParams(
        gamma_phi_Xp_1_s=1e9,
        gamma_phi_Xm_1_s=1e9,
        gamma_phi_XX_1_s=1e9,
        gamma_phi_eid_scale=0.0,
        enable_exciton_relaxation=False,
    )

    # 3) EID only (no static). Choose a scale that gives a noticeable effect.
    # NOTE: with Omega_phys ~ 1e10 rad/s, if scale ~ 1e-12 s then gamma ~ 1e8 1/s.
    pp_eid = PhononParams(
        gamma_phi_Xp_1_s=0.0,
        gamma_phi_Xm_1_s=0.0,
        gamma_phi_XX_1_s=0.0,
        gamma_phi_eid_scale=1e-12,
        enable_exciton_relaxation=False,
    )

    cases = [
        ("EID", pp_eid),
        ("baseline", pp_none),
        (r"static $\varphi$", pp_static),
    ]

    # --- Define three pulses (we will output 3 PDFs, each has 3 panels) ---
    pulses = [
        ("pi", make_drive(area=np.pi, detuning=0.0), r"$\pi$ pulse"),
        (
            "5pi",
            make_drive(area=5 * np.pi, detuning=0.0, sigma=3 * SIGMA),
            r"$5\pi$ pulse (longer)",
        ),
        (
            "detuned",
            make_drive(area=np.pi, detuning=2e10),
            r"detuned $\pi$ pulse",
        ),
    ]

    for key, drv, pulse_label in pulses:
        traces_row: List[QDTraces] = []
        titles_row: List[str] = []
        diags_for_table: List[Tuple[str, dict]] = []

        for name, pp in cases:
            print(key, name)
            qd = make_dot(pp)
            print(qd._phonon_params)
            tr, diag = run_once(drv, qd)
            traces_row.append(tr)
            titles_row.append(f"{pulse_label} — {name}")
            diags_for_table.append((name, diag))

        print_metrics_table(pulse_label, diags_for_table)
        plot_row(traces_row, titles_row, filename=f"phonon_test_{key}.pdf")


if __name__ == "__main__":
    main()
