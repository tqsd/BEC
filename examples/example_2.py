from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from smef.core.drives.types import DriveSpec
from smef.core.units import Q, hbar, magnitude
from smef.engine import SimulationEngine, UnitSystem

from bec.light.classical import carrier_profiles
from bec.light.classical.carrier import Carrier
from bec.light.classical.factories import gaussian_field_drive
from bec.light.classical.field_drive import ClassicalFieldDriveU
from bec.quantum_dot.dot import QuantumDot
from bec.quantum_dot.enums import QDState, TransitionPair
from bec.quantum_dot.smef.initial_state import rho0_qd_vacuum
from bec.quantum_dot.spec.dipole_params import DipoleParams
from bec.quantum_dot.spec.energy_structure import EnergyStructure
from bec.quantum_dot.spec.phonon_params import (
    PhenomenologicalPhononParams,
    PhononCouplings,
    PhononModelKind,
    PhononParams,
    PolaronLAParams,
)
from bec.reporting.plotting.api import plot_run
from bec.reporting.plotting.grid import PlotConfig


def safe_print_audit(res: object) -> None:
    """
    Try a few common places where audit text might live.
    Adjust as needed once you see what your engine returns.
    """
    for attr in ["audit", "audit_text", "report", "report_text"]:
        if hasattr(res, attr):
            val = getattr(res, attr)
            if isinstance(val, str) and val.strip():
                print("\n=== AUDIT (%s) ===\n%s" % (attr, val))
                return
    print(
        "\n[warn] Could not find audit text on result object. "
        "Inspect `dir(res)` to locate it."
    )


def sample_omega_L(
    drive: ClassicalFieldDriveU, t_phys_s: np.ndarray
) -> np.ndarray:
    out = np.empty(t_phys_s.size, dtype=float)
    for i in range(t_phys_s.size):
        w = drive.omega_L_rad_s(float(t_phys_s[i]))
        if w is None:
            raise RuntimeError(
                "Drive has no carrier; omega_L_rad_s returned None"
            )
        out[i] = float(w)
    return out


def sample_E_env(
    drive: ClassicalFieldDriveU, t_phys_s: np.ndarray
) -> np.ndarray:
    out = np.empty(t_phys_s.size, dtype=float)
    for i in range(t_phys_s.size):
        out[i] = float(drive.E_env_V_m(float(t_phys_s[i])))
    return out


def compute_omega_solver_and_detuning(
    *,
    qd: QuantumDot,
    drive: ClassicalFieldDriveU,
    pair: TransitionPair,
    tlist_solver: np.ndarray,
    time_unit_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mirrors the logic you use in the emitter:
      - omega_solver(t) from drive strength model
      - detuning_rad_s(t) = mult*omega_L(t) - omega_ref_rad_s(pair)
    """
    tlist_solver = np.asarray(tlist_solver, dtype=float).reshape(-1)
    s = float(time_unit_s)
    t_phys_s = s * tlist_solver

    derived = qd.derived_view

    # Determine which directed transition is the absorption direction for the pair.
    fwd, _ = derived.t_registry.directed(pair)

    # E field envelope in V/m
    E_t = sample_E_env(drive, t_phys_s)

    # mu in C*m, hbar in J*s
    mu_Cm = float(derived.mu_Cm(fwd))
    hbar_Js = float(magnitude(hbar, "J*s"))

    # polarization overlap (optional)
    pol = 1.0 + 0.0j
    pol_vec = drive.effective_pol()
    if pol_vec is not None:
        pol = complex(derived.drive_projection(fwd, pol_vec))

    # Omega_phys(t) in rad/s
    omega_rad_s = (mu_Cm * E_t) / hbar_Js

    # Convert to solver units and apply pol overlap
    omega_solver = (omega_rad_s * s).astype(complex) * pol

    # Apply polaron renormalization B(tr) (per directed transition)
    B = float(derived.polaron_B(fwd))
    omega_solver = omega_solver * (B + 0.0j)

    # Detuning
    omega_ref = float(derived.omega_ref_rad_s(pair))
    kind = drive.preferred_kind or "1ph"
    mult = 2.0 if str(kind) == "2ph" else 1.0

    omega_L = sample_omega_L(drive, t_phys_s)
    detuning_rad_s = (mult * omega_L) - omega_ref

    return omega_solver, detuning_rad_s


def main() -> None:
    # --- QD spec ---
    energy = EnergyStructure(
        X1=Q(1.201, "eV"),
        X2=Q(1.201, "eV"),
        XX=Q(2.600, "eV"),
    )
    dipoles = DipoleParams(mu_default=Q(1e-27, "C*m"))

    # --- phonons: turn on polaron renorm + enable EID + set a nonzero EID scale ---
    # Notes:
    # - alpha and omega_c must be > 0 for your J(w) to be nonzero.
    # - gamma_phi_eid_scale is "dimensionless" in your contract but it effectively
    #   sets strength of the EID collapse. Start bigger if you don't see anything.
    phonons = PhononParams(
        kind=PhononModelKind.POLARON_LA,
        temperature=Q(4.0, "K"),
        couplings=PhononCouplings(
            phi_g=0.0, phi_x1=1.0, phi_x2=1.0, phi_xx=2.0
        ),
        polaron_la=PolaronLAParams(
            enable_polaron_renorm=True,
            enable_exciton_relaxation=False,
            enable_eid=True,
            alpha=Q(1.0e-26, "s**2"),
            omega_c=Q(1.0e12, "rad/s"),
        ),
        phenomenological=PhenomenologicalPhononParams(
            gamma_phi_eid_scale=1.0e-20,
        ),
    )

    qd = QuantumDot(energy=energy, dipoles=dipoles, phonons=phonons)

    # --- solver grid ---
    time_unit_s = float(Q(1.0, "ps").to("s").magnitude)
    tlist = np.linspace(0.0, 200.0, 2001)  # solver units
    units = UnitSystem(time_unit_s=time_unit_s)

    # --- choose 2ph target ---
    pair = TransitionPair.G_XX
    omega_ref = float(qd.derived_view.omega_ref_rad_s(pair))
    omega0 = 0.5 * omega_ref

    # --- build drive (2ph + chirp) ---
    base = gaussian_field_drive(
        t0=Q(60, "ps"),
        sigma=Q(8, "ps"),
        E0=Q(2e4, "V/m"),
        energy=Q(1.3, "eV"),
        delta_omega=Q(0.0, "rad/s"),
        pol_state=None,
        preferred_kind="2ph",
        label="drive_base",
    )

    delta_fn = carrier_profiles.linear_chirp(
        rate=Q(6.0e22, "rad/s^2"),
        t0=Q(60, "ps"),
    )
    carrier = Carrier(omega0=Q(omega0, "rad/s"), delta_omega=delta_fn)

    drive = ClassicalFieldDriveU(
        envelope=base.envelope,
        amplitude=base.amplitude,
        carrier=carrier,
        pol_state=base.pol_state,
        pol_transform=base.pol_transform,
        preferred_kind=base.preferred_kind,
        label="chirped_linear_eid_on",
    )

    # --- compute and plot omega_L(t), detuning(t), and gamma_eid(t) ---
    t_phys = (time_unit_s * tlist).astype(float)
    omega_L = sample_omega_L(drive, t_phys)

    omega_solver, detuning_rad_s = compute_omega_solver_and_detuning(
        qd=qd,
        drive=drive,
        pair=pair,
        tlist_solver=tlist,
        time_unit_s=time_unit_s,
    )

    # Pull polaron_rates from phonon outputs (created when enable_eid=True)
    po = qd.derived_view.phonon_outputs
    polaron_rates = getattr(po, "polaron_rates", None)

    if polaron_rates is None or not bool(
        getattr(polaron_rates, "enabled", False)
    ):
        print(
            "[warn] polaron_rates not enabled. You will fall back to phenomenological EID."
        )
        gamma_1_s = np.zeros_like(detuning_rad_s, dtype=float)
    else:
        eid_scale = float(qd.derived_view.gamma_phi_eid_scale)
        gamma_1_s = polaron_rates.gamma_eid_1_s(
            omega_solver=omega_solver,
            detuning_rad_s=detuning_rad_s,
            time_unit_s=float(time_unit_s),
            scale=float(eid_scale),
        )
        gamma_1_s = np.asarray(gamma_1_s, dtype=float).reshape(-1)

    gamma_solver = gamma_1_s * float(time_unit_s)

    plt.figure()
    plt.plot(t_phys * 1e12, omega_L)
    plt.xlabel("t (ps)")
    plt.ylabel("omega_L (rad/s)")
    plt.title("Carrier omega_L(t)")

    plt.figure()
    plt.plot(t_phys * 1e12, detuning_rad_s)
    plt.xlabel("t (ps)")
    plt.ylabel("detuning Delta (rad/s)")
    plt.title("Detuning Delta(t) = mult*omega_L - omega_ref")

    plt.figure()
    plt.plot(t_phys * 1e12, gamma_1_s)
    plt.xlabel("t (ps)")
    plt.ylabel("gamma_eid (1/s)")
    plt.title("Drive-induced dephasing rate gamma_eid(t) in physical units")

    plt.figure()
    plt.plot(t_phys * 1e12, gamma_solver)
    plt.xlabel("t (ps)")
    plt.ylabel("gamma_eid_solver (dimensionless / solver rate)")
    plt.title(
        "Drive-induced dephasing gamma in solver units (gamma_1_s * time_unit_s)"
    )

    # --- initial state ---
    bundle = qd.compile_bundle(units=units)
    dims = bundle.modes.dims()
    rho0 = rho0_qd_vacuum(dims=dims, qd_state=QDState.G)

    # --- run with audit enabled ---
    engine = SimulationEngine(audit=True)
    specs = [DriveSpec(payload=drive, drive_id="chirped_linear_eid_on")]

    qutip_options = {
        "method": "bdf",
        "atol": 1e-10,
        "rtol": 1e-8,
        "nsteps": 200000,
        "max_step": 0.02,
        "progress_bar": "tqdm",
    }

    res = engine.run(
        qd,
        tlist=tlist,
        time_unit_s=time_unit_s,
        rho0=rho0,
        drives=specs,
        solve_options={"qutip_options": qutip_options},
    )

    safe_print_audit(res)

    # --- plotting your run summary (includes coupling panel, omega_L, etc.) ---
    fig = plot_run(
        res,
        units=units,
        drives=[drive],
        qd=qd,
        cfg=PlotConfig(
            title="Chirped drive with EID enabled",
            show_omega_L=True,
            show_coupling_panel=True,
            coupling_mode="abs",
        ),
    )

    plt.show()


if __name__ == "__main__":
    main()
