"""
Quantum Dot QSI wrapper
-----------------------
Exposes a 4-level quantum dot (G, X1, X2, XX) over the QSI protocol.
- param_query / param_set configure device & environment.
- state_init declares no internal state.
- channel_query accepts a classical 2γ drive (Gaussian/Tabulated/Symbolic),
  runs a short simulation, and returns Kraus operators that *prepare* the
  resulting photonic state from a scalar input (|0> domain).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj, basis

# --- QSI infra ---------------------------------------------------------------
from qsi.qsi import QSI
from bec.plots.quick import plot_traces
from qsi.helpers import numpy_to_json

# not used for internal states here, but kept for parity

# --- BEC / QD stack ----------------------------------------------------------
from bec.light.envelopes import (
    SerializableEnvelope,
    GaussianEnvelope,
    TabulatedEnvelope,
    SymbolicEnvelope,
    envelope_from_json,
)
from bec.light.classical import ClassicalTwoPhotonDrive
from bec.operators.qd_operators import QDState
from bec.params.energy_levels import EnergyLevels
from bec.params.cavity_params import CavityParams
from bec.params.dipole_params import DipoleParams
from bec.quantum_dot.dot import QuantumDot

from bec.simulation.engine import SimulationEngine, SimulationConfig
from bec.simulation.scenarios import ClassicalDriveScenario
from bec.simulation.solvers import QutipMesolveBackend, MesolveOptions

# --- QSI server --------------------------------------------------------------
qsi = QSI()

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------


@dataclass
class Params:
    # Energies (eV)
    exciton_eV: float = 1.300
    biexciton_eV: float = 2.7  # exciton - binding; just a safe default
    fss_eV: float = 0.0
    delta_prime_eV: float = 0.0

    # Dipole (C·m)
    dipole_moment_Cm: float = 10.0 * 3.33564e-30  # ~10 Debye

    # Cavity
    cavity_Q: float = 5e4
    cavity_V_eff_um3: float = 0.5
    cavity_lambda_nm: float = 930.0
    cavity_n: float = 3.4

    # Simulation (can be overridden later if you like)
    trunc_per_pol: int = 2
    time_unit_s: float = 1e-10
    t_start_s: float = 0.0
    t_stop_s: float = 2.0e-8  # 20 ns
    num_t: int = 20001

    # plot
    plot_show: bool = False
    plot_save: str = ""


# Global mutable config
PARAMS = Params()

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _env_from_signal(signal: Dict[str, Any]) -> SerializableEnvelope:
    """
    Build an envelope from a 'signal' dict.
    Supports either explicit Envelope JSON (preferred), or ad-hoc Gaussian fields.
    Expected forms:
      signal = {
        "type": "classical_2g",
        "envelope": {"type": "gaussian"|"tabulated"|"symbolic", ...},
        ... (other drive fields)
      }
    Or (legacy quick form):
      signal = {"type": "classical_2g", "t0": ..., "sigma": ..., "area": ...}
    """
    if "envelope" in signal and isinstance(signal["envelope"], dict):
        return envelope_from_json(signal["envelope"])

    # Legacy Gaussian convenience
    if all(k in signal for k in ("t0", "sigma", "area")):
        return GaussianEnvelope(
            t0=float(signal["t0"]),
            sigma=float(signal["sigma"]),
            area=float(signal["area"]),
        )
    raise ValueError(
        "Signal is missing 'envelope' or Gaussian fields (t0, sigma, area)."
    )


def _drive_from_signal(signal: Dict[str, Any]) -> ClassicalTwoPhotonDrive:
    """
    Create a ClassicalTwoPhotonDrive from the signal specification.
    Required:
      - type: "classical_2g"
      - envelope: SerializableEnvelope JSON (gaussian/tabulated/symbolic)
    Optional:
      - omega0 (rad/s), detuning (rad/s), label, laser_omega (rad/s)
    """
    stype = str(signal.get("type", "")).lower()
    if stype not in ("classical_2g", "classical-two-photon", "2g"):
        raise ValueError(
            f"Unsupported signal type '{
                stype}' (expected 'classical_2g')."
        )

    env = _env_from_signal(signal)

    omega0 = float(signal.get("omega0", 1.0e10))
    detuning = float(signal.get("detuning", 0.0))
    label = str(signal.get("label", "2g"))
    laser_omega = float(signal.get("laser_omega", 0.0))  # optional cosmetic

    return ClassicalTwoPhotonDrive(
        envelope=env,
        omega0=omega0,
        detuning=detuning,
        label=label,
        laser_omega=laser_omega,
    )


def _build_qd(params: Params) -> QuantumDot:
    EL = EnergyLevels(
        exciton=params.exciton_eV,
        biexciton=params.biexciton_eV,
        fss=params.fss_eV,
        delta_prime=params.delta_prime_eV,
    )
    CP = CavityParams(
        Q=params.cavity_Q,
        Veff_um3=params.cavity_V_eff_um3,
        lambda_nm=params.cavity_lambda_nm,
        n=params.cavity_n,
    )
    DP = DipoleParams(dipole_moment_Cm=params.dipole_moment_Cm)

    qd = QuantumDot(
        EL,
        cavity_params=CP,
        dipole_params=DP,
        time_unit_s=params.time_unit_s,
        initial_state=QDState.G,
    )
    return qd


def _run_simulation(
    qd: QuantumDot, drive: ClassicalTwoPhotonDrive, params: Params
) -> Qobj:
    """Run a mesolve-based scenario; return the final *photonic* density matrix."""
    tlist = np.linspace(
        params.t_start_s / params.time_unit_s,
        params.t_stop_s / params.time_unit_s,
        params.num_t,
    )
    dt_s = (params.t_stop_s - params.t_start_s) / (params.num_t - 1)
    print(
        f"mesolve: {len(tlist)} samples, dt = {dt_s:.2e} s (~{
            dt_s/params.time_unit_s:.3g} solver units)",
        flush=True,
    )
    cfg = SimulationConfig(
        tlist=tlist,
        trunc_per_pol=params.trunc_per_pol,
        time_unit_s=params.time_unit_s,
    )

    backend = QutipMesolveBackend(
        MesolveOptions(
            nsteps=1_000_000,
            rtol=1e-6,
            atol=1e-8,
            progress_bar="text",
            store_final_state=True,
            max_step=1e-1,
        )
    )
    engine = SimulationEngine(solver=backend)
    scenario = ClassicalDriveScenario(drive=drive)

    # returns traces, rho_final, rho_phot_final
    traces, _, rho_phot = engine.run_with_state(qd, scenario, cfg)
    save_path = (
        params.plot_save.strip() if isinstance(params.plot_save, str) else ""
    )
    fig = plot_traces(
        traces,
        title=r"Biexciton classical 2$\gamma$ drive (Gaussian)",
    )
    if params.plot_save:
        print("SAVING THE PLOT")
        fig.savefig(save_path)
    if params.plot_show:
        plt.show()
    else:
        # if not showing, close to free memory
        plt.close(fig)
    return rho_phot


def _kraus_prepare_from_scalar(rho: Qobj) -> List[Qobj]:
    """
    Build Kraus ops that prepare 'rho' from a scalar input.
      K_i = sqrt(p_i) |psi_i><0|
    Each K has dims_out x 1. The coordinator should accept non-square Kraus.
    """
    dims_out = rho.dims[0]
    ket0 = basis(1, 0)

    Ks: List[Qobj] = []
    evals, vecs = rho.eigenstates()
    for p, psi in zip(evals, vecs):
        p = float(p)
        if p > 1e-14:
            K = np.sqrt(p) * (psi * ket0.dag())
            # shape bookkeeping: out × in
            K.dims = [[dims_out], [[1]]]
            Ks.append(K)
    return Ks


def _serialize_kraus(Ks: List[Qobj]) -> List[List[List[List[float]]]]:
    """
    Convert a list of Qobj Kraus operators to the nested JSON form expected by QSI.
    Uses qsi.helpers.numpy_to_json to map complex -> [re, im].
    """
    return [numpy_to_json(K.full()) for K in Ks]


# -----------------------------------------------------------------------------
# QSI message handlers
# -----------------------------------------------------------------------------


@qsi.on_message("param_query")
def _param_query(msg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "msg_type": "param_query_response",
        "sent_from": msg.get("sent_from", 0),
        "params": {
            # name: type
            "exciton_eV": "number",
            "biexciton_eV": "number",
            "fss_eV": "number",
            "delta_prime_eV": "number",
            "dipole_moment_Cm": "number",
            "cavity_Q": "number",
            "cavity_V_eff_um3": "number",
            "cavity_lambda_nm": "number",
            "cavity_n": "number",
            # optional sim controls (exposed for convenience)
            "trunc_per_pol": "integer",
            "time_unit_s": "number",
            "t_start_s": "number",
            "t_stop_s": "number",
            "num_t": "integer",
            "plot_show": "string",
            "plot_save": "string",
        },
    }


# --- in param_set ------------------------------------------------------------


@qsi.on_message("param_set")
def _param_set(msg: Dict[str, Any]) -> Dict[str, Any]:
    global PARAMS
    print("SETTING PARAMETERS")

    def _str_to_bool(s: str) -> bool:
        v = s.strip().lower()
        if v in ("1", "true", "t", "yes", "y", "on", "True"):
            return True
        if v in ("0", "false", "f", "no", "n", "off", "False"):
            return False
        raise ValueError(f"cannot parse boolean from '{s}'")

    def _coerce_like(template, value):
        # per-key overrides first
        return value  # default pass-through; we only special-case below

    incoming = msg.get("params", {}) or {}
    errors: List[str] = []

    for key, spec in incoming.items():
        try:
            if not hasattr(PARAMS, key):
                continue
            cur = getattr(PARAMS, key)
            value = spec.get("value", None)

            # ---- per-key coercion ----
            if key == "plot_show":
                if isinstance(value, str):
                    coerced = _str_to_bool(value)
                else:
                    coerced = bool(value)
            elif key == "plot_save":
                # Treat None / "", "none", "null" as "no file"
                if value is None:
                    coerced = ""
                else:
                    s = str(value).strip()
                    coerced = "" if s.lower() in ("", "none", "null") else s
            else:
                # generic coercion aligned with template type
                if isinstance(cur, bool):
                    coerced = bool(value)
                elif isinstance(cur, int):
                    coerced = int(value)
                elif isinstance(cur, float):
                    coerced = float(value)
                elif isinstance(cur, str):
                    coerced = str(value)
                else:
                    coerced = value
            # --------------------------

            setattr(PARAMS, key, coerced)

        except Exception as e:
            errors.append(f"{key}: {e}")
        try:
            _ = _build_qd(PARAMS)
        except Exception as e:
            errors.append(f"QD ERR: {e}")

    return {
        "msg_type": "param_set_response",
        "sent_from": msg.get("sent_from", 0),
        "error": 1 if errors else 0,
        "message": "; ".join(errors) if errors else "ok",
        "current": asdict(PARAMS),
    }


@qsi.on_message("state_init")
def _state_init(msg: Dict[str, Any]) -> Dict[str, Any]:
    # This model declares NO internal state for the coordinator to track.
    # (It acts like a source channel that prepares a photonic state.)
    return {
        "msg_type": "state_init_response",
        "sent_from": msg.get("sent_from", 0),
        "states": [],
        "state_ids": [],
    }


@qsi.on_message("channel_query")
def _channel_query(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expects:
      msg = {
        "msg_type": "channel_query",
        "signals": [ { ... drive spec ... } ],
        "time": <float>,            # ignored here
        "sent_from": <int>
      }
    Returns channel that *prepares* rho_phot_final from a scalar input.
    """
    try:
        signal = (msg.get("signals") or [None])[0]
        if not isinstance(signal, dict):
            raise ValueError(
                "signals[0] must be an object with the drive spec."
            )

        drive = _drive_from_signal(signal)

        qd = _build_qd(PARAMS)
        print(drive)

        rho_phot = _run_simulation(qd, drive, PARAMS)

        Ks = _kraus_prepare_from_scalar(rho_phot)
        kraus_json = _serialize_kraus(Ks)

        return {
            "msg_type": "channel_query_response",
            "sent_from": msg.get("sent_from", 0),
            "kraus_operators": kraus_json,
            "kraus_state_indices": [],  # scalar input → no bound states
            "error": 0.0,
            "diagnostics": qd.diagnostics.mode_layout_summary(
                rho_phot=rho_phot
            ),
            "retrigger": False,
            "operation_time": PARAMS.time_unit_s,  # nominal op time in seconds
        }

    except Exception as e:
        return {
            "msg_type": "channel_query_response",
            "sent_from": msg.get("sent_from", 0),
            "message": f"{type(e).__name__}: {e}",
        }


@qsi.on_message("terminate")
def _terminate(msg: Dict[str, Any]) -> Dict[str, Any] | None:
    qsi.terminate()
    return None


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    qsi.run()
    # keep process alive very briefly to ensure startup message plumbing
    time.sleep(1)
