"""
Example: QD source via QSI (Gaussian envelope, prepare-from-scalar)
-------------------------------------------------------------------
- Registers ../../bec/qsi/classical.py (your QSI wrapper module)
- Sets device & sim parameters
- Uses a single-dimensional input state (truncation=1)
- Sends a classical 2γ drive with a *Gaussian* envelope (JSON form)
- Prints the channel response and basic info about returned Kraus ops
"""

import numpy as np
from scipy.constants import e as _e, hbar as _hbar, pi, c as _c

from qsi.coordinator import Coordinator
from qsi.state import State, StateProp

from bec.qsi.utilities import (
    apply_prepare_from_scalar,
    stateprops_from_qd_diagnostics,
)

# 1) Start the coordinator + register your wrapper module
coordinator = Coordinator()

qd_source = coordinator.register_component(
    module="../../bec/qsi/classical.py",  # path to your QSI wrapper
    runtime="python",
)

# 2) Spin up the coordinator; this triggers param_query on modules
coordinator.run()

# 3) Configure device/physics + sim params
qd_source.set_param("exciton_eV", 1.35)
qd_source.set_param("biexciton_eV", 2.70)  # example value
qd_source.set_param("fss_eV", 0e-6)
qd_source.set_param("dipole_moment_Cm", 3.0e-29)

qd_source.set_param("cavity_Q", 5e4)
qd_source.set_param("cavity_V_eff_um3", 0.5)
qd_source.set_param("cavity_lambda_nm", 930.0)
qd_source.set_param("cavity_n", 3.4)

# Optional sim controls exposed by the wrapper
qd_source.set_param("trunc_per_pol", 2)
qd_source.set_param("time_unit_s", 1e-9)
qd_source.set_param("t_start_s", 0.0)
qd_source.set_param("t_stop_s", 4e-9)
qd_source.set_param("num_t", 5001)

# Optional plot controls
qd_source.set_param("plot_show", "True")
qd_source.set_param("plot_save", "test.png")

# Push all configured params to the device (param_set)
qd_source.send_params()

# 4) Single-dimensional input state (acts as a scalar domain)
scalar_input = State(
    StateProp(
        state_type="light",  # placeholder; wrapper doesn't use the content
        truncation=1,  # <- IMPORTANT: 1D input so channel prepares output
        wavelength=0,
        polarization="H",
        bandwidth=0,
    )
)

# 5) Build a Gaussian envelope in JSON form (as expected by envelope_from_json)
sigma = 5e-11  # pulse width (s)
t0 = 1e-9  # pulse center (s)
omega0 = 1e10  # Rabi amplitude (rad/s)

env_json = {
    "type": "gaussian",
    "t0": 1e-9,
    "sigma": sigma,
    "area": float(np.pi / omega0),
}

w_xxg = 2.70 * _e / _hbar
detuning = 0e10
wL = 0.5 * w_xxg + detuning

drive = {
    "type": "classical_2g",
    "omega0": 1.0e10,
    "detuning": 0.0,
    "label": "symbolic 2g",
    "laser_omega": wL,
    "envelope": {
        "type": "symbolic",
        # expression in variable `t` (seconds)
        "expr": "np.exp(-((t - t0)**2)/(2*sigma**2))",
        "params": {"t0": 1.0e-9, "sigma": 5.0e-11},
        # optional domain hints (if your impl samples internally)
        "t_start": 0.0,
        "t_stop": 4.0e-9,
        "num": 5001,
        # optional area normalization/override
        "normalize": False,
        "area": float(np.pi / omega0),
    },
}

# 6) Request a channel; map the port name "input" to our scalar state UUID
response, kraus_ops = qd_source.channel_query(
    scalar_input,
    {"input": scalar_input.state_props[0].uuid},
    signals=[drive],
)

print(response.get("diagnostics"))
# 1) Build factor layout from diagnostics (order matches the QD registry)
diag = response["diagnostics"]  # dict returned by the wrapper
props = stateprops_from_qd_diagnostics(
    diag,
    trunc_per_pol=2,
    default_bandwidth_Hz=1e9,  # fallback if none in diagnostics
    pol_map={"+": "H", "−": "V"},
)

prepared = apply_prepare_from_scalar(
    kraus_ops,
    props,
    normalize=True,
)

# 3) Inspect/container is ready
for p in prepared.state_props:
    print(p)
print("Prepared state dims:", [p.truncation for p in prepared.state_props])
print("Total Hilbert size:", prepared.dimensions)
print("Trace(ρ):", np.trace(prepared.state).real)


coordinator.terminate()
