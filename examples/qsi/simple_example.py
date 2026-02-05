"""
Minimal QSI example: QD source (TPE, no phonons), robust against schema + operator unpacking
"""

import json
import numpy as np

from qsi.coordinator import Coordinator
from qsi.state import State, StateProp

from bec.light.classical.field_drive import ClassicalFieldDriveU
from bec.light.classical.carrier import Carrier
from bec.light.envelopes.gaussian import GaussianEnvelopeU
from bec.light.classical.amplitude import FieldAmplitude

from bec.qsi.utils import apply_prepare_from_scalar
import qutip as qt


def log_negativity_early_late(rho_16x16):
    rho = np.asarray(rho_16x16, dtype=complex)
    # dims: [GX_H, GX_V, XX_H, XX_V] each trunc=2
    dims = [2, 2, 2, 2]
    q = qt.Qobj(rho, dims=[dims, dims])

    # Partial transpose on one side of the bipartition.
    # Early=XX modes are indices 2,3 in this ordering.
    pt = qt.partial_transpose(q, [0, 0, 1, 1])

    evals = np.linalg.eigvalsh(pt.full())
    neg = float(np.sum(np.abs(evals[evals < 0.0])))

    # Log-negativity (base 2)
    # E_N = log2(||rho^T||_1) = log2(1 + 2 * negativity)
    return float(np.log2(1.0 + 2.0 * neg)), neg


def main() -> None:
    coord = Coordinator(port=2500)

    qd_source = coord.register_component(
        module="../../bec/qsi/qd_source.py",
        runtime="python",
    )

    coord.run()

    # ----- set params (must match param_query types) -----
    qd_source.set_param("exciton_eV", 1.300)
    qd_source.set_param("binding_eV", 0.003)
    qd_source.set_param("fss_eV", 0.0)
    qd_source.set_param("mu_default_Cm", 3.0e-29)

    qd_source.set_param("cavity_Q", 5.0e4)
    qd_source.set_param("cavity_Veff_um3", 0.5)
    qd_source.set_param("cavity_lambda_nm", 930.0)
    qd_source.set_param("cavity_n", 3.4)

    qd_source.set_param("phonon_kind", "none")

    qd_source.set_param("time_unit_s", 1e-9)
    qd_source.set_param("t_start_s", 0.0)
    qd_source.set_param("t_stop_s", 2.0e-9)
    qd_source.set_param("num_t", 2001)
    qd_source.set_param("fock_dim", 2)

    qd_source.set_param("audit_flag", 0)

    # solve options must be a STRING (JSON)
    solve_opts = {
        "qutip_options": {
            "method": "bdf",
            "atol": 1e-10,
            "rtol": 1e-8,
            "nsteps": 200000,
            "max_step": 0.01,
            "progress_bar": "tqdm",
            "store_final_state": True,
        }
    }
    qd_source.set_param("solve_options_json", json.dumps(solve_opts))

    qd_source.send_params()

    # ----- scalar input (prepare-from-scalar) -----
    scalar_input = State(
        StateProp(
            state_type="light",
            truncation=1,
            wavelength=1.0,
            polarization="H",
            bandwidth=1.0,
        )
    )

    # ----- build drive object, then serialize -----
    # crude TPE carrier guess for example purposes
    omega_xxg = 2.6 * 1.602176634e-19 / 1.054571817e-34
    omega_L = 0.5 * omega_xxg

    drive_obj = ClassicalFieldDriveU(
        envelope=GaussianEnvelopeU(t0=1.0e-9, sigma=80.0e-12),
        amplitude=FieldAmplitude(E0=1.0e8),
        carrier=Carrier(omega0=omega_L, delta_omega=0.0, phi0=0.0),
        preferred_kind="2ph",
        label="tpe_gaussian",
    )

    signal = {
        "drive_id": "tpe",
        "payload": drive_obj.to_dict(),
    }

    response, kraus_ops = qd_source.channel_query(
        scalar_input,
        {"input": scalar_input.state_props[0].uuid},
        signals=[signal],
    )

    # Always safe now: kraus_ops exists (maybe empty list).
    if float(response.get("error", 1.0)) != 0.0:
        raise RuntimeError(response.get("message", "channel_query failed"))

    props = [StateProp(**p) for p in response["stateprops"]]
    prepared = apply_prepare_from_scalar(
        kraus_ops,
        props,
        normalize=True,
    )
    E_N, neg = log_negativity_early_late(prepared.state)
    print(neg)
    print(E_N)

    coord.terminate()


if __name__ == "__main__":
    main()
