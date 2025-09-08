"""
Quantum Dot, which can be excited by classical laser
"""

from dataclasses import dataclass

from qsi.qsi import QSI
from qsi.helpers import numpy_to_json, pretty_print_dict
from qsi.state import State, StateProp
import time
import numpy as np
import uuid

from bec.operators.qd_operators import QDState
from bec.quantum_dot.params import CavityParams, DipoleParams, EnergyLevels
from bec.quantum_dot.qd_photon_weave import QuantumDotSystem

from bec.qsi.helpers import build_drive_from_signal

qsi = QSI()
uid = str(uuid.uuid4())


@dataclass
class Params:
    exciton: float
    biexciton: float
    fss: float
    dipole_moment: float
    cavity_Q: float
    cavity_V_eff_um3: float
    cavity_lambda_nm: float
    cavity_n: float


@qsi.on_message("param_query")
def param_query(msg):
    """
    Single Photon Source
    declares no parameters
    """
    return {
        "msg_type": "param_query_response",
        "params": {
            "exciton_eV": "number",
            "biexciton_eV": "number",
            "fss_eV": "number",
            "dipole_moment_Cm": "number",
            "cavity_Q": "number",
            "cavity_V_eff_um3": "number",
            "cavity_lambda_nm": "number",
            "cavity_n": "number",
        },
    }


params: Params | None = None
QD: QuantumDotSystem | None = None


@qsi.on_message("param_set")
def param_set(msg):
    """
    Update (or create) the global Params from the incoming message.
    Accepts partial updates. Values are taken from msg["params"][<key>]["value"].
    """
    global params

    key_map = {
        "exciton_eV": "exciton",
        "biexciton_eV": "biexciton",
        "fss_eV": "fss",
        "dipole_moment_Cm": "dipole_moment",
        "cavity_Q": "cavity_Q",
        "cavity_V_eff_um3": "cavity_V_eff_um3",
        "cavity_lambda_nm": "cavity_lambda_nm",
        "cavity_n": "cavity_n",
    }

    incoming = msg.get("params", {}) or {}
    updates = {}
    errors = []

    for incoming_key, field_name in key_map.items():
        if incoming_key in incoming:
            try:
                val = float(incoming[incoming_key]["value"])
                updates[field_name] = val
            except Exception as e:
                errors.append(f"{incoming_key}: {e}")

    # Initialize if needed (use 0.0 defaults unless you prefer None/raise)
    if params is None:
        params = Params(
            exciton=0.0,
            biexciton=0.0,
            fss=0.0,
            dipole_moment=0.0,
            cavity_Q=0.0,
            cavity_V_eff_um3=0.0,
            cavity_lambda_nm=(
                0.0 if not hasattr(Params, "cavity_lambda_nm") else 0.0
            ),
            cavity_n=0.0,
        )

    field_name_lambda = (
        "cavity_lambda_nm"
        if hasattr(params, "cavity_lambda_nm")
        else "cavity_lamdba_nm"
    )
    if (
        "cavity_lambda_nm" in updates
        and field_name_lambda != "cavity_lambda_nm"
    ):
        updates[field_name_lambda] = updates.pop("cavity_lambda_nm")

    # Apply updates
    for k, v in updates.items():
        setattr(params, k, v)

    return {
        "msg_type": "param_set_response",
        "error": 1 if errors else 0,
        "message": "; ".join(errors) if errors else "ok",
    }


@qsi.on_message("state_init")
def state_init(msg):
    """
    This Quantum Dot model declares no internal state
    """
    msg = {"msg_type": "state_init_response", "states": [], "state_ids": []}
    return msg


@qsi.on_message("channel_query")
def channel_query(msg):
    try:
        drive = msg["signals"][0]
    except (KeyError, IndexError) as e:
        print("ERROR")
        return {
            "msg_type": "channel_query_response",
            "message": f"Missing 'signals[0]': {e}",
        }

    try:
        drive = build_drive_from_signal(drive)
        print(drive)

        EL = EnergyLevels(
            exciton=params.exciton, biexciton=params.biexciton, fss=params.fss
        )
        DP = DipoleParams(dipole_moment_Cm=params.dipole_moment)
        CP = CavityParams(
            Q=params.cavity_Q,
            Veff_um3=params.cavity_V_eff_um3,
            lambda_nm=params.cavity_lambda_nm,
            n=params.cavity_n,
        )
        QD = QuantumDotSystem(
            EL, initial_state=QDState.G, cavity_params=CP, dipole_params=DP
        )

    except Exception as e:
        return {
            "msg_type": "channel_query_response",
            "message": f"Invalid drive/envelope spec: {e}",
        }

    kraus_operators = [np.eye(2)]
    return {
        "msg_type": "channel_query_response",
        "kraus_operators": [numpy_to_json(x) for x in kraus_operators],
        "kraus_state_indices": [],
        "error": 0,
        "retrigger": False,
        "operation_time": 1e-10,
    }


@qsi.on_message("terminate")
def terminate(msg):
    qsi.terminate()


qsi.run()
time.sleep(1)


class WrongStateTypeException(Exception):
    pass
