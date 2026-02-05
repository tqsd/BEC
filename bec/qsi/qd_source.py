from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from qsi.qsi import QSI
from qsi.helpers import numpy_to_json

from smef.core.drives.types import DriveSpec
from smef.core.units import Q, UnitSystem
from smef.engine import SimulationEngine

from bec.light.classical.field_drive import ClassicalFieldDriveU
from bec.qsi.utils import stateprops_photons_from_qd
from bec.quantum_dot.dot import QuantumDot
from bec.quantum_dot.enums import QDState
from bec.quantum_dot.smef.initial_state import rho0_qd_vacuum
from bec.quantum_dot.smef.modes import QDModes
from bec.quantum_dot.spec.cavity_params import CavityParams
from bec.quantum_dot.spec.dipole_params import DipoleParams
from bec.quantum_dot.spec.energy_structure import EnergyStructure
from bec.quantum_dot.spec.exciton_mixing_params import ExcitonMixingParams
from bec.quantum_dot.spec.phonon_params import (
    PhenomenologicalPhononParams,
    PhononCouplings,
    PhononModelKind,
    PhononParams,
    PolaronLAParams,
    SpectralDensityKind,
)

qsi = QSI()


_DEFAULT_SOLVE_OPTIONS: Dict[str, Any] = {
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


def _as_int_flag(x: Any) -> int:
    try:
        v = int(x)
    except Exception:
        v = 0
    return 1 if v != 0 else 0


def _as_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _as_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _parse_json_object(text: Any) -> Optional[Dict[str, Any]]:
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise ValueError("json must decode to an object")
    return obj


def _merge_solve_options(
    base: Mapping[str, Any], override: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base)
    if not override:
        return out

    for k, v in override.items():
        if k == "qutip_options" and isinstance(v, Mapping):
            q0 = dict(out.get("qutip_options", {}))
            for kk, vv in v.items():
                q0[kk] = vv
            out["qutip_options"] = q0
        else:
            out[k] = v
    return out


@dataclass
class SimParams:
    time_unit_s: float = 1e-9
    t_start_s: float = 0.0
    t_stop_s: float = 1.0e-9
    num_t: int = 2001
    fock_dim: int = 2

    # QSI schema only supports integer/number/string/complex in param registry.
    # So expose flags as integers 0/1 and JSON objects as strings.
    audit_flag: int = 0
    solve_options_json: Optional[str] = None


@dataclass
class QDParams:
    exciton_eV: float = 1.300
    binding_eV: float = 3.0e-3
    fss_eV: float = 5.0e-6

    mu_default_Cm: float = 3.33564e-29

    cavity_Q: float = 5.0e4
    cavity_Veff_um3: float = 0.5
    cavity_lambda_nm: float = 930.0
    cavity_n: float = 3.4

    delta_prime_eV: float = 0.0

    # phonons: kind is string; enable flags are integers (0/1)
    phonon_kind: str = "none"
    temperature_K: float = 4.0

    phi_g: float = 0.0
    phi_x1: float = 1.0
    phi_x2: float = 1.0
    phi_xx: float = 2.0

    enable_polaron_renorm_flag: int = 1
    enable_exciton_relaxation_flag: int = 0
    enable_eid_flag: int = 0
    enable_polaron_scattering_flag: int = 1

    spectral_density: str = "superohmic_gaussian"
    alpha_s2: float = 0.0
    omega_c_rad_s: float = 1.0e12

    gamma_phi_x1_1_s: float = 0.0
    gamma_phi_x2_1_s: float = 0.0
    gamma_phi_xx_1_s: float = 0.0
    gamma_relax_x1_x2_1_s: float = 0.0
    gamma_relax_x2_x1_1_s: float = 0.0
    gamma_phi_eid_scale: float = 0.0


@dataclass
class Params:
    qd: QDParams = field(default_factory=QDParams)
    sim: SimParams = field(default_factory=SimParams)


PARAMS = Params()


def _build_qd(p: Params) -> QuantumDot:
    q = p.qd

    energy = EnergyStructure.from_params(
        exciton=Q(float(q.exciton_eV), "eV"),
        fss=Q(float(q.fss_eV), "eV"),
        binding=Q(float(q.binding_eV), "eV"),
    )

    dipoles = DipoleParams.biexciton_cascade_from_fss(
        mu_default_Cm=Q(float(q.mu_default_Cm), "C*m"),
        fss=Q(float(q.fss_eV), "eV"),
        delta_prime=Q(float(q.delta_prime_eV), "eV"),
    )

    cavity = CavityParams.from_values(
        Q=float(q.cavity_Q),
        Veff_um3=float(q.cavity_Veff_um3),
        lambda_nm=float(q.cavity_lambda_nm),
        n=float(q.cavity_n),
    )

    mixing = ExcitonMixingParams.from_values(
        delta_prime_eV=float(q.delta_prime_eV)
    )

    phonons: Optional[PhononParams] = None
    kind = str(q.phonon_kind).lower().strip()
    if kind and kind != "none":
        if kind not in ("polaron_la", "polaron-la"):
            raise ValueError("Unsupported phonon_kind: %r" % kind)

        couplings = PhononCouplings(
            phi_g=float(q.phi_g),
            phi_x1=float(q.phi_x1),
            phi_x2=float(q.phi_x2),
            phi_xx=float(q.phi_xx),
        )

        sd = str(q.spectral_density).lower().strip()
        if sd in ("superohmic_gaussian", "super-ohmic-gaussian"):
            sd_kind = SpectralDensityKind.SUPER_OHMIC_GAUSSIAN
        else:
            raise ValueError("Unsupported spectral_density: %r" % sd)

        polaron = PolaronLAParams(
            enable_polaron_renorm=bool(
                _as_int_flag(q.enable_polaron_renorm_flag)
            ),
            enable_exciton_relaxation=bool(
                _as_int_flag(q.enable_exciton_relaxation_flag)
            ),
            enable_eid=bool(_as_int_flag(q.enable_eid_flag)),
            enable_polaron_scattering=bool(
                _as_int_flag(q.enable_polaron_scattering_flag)
            ),
            spectral_density=sd_kind,
            alpha=Q(float(q.alpha_s2), "s**2"),
            omega_c=Q(float(q.omega_c_rad_s), "rad/s"),
        )

        phenomenological = PhenomenologicalPhononParams(
            gamma_phi_x1=Q(float(q.gamma_phi_x1_1_s), "1/s"),
            gamma_phi_x2=Q(float(q.gamma_phi_x2_1_s), "1/s"),
            gamma_phi_xx=Q(float(q.gamma_phi_xx_1_s), "1/s"),
            gamma_relax_x1_x2=Q(float(q.gamma_relax_x1_x2_1_s), "1/s"),
            gamma_relax_x2_x1=Q(float(q.gamma_relax_x2_x1_1_s), "1/s"),
            gamma_phi_eid_scale=float(q.gamma_phi_eid_scale),
        )

        phonons = PhononParams(
            kind=PhononModelKind.POLARON_LA,
            temperature=Q(float(q.temperature_K), "K"),
            couplings=couplings,
            polaron_la=polaron,
            phenomenological=phenomenological,
        )

    return QuantumDot(
        energy=energy,
        dipoles=dipoles,
        cavity=cavity,
        phonons=phonons,
        mixing=mixing,
    )


def _make_rho0(qd: QuantumDot, *, time_unit_s: float) -> np.ndarray:
    units = UnitSystem(time_unit_s=float(time_unit_s))
    bundle = qd.compile_bundle(units=units)
    dims = bundle.modes.dims()
    return rho0_qd_vacuum(dims=dims, qd_state=QDState.G)


def _drive_from_signal(
    signal: Mapping[str, Any],
) -> Tuple[str, ClassicalFieldDriveU]:
    if not isinstance(signal, Mapping):
        raise ValueError("signal must be an object")

    drive_id = str(signal.get("drive_id", "qsi_drive"))

    payload: Any = signal.get("payload", None)
    if payload is None:
        payload_json = signal.get("payload_json", None)
        if payload_json is None:
            raise ValueError(
                "signal must include payload (object) or payload_json (string)"
            )
        payload = _parse_json_object(payload_json)

    if not isinstance(payload, Mapping):
        raise ValueError("payload must be an object")

    drv = ClassicalFieldDriveU.from_dict(dict(payload))

    if drv.carrier is None:
        raise ValueError(
            "Drive carrier is required (carrier must not be None)")
    return drive_id, drv


def _rho_phot_from_total(rho_total: np.ndarray, *, dims: Sequence[int]) -> Any:
    import qutip as qt

    dims_list = [int(d) for d in dims]
    rho_q = qt.Qobj(
        np.asarray(rho_total, dtype=complex), dims=[dims_list, dims_list]
    )
    return rho_q.ptrace([1, 2, 3, 4])


def _kraus_prepare_from_scalar(rho_phot: Any) -> List[Any]:
    import qutip as qt

    evals, evecs = rho_phot.eigenstates()
    ket0 = qt.basis(1, 0)

    Ks: List[Any] = []
    for p, psi in zip(evals, evecs):
        pp = float(np.real(p))
        if pp > 1e-14:
            Ks.append(np.sqrt(pp) * (psi * ket0.dag()))
    return Ks


def _serialize_kraus(Ks: Sequence[Any]) -> List[Any]:
    return [numpy_to_json(K.full()) for K in Ks]


def _flat_current_params() -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    d.update(asdict(PARAMS.qd))
    d.update(asdict(PARAMS.sim))
    return d


def _set_param(key: str, value: Any) -> None:
    q = PARAMS.qd
    s = PARAMS.sim

    if hasattr(q, key):
        cur = getattr(q, key)
        if isinstance(cur, int):
            setattr(q, key, _as_int(value, cur))
        elif isinstance(cur, float):
            setattr(q, key, _as_float(value, cur))
        else:
            setattr(q, key, str(value))
        return

    if hasattr(s, key):
        if key == "solve_options_json":
            if value is None:
                s.solve_options_json = None
            else:
                obj = _parse_json_object(value)
                if obj is None:
                    s.solve_options_json = None
                else:
                    s.solve_options_json = json.dumps(obj)
            return

        cur = getattr(s, key)
        if isinstance(cur, int):
            setattr(s, key, _as_int(value, cur))
        else:
            setattr(s, key, _as_float(value, cur))
        return

    raise KeyError("Unknown param: %r" % key)


@qsi.on_message("param_query")
def _param_query(msg: Dict[str, Any]) -> Dict[str, Any]:
    # Respect QSI schema: only integer/number/string/complex allowed as param types.
    params: Dict[str, str] = {}

    for k, v in asdict(PARAMS.qd).items():
        if isinstance(v, int):
            params[k] = "integer"
        elif isinstance(v, float):
            params[k] = "number"
        else:
            params[k] = "string"

    for k, v in asdict(PARAMS.sim).items():
        if k == "solve_options_json":
            params[k] = "string"
        elif isinstance(v, int):
            params[k] = "integer"
        else:
            params[k] = "number"

    return {
        "msg_type": "param_query_response",
        "sent_from": msg.get("sent_from", 0),
        "params": params,
    }


@qsi.on_message("param_set")
def _param_set(msg: Dict[str, Any]) -> Dict[str, Any]:
    incoming = msg.get("params", {}) or {}
    errors: List[str] = []

    for key, spec in incoming.items() if isinstance(incoming, Mapping) else []:
        try:
            if not isinstance(spec, Mapping):
                continue
            if "value" not in spec:
                continue
            _set_param(str(key), spec["value"])
        except Exception as e:
            errors.append("%s: %s" % (str(key), str(e)))

    # validate QD build
    try:
        _ = _build_qd(PARAMS)
    except Exception as e:
        errors.append("QD ERR: %s" % str(e))

    return {
        "msg_type": "param_set_response",
        "sent_from": msg.get("sent_from", 0),
        "error": 1 if errors else 0,
        "message": "; ".join(errors) if errors else "ok",
        "current": _flat_current_params(),
    }


@qsi.on_message("state_init")
def _state_init(msg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "msg_type": "state_init_response",
        "sent_from": msg.get("sent_from", 0),
        "states": [],
        "state_ids": [],
    }


@qsi.on_message("channel_query")
def _channel_query(msg: Dict[str, Any]) -> Dict[str, Any]:
    try:
        signals = msg.get("signals", None)
        if not isinstance(signals, Sequence) or not signals:
            raise ValueError("signals must be a non-empty list")

        signal0 = signals[0]
        if not isinstance(signal0, Mapping):
            raise ValueError("signals[0] must be an object")

        qd = _build_qd(PARAMS)

        # Per-call overrides (do not mutate PARAMS)
        time_unit_s = _as_float(
            msg.get("time_unit_s", PARAMS.sim.time_unit_s),
            PARAMS.sim.time_unit_s,
        )
        if time_unit_s <= 0.0:
            raise ValueError("time_unit_s must be > 0")

        t_start_s = _as_float(
            msg.get("t_start_s", PARAMS.sim.t_start_s), PARAMS.sim.t_start_s
        )
        t_stop_s = _as_float(
            msg.get("t_stop_s", PARAMS.sim.t_stop_s), PARAMS.sim.t_stop_s
        )
        num_t = _as_int(msg.get("num_t", PARAMS.sim.num_t), PARAMS.sim.num_t)
        if num_t < 2:
            raise ValueError("num_t must be >= 2")
        if t_stop_s <= t_start_s:
            raise ValueError("t_stop_s must be > t_start_s")

        fock_dim = _as_int(
            msg.get("fock_dim", PARAMS.sim.fock_dim), PARAMS.sim.fock_dim
        )
        if fock_dim < 1:
            raise ValueError("fock_dim must be >= 1")

        audit_flag = _as_int_flag(msg.get("audit_flag", PARAMS.sim.audit_flag))

        # Solve options: DEFAULT <- PARAMS.sim.solve_options_json <- msg.solve_options_json
        solve_param = (
            _parse_json_object(PARAMS.sim.solve_options_json)
            if PARAMS.sim.solve_options_json
            else None
        )
        solve_msg = (
            _parse_json_object(msg.get("solve_options_json", None))
            if msg.get("solve_options_json", None) is not None
            else None
        )

        solve_opts = _merge_solve_options(
            _DEFAULT_SOLVE_OPTIONS,
            _merge_solve_options(solve_param or {}, solve_msg or {}),
        )

        # Time grid in solver units
        tlist_phys = np.linspace(float(t_start_s), float(t_stop_s), int(num_t))
        tlist_solver = tlist_phys / float(time_unit_s)

        rho0 = _make_rho0(qd, time_unit_s=float(time_unit_s))

        drive_id, drive = _drive_from_signal(signal0)
        specs = [DriveSpec(payload=drive, drive_id=drive_id)]

        engine = SimulationEngine(audit=bool(audit_flag))
        res = engine.run(
            qd,
            tlist=tlist_solver,
            time_unit_s=float(time_unit_s),
            rho0=rho0,
            drives=specs,
            solve_options=solve_opts,
        )

        if res.states is None:
            raise ValueError("Solver did not return final state in res.states")

        # Reduce to photonic state using QDModes ordering
        dims = QDModes(fock_dim=int(fock_dim)).dims()
        rho_phot = _rho_phot_from_total(res.states, dims=dims)

        Ks = _kraus_prepare_from_scalar(rho_phot)
        kraus_json = _serialize_kraus(Ks)

        # Provide stateprops (GX_H, GX_V, XX_H, XX_V)
        props = stateprops_photons_from_qd(
            qd,
            trunc_per_pol=int(fock_dim),
            default_bandwidth_Hz=1.0,
        )

        return {
            "msg_type": "channel_query_response",
            "sent_from": msg.get("sent_from", 0),
            "kraus_operators": kraus_json,
            "kraus_state_indices": [],
            "error": 0.0,
            "message": "ok",
            "stateprops": [p.__dict__ for p in props],
            "retrigger": False,
            "operation_time": float(t_stop_s - t_start_s),
            "time_unit_s": float(time_unit_s),
            "solve_options_effective": solve_opts,
        }

    except Exception as e:
        return {
            "msg_type": "channel_query_response",
            "sent_from": msg.get("sent_from", 0),
            "kraus_operators": [],
            "kraus_state_indices": [],
            "error": 1.0,
            "message": "%s: %s" % (type(e).__name__, str(e)),
        }


@qsi.on_message("terminate")
def _terminate(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    qsi.terminate()
    return None


if __name__ == "__main__":
    qsi.run()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
