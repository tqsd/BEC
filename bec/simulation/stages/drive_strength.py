from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.constants import hbar as _hbar

from bec.light.classical import ClassicalCoherentDrive
from bec.params.transitions import Transition
from bec.quantum_dot.dot import QuantumDot
from bec.quantum_dot.me.coeffs import CoeffExpr, as_coeff
from bec.quantum_dot.models.phonon_model import PhononOutputs
from bec.simulation.types import DriveCoefficients, ResolvedDrive


def _ensure_e_pol(drive: ClassicalCoherentDrive) -> np.ndarray:
    E = drive.effective_polarization()
    if E is None:
        E = np.array([1.0 + 0j, 0.0 + 0j], dtype=complex)
    E = np.asarray(E, dtype=complex).reshape(2)
    n = np.linalg.norm(E)
    if n != 0:
        E = E / n
    return E


def _dipole_overlap_Cm(d_vec_hv: np.ndarray, E_pol_hv: np.ndarray) -> complex:
    return complex(
        np.vdot(
            np.asarray(d_vec_hv, dtype=complex).reshape(2),
            np.asarray(E_pol_hv, dtype=complex).reshape(2),
        )
    )


def _rd_components(rd: Any) -> Tuple[Tuple[Transition, complex], ...]:
    comps = tuple(getattr(rd, "components", ()) or ())
    if comps:
        return comps
    tr = getattr(rd, "transition", None)
    if tr is not None:
        return ((tr, 1.0 + 0j),)
    raise ValueError("ResolvedDrive has neither components nor transition.")


def _as_float(x: Any, *, name: str) -> float:
    try:
        return float(x)
    except Exception as e:
        raise TypeError(f"{name} must be float-like, got {type(x)!r}") from e


def _get_rd_kind(rd: ResolvedDrive) -> str:
    # You may or may not have rd.kind in your current ResolvedDrive.
    k = getattr(rd, "kind", None)
    if k is None:
        # Backwards compatible default: treat as 1-photon drive.
        return "1ph"
    return str(k)


def _scale_coeff(c: CoeffExpr, s: float) -> CoeffExpr:
    if s == 1.0:
        return c

    def scaled(t: float, args=None, *, _c=c, _s=float(s)) -> complex:
        return _s * _c(t, args)

    return as_coeff(scaled)


@dataclass(frozen=True)
class DefaultDriveStrengthModel:
    """
    Builds solver-ready coefficients.

    Extensions added for TPE:
    - If rd.kind == "2ph", compute DriveCoefficients.omega_2ph
      (and leave omega_by_transition empty).
    """

    # floor to avoid division by ~0 when computing effective 2ph coupling
    detuning_floor_phys: float = 1.0e6  # rad/s

    def build(
        self,
        *,
        qd: QuantumDot,
        rd: ResolvedDrive,
        time_unit_s: float,
        phonons: Optional[PhononOutputs] = None,
    ) -> DriveCoefficients:
        phonons = phonons or qd.phonon_model.compute()
        time_unit_s = _as_float(time_unit_s, name="time_unit_s")
        drive = rd.physical
        amp_kind = getattr(drive, "amplitude_kind", "rabi")

        rd_kind = _get_rd_kind(rd)
        if rd_kind == "2ph":
            return self._build_2ph(
                qd=qd,
                drive=drive,
                rd=rd,
                time_unit_s=time_unit_s,
                amp_kind=amp_kind,
                phonons=phonons,
            )

        # default: 1-photon mapping
        if amp_kind == "rabi":
            return self._build_rabi(
                drive=drive, rd=rd, time_unit_s=time_unit_s, phonons=phonons
            )
        if amp_kind == "field":
            return self._build_field(
                qd=qd,
                drive=drive,
                rd=rd,
                time_unit_s=time_unit_s,
                phonons=phonons,
            )
        raise ValueError(f"Unknown amplitude_kind={amp_kind!r}")

    def build_many(
        self,
        *,
        qd: QuantumDot,
        resolved: tuple[ResolvedDrive, ...],
        time_unit_s: float,
    ) -> Dict[str, DriveCoefficients]:
        phonons: PhononOutputs = qd.phonon_model.compute()
        return {
            rd.drive_id: self.build(
                qd=qd, rd=rd, phonons=phonons, time_unit_s=time_unit_s
            )
            for rd in resolved
        }

    def _build_rabi(
        self,
        *,
        drive: ClassicalCoherentDrive,
        rd: ResolvedDrive,
        time_unit_s: float,
        phonons: Optional[PhononOutputs] = None,
    ) -> DriveCoefficients:
        omega_by_tr: Dict[Transition, CoeffExpr] = {}

        for tr, w in _rd_components(rd):
            w_c = complex(w)

            def coeff(
                t: float, args=None, *, _w=w_c, _tu=time_unit_s
            ) -> complex:
                return _w * drive.omega_solver(float(t), time_unit_s=float(_tu))

            c = as_coeff(coeff)

            if phonons is not None:
                B = float(phonons.B_polaron_per_transition.get(tr, 1.0))
                c = _scale_coeff(c, B)
            omega_by_tr[tr] = c

        return DriveCoefficients(
            omega_by_transition=omega_by_tr, meta={"kind": "rabi"}
        )

    def _build_field(
        self,
        *,
        qd: QuantumDot,
        drive: ClassicalCoherentDrive,
        rd: ResolvedDrive,
        time_unit_s: float,
        phonons: Optional[PhononOutputs] = None,
    ) -> DriveCoefficients:
        E_pol = _ensure_e_pol(drive)

        omega_by_tr: Dict[Transition, CoeffExpr] = {}
        overlaps: Dict[str, complex] = {}

        for tr, w in _rd_components(rd):
            w_c = complex(w)

            d_vec = qd.dipole_params.d_vec_hv(tr)
            overlap = _dipole_overlap_Cm(d_vec, E_pol)
            overlaps[str(tr)] = overlap

            def coeff(
                t: float,
                args=None,
                *,
                _w=w_c,
                _ov=overlap,
                _tu=time_unit_s,
            ) -> complex:
                t_phys = float(_tu) * float(t)
                E_scalar = drive.field_phys(t_phys)  # V/m
                Omega_phys = (complex(E_scalar) * _ov) / _hbar  # rad/s
                return _w * Omega_phys * float(_tu)  # solver units

            c = as_coeff(coeff)

            if phonons is not None:
                B = float(phonons.B_polaron_per_transition.get(tr, 1.0))
                c = _scale_coeff(c, B)
            omega_by_tr[tr] = c

        return DriveCoefficients(
            omega_by_transition=omega_by_tr,
            meta={"kind": "field", "E_pol": E_pol, "overlaps_Cm": overlaps},
        )

    def _build_2ph(
        self,
        *,
        qd: QuantumDot,
        drive: ClassicalCoherentDrive,
        rd: ResolvedDrive,
        time_unit_s: float,
        amp_kind: str,
        phonons: Optional[PhononOutputs] = None,
    ) -> DriveCoefficients:
        if amp_kind == "rabi":

            def coeff(t: float, args=None, *, _tu=time_unit_s) -> complex:
                return drive.omega_solver(float(t), time_unit_s=float(_tu))

            c = as_coeff(coeff)

            if phonons is not None:
                B2 = float(
                    phonons.B_polaron_per_transition.get(Transition.G_XX, 1.0)
                )
                c = _scale_coeff(c, B2)

            return DriveCoefficients(
                omega_by_transition={},
                omega_2ph=c,
                meta={"kind": "2ph_rabi"},
            )

        if amp_kind != "field":
            raise ValueError(
                f"2ph drive requires amplitude_kind 'rabi' or 'field', got {
                    amp_kind!r}"
            )

        E_pol = _ensure_e_pol(drive)

        ov_g_x1 = _dipole_overlap_Cm(
            qd.dipole_params.d_vec_hv(Transition.G_X1), E_pol
        )
        ov_x1_xx = _dipole_overlap_Cm(
            qd.dipole_params.d_vec_hv(Transition.X1_XX), E_pol
        )
        ov_g_x2 = _dipole_overlap_Cm(
            qd.dipole_params.d_vec_hv(Transition.G_X2), E_pol
        )
        ov_x2_xx = _dipole_overlap_Cm(
            qd.dipole_params.d_vec_hv(Transition.X2_XX), E_pol
        )

        # reference transition frequencies (phys rad/s)
        el = qd.energy_levels
        omega_gx1 = float(el.X1) * 1.602176634e-19 / 1.054571817e-34
        omega_gx2 = float(el.X2) * 1.602176634e-19 / 1.054571817e-34

        floor = float(getattr(self, "detuning_floor_phys", 1.0e6))
        floor = float(rd.meta.get("detuning_floor_phys", floor))

        def _safe_det(d: float) -> float:
            if abs(d) >= floor:
                return d
            return floor if d >= 0.0 else -floor

        def _laser_omega_phys(t_phys: float) -> float:
            w0 = drive.laser_omega0
            if w0 is None:
                raise ValueError(
                    "drive.laser_omega0 is None (required for 2ph)."
                )
            w = float(w0)
            dw = drive.delta_omega
            if callable(dw):
                w += float(dw(float(t_phys)))
            else:
                w += float(dw)
            return w

        def coeff(
            t: float,
            args=None,
            *,
            _tu=time_unit_s,
            _ov11=ov_g_x1,
            _ov12=ov_x1_xx,
            _ov21=ov_g_x2,
            _ov22=ov_x2_xx,
            _wgx1=omega_gx1,
            _wgx2=omega_gx2,
        ) -> complex:
            t_phys = float(_tu) * float(t)

            # field envelope (phys)
            E = complex(drive.field_phys(t_phys))

            # chirped laser frequency
            wL = _laser_omega_phys(t_phys)

            # instantaneous one-photon detunings for the two virtual paths
            d1 = _safe_det(float(wL) - float(_wgx1))
            d2 = _safe_det(float(wL) - float(_wgx2))

            Omega2_phys = (E * E / (_hbar * _hbar)) * (
                (_ov11 * _ov12) / d1 + (_ov21 * _ov22) / d2
            )

            return Omega2_phys * float(_tu)

        c = as_coeff(coeff)

        if phonons is not None:
            B2 = float(
                phonons.B_polaron_per_transition.get(Transition.G_XX, 1.0)
            )
            c = _scale_coeff(c, B2)

        return DriveCoefficients(
            omega_by_transition={},
            omega_2ph=c,
            meta={
                "kind": "2ph_field",
                "E_pol": E_pol,
                "detuning_floor_phys": floor,
            },
        )
