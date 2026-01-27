from __future__ import annotations

import numpy as np

from bec.light.classical.profiles import linear_chirp
from bec.quantum_dot.compile import compile_qd
from bec.simulation.compiler import MECompiler
from bec.simulation.drive_decode.decoder import DecodePolicy
from bec.units import Q

# --- QD ---
from bec.quantum_dot.dot import QuantumDot
from bec.quantum_dot.parameters.energy_structure import EnergyStructure
from bec.quantum_dot.parameters.phonons import PhononParams
from bec.quantum_dot.parameters.dipole import DipoleParams
from bec.quantum_dot.enums import Transition

# --- Light ---
from bec.light.envelopes.gaussian import GaussianEnvelope  # adjust if different
from bec.light.classical.amplitude import FieldAmplitude
from bec.light.classical.carrier import Carrier
from bec.light.core.polarization import JonesState
from bec.light.classical import ClassicalFieldDrive

# --- Simulation drive-decode ---
from bec.simulation.drive_decode import DriveDecodeContext, DefaultDriveDecoder
from bec.simulation.drive_decode.protocols import TransitionKey
from bec.quantum_dot.adapters.transition_registry_adapter import (
    QDTransitionRegistryAdapter,
)
from bec.quantum_dot.adapters.polarization_adapter import QDPolarizationAdapter

# ----------------------------
# QD-side adapters (live in simulation package)
# ----------------------------


class QDPhononRenormAdapter:
    def __init__(self, qd: QuantumDot):
        self._d = qd.derived

    def polaron_B(self) -> float:
        return float(self._d.polaron_B)


# ----------------------------
# Demo
# ----------------------------


def main():
    # 1) Build a QD
    es = EnergyStructure(
        X1=Q(1.300, "eV"),
        X2=Q(1.300, "eV"),
        XX=Q(2.600, "eV"),
    )

    qd = QuantumDot(
        energy_structure=es,
        phonon_params=PhononParams(),  # optional
        dipole_params=DipoleParams.biexciton_cascade_defaults(
            mu_default_Cm=1e-29
        ),
    )
    qd.derived.report()

    # 2) Make decode context (simulation-side)
    ctx = DriveDecodeContext(
        transitions=QDTransitionRegistryAdapter(qd),
        pol=QDPolarizationAdapter(qd),
        phonons=QDPhononRenormAdapter(qd),  # optional; harmless
        bandwidth=None,  # keep fallback for now
    )

    # 3) Solver time grid
    tlist = np.linspace(0.0, 10.0, 2001)  # solver units
    time_unit_s = 1e-12  # 1 solver unit = 1 ps

    # 4) Build two drives: 1ph resonant to G->X1, 2ph resonant to G->XX via half-frequency
    w_gx1 = qd.derived.omega_ref_rad_s(Transition.G_X1)  # rad/s
    w_gxx = qd.derived.omega_ref_rad_s(Transition.G_XX)  # rad/s

    env = GaussianEnvelope(
        t0=Q(5.0e-12, "s"),  # center at 5 ps (physical)
        sigma=Q(1.0e-12, "s"),  # 1 ps width
    )

    amp = FieldAmplitude(E0=Q(1e5, "V/m"))

    drv_1ph = ClassicalFieldDrive(
        envelope=env,
        amplitude=amp,
        carrier=Carrier(omega0=Q(w_gx1, "rad/s")),
        pol_state=JonesState.H(),  # couple mostly to X1 in your default dipoles
        label="drive_1ph_GX1",
    )

    drv_2ph = ClassicalFieldDrive(
        envelope=env,
        amplitude=amp,
        carrier=Carrier(omega0=Q(0.5 * w_gxx, "rad/s")),
        pol_state=JonesState.H(),  # pol doesn’t matter for 2ph in decoder
        label="drive_2ph_GXX",
        preferred_kind="2ph",
    )

    chirp_rate = Q(1e23, "rad/s^2")

    chirp = linear_chirp(
        rate=chirp_rate, t0=Q(5e-12, "s")
    )  # center around pulse center

    drv_1ph_chirped = ClassicalFieldDrive(
        envelope=env,
        amplitude=amp,
        carrier=Carrier(
            omega0=Q(w_gx1, "rad/s"),
            delta_omega=chirp,  # <-- callable!
            label="GX1_linear_chirp",
        ),
        pol_state=JonesState.H(),
        label="drive_1ph_GX1_chirped",
    )

    E = drv_1ph.effective_pol()
    dec = DefaultDriveDecoder(
        policy=DecodePolicy(sample_points=21, k_bandwidth=3.0)
    )
    resolved = dec.decode(
        ctx=ctx,
        drives=[drv_1ph_chirped, drv_2ph],
        tlist=tlist,
        time_unit_s=time_unit_s,
    )
    print(resolved)

    me = compile_qd(
        qd=qd,
        drives=[drv_1ph_chirped],
        tlist=tlist,
        time_unit_s=1e-12,
        report=True,
    )


if __name__ == "__main__":
    main()
