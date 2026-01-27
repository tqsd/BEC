import numpy as np

from bec.quantum_dot.dot import QuantumDot
from bec.quantum_dot.enums import Transition
from bec.quantum_dot.parameters.energy_structure import EnergyStructure
from bec.quantum_dot.parameters.exciton_mixing import ExcitonMixingParams
from bec.quantum_dot.parameters.cavity import CavityParams
from bec.quantum_dot.parameters.dipole import DipoleParams
from bec.quantum_dot.parameters.phonons import PhononParams, PhononModelType

# --- Energies ---
E = EnergyStructure.from_params(
    exciton=1.33,  # eV
    fss=60e-6,  # 60 µeV -> visible splitting
    binding=3e-3,  # 3 meV
)

# --- Exciton mixing (delta_prime) ---
mix = ExcitonMixingParams.from_values(
    delta_prime_eV=40e-6  # comparable to fss -> substantial rotation
)

# --- Cavity ---
cav = CavityParams.from_values(
    Q=25_000,
    Veff_um3=0.6,
    lambda_nm=932.0,
    n=3.5,
)

# --- Dipoles: give different polarization characters per transition ---
# H = (1,0), V = (0,1), + = (1,1)/sqrt2, - = (1,-1)/sqrt2
rt2 = np.sqrt(2)

dip = DipoleParams.biexciton_cascade_defaults()

# --- Phonons: make B(T) clearly < 1 (choose alpha > 0) ---
ph = PhononParams(
    model=PhononModelType.POLARON,
    temperature=40.0,
    enable_polaron_renorm=True,
    alpha=0.03e-24,  # tune to your convention; just nonzero
    omega_c=1.2e12,
)

qd = QuantumDot(
    energy_structure=E,
    exciton_mixing=mix,
    cavity_params=cav,
    dipole_params=dip,
    phonon_params=ph,
)

# print(qd.derived.report_plain())
qd.derived.report()
cat = qd.observables_catalog
print("==== Observables (IR) ====")
for i, t in enumerate(cat.all_terms):
    # term basics
    print(f"[{i:02d}] {t.kind.value}  label={t.label}")
    if t.pretty:
        print(f"     pretty: {t.pretty}")

    # op summary
    op = t.op
    print(f"     op.kind: {op.kind.value}")

    prim = op.primitive
    if prim is not None:
        qdref = prim.qd.key if prim.qd.key is not None else "<mat>"
        if prim.fock is None:
            print(f"     primitive: qd={qdref}, fock=None")
        else:
            f = prim.fock
            print(
                f"     primitive: qd={
                  qdref}, fock=({f.kind.value}, key={f.key})"
            )

    if t.meta:
        # keep it short
        keys = ("type", "state", "channel_index")
        meta_short = {k: t.meta[k] for k in keys if k in t.meta}
        if meta_short:
            print(f"     meta: {meta_short}")

    print("")
