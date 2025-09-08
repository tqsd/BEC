from qutip import tensor, basis, Qobj
from typing import Sequence, List


def reduce_by_spectators_vacuum(
    kraus_full: Sequence[Qobj],
    dims: List[int],
    spect_indices: Sequence[int],
    spec_dims: Sequence[int],
) -> List[Qobj]:
    """
    Freeze only 'spectator' subsystems to |vac> and remove them:
    - spect_indices: positions of spectator subsystems in your dims list
    - spec_dims:     their Hilbert dims (match dims entries)
    Returns Kraus operators acting on the remaining subsystems.
    """
    # Build |vac> for spectators; here we assume '0' is the vacuum Fock basis index
    kets = [basis(d, 0) for d in spec_dims]
    ket_spec = tensor(kets)
    bra_spec = ket_spec.dag()

    reduced = []
    for K in kraus_full:
        # K acts on full H; project spectators
        M = bra_spec * K * ket_spec
        reduced.append(M)
    return reduced
