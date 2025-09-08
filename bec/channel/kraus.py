import numpy as np
from typing import List, Sequence, Optional
from qutip import Qobj, Options, propagator, to_choi, tensor, basis

from bec.plots.plotter import QDPlotter


def kraus_from_superop(
    S: Qobj, dims_op: List[int], tol: float = 1e-12
) -> List[Qobj]:
    """
    Turn a Liouville superoperator S into a Kraus set {K_a}.
    - dims_op: the list of subsystem dims for the operator space, e.g. [4, d+, d-, ...]
    Notes:
      * We assume column-stacking vectorization, so we reshape eigenvectors with order='F'.
      * Tiny negative eigenvalues are clipped to 0 for numerical stability.
    """
    J = to_choi(S).full()
    w, V = np.linalg.eigh(J)

    d = int(np.prod(dims_op))
    kraus = []
    for lam, vec in zip(w, V.T):
        if lam > tol:
            Kmat = np.sqrt(max(lam, 0.0)) * vec.reshape((d, d), order="F")
            kraus.append(Qobj(Kmat, dims=[dims_op, dims_op]))
    return kraus


def check_completeness(kraus: Sequence[Qobj], dims_op: List[int]) -> float:
    """
    Return || Σ K†K − I ||_F as a sanity check (should be ~1e−10 … 1e−6).
    """
    Id = Qobj(np.eye(int(np.prod(dims_op))), dims=[dims_op, dims_op])
    S = sum(K.dag() * K for K in kraus)
    return float((S - Id).norm("fro"))


# ---------- Mixin that adds Kraus features on top of your builder ----------
class KrausMixin:
    def superop(
        self,
        *,
        t_index: Optional[int] = None,
        solver_opts: Optional[Options] = None
    ) -> Qobj:
        """
        Return Liouville-space propagator S(t,0).
        - If t_index is None: returns S at final time.
        - Uses self.H, self.C_OPS, self.tlist, and self.DIMENSIONS.
        """
        if self.H is None or self.C_OPS is None or self.rho0 is None:
            # ensure QuTiP objects exist
            self._build_qutip_space()
            self._build_hamiltonian_and_collapse()

        print("Solving propagator")
        U_or_S = propagator(
            self.H,
            self.tlist,
            c_ops=self.C_OPS,
            options=solver_opts or Options(nsteps=5000),
        )
        if isinstance(U_or_S, (list, tuple)):
            idx = -1 if t_index is None else int(t_index)
            U_or_S = U_or_S[idx]
        # Convert unitary to super if no dissipation
        return U_or_S if U_or_S.type == "super" else to_super(U_or_S)

    def kraus(
        self, *, t_index: Optional[int] = None, solver_opts=None
    ) -> List[Qobj]:
        """Kraus on the full space (QD + all modes you included)."""
        S = self.superop(t_index=t_index, solver_opts=solver_opts)
        Ks = kraus_from_superop(S, self.DIMENSIONS)
        # optional: print sanity
        # print("Completeness error:", _check_completeness(Ks, self.DIMENSIONS))
        return Ks

    def kraus_keep_qd_reduce_spectators(
        self, *, spectator_dims: Sequence[int] = ()
    ) -> List[Qobj]:
        """
        Keep QD dynamical; freeze true spectators to |vac>.
        Provide a list of their local dimensions (in the same order you’d tensor them).
        """
        Ks = self.kraus()
        if not spectator_dims:
            return Ks
        ket_spec = tensor([basis(d, 0) for d in spectator_dims])
        bra_spec = ket_spec.dag()
        return [bra_spec * K * ket_spec for K in Ks]

    def kraus_optical_only_from_G(
        self, *, spectator_dims: Sequence[int] = ()
    ) -> List[Qobj]:
        """
        Optical-only channel (QD projected to |G⟩) and spectators to |vac⟩ at the END.
        Use this for “one-shot source” models.
        """
        Ks = self.kraus()
        # build projector |G,vac_spect><G,vac_spect|
        # QD is the first subsystem (dim 4). Ground = basis(4, 0).
        ket_blocks = [basis(4, 0)]  # |G>
        for d in spectator_dims:
            ket_blocks.append(basis(d, 0))
        ket = tensor(ket_blocks)
        bra = ket.dag()
        return [bra * K * ket for K in Ks]

    def kraus_completeness_error(self, Ks: Sequence[Qobj]) -> float:
        return check_completeness(Ks, self.DIMENSIONS)


# ---------- A concrete class you can use like QDPlotter ----------
class QDKraus(QDPlotter, KrausMixin):
    """
    Same interface as QDPlotter (builds the space/ops),
    plus .kraus(), .kraus(t_index=...), and reduction helpers.
    """

    pass
