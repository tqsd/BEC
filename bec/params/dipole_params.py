from dataclasses import dataclass


@dataclass
class DipoleParams:
    """
    Transition dipole moment parameters for a quantum dot.

    Parameters
    ----------
    dipole_moment_Cm : float
        Transition dipole moment in Coulomb–meters (C·m). This quantity
        characterizes the strength of light–matter coupling and enters
        directly into spontaneous emission rate calculations.

    Notes
    -----
    The spontaneous emission rate in free space scales with μ^2, where μ is the
    transition dipole moment. In SI units, typical values for quantum dots are
    on the order of 10^-29 to 10^-28 C·m.
    """

    dipole_moment_Cm: float  # in C·m
