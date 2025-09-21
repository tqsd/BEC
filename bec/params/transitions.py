from enum import Enum


class Transition(Enum):
    G_X1 = 0
    G_X2 = 1
    X1_XX = 2
    X2_XX = 3
    X_XX = 4
    G_X = 5

    def tex(self) -> str:
        """Return a LaTeX string for the transition with order reversed."""
        mapping = {
            Transition.G_X1: (r"\mathrm{X}_{1}", r"\mathrm{G}"),
            Transition.G_X2: (r"\mathrm{X}_{2}", r"\mathrm{G}"),
            Transition.X1_XX: (r"\mathrm{XX}", r"\mathrm{X}_{1}"),
            Transition.X2_XX: (r"\mathrm{XX}", r"\mathrm{X}_{2}"),
            Transition.X_XX: (r"\mathrm{XX}", r"\mathrm{X}"),
            Transition.G_X: (r"\mathrm{X}", r"\mathrm{G}"),
        }
        if self not in mapping:
            return self.name
        left, right = mapping[self]
        return rf"${left}\!\leftrightarrow\!{right}$"


class TransitionType(Enum):
    EXTERNAL = "external"
    INTERNAL = "internal"


class TransitionRole(Enum):
    TPE = 0
    SINGLE = 1
