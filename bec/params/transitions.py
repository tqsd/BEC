from enum import Enum


class Transition(Enum):
    G_X1 = 0
    G_X2 = 1
    X1_XX = 2
    X2_XX = 3
    X_XX = 4
    G_X = 5


class TransitionType(Enum):
    EXTERNAL = "external"
    INTERNAL = "internal"


class TransitionRole(Enum):
    TPE = 0
    SINGLE = 1
