from typing import Callable

from bec.light.envelopes.base import Envelope


def compile_envelope(
    env: Envelope, *, time_unit_s: float
) -> Callable[[float], float]:
    s = float(time_unit_s)

    # If the env exposes a fast seconds evaluator, use it.
    eval_s = getattr(env, "_eval_seconds", None)
    if callable(eval_s):

        def env_solver(t_solver: float) -> float:
            return float(eval_s(s * float(t_solver)))

        return env_solver

    # Fallback: still avoid Pint by passing float seconds into __call__
    # (because your Envelope contract treats numbers as seconds).
    def env_solver(t_solver: float) -> float:
        return float(env(s * float(t_solver)))

    return env_solver
