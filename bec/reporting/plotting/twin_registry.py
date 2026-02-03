from __future__ import annotations

from typing import List, Iterable
from weakref import WeakKeyDictionary

from matplotlib.axes import Axes


_TWIN_AXES: WeakKeyDictionary[Axes, List[Axes]] = WeakKeyDictionary()


def register_twin_axes(ax: Axes, twin: Axes) -> None:
    twins = _TWIN_AXES.get(ax)
    if twins is None:
        twins = []
        _TWIN_AXES[ax] = twins
    twins.append(twin)


def clear_twin_axes(ax: Axes) -> None:
    twins = _TWIN_AXES.get(ax)
    if twins is not None:
        twins.clear()


def iter_axes_for_legend(ax: Axes) -> Iterable[Axes]:
    yield ax
    for t in _TWIN_AXES.get(ax, []):
        yield t
