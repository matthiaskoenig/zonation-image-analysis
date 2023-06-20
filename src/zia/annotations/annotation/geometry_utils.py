"""Utilities for manipulating geometry and coordinates."""

from typing import List, Tuple


def rescale_coords(
    coords: List[Tuple[float, float]],
    factor: float = 1.0,
    offset: Tuple[int, int] = (0, 0),
) -> List[Tuple[float, float]]:
    """Rescaling of coordinates by factor.

    :param factor: muliplicative factor
    :param offset: offset to shift coordinates
    """
    return [((x - offset[0]) * factor, (y - offset[1]) * factor) for x, y in coords]
