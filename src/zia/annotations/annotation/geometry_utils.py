from typing import List, Tuple


def rescale_coords(coords: List[Tuple[float, float]], factor: float) -> List[Tuple[float, float]]:
    return [(x * factor, y * factor) for x, y in
            coords]
