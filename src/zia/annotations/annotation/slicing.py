from typing import List, Tuple

import numpy as np


def get_tile_slices(
    shape: Tuple[int, int], tile_size=2**13
) -> List[Tuple[slice, slice]]:
    r, c = shape
    num_col = int(np.ceil(c / tile_size))
    num_row = int(np.ceil(r / tile_size))
    slices = []
    for i in range(num_col):
        for k in range(num_row):
            col_end = min(c, (i + 1) * tile_size)
            cs = slice(i * tile_size, col_end)

            row_end = min(r, (k + 1) * tile_size)
            rs = slice(k * tile_size, row_end)

            slices.append((rs, cs))
    return slices
