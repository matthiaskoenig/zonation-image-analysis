import os
from typing import List, Tuple

from PIL.Image import Image

from zia.annotations import OPENSLIDE_PATH


if hasattr(os, "add_dll_directory"):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


def rescale_coords(
    coords: List[Tuple[float, float]], factor: float, offset=(0, 0)
) -> List[Tuple[float, float]]:
    return [((x - offset[0]) * factor, (y - offset[1]) * factor) for x, y in coords]


def read_full_image_from_slide(slide: openslide.OpenSlide, level: int) -> Image:
    w, h = slide.dimensions
    factor = slide.level_downsamples[level]
    return slide.read_region(
        location=(0, 0), level=level, size=(int(w / factor), int(h / factor))
    )
