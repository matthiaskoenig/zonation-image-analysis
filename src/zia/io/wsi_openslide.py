"""Read whole-slide images with openslide."""

import os
from pathlib import Path

from PIL.Image import Image

from zia.console import console
from zia.io.utils import check_image_path


openslide = None


if hasattr(os, "add_dll_directory"):
    from zia import OPENSLIDE_PATH

    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


def read_wsi(image_path: Path) -> openslide.OpenSlide:
    """Read image with openslide library."""
    check_image_path(image_path)
    return openslide.OpenSlide(image_path)


def read_full_image_from_slide(slide: openslide.OpenSlide, level: int) -> Image:
    """Read the complete image for given level."""
    w, h = slide.dimensions
    factor = slide.level_downsamples[level]
    return slide.read_region(
        location=(0, 0), level=level, size=(int(w / factor), int(h / factor))
    )


if __name__ == "__main__":
    from zia import BASE_PATH

    example_ndpi = BASE_PATH / "data" / "ndpi" / "LQF1_LM_HE_PVL.ndpi"
    open_slide = read_wsi(example_ndpi)
    console.print(open_slide)
