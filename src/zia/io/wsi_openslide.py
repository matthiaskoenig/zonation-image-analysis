"""Read whole-slide images with openslide."""

import os
from pathlib import Path

from PIL.Image import Image

from zia.config import read_config
from zia.console import console
from zia.io.utils import check_image_path
from zia import BASE_PATH

openslide = None


if hasattr(os, "add_dll_directory"):
    config = read_config(BASE_PATH / "configuration.ini")
    # Python >= 3.8 on Windows
    #with os.add_dll_directory(str(config.openslide_path)):
     #   import openslide
else:
    import openslide


def read_wsi(image_path: Path) -> openslide.OpenSlide:
    """Read image with openslide library."""
    check_image_path(Path(image_path))
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
