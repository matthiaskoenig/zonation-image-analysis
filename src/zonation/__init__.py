from pathlib import Path

CZI_PATH = Path(__file__).parent.parent.parent / "data" / "czi"
CZI_IMAGES = sorted([p for p in CZI_PATH.glob("*.czi")])

IMAGE_PATH = Path(__file__).parent.parent.parent / "data" / "images"
IMAGES = sorted([p for p in IMAGE_PATH.glob("*.czi")])