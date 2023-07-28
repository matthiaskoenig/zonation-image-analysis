"""Read images."""
from pathlib import Path


def check_image_path(image_path: Path) -> None:
    """Check that path exists and is file."""
    if not image_path.exists():
        raise IOError(f"Image file does not exist: {image_path}")
    if not image_path.is_file():
        raise IOError(f"Image file is not a file: {image_path}")
