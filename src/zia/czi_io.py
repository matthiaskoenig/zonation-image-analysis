"""Reading CZI fluorescence images.

Using bioformats with python-javabridge

https://pypi.org/project/python-javabridge/

This requires a working JAVA version
https://pythonhosted.org/javabridge/installation.html

sudo apt-get install openjdk-11-jdk
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
"""
from __future__ import annotations

import logging
import pickle
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import bioformats
import javabridge
import numpy as np
import xmltodict
from rich.console import Console


console = Console()
logger = logging.getLogger(__name__)


def read_czi(path: Path) -> Tuple[np.ndarray, Dict]:
    """Read czi_data with OME metadata."""

    with bioformats.ImageReader(str(path)) as reader:
        data = reader.read()

    omexml: str = bioformats.get_omexml_metadata(str(path))
    ome: Dict = xmltodict.parse(omexml)

    return data, ome


class Fluorophor(str, Enum):
    """Fluorophors for channels."""

    ALEXA_FLUOR_488 = "Alexa Fluor 488"
    DAPI = "DAPI"
    CY3 = "Cy3"


class FluorescenceImage:
    """Create image and serialize."""

    def __init__(self, sid: str, data: np.ndarray, ome: Dict):
        """Create image from data and OME metadata."""
        self.sid: str = sid
        self.data: np.ndarray = data
        self.metadata: Dict[str, Any] = self.parse_metadata(ome)

    def get_channel_data(self, fluorophor: Fluorophor) -> Optional[np.ndarray]:
        """Get image data for channel by fluorophor."""

        for k, channel in enumerate(self.metadata["Channels"]):
            if channel["@Fluor"] == fluorophor.value:
                return self.data[..., k]
        else:
            logger.error("Channel could not be found.")
            return None

    @staticmethod
    def from_file(path: Path) -> FluorescenceImage:
        """Read images from file using pickle."""
        with open(path, "rb") as f:
            image = pickle.load(f)
        return image

    def to_file(self, path: Path) -> None:
        """Save to file with pickle."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def parse_metadata(ome_dict: dict) -> Dict:
        """Parse metadata from OME dictionary."""

        info = {
            "NominalMagnification": float(
                ome_dict["OME"]["Instrument"]["Objective"]["@NominalMagnification"]
            ),
            "AquisitionDate": ome_dict["OME"]["Image"]["AcquisitionDate"],
            "SizeX": ome_dict["OME"]["Image"]["Pixels"]["@SizeX"],
            "SizeY": ome_dict["OME"]["Image"]["Pixels"]["@SizeY"],
            "PhysicalSizeX": float(
                ome_dict["OME"]["Image"]["Pixels"]["@PhysicalSizeX"]
            ),
            "PhysicalSizeXUnit": ome_dict["OME"]["Image"]["Pixels"][
                "@PhysicalSizeXUnit"
            ],
            "PhysicalSizeY": float(
                ome_dict["OME"]["Image"]["Pixels"]["@PhysicalSizeY"]
            ),
            "PhysicalSizeYUnit": ome_dict["OME"]["Image"]["Pixels"][
                "@PhysicalSizeYUnit"
            ],
            "Channels": ome_dict["OME"]["Image"]["Pixels"]["Channel"],
        }
        return info


def read_czi_images(czi_images: Iterable[Path]) -> List[Path]:
    """Reads and stores CZI image data.

    Uses pickle to store the matrices and metadata.
    Returns list of pickle paths.
    """
    pickle_paths: List[Path] = []
    javabridge.start_vm(class_path=bioformats.JARS)

    for czi_path in czi_images:

        sid = czi_path.stem
        data, ome = read_czi(czi_path)
        image = FluorescenceImage(sid=sid, data=data, ome=ome)

        pickle_path = czi_path.parent / f"{sid}.pickle"
        pickle_paths.append(pickle_path)
        image.to_file(pickle_path)

    javabridge.kill_vm()
    return pickle_paths


if __name__ == "__main__":

    from zia import CZI_EXAMPLE, CZI_IMAGES

    # read_czi_images(czi_images=[CZI_EXAMPLE])
    read_czi_images(czi_images=CZI_IMAGES)