from tifffile import tifffile

from zia import BASE_PATH
from zia.config import read_config
from zia.io.wsi_openslide import read_wsi

if __name__ == "__main__":
    config = read_config(BASE_PATH / "configuration.ini")
    subject = "SSES2021 9"
    path = config.data_path / "pig" / "CYP2D6" / "SSES2021 9_Pig_J-21-150_CYP2D6- 1 2000_Run 18__MAA_004.ndpi"
    store: tifffile.ZarrTiffStore = tifffile.imread(path, aszarr=True)
    slide = read_wsi(path)
    print(slide.properties["openslide.mpp-x"])
