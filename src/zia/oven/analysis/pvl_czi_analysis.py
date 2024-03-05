"""Analysis of the CZI patterns."""
from zia import BASE_PATH


CZI_IMAGES_INITIAL = sorted(
    [p for p in (BASE_PATH / "data" / "czi" / "Initial").glob("*.czi")]
)
CZI_IMAGES_AXIOS = sorted(
    [p for p in (BASE_PATH / "data" / "czi" / "AxioScanZeiss").glob("*.czi")]
)
