from pprint import pprint

import dask.array as da
import zarr
import napari

from napari_lazy_openslide import OpenSlideStore
from zia import RESOURCES_PATH
from openslide import OpenSlide, OpenSlideUnsupportedFormatError
from tifffile import imread
if __name__ == "__main__":
    #print(OME_TIFF_EXAMPLE)
    #store = OpenSlideStore(OME_TIFF_EXAMPLE)
    #grp = zarr.open(store, mode="r")
    #with napari.gui_qt():
    #    viewer = napari.Viewer()
    #    viewer.open(OME_TIFF_EXAMPLE)

    #OpenSlideStore(OME_TIFF_EXAMPLE)
    file_path_luca = RESOURCES_PATH / "LuCa-7color_Scan1.ome.tiff"
    file_path_npdi = RESOURCES_PATH / "LQF2_LM_HE_PVL.ndpi"
    file_path_lqf2 = RESOURCES_PATH / "LQF2_RM_HE_PVL2.ome.tif"
    file_path_czi = RESOURCES_PATH / "RH0422_1x2.5.czi"

    file_path_tiff_test1 = RESOURCES_PATH / "test_001.tif"
    file_path_tiff_test1_exported = RESOURCES_PATH / "test_001_exported.ome.tif"
    file_path_tiff_test1_exported2 = RESOURCES_PATH / "test_001_exported2.ome.tif"

    for file_path in [file_path_lqf2, file_path_npdi, file_path_tiff_test1, file_path_czi, file_path_tiff_test1_exported2]:
        try:
            s = OpenSlideStore(file_path)
            grp = zarr.open(s, mode="r")
            # The OpenSlideStore implements the multiscales extension
            # https://forum.image.sc/t/multiscale-arrays-v0-1/37930
            pprint(s._slide.properties)
            datasets = grp.attrs["multiscales"][0]["datasets"]
            pyramid = [grp.get(d["path"]) for d in datasets]
            print(pyramid)
        except OpenSlideUnsupportedFormatError:
            print(f"OpenSlideStore failed for {file_path}")
            s = None



    #imread(file_path_npdi)
    #print(file_path)
    slide =OpenSlide(file_path_tiff_test1_exported)
    slide1 =OpenSlide(file_path_tiff_test1_exported2)
    slide2 =OpenSlide(file_path_tiff_test1)

    for s in [slide, slide1, slide2, file_path_npdi]:
        #slide1 =OpenSlide(file_path_npdi)
        print(s.level_dimensions)
        print(s.dimensions)




    slide1 =OpenSlide(file_path_npdi)

    print(slide.level_dimensions)
    print(slide.dimensions)


    #with napari.gui_qt():
    #    viewer = napari.Viewer()
    #    viewer.open(OME_TIFF_EXAMPLE)
    #    viewer.add_image(grp["0"], name="0")
    #    viewer.add_image(grp["1"], name="1")
    #    viewer.add_image(grp["2"], name="2")

