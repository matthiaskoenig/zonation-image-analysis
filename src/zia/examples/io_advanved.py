from pathlib import Path
from typing import List
from xml.etree.ElementTree import XML

from aicspylibczi import CziFile
import bioformats
import zarr
import dask.array as da

from tifffile import tifffile, ZarrStore, TiffFileError, tiffcomment
from aicsimageio import AICSImage, readers

from zia import TEST_FILES_PATHS, file_path_npdi, file_path_tiff_test1, file_path_czi, \
    file_path_czi2, file_path_tiff_exported2
import napari

def read_file(file_path: Path) -> (XML, List[da.Array]):
    try:
        print(f"Reading file: {file_path.name}")
        print(file_path.is_file())
        store = tifffile.imread(file_path, aszarr=True)


        #ome_xml = bioformats.get_omexml_metadata(file_path)
        grp = zarr.open(store, mode="r")
        datasets = grp.attrs["multiscales"][0]["datasets"]
        #print(grp.attrs)
        #layers_n = 3
        #layers = {f"layer{x}":[] for x in range(layers_n)}
        #for d in datasets:
        #    data = da.from_zarr(store, component=d["path"])
        #    for layer_n in range(layers_n):
        #        layers[f"layer{layer_n}"].append(data[layer_n,:,:])
        data = [da.from_zarr(store, component=d["path"]) for d in datasets]

        #data = [da.moveaxis(da.from_zarr(store, component=d["path"]),0,2) for d in datasets]



        return data
    except TiffFileError as e:
        img = readers.czi_reader.CziReader(file_path)
        print(f"Error reading file: {file_path.name}")

        #czi = CziFile(file_path)
        #dimensions = czi.get_dims_shape()
        #print(dimensions)

        #img = AICSImage(file_path, reconstruct_mosaic=False)  # selects the first scene found

        properties = [img.dims,  # returns a Dimensions object
         img.dims.order,  # returns string "TCZYX"
         img.dims.X, # returns size of X dimension
         img.shape,  # returns tuple of dimension sizes in TCZYX order
         img.scenes,  # returns total number of pixels
         ]

        print(properties)
        # print(img.get_image_data("TCZYXS"))

        return img.get_image_data("TCZYXS")



        #raise e

def napari_view(file_path: Path) -> None:
    data= read_file(file_path)
    #ome = ome_types.from_xml(ome_xml)
    #ome =ome_types.from_xml(ome_xml)
    #print(ome)

    #with napari.gui_qt():
    viewer = napari.Viewer()
    colormaps = {"what_ever_1": "red", "what_ever_2": "green", "what_ever_3": "blue"}
    for n, (name,color) in enumerate(colormaps.items()):
        viewer.add_image([data_single[:, :, n] for data_single in data], blending="additive", colormap=color, name=name)

    #viewer = napari.view_image(data, rgb=True, name=file_path.name, )
    napari.run()

def napari_view_ndpi(file_path: Path) -> None:
    data= read_file(file_path,)

    viewer = napari.Viewer()
    viewer.add_image(data)

    #viewer = napari.view_image(data, rgb=True, name=file_path.name, )
    napari.run()


if __name__ == "__main__":
    #napari_view(file_path_tiff_exported2,)
    #napari_view_ndpi(file_path_npdi,)

    #napari_view(file_path_tiff_test1)
    napari_view_ndpi(file_path_czi2)



#for path in TEST_FILES_PATHS:
    #    store = read_file(path)










