import geojson as gj
import matplotlib.pyplot as plt
import napari
import numpy as np
import openslide
import zarr
from PIL import Image, ImageDraw
from shapely.geometry import MultiPolygon, Polygon, shape


PATH_TO_FILE = "../resources/geojsons/J-12-00350_NOR-022_Lewis_CYP2E1- 1 300_Run 14_ML, Sb, rk_MAA_006.geojson"
PATH_TO_ZARR = "../resources/zarr_files/img2.zarr"
if __name__ == "__main__":
    with open(PATH_TO_FILE) as f:
        data = gj.load(f)

    # extracting features from the dict
    features = data["features"]

    geometries = [shape(feature["geometry"]) for feature in features]

    mask = Image.new("L", (71424, 83456), 0)

    draw = ImageDraw.Draw(mask)
    for geometry in geometries:
        if isinstance(geometry, Polygon):
            coords = list(geometry.exterior.coords)
            draw.polygon(coords, fill=255)
        elif isinstance(geometry, MultiPolygon):
            for poly in geometry.geoms:
                coords = list(poly.exterior.coords)
                draw.polygon(coords, fill=255)

    mask_array = np.array(mask)

    zarray = zarr.array(mask_array, store=PATH_TO_ZARR)

    # zarr.convenience.save_array(PATH_TO_ZARR, mask_array)

    # viewer = napari.Viewer()
    # viewer.add_image(mask_array, name="mask")
    # napari.run()
