import cv2
import pandas as pd

from zia import BASE_PATH
from zia.config import read_config
from zia.oven.data_store import ZarrGroups
from zia.io.wsi_tifffile import read_ndpi
from zia.statistics.expression_profile.expression_profile_gradient import open_protein_arrays
from zia.statistics.expression_profile.validation_images import get_level_seven_array
from zia.statistics.utils.data_provider import SlideStatsProvider

if __name__ == "__main__":
    subject = "UKJ-19-026_Human"
    roi = 0
    rois = [0, 0, 0, 0]
    config = read_config(BASE_PATH / "configuration.ini")
    report_path = config.reports_path / "algo_figure"
    report_path.mkdir(exist_ok=True, parents=True)
    protein_order = ["CYP1A2", "CYP2D6", "CYP2E1", "CYP3A4", "GS", "HE"]
    species_order = SlideStatsProvider.species_order
    colors = SlideStatsProvider.species_colors
    df = pd.read_csv(config.reports_path / "lobule_distances.csv", sep=",", index_col=False)
    species_gb = df.groupby("species")

    slide_path = None

    slide_stats_dict = get_slide_stats()

    protein = "CYP3A4"
    slide_dir = config.image_data_path / "rois_registered" / f"{subject}" / f"{roi}"
    for file in slide_dir.iterdir():
        if file.is_file() and protein in file.stem:
            slide_path = file

    slide_array = read_ndpi(slide_path)

    he_array = get_level_seven_array(slide_array)

    dab_arrays = open_protein_arrays(address=config.image_data_path / "stain_separated" / f"{subject}.zarr",
                                     path=f"{ZarrGroups.STAIN_1.value}/{roi}",
                                     level=slide_stats_dict[subject][str(roi)].meta_data["level"],
                                     excluded=[])

    h_array = open_protein_arrays(address=config.image_data_path / "stain_separated" / f"{subject}.zarr",
                                  path=f"{ZarrGroups.STAIN_0.value}/{roi}",
                                  level=slide_stats_dict[subject][str(roi)].meta_data["level"],
                                  excluded=[])

    cv2.imwrite(str(report_path / f"{protein}.png"), cv2.cvtColor(he_array, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(report_path / f"{protein}_dab.png"), 255 - dab_arrays[protein])
    cv2.imwrite(str(report_path / f"{protein}_h.png"), 255- h_array[protein])
