import cv2
import numpy as np
import pandas as pd

from zia.oven.annotations.workflow_visualizations.util.image_plotting import plot_pic
from zia.pipeline.common.project_config import get_project_config
from zia.pipeline.pipeline_components.portality_mapping_component import PortalityMappingComponent
from zia.pipeline.pipeline_components.roi_extraction_component import RoiExtractionComponent
from zia.statistics.lobulus_geometry.species_comparison import plot_species_comparison
from image_utils.io.tiffile import read_ndpi

from zia.statistics.steatosis.utils import map_to_group

project_config = get_project_config("steatosis")

steatosis_stats_path = project_config.image_data_path / "SteatosisStats"

DIET = {
    "FLR-167": "2",
    "FLR-168": "2",
    "FLR-169": "2",
    "FLR-170": "2",
    "FLR-171": "2",
    "FLR-172": "2",

    "FLR-179": "4",
    "FLR-180": "4",
    "FLR-181": "4",
    "FLR-199": "4",
    "FLR-200": "4",
    "FLR-201": "4",

    "MNT-031": "2",
    "MNT-032": "2",
    "MNT-027": "2",
    "MNT-033": "2",
    "MNT-034": "2",
    "MNT-035": "2",
    "MNT-036": "2",

    "MNT-041": "4",
    "MNT-042": "4",
    "MNT-043": "4",
    "MNT-044": "4",
    "MNT-045": "4",
    "MNT-046": "4",
}


def get_species_from_subject(subject: str) -> str:
    if "FLR" in subject:
        return "rat"
    if "MNT" in subject:
        return "mouse"
    if "Human" in subject:
        return "human"
    return "unknown"


def get_steatosis_stats() -> pd.DataFrame:
    dfs = []
    for subject_dir in steatosis_stats_path.iterdir():
        subject = subject_dir.stem
        for roi_dir in subject_dir.iterdir():
            roi = roi_dir.stem

            df = pd.read_csv(roi_dir / "steatosis_statistics.csv")

            df["subject"] = subject
            df["roi"] = roi

            species = get_species_from_subject(subject)
            df["species"] = species

            if species in ["mouse", "rat"]:
                df["diet"] = DIET[subject]

            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def map_diet_on_df(df: pd.DataFrame) -> None:
    df["group"] = list(map(map_to_group, df['species'], df['diet']))


def get_density_stats(px_size=0.2272) -> pd.DataFrame:
    data_points = []

    for subject_dir in (project_config.image_data_path / RoiExtractionComponent.dir_name).iterdir():
        for roi_dir in subject_dir.iterdir():

            for wsi_path in roi_dir.iterdir():
                if not "HE" in wsi_path.stem:
                    continue
                wsis = read_ndpi(wsi_path)

                lowest_key = max(wsis.keys())

                image = cv2.cvtColor(wsis[lowest_key][:], cv2.COLOR_RGB2GRAY)

                blur = cv2.GaussianBlur(image, (15, 15), 5)

                _, th = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)

                area = np.count_nonzero(th == 0) * (px_size * 2 ** lowest_key) ** 2

                data_points.append(
                    dict(
                        subject=subject_dir.stem,
                        roi=roi_dir.stem,
                        species=get_species_from_subject(subject_dir.stem),
                        area=area,
                        diet=DIET.get(subject_dir.stem)
                    )
                )

    return pd.DataFrame(data_points)


def get_droplet_df(droplet_data: pd.DataFrame, overwrite: bool = True) -> pd.DataFrame:
    result_df_path = project_config.image_data_path / PortalityMappingComponent.dir_name / "lobule_droplets.csv"

    if not overwrite:
        if result_df_path.exists():
            df = pd.read_csv(result_df_path)
            return df

    portality_df = pd.read_csv(project_config.image_data_path / PortalityMappingComponent.dir_name / "lobule_distances.csv")


    portality_df["diet"] = portality_df["subject"].map(DIET)
    map_diet_on_df(portality_df)

    portality_df = portality_df[portality_df["protein"] == "he"]

    droplet_groupy = droplet_data.groupby(["subject", "roi"])

    # print(droplet_groupy.groups.keys())
    result_dfs = []

    for (subject, roi), pgroup_df in portality_df.groupby(["subject", "roi"]):
        droplet_group_df = droplet_groupy.get_group((subject, str(roi))).copy()
        # print(subject, roi)
        # print("portality_df", len(pgroup_df))
        # print("droplet_df", len(droplet_group_df))
        droplet_group_df["y_idx"] = np.ceil((droplet_group_df["cy"] / 2 ** 7)).astype(int)
        droplet_group_df["x_idx"] = np.ceil((droplet_group_df["cx"] / 2 ** 7)).astype(int)

        grouped_by_idx = droplet_group_df.groupby(["x_idx", "y_idx"]).agg(mean_droplet_area=("area", "mean"), droplet_count=("area", "count"),
                                                                          total_droplet_area=("area", "sum"),
                                                                          diet=("diet", lambda x: pd.unique(x)[0]))

        result_dfs.append(
            pd.merge(pgroup_df.copy(), grouped_by_idx, left_on=("height", "width"), right_on=("y_idx", "x_idx"), how="left")
        )

        #print("merged_df", len(result_dfs[-1]))


    portality_droplet_df = pd.concat(result_dfs, ignore_index=True)
    #print(portality_droplet_df.columns)


    portality_droplet_df = portality_droplet_df.drop(columns="diet_y")
    portality_droplet_df = portality_droplet_df.rename(columns={"diet_x": "diet"})

    portality_droplet_df[['mean_droplet_area', 'droplet_count', 'total_droplet_area']] = portality_droplet_df[['mean_droplet_area', 'droplet_count', 'total_droplet_area']].fillna(0)
    #print(len(portality_droplet_df))

    # portality_droplet_df = portality_droplet_df.astype(dict(diet="Int64"))

    portality_droplet_df.to_csv(project_config.image_data_path / PortalityMappingComponent.dir_name / "lobule_droplets.csv", index=False)

    return portality_droplet_df
