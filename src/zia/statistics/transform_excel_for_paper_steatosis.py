import pandas as pd

from zia.pipeline.common.project_config import get_project_config

if __name__ == "__main__":
    project_config = get_project_config("steatosis")

    excel_sheets = pd.read_excel(
        project_config.reports_path / "descriptive-stats.xlsx",
        sheet_name=None)

    value_cols = ["median", "q1", "q3", "min", "max", "mean", "std"]

    sheet_unit_dict = {
        "species-comparison-lobule-geometry-steatosis": [r"\unit{\milli\metre}", r"\unit{\square\milli\metre}", "-", r"\unit{\milli\metre}"],
        "species-comparison-droplet-density": [r"\unit{\per\square\milli\metre}", r"\unit{\percent}"],
        "species-comparison-droplet-geometry": [r"\unit{\micro\metre}", r"\unit{\square\micro\metre}", r"\unit{\micro\metre}"]
    }

    sheet_condition_dict = {
        "species-comparison-lobule-geometry-steatosis": ["perimeter", "area", "compactness", "minimum_bounding_radius"],
        "species-comparison-droplet-density": ["Average droplet density", "Surface coverage"],
        "species-comparison-droplet-geometry": ["perimeter", "area", "compactness", "minimum_bounding_radius"]
    }

    sheet_factor_dict = {
        "species-comparison-lobule-geometry-steatosis": [1000, 1e6, 1, 1000],
        "species-comparison-droplet-density": [1, 1, 1, 1],
        "species-comparison-droplet-geometry": [1, 1, 1, 1]
    }

    for sheet_name in sheet_factor_dict:

        df = excel_sheets[sheet_name]

        if sheet_name == "species-comparison-lobule-geometry-steatosis":
            df = df[df["group"] != "Control"]

        name_mapping = {"species": "Species", "group": "Group", "attr": "Attribute", "unit": "Unit", "n": "n",
                        "median": "Median", "q1": "Q1", "q3": "Q3", "min": "Min", "max": "Max", "mean": "Mean", "std": "SD"}

        columns_to_drop = ["iqr", "geo_mean", "log_std", "cv", "log_cv"]

        df = df.drop(columns=columns_to_drop)

        conditions = [df["attr"] == cond for cond in sheet_condition_dict[sheet_name]]
        factors = sheet_factor_dict[sheet_name]
        units = sheet_unit_dict[sheet_name]

        for condition, factor, unit in zip(conditions, factors, units):
            df.loc[condition, value_cols] = (df.loc[condition, value_cols] / factor).round(3 if sheet_name == "species-comparison-lobule-geometry-steatosis" else 1)
            df.loc[condition, "unit"] = unit

        df[value_cols] = df[value_cols].applymap(lambda x: f"{x:.3f}" if sheet_name == "species-comparison-lobule-geometry-steatosis" else f"{x:.1f}")


        df.loc[df["attr"] == "minimum_bounding_radius", "attr"] = "min radius"
        df.loc[df["attr"] == "min_enclosing_circle", "attr"] = "min radius"


        df = df.sort_values(by=['attr', 'species', 'group'])

        df = df[list(name_mapping.keys())]

        df = df.rename(columns=name_mapping)

        df["Group"] = df["Group"].apply(lambda x: x + " HDF" if isinstance(x, str) and (x == '4W' or x == '2W') else x)
        df["Group"] = df["Group"].apply(lambda x: "Steatosis" if isinstance(x, str) and (x == "Stea.") else x)

        df.to_csv(project_config.reports_path / f"{sheet_name}.csv", index=False)
