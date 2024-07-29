import pandas as pd

from zia.pipeline.common.project_config import get_project_config

if __name__ == '__main__':

    project_config = get_project_config("control")

    excel_sheets = pd.read_excel(
        project_config.reports_path.joinpath("plots/manuscript/descriptive-stats/descriptive-stats.xlsx"),
        sheet_name=None)

    value_cols = ["median", "q1", "q3", "min", "max", "mean", "std", "se"]
    col_oder = ["group", "attr", "unit", "n"] + value_cols

    for sheet_name, df in excel_sheets.items():
        print(sheet_name)

        conditions = [
            df["attr"] == "perimeter",
            df["attr"] == "area",
            df["attr"] == "compactness",
            df["attr"] == "minimum_bounding_radius"
        ]

        factors = [
            1000,
            1e6,
            1,
            1000
        ]

        unit = [r"\unit{\milli\metre}", r"\unit{\square\milli\metre}", "-", r"\unit{\milli\metre}"]

        for condition, factor, unit in zip(conditions, factors, unit):
            df.loc[condition, value_cols] = (df.loc[condition, value_cols] / factor).round(3)
            df.loc[condition, "unit"] = unit

        df[value_cols] = df[value_cols].applymap(lambda x: f"{x:.3f}")

        df.loc[df["attr"] == "minimum_bounding_radius", "attr"] = "min radius"

        df = df.sort_values(by=['attr', 'group'])
        df = df.drop(columns=["nominal_var"])


        df = df[col_oder]

        df = df.rename(columns={"group": "Group",
                                "attr": "Attribute",
                                "unit": "Unit",
                                "n": "n",
                                "median": "Median",
                                "q1": "Q1",
                                "q3": "Q3",
                                "min": "Min",
                                "max": "Max",
                                "mean": "Mean",
                                "std": "SD",
                                "se": "SE"})

        df = df.drop(columns=["SE"])
        df.to_csv(project_config.reports_path / "plots" / "manuscript" / "descriptive-stats" / f"{sheet_name}.csv", index=False)
