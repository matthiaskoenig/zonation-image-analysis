import matplotlib.pyplot as plt

from zia.pipeline.common.project_config import get_project_config
from zia.pipeline.pipeline_components.algorithm.segementation.lobulus_statistics import SlideStats
from zia.pipeline.pipeline_components.segementation_component import SegmentationComponent

if __name__ == "__main__":
    config = get_project_config("control")
    subject = "SSES2021 14"
    roi = "0"
    slide_stats = SlideStats.load_from_file_system(config.image_data_path / SegmentationComponent.dir_name / subject / roi)

    slide_stats.plot()
