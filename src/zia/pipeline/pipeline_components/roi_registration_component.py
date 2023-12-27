from pathlib import Path
from typing import List

from zia.pipeline.file_management.file_management import SlideFileManager
from zia.pipeline.pipeline_components.pipeline import IPipelineComponent
from zia.pipeline.common.project_config import Configuration
from zia.pipeline.pipeline_components.roi_extraction_component import RoiExtractionComponent
from zia.log import get_logger
import os

from zia import BASE_PATH
from zia.config import read_config

config = read_config(BASE_PATH / "configuration.ini")
vipsbin = str(config.libvips_path)
# print(vipsbin)

os.environ['PATH'] = vipsbin + ';' + os.environ['PATH']
os.add_dll_directory(vipsbin)

for s in os.environ["PATH"].split(";"):
    print(s)

from valis import registration

logger = get_logger(__name__)


def register(roi_dir: Path, registration_path: Path, registration_result_path: Path) -> None:
    # Create a Valis object and use it to register the slides in slide_src_dir
    registrar = registration.Valis(str(roi_dir), str(registration_path))
    rigid_registrar, non_rigid_registrar, error_df = registrar.register()

    # Save all registered slides as ome.tiff
    registrar.warp_and_save_slides(
        str(registration_result_path),
        crop="reference",
        level=0,
        compression="jpeg",
        tile_wh=2 ** 11
    )


class SlideRegistrationComponent(IPipelineComponent):
    """Pipeline step for registration of ROIs"""
    dir_name = "RoiRegistration"

    def __init__(self, project_config: Configuration, file_manager: SlideFileManager, overwrite: bool = False):
        super().__init__(project_config, file_manager, SlideRegistrationComponent.dir_name, overwrite)


    def get_roi_dir(self, subject: str) -> List[Path]:
        p = self.project_config.image_data_path / RoiExtractionComponent.dir_name / subject
        if not p.exists():
            raise FileNotFoundError(f"No extracted ROI directory found for {subject}")

        sub_dirs = [sub_dir for sub_dir in p.iterdir()]

        if len(sub_dirs) == 0:
            raise FileExistsError(f"No ROI tiff files found for {subject}")

        return sub_dirs

    def get_registration_path(self, subject: str, lobe_id: str) -> Path:
        p = self.image_data_path / subject / lobe_id
        p.mkdir(exist_ok=True, parents=True)
        return p

    def get_registration_results_path(self, subject: str, lobe_id: str) -> Path:
        p = self.report_path / subject / lobe_id
        p.mkdir(exist_ok=True, parents=True)
        return p

    def check_exists(self, registration_path: Path) -> bool:
        if any(registration_path.iterdir()) and not self.overwrite:
            return False
        return True

    def run(self):
        try:
            for subject, slides in self.file_manager.group_by_subject().items():
                roi_dirs = self.get_roi_dir(subject)
                for roi_dir in roi_dirs:
                    lobe_id = roi_dir.name
                    registration_path = self.get_registration_path(subject, lobe_id)
                    registration_result_path = self.get_registration_results_path(subject, lobe_id)
                    if self.check_exists(registration_path):
                        continue

                    register(roi_dir, registration_path, registration_result_path)
        finally:
            registration.kill_jvm()


