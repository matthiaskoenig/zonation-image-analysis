from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

from zia.pipeline.file_management.file_utils import get_subject_from_path


@dataclass
class Slide:
    name: str
    protein: str
    subject: str
    species: str


class SlideFileManager:
    def __init__(self, slide_dir: Path, extension: str = "ndpi"):
        self.slides = SlideFileManager._from_slide_dir(slide_dir, extension)

    def group_by_subject(self) -> Dict[str, List[Slide]]:
        subject_slides_dict = {}
        for slide in self.slides:
            if slide.subject not in subject_slides_dict.keys():
                subject_slides_dict[slide.subject] = []

            subject_slides_dict[slide.subject].append(slide)

        return subject_slides_dict

    @classmethod
    def _from_slide_dir(cls, slide_dir: Path, extension: str) -> List[Slide]:
        return [cls._create_slide(p) for p in slide_dir.glob(f"**/*.{extension}")]

    @classmethod
    def _create_slide(cls, slide_path: Path) -> Slide:
        species = slide_path.parent.parent.name.lower()
        protein = slide_path.parent.name.lower()
        image_id = slide_path.stem
        subject = get_subject_from_path(image_id, species)

        return Slide(image_id, protein, subject, species)
