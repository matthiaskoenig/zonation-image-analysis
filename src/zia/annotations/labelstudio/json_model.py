from dataclasses import dataclass, field
from typing import List, Union, Optional, Tuple


@dataclass
class RectangleValue:
    x: float
    y: float
    width: float
    height: float
    rotation: float = 0.0


@dataclass
class PolygonValue:
    points: List[Tuple[float, float]]
    polygonlabels: List[str]


@dataclass
class EllipseValue:
    x: float
    y: float
    radiusX: float
    radiusY: float
    rotation: float = 0.0


@dataclass
class KeypointValue:
    x: float
    y: float
    width: float
    keypointlabels: List[str]


@dataclass
class Result:
    original_width: int
    original_height: int
    image_rotation: int
    value: Union[RectangleValue, PolygonValue, EllipseValue, KeypointValue]
    id: str
    from_name: str
    to_name: str
    type: str


@dataclass
class Annotation:
    unique_id: str
    completed_by: int
    result: List[Result] = field(default_factory=list)
    was_cancelled: bool = False
    ground_truth: bool = False
    draft_created_at: Optional[str] = None
    lead_time: Optional[int] = None
    import_id: Optional[int] = None
    last_action: Optional[str] = None
    task: Optional[int] = None
    project: Optional[int] = None
    updated_by: Optional[int] = None
    parent_prediction: Optional[int] = None
    parent_annotation: Optional[int] = None
    last_created_by: Optional[int] = None


@dataclass
class Prediction:
    model_version: str
    task: int
    result: Optional[List[Result]] = field(default_factory=list)
    score: Optional[float] = None
    cluster: Optional[int] = None
    neighbors: Optional[list] = field(default_factory=list)
    mislabeling: Optional[float] = None
    project: Optional[int] = None


@dataclass
class DataItem:
    data: dict
    predictions: List[Prediction]


@dataclass
class Task:
    pass