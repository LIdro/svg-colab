from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DetectMethod(str, Enum):
    gdino = "gdino"
    yolo26l = "yolo26l"
    yolo26x = "yolo26x"


class InpaintProvider(str, Enum):
    openrouter = "openrouter"
    big_lama = "big-lama"
    qualcomm_lama_dilated = "qualcomm-lama-dilated"


class DetectRequest(BaseModel):
    image_data: str
    text: str
    method: DetectMethod = DetectMethod.yolo26l
    min_score: float = 0.3
    max_results: int = 5
    return_masks: bool = True


class DetectBox(BaseModel):
    box: List[float] = Field(description="[x, y, width, height]")
    score: float
    mask_png: Optional[str] = None
    label: Optional[str] = None


class DetectResponse(BaseModel):
    boxes: List[DetectBox]


class ManualBoxRequest(BaseModel):
    image_data: str
    box: List[float] = Field(description="[x1, y1, x2, y2]")
    label: str


class ManualBoxResponse(BaseModel):
    box: List[float]
    label: str
    mask_png: str
    width: int
    height: int


class ZOrderObject(BaseModel):
    id: str
    label: str
    bbox: List[float] = Field(description="[x1, y1, x2, y2]")
    mask_data: Optional[str] = None


class ZOrderRequest(BaseModel):
    image_width: int
    image_height: int
    objects: List[ZOrderObject]


class ZOrderResult(BaseModel):
    id: str
    label: str
    z_score: float
    rank: int
    reasoning: str


class ZOrderResponse(BaseModel):
    ordered_objects: List[ZOrderResult]


class SequentialInpaintRequest(BaseModel):
    image_data: str
    objects: List[ZOrderObject]
    model: Optional[str] = None
    provider: Optional[InpaintProvider] = InpaintProvider.big_lama
    api_key: Optional[str] = None
    use_z_order: bool = True


class InpaintedLayer(BaseModel):
    object_id: str
    label: str
    rank: int
    inpainted_image: str
    processing_time: float


class SequentialInpaintResponse(BaseModel):
    success: bool
    layers: List[InpaintedLayer]
    final_background: str
    z_order_used: List[ZOrderResult]
    total_processing_time: float


class TraceOptions(BaseModel):
    color_mode: str = "color"
    mode: str = "spline"
    filter_speckle: int = 0
    corner_threshold: float = 0.0
    color_precision: int = 8
    path_precision: int = 10
    splice_threshold: float = 6.0
    length_threshold: float = 1.0
    max_iterations: int = 12


class TraceLayerRequest(BaseModel):
    id: str
    label: str
    image_data: str
    source_image_data: Optional[str] = None
    input_offset: Optional[Dict[str, float]] = None
    options: Optional[TraceOptions] = None


class TraceBatchRequest(BaseModel):
    layers: List[TraceLayerRequest]
    options: Optional[TraceOptions] = None


class TraceLayerResponse(BaseModel):
    id: str
    label: str
    width: int
    height: int
    svg_paths: str
    svg_full: str
    stats: Dict[str, Any]
    offset: Optional[Dict[str, float]] = None


class TraceBatchResponse(BaseModel):
    layers: List[TraceLayerResponse]


class AssembleLayer(BaseModel):
    id: str
    label: str
    svg_paths: Optional[str] = None
    svg_full: Optional[str] = None
    z_index: int = 0
    hidden: bool = False


class AssembleRequest(BaseModel):
    width: int
    height: int
    layers: List[AssembleLayer]
    optimize: bool = False


class AssembleResponse(BaseModel):
    svgText: str


class HealthResponse(BaseModel):
    status: str
