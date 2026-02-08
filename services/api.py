from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .contracts import (
    AssembleRequest,
    AssembleResponse,
    DetectRequest,
    DetectResponse,
    HealthResponse,
    ManualBoxRequest,
    ManualBoxResponse,
    SequentialInpaintRequest,
    SequentialInpaintResponse,
    TraceBatchRequest,
    TraceBatchResponse,
    ZOrderRequest,
    ZOrderResponse,
)
from .pipeline import pipeline

app = FastAPI(title="svg-repair-colab", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/detect", response_model=DetectResponse)
def detect(payload: DetectRequest) -> DetectResponse:
    return pipeline.detect(payload)


@app.post("/segment-manual-box", response_model=ManualBoxResponse)
def segment_manual_box(payload: ManualBoxRequest) -> ManualBoxResponse:
    result = pipeline.segment_manual_box(payload.image_data, payload.box, payload.label)
    return ManualBoxResponse(**result)


@app.post("/compute-z-order", response_model=ZOrderResponse)
def compute_z_order(payload: ZOrderRequest) -> ZOrderResponse:
    return pipeline.compute_z_order(payload.objects, payload.image_width, payload.image_height)


@app.post("/inpaint-sequential", response_model=SequentialInpaintResponse)
def inpaint_sequential(payload: SequentialInpaintRequest) -> SequentialInpaintResponse:
    return pipeline.inpaint_sequential(payload)


@app.post("/trace-batch", response_model=TraceBatchResponse)
def trace_batch(payload: TraceBatchRequest) -> TraceBatchResponse:
    return pipeline.trace_batch(payload)


@app.post("/assemble", response_model=AssembleResponse)
def assemble(payload: AssembleRequest) -> AssembleResponse:
    return AssembleResponse(**pipeline.assemble(payload))
