from __future__ import annotations

import base64
import io

from PIL import Image

from colab_demo.services.api import assemble, detect, health
from colab_demo.services.contracts import AssembleRequest, DetectRequest
from colab_demo.services.pipeline import pipeline


def _tiny_png_data_uri() -> str:
    img = Image.new("RGBA", (32, 24), (255, 255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def test_health():
    r = health()
    assert r.status == "ok"


def test_detect_min_payload():
    # Keep the smoke test sandbox-safe by skipping heavy model import.
    pipeline._loaded = True
    pipeline.sam_model = None
    pipeline.detectors = {}
    payload = DetectRequest(
        image_data=_tiny_png_data_uri(),
        text="object",
        method="yolo26l",
        min_score=0.3,
        max_results=3,
        return_masks=True,
    )
    r = detect(payload)
    assert isinstance(r.boxes, list)
    assert len(r.boxes) >= 1


def test_assemble_min_payload():
    r = assemble(
        AssembleRequest(
            width=64,
            height=64,
            layers=[
                {
                    "id": "bg",
                    "label": "background",
                    "svg_paths": "<rect x='0' y='0' width='64' height='64' fill='white' />",
                    "z_index": 0,
                    "hidden": False,
                }
            ],
            optimize=False,
        )
    )
    assert "<svg" in r.svgText
