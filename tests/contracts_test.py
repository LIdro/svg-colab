from __future__ import annotations

from colab_demo.services.contracts import (
    AssembleRequest,
    DetectRequest,
    ManualBoxRequest,
    SequentialInpaintRequest,
    TraceBatchRequest,
    ZOrderRequest,
)


def test_detect_contract_defaults():
    req = DetectRequest(image_data="data:image/png;base64,AAA", text="logo")
    assert req.min_score == 0.3
    assert req.max_results == 5


def test_manual_segment_contract():
    req = ManualBoxRequest(image_data="data:image/png;base64,AAA", box=[1, 2, 30, 40], label="text")
    assert req.box[0] == 1


def test_z_order_contract_shape():
    req = ZOrderRequest(
        image_width=100,
        image_height=80,
        objects=[{"id": "a", "label": "text", "bbox": [1, 2, 30, 40], "mask_data": "data:image/png;base64,AAA"}],
    )
    assert len(req.objects) == 1


def test_inpaint_sequential_contract_shape():
    req = SequentialInpaintRequest(
        image_data="data:image/png;base64,AAA",
        objects=[{"id": "a", "label": "text", "bbox": [1, 2, 30, 40], "mask_data": "data:image/png;base64,AAA"}],
    )
    assert req.use_z_order is True


def test_trace_batch_contract_shape():
    req = TraceBatchRequest(layers=[{"id": "a", "label": "obj", "image_data": "data:image/png;base64,AAA"}])
    assert req.layers[0].id == "a"


def test_assemble_contract_shape():
    req = AssembleRequest(
        width=300,
        height=200,
        layers=[{"id": "l1", "label": "Layer", "svg_paths": "<path d='M0 0'/>", "z_index": 0}],
    )
    assert req.width == 300
