from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab_demo.services.semantic_labels import assign_text_component_labels, semanticize_non_text_objects


def test_assign_text_component_labels_with_dot():
    components = [
        {"bbox_xyxy": [0, 0, 10, 20], "area": 200},  # l
        {"bbox_xyxy": [12, 0, 22, 20], "area": 200},  # u
        {"bbox_xyxy": [24, 0, 38, 20], "area": 280},  # m
        {"bbox_xyxy": [40, 0, 46, 20], "area": 120},  # i stem
        {"bbox_xyxy": [40, -6, 46, -2], "area": 24},  # dot over i
        {"bbox_xyxy": [48, 0, 58, 20], "area": 200},  # l
        {"bbox_xyxy": [60, 0, 72, 20], "area": 220},  # o
        {"bbox_xyxy": [74, 0, 84, 20], "area": 200},  # u
        {"bbox_xyxy": [86, 0, 98, 20], "area": 240},  # p
    ]

    labels = assign_text_component_labels("lumiloup text", components)

    assert labels[0] == "first l in lumiloup"
    assert labels[1] == "first u in lumiloup"
    assert labels[2] == "first m in lumiloup"
    assert labels[3] == "first i in lumiloup"
    assert labels[4] == "dot over first i in lumiloup"
    assert labels[5] == "second l in lumiloup"
    assert labels[6] == "first o in lumiloup"
    assert labels[7] == "second u in lumiloup"
    assert labels[8] == "first p in lumiloup"


def test_semanticize_non_text_objects_adds_position():
    objects = [
        {"id": "a", "label": "icon", "bbox_xyxy": [0, 0, 10, 10]},
        {"id": "b", "label": "icon", "bbox_xyxy": [90, 0, 100, 10]},
    ]

    updated = semanticize_non_text_objects(objects, (100, 100))
    labels = {obj["id"]: obj["label"] for obj in updated}

    assert labels["a"] == "first icon (top-left)"
    assert labels["b"] == "second icon (top-right)"
