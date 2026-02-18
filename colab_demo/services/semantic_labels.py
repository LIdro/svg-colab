from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple


_TEXT_KEYWORDS = {
    "text",
    "logo",
    "word",
    "words",
    "letter",
    "letters",
    "title",
    "caption",
    "label",
    "watermark",
}

_ORDINAL_WORDS = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
}


def ordinal_word(value: int) -> str:
    if value in _ORDINAL_WORDS:
        return _ORDINAL_WORDS[value]
    suffix = "th"
    if 10 <= value % 100 <= 20:
        suffix = "th"
    elif value % 10 == 1:
        suffix = "st"
    elif value % 10 == 2:
        suffix = "nd"
    elif value % 10 == 3:
        suffix = "rd"
    return f"{value}{suffix}"


def normalize_label(label: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", " ", label or "").strip().lower()
    return cleaned or "object"


def extract_word_from_label(label: str) -> str:
    tokens = [token for token in re.split(r"[^a-zA-Z0-9]+", label or "") if token]
    candidates = [token for token in tokens if token.lower() not in _TEXT_KEYWORDS]
    if not candidates:
        return "text"
    return max(candidates, key=len).lower()


def position_descriptor(bbox_xyxy: Iterable[float], image_size: Tuple[int, int]) -> str:
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    except Exception:
        return ""
    width, height = image_size
    if width <= 0 or height <= 0:
        return ""

    cx = (x1 + x2) / 2.0 / float(width)
    cy = (y1 + y2) / 2.0 / float(height)

    if cx < 0.33:
        horiz = "left"
    elif cx > 0.67:
        horiz = "right"
    else:
        horiz = "center"

    if cy < 0.33:
        vert = "top"
    elif cy > 0.67:
        vert = "bottom"
    else:
        vert = "middle"

    if horiz == "center" and vert == "middle":
        return "center"
    if vert == "middle":
        return horiz
    if horiz == "center":
        return vert
    return f"{vert}-{horiz}"


def _component_metrics(component: Dict[str, Any]) -> Dict[str, float]:
    x1, y1, x2, y2 = [float(v) for v in component.get("bbox_xyxy", [0, 0, 0, 0])]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    area = float(component.get("area") or (width * height))
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "cx": (x1 + x2) / 2.0,
        "cy": (y1 + y2) / 2.0,
        "width": width,
        "height": height,
        "area": area,
    }


def _x_overlap_ratio(a: Dict[str, float], b: Dict[str, float]) -> float:
    left = max(a["x1"], b["x1"])
    right = min(a["x2"], b["x2"])
    overlap = max(0.0, right - left)
    denom = max(1.0, min(a["width"], b["width"]))
    return overlap / denom


def assign_text_component_labels(label: str, components: List[Dict[str, Any]]) -> List[str]:
    if not components:
        return []

    word = extract_word_from_label(label)
    letters = list(word)

    enriched = []
    for idx, component in enumerate(components):
        metrics = _component_metrics(component)
        metrics["index"] = idx
        enriched.append(metrics)

    areas = sorted(m["area"] for m in enriched)
    median_area = areas[len(areas) // 2] if areas else 0.0
    dot_threshold = median_area * 0.35 if median_area > 0 else 0.0

    dot_indices: set[int] = set()
    dot_map: Dict[int, int] = {}

    for dot in enriched:
        if dot_threshold > 0 and dot["area"] > dot_threshold:
            continue
        best_base = None
        best_score = None
        for base in enriched:
            if base is dot:
                continue
            if base["area"] <= dot_threshold:
                continue
            if dot["cy"] >= base["cy"]:
                continue
            if _x_overlap_ratio(dot, base) < 0.35:
                continue
            score = base["cy"] - dot["cy"]
            if best_score is None or score < best_score:
                best_score = score
                best_base = base
        if best_base is not None:
            dot_indices.add(dot["index"])
            if best_base["index"] not in dot_map:
                dot_map[best_base["index"]] = dot["index"]

    base_components = [comp for comp in enriched if comp["index"] not in dot_indices]
    base_components.sort(key=lambda item: item["cx"])

    labels = ["" for _ in components]
    letter_counts: Dict[str, int] = defaultdict(int)
    letter_meta: Dict[int, Tuple[str, str]] = {}

    for pos, comp in enumerate(base_components):
        if pos < len(letters):
            letter = letters[pos]
            letter_counts[letter] += 1
            ordinal = ordinal_word(letter_counts[letter])
            labels[comp["index"]] = f"{ordinal} {letter} in {word}"
            letter_meta[comp["index"]] = (letter, ordinal)
        else:
            ordinal = ordinal_word(pos + 1)
            labels[comp["index"]] = f"{ordinal} segment of {word}"

    for base_idx, dot_idx in dot_map.items():
        letter, ordinal = letter_meta.get(base_idx, ("letter", ordinal_word(1)))
        labels[dot_idx] = f"dot over {ordinal} {letter} in {word}"

    for idx, current in enumerate(labels):
        if not current:
            ordinal = ordinal_word(idx + 1)
            labels[idx] = f"{ordinal} accent of {word}"

    return labels


def semanticize_non_text_objects(objects: List[Dict[str, Any]], image_size: Tuple[int, int]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for obj in objects:
        if obj.get("semantic_kind") == "text_component":
            continue
        base = normalize_label(obj.get("label", "object"))
        grouped[base].append(obj)

    labels_by_id: Dict[str, str] = {}
    for base, items in grouped.items():
        ordered = sorted(items, key=lambda o: (o.get("bbox_xyxy", [0, 0, 0, 0])[0], o.get("id", "")))
        for idx, obj in enumerate(ordered, start=1):
            position = position_descriptor(obj.get("bbox_xyxy", [0, 0, 0, 0]), image_size)
            if len(ordered) > 1:
                label = f"{ordinal_word(idx)} {base}"
                if position and position != "center":
                    label = f"{label} ({position})"
            else:
                label = base
                if position and position != "center":
                    label = f"{position} {label}"
            labels_by_id[obj.get("id", str(idx))] = label

    updated: List[Dict[str, Any]] = []
    for obj in objects:
        if obj.get("semantic_kind") == "text_component":
            updated.append(dict(obj))
            continue
        label = labels_by_id.get(obj.get("id", ""))
        if label:
            updated_obj = dict(obj)
            updated_obj["label"] = label
            updated.append(updated_obj)
        else:
            updated.append(dict(obj))
    return updated
