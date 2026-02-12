from __future__ import annotations

import base64
import html
import io
import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter

API_BASE = os.getenv("COLAB_API_BASE", "http://127.0.0.1:5700")
VISION_SOC_URL = os.getenv("VISION_SOC_URL", "http://127.0.0.1:5050")
_PREPARED_INPAINT_CACHE: Dict[str, Dict[str, Any]] = {}
STATE_DIR = Path(os.getenv("COLAB_STATE_DIR", ".dev_state"))
STATE_INDEX = STATE_DIR / "states_index.json"

try:
    import cv2
except Exception:
    cv2 = None


def pil_to_data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def data_uri_to_pil(data_uri: str) -> Image.Image:
    b64 = data_uri.split(",", 1)[1] if "," in data_uri else data_uri
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")


def _image_to_data_uri(image: Any) -> Optional[str]:
    if image is None:
        return None
    if isinstance(image, str):
        if image.startswith("data:image/"):
            return image
        if os.path.exists(image):
            try:
                return pil_to_data_uri(Image.open(image).convert("RGBA"))
            except Exception:
                return None
        return None
    if isinstance(image, Image.Image):
        return pil_to_data_uri(image.convert("RGBA"))
    if isinstance(image, np.ndarray):
        try:
            return pil_to_data_uri(Image.fromarray(image).convert("RGBA"))
        except Exception:
            return None
    return None


def _data_uri_to_image(value: Optional[str]) -> Optional[Image.Image]:
    if not value:
        return None
    try:
        return data_uri_to_pil(value)
    except Exception:
        return None


def _gallery_to_serializable(gallery: Any) -> List[Dict[str, str]]:
    if not isinstance(gallery, list):
        return []
    out: List[Dict[str, str]] = []
    for item in gallery:
        caption = ""
        image_part = None
        if isinstance(item, (tuple, list)) and item:
            image_part = item[0]
            if len(item) > 1 and isinstance(item[1], str):
                caption = item[1]
        elif isinstance(item, dict):
            image_part = item.get("image") or item.get("value")
            if isinstance(item.get("caption"), str):
                caption = item["caption"]
        else:
            image_part = item
        image_data = _image_to_data_uri(image_part)
        if image_data:
            out.append({"image_data": image_data, "caption": caption})
    return out


def _gallery_from_serialized(items: Any) -> List[tuple[Image.Image, str]]:
    if not isinstance(items, list):
        return []
    out: List[tuple[Image.Image, str]] = []
    for row in items:
        if not isinstance(row, dict):
            continue
        image = _data_uri_to_image(row.get("image_data"))
        if image is None:
            continue
        out.append((image, str(row.get("caption", ""))))
    return out


def _ensure_state_store() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if not STATE_INDEX.exists():
        STATE_INDEX.write_text("[]", encoding="utf-8")


def _load_state_index() -> List[Dict[str, Any]]:
    _ensure_state_store()
    try:
        data = json.loads(STATE_INDEX.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_state_index(rows: List[Dict[str, Any]]) -> None:
    _ensure_state_store()
    STATE_INDEX.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _state_choices() -> List[str]:
    rows = _load_state_index()
    choices = []
    for row in sorted(rows, key=lambda r: r.get("saved_at", ""), reverse=True):
        sid = row.get("id", "")
        name = row.get("name", "state")
        ts = row.get("saved_at", "")
        choices.append(f"{sid} | {name} | {ts}")
    return choices


def _state_id_from_choice(choice: str) -> str:
    if not choice:
        return ""
    return choice.split(" | ", 1)[0].strip()


def _svg_to_temp_file(svg_text: str) -> Optional[str]:
    if not svg_text:
        return None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".svg") as f:
        f.write(svg_text.encode("utf-8"))
        return f.name


def _api_error_text(exc: requests.RequestException) -> str:
    msg = str(exc)
    if isinstance(exc, requests.ConnectionError) or "Connection refused" in msg or "Failed to establish a new connection" in msg:
        return (
            f"API unreachable at {API_BASE}. Start the API first: "
            "python -m uvicorn colab_demo.services.api:app --host 0.0.0.0 --port 5700"
        )

    response = getattr(exc, "response", None)
    if response is None:
        return msg
    try:
        payload = response.json()
        if isinstance(payload, dict):
            detail = payload.get("detail")
            if detail:
                return str(detail)
            error = payload.get("error")
            if error:
                return str(error)
    except Exception:
        pass
    text = response.text.strip()
    if text:
        return f"HTTP {response.status_code}: {text[:300]}"
    return f"HTTP {response.status_code}"


def api_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(f"{API_BASE}{path}", json=payload, timeout=180)
    r.raise_for_status()
    return r.json()


def image_to_overlay(image: Image.Image, objects: List[Dict[str, Any]]) -> Image.Image:
    out = image.convert("RGBA").copy()
    # Blend segmentation masks first so bounding boxes remain clearly visible on top.
    for obj in objects:
        mask_data = obj.get("mask_data")
        if not mask_data:
            continue
        try:
            mask_img = data_uri_to_pil(mask_data)
            alpha = mask_img.getchannel("A")
            tint = Image.new("RGBA", out.size, (255, 80, 80, 95))
            out = Image.composite(tint, out, alpha)
        except Exception:
            # Keep rendering even if one mask is malformed.
            continue

    draw = ImageDraw.Draw(out)
    for i, obj in enumerate(objects):
        bbox = obj.get("bbox_xyxy") or [0, 0, 0, 0]
        x1, y1, x2, y2 = bbox
        color = (255, 80, 80, 255)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        draw.text((x1 + 4, max(0, y1 - 16)), f"{i+1}:{obj.get('label','obj')}", fill=color)
    return out


def objects_to_table(objects: List[Dict[str, Any]]) -> List[List[Any]]:
    rows = []
    for obj in objects:
        x1, y1, x2, y2 = obj.get("bbox_xyxy", [0, 0, 0, 0])
        rows.append([obj["id"], obj.get("label", ""), round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)])
    return rows


def object_choices(objects: List[Dict[str, Any]]) -> List[str]:
    return [f"{obj['id']} | {obj.get('label', '')}" for obj in objects]


def object_id_from_choice(choice: str) -> str:
    if not choice:
        return ""
    return choice.split(" | ", 1)[0].strip()


def manager_dropdown_update(objects: List[Dict[str, Any]], preferred: str | None = None):
    choices = object_choices(objects)
    value = None
    if preferred and preferred in choices:
        value = preferred
    elif choices:
        value = choices[0]
    return gr.update(choices=choices, value=value)


def add_detected_objects(
    image: Image.Image,
    prompt: str,
    method: str,
    min_score: float,
    max_results: int,
    selected_objects: List[Dict[str, Any]],
):
    if image is None:
        return None, selected_objects, objects_to_table(selected_objects), manager_dropdown_update(selected_objects), "Upload an image first."
    if not prompt.strip():
        return image, selected_objects, objects_to_table(selected_objects), manager_dropdown_update(selected_objects), "Enter a prompt first."

    if method == "gdino":
        try:
            ensure = api_post("/models/gdino/ensure", {"auto_download": True, "force_reload": False})
            if not ensure.get("loaded", False):
                return (
                    image,
                    selected_objects,
                    objects_to_table(selected_objects),
                    manager_dropdown_update(selected_objects),
                    f"GDINO setup failed: {ensure.get('error') or 'unknown error'}",
                )
        except requests.RequestException as exc:
            return (
                image,
                selected_objects,
                objects_to_table(selected_objects),
                manager_dropdown_update(selected_objects),
                f"GDINO setup failed: {_api_error_text(exc)}",
            )

    image_data = pil_to_data_uri(image)
    try:
        result = api_post(
            "/detect",
            {
                "image_data": image_data,
                "text": prompt,
                "method": method,
                "min_score": min_score,
                "max_results": int(max_results),
                "return_masks": True,
            },
        )
    except requests.RequestException as exc:
        return image, selected_objects, objects_to_table(selected_objects), manager_dropdown_update(selected_objects), f"Detect failed: {_api_error_text(exc)}"

    added = 0
    for box_item in result.get("boxes", []):
        x, y, w, h = box_item["box"]
        selected_objects.append(
            {
                "id": str(uuid.uuid4())[:8],
                "label": box_item.get("label") or prompt,
                "bbox_xyxy": [x, y, x + w, y + h],
                "mask_data": box_item.get("mask_png"),
            }
        )
        added += 1

    overlay = image_to_overlay(image, selected_objects)
    return overlay, selected_objects, objects_to_table(selected_objects), manager_dropdown_update(selected_objects), f"Added {added} object(s)."


def add_manual_box(
    image: Image.Image,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    label: str,
    selected_objects: List[Dict[str, Any]],
):
    if image is None:
        return None, selected_objects, objects_to_table(selected_objects), manager_dropdown_update(selected_objects), "Upload an image first."
    if not label.strip():
        return image, selected_objects, objects_to_table(selected_objects), manager_dropdown_update(selected_objects), "Enter label for manual box."

    payload = {
        "image_data": pil_to_data_uri(image),
        "box": [x1, y1, x2, y2],
        "label": label,
    }
    try:
        result = api_post("/segment-manual-box", payload)
    except requests.RequestException as exc:
        return image, selected_objects, objects_to_table(selected_objects), manager_dropdown_update(selected_objects), f"Manual segment failed: {_api_error_text(exc)}"

    selected_objects.append(
        {
            "id": str(uuid.uuid4())[:8],
            "label": result["label"],
            "bbox_xyxy": result["box"],
            "mask_data": result["mask_png"],
        }
    )
    overlay = image_to_overlay(image, selected_objects)
    return overlay, selected_objects, objects_to_table(selected_objects), manager_dropdown_update(selected_objects), "Manual segment added."


def remove_selected_object(
    image: Image.Image,
    selected_objects: List[Dict[str, Any]],
    object_choice: str,
):
    if image is None:
        return None, selected_objects, objects_to_table(selected_objects), manager_dropdown_update(selected_objects), "Upload an image first."
    if not selected_objects:
        return image, selected_objects, [], manager_dropdown_update(selected_objects), "No selected objects."
    object_id = object_id_from_choice(object_choice)
    if not object_id:
        return image, selected_objects, objects_to_table(selected_objects), manager_dropdown_update(selected_objects), "Select an object to remove."

    before = len(selected_objects)
    selected_objects = [obj for obj in selected_objects if obj.get("id") != object_id]
    removed = before - len(selected_objects)
    overlay = image_to_overlay(image, selected_objects)
    message = "Object removed." if removed else "Object not found."
    return overlay, selected_objects, objects_to_table(selected_objects), manager_dropdown_update(selected_objects), message


def relabel_selected_object(
    image: Image.Image,
    selected_objects: List[Dict[str, Any]],
    object_choice: str,
    new_label: str,
):
    if image is None:
        return None, selected_objects, objects_to_table(selected_objects), manager_dropdown_update(selected_objects), "Upload an image first."
    if not selected_objects:
        return image, selected_objects, [], manager_dropdown_update(selected_objects), "No selected objects."
    if not new_label.strip():
        return image, selected_objects, objects_to_table(selected_objects), manager_dropdown_update(selected_objects), "Enter a new label."

    object_id = object_id_from_choice(object_choice)
    if not object_id:
        return image, selected_objects, objects_to_table(selected_objects), manager_dropdown_update(selected_objects), "Select an object to relabel."

    updated = False
    for obj in selected_objects:
        if obj.get("id") == object_id:
            obj["label"] = new_label.strip()
            updated = True
            break

    overlay = image_to_overlay(image, selected_objects)
    preferred = f"{object_id} | {new_label.strip()}" if updated else object_choice
    message = "Object relabeled." if updated else "Object not found."
    return overlay, selected_objects, objects_to_table(selected_objects), manager_dropdown_update(selected_objects, preferred), message


def clear_selected_objects(image: Image.Image):
    if image is None:
        return None, [], [], manager_dropdown_update([]), "Cleared selected objects."
    return image.convert("RGBA"), [], [], manager_dropdown_update([]), "Cleared selected objects."


def _should_upscale(label: str, upscale_mode: str) -> bool:
    if upscale_mode == "none":
        return False
    if upscale_mode == "all_objects":
        return True
    text_keywords = ["text", "logo", "title", "label", "word", "letter"]
    return any(k in label.lower() for k in text_keywords)


def _is_text_like_label(label: str) -> bool:
    lower = (label or "").lower()
    return any(k in lower for k in ["text", "logo", "word", "letter", "title", "caption", "label", "watermark"])


def _split_text_object_layers(
    obj: Dict[str, Any],
    image_size: tuple[int, int],
    min_component_area: int = 12,
) -> List[Dict[str, Any]]:
    if not obj.get("mask_data") or not _is_text_like_label(obj.get("label", "")) or cv2 is None:
        return [obj]

    width, height = image_size
    try:
        mask_img = data_uri_to_pil(obj["mask_data"])
    except Exception:
        return [obj]
    if mask_img.size != (width, height):
        mask_img = mask_img.resize((width, height), Image.Resampling.LANCZOS)

    alpha = np.array(mask_img.getchannel("A"), dtype=np.uint8)
    binary = (alpha > 0).astype(np.uint8)
    if binary.max() == 0:
        return [obj]

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels <= 2:
        return [obj]

    components = []
    for idx in range(1, n_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue

        left = int(stats[idx, cv2.CC_STAT_LEFT])
        top = int(stats[idx, cv2.CC_STAT_TOP])
        comp_w = int(stats[idx, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[idx, cv2.CC_STAT_HEIGHT])

        component = np.zeros((height, width), dtype=np.uint8)
        component[labels == idx] = 255
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[:, :, 3] = component
        comp_mask = pil_to_data_uri(Image.fromarray(rgba, mode="RGBA"))

        components.append(
            {
                "id": f"{obj['id']}-c{len(components) + 1}",
                "label": f"{obj.get('label', 'text')}_{len(components) + 1}",
                "bbox_xyxy": [left, top, left + comp_w, top + comp_h],
                "mask_data": comp_mask,
            }
        )

    if not components:
        return [obj]

    components.sort(key=lambda item: item["bbox_xyxy"][0])
    return components


def _expand_objects_for_trace(
    selected_objects: List[Dict[str, Any]],
    image_size: tuple[int, int],
    split_text_layers: bool,
) -> List[Dict[str, Any]]:
    if not split_text_layers:
        return selected_objects
    expanded: List[Dict[str, Any]] = []
    for obj in selected_objects:
        expanded.extend(_split_text_object_layers(obj, image_size))
    return expanded


def _upscale_data_uri(image_data: str, mode: str) -> str:
    try:
        b64 = image_data.split(",", 1)[1] if "," in image_data else image_data
        raw = base64.b64decode(b64)
        files = {"file": ("layer.png", raw, "image/png")}
        resp = requests.post(f"{VISION_SOC_URL}/process", files=files, data={"mode": mode}, timeout=180)
        if resp.status_code >= 400:
            return image_data
        try:
            j = resp.json()
            out = j.get("output_image") or j.get("processed_image")
            if isinstance(out, str) and out.startswith("data:image"):
                return out
        except ValueError:
            pass
        return "data:image/jpeg;base64," + base64.b64encode(resp.content).decode("ascii")
    except Exception:
        return image_data


def _dilate_mask_data_uri(mask_data: str, pixels: int) -> str:
    if pixels <= 0:
        return mask_data
    mask = data_uri_to_pil(mask_data)
    alpha = np.array(mask.getchannel("A"))
    binary = (alpha > 0).astype(np.uint8) * 255
    m = Image.fromarray(binary, mode="L")
    m = m.filter(ImageFilter.MaxFilter(size=2 * pixels + 1))
    rgba = np.zeros((m.height, m.width, 4), dtype=np.uint8)
    rgba[:, :, 3] = np.array(m)
    return pil_to_data_uri(Image.fromarray(rgba, mode="RGBA"))


def _mask_alpha(mask_data: str, size: tuple[int, int]) -> np.ndarray:
    mask = data_uri_to_pil(mask_data)
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.LANCZOS)
    return np.array(mask.getchannel("A"), dtype=np.uint8)


def _holed_preview(source: Image.Image, alpha: np.ndarray) -> Image.Image:
    arr = np.array(source.convert("RGBA"))
    hole = arr.copy()
    region = alpha > 0
    hole[region, 0] = 255
    hole[region, 1] = 255
    hole[region, 2] = 255
    hole[region, 3] = 255
    return Image.fromarray(hole, mode="RGBA")


def _build_inpaint_debug_gallery(
    original_image: Image.Image,
    selected_objects: List[Dict[str, Any]],
    ordered_ids: List[str],
    layers_map: Optional[Dict[str, str]] = None,
) -> List[tuple[Image.Image, str]]:
    gallery: List[tuple[Image.Image, str]] = []
    if original_image is None or not selected_objects:
        return gallery

    object_map = {obj["id"]: obj for obj in selected_objects}
    z_ids = ordered_ids or [obj["id"] for obj in selected_objects]
    layers_map = layers_map or {}

    current = original_image.convert("RGBA")
    gallery.append((current, "Step 0: Source image"))

    for step, object_id in enumerate(z_ids, start=1):
        obj = object_map.get(object_id)
        if not obj or not obj.get("mask_data"):
            continue

        label = obj.get("label", object_id)
        mask_data = obj["mask_data"]
        if _is_text_like_label(label):
            mask_data = _dilate_mask_data_uri(mask_data, 6)

        alpha = _mask_alpha(mask_data, current.size)
        mask_rgba = np.zeros((current.height, current.width, 4), dtype=np.uint8)
        mask_rgba[:, :, 3] = alpha
        mask_img = Image.fromarray(mask_rgba, mode="RGBA")
        hole_img = _holed_preview(current, alpha)

        gallery.append((current.copy(), f"Step {step} input: {label}"))
        gallery.append((mask_img, f"Step {step} mask: {label}"))
        gallery.append((hole_img, f"Step {step} holed preview: {label}"))

        inpainted_data = layers_map.get(object_id)
        if inpainted_data:
            try:
                inpainted = data_uri_to_pil(inpainted_data)
                gallery.append((inpainted, f"Step {step} output: {label}"))
                current = inpainted.convert("RGBA")
            except Exception:
                current = hole_img
        else:
            current = hole_img

    return gallery


def _objects_payload(selected_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    objects_payload = []
    for obj in selected_objects:
        objects_payload.append(
            {
                "id": obj["id"],
                "label": obj["label"],
                "bbox": obj["bbox_xyxy"],
                "mask_data": obj["mask_data"],
            }
        )
    return objects_payload


def _ordered_ids_from_payload(image: Image.Image, objects_payload: List[Dict[str, Any]], use_z_order: bool) -> tuple[List[str], str]:
    if not use_z_order:
        z_rows = [
            {"id": obj["id"], "label": obj["label"], "z_score": 0.0, "rank": i + 1, "reasoning": "user order"}
            for i, obj in enumerate(objects_payload)
        ]
        return [obj["id"] for obj in objects_payload], json.dumps(z_rows, indent=2)

    z_result = api_post(
        "/compute-z-order",
        {
            "image_width": image.width,
            "image_height": image.height,
            "objects": objects_payload,
        },
    )
    z_rows = z_result.get("ordered_objects", [])
    ordered_ids = [z.get("id") for z in z_rows if isinstance(z, dict) and z.get("id")]
    if not ordered_ids:
        ordered_ids = [obj["id"] for obj in objects_payload]
    return ordered_ids, json.dumps(z_rows, indent=2)


def prepare_inpaint(
    image: Image.Image,
    selected_objects: List[Dict[str, Any]],
    provider: str,
    model: str,
    api_key: str,
    use_z_order: bool,
):
    if image is None:
        return None, None, "Upload an image first.", "", [], None
    if not selected_objects:
        return None, None, "No selected objects.", "", [], None

    objects_payload = _objects_payload(selected_objects)
    try:
        ordered_ids, z_json = _ordered_ids_from_payload(image, objects_payload, use_z_order)
    except requests.RequestException as exc:
        return None, None, f"Prepare failed: {_api_error_text(exc)}", "", [], None

    debug_gallery = _build_inpaint_debug_gallery(image, selected_objects, ordered_ids, None)
    payload = {
        "image_data": pil_to_data_uri(image),
        "objects": objects_payload,
        "provider": provider,
        "model": model or None,
        "api_key": api_key or None,
        "use_z_order": use_z_order,
    }
    cache_key = str(uuid.uuid4())
    _PREPARED_INPAINT_CACHE[cache_key] = {
        "payload": payload,
        "ordered_ids": ordered_ids,
        "selected_objects": selected_objects,
        "original_image_data": pil_to_data_uri(image),
    }
    return None, None, "Inputs prepared. Review Z-order/debug frames, then click Run Inpaint.", z_json, debug_gallery, cache_key


def run_inpaint(prepared_state_key: Optional[str]):
    if not prepared_state_key:
        return None, None, "Prepare inputs first, then click Run Inpaint.", "", []
    prepared_state = _PREPARED_INPAINT_CACHE.get(prepared_state_key)
    if not prepared_state or not prepared_state.get("payload"):
        return None, None, "Prepare inputs first, then click Run Inpaint.", "", []
    try:
        result = api_post("/inpaint-sequential", prepared_state["payload"])
    except requests.RequestException as exc:
        return None, None, f"Process failed: {_api_error_text(exc)}", "", []

    bg_data = result["final_background"]
    bg_image = data_uri_to_pil(bg_data)
    layers_map: Dict[str, str] = {}
    for layer in result.get("layers", []):
        if not isinstance(layer, dict):
            continue
        object_id = layer.get("object_id")
        inpainted = layer.get("inpainted_image")
        if object_id and inpainted:
            layers_map[object_id] = inpainted

    try:
        original_image = data_uri_to_pil(prepared_state.get("original_image_data", ""))
    except Exception:
        original_image = bg_image

    debug_gallery = _build_inpaint_debug_gallery(
        original_image=original_image,
        selected_objects=prepared_state.get("selected_objects", []),
        ordered_ids=prepared_state.get("ordered_ids", []),
        layers_map=layers_map,
    )
    status = f"Processed {len(result.get('layers', []))} object(s) in {result.get('total_processing_time', 0):.2f}s"
    return bg_image, bg_data, status, json.dumps(result.get("z_order_used", []), indent=2), debug_gallery


def refresh_saved_states():
    choices = _state_choices()
    value = choices[0] if choices else None
    return gr.update(choices=choices, value=value), "Saved states list refreshed."


def _snapshot_data_from_inputs(
    input_image: Image.Image,
    detect_preview_image: Image.Image,
    selected_objects: List[Dict[str, Any]],
    processed_background_data: Optional[str],
    prepared_inpaint_key: Optional[str],
    inpaint_preview_image: Image.Image,
    z_order_text: str,
    inpaint_debug_gallery: Any,
    svg_preview_html: str,
    svg_code_text: str,
    metadata_text: str,
    prompt: str,
    detect_method: str,
    min_score: float,
    max_results: int,
    mx1: float,
    my1: float,
    mx2: float,
    my2: float,
    mlabel: str,
    provider: str,
    model: str,
    api_key: str,
    use_z_order: bool,
    upscale_mode: str,
    upscale_quality: str,
    split_text_layers: bool,
    svg_code_mode: str,
) -> Dict[str, Any]:
    prepared_payload = _PREPARED_INPAINT_CACHE.get(prepared_inpaint_key or "", None)
    return {
        "input_image": _image_to_data_uri(input_image),
        "detect_preview_image": _image_to_data_uri(detect_preview_image),
        "selected_objects": selected_objects or [],
        "processed_background_data": processed_background_data,
        "prepared_payload": prepared_payload,
        "inpaint_preview_image": _image_to_data_uri(inpaint_preview_image),
        "z_order_text": z_order_text or "",
        "inpaint_debug_gallery": _gallery_to_serializable(inpaint_debug_gallery),
        "svg_preview_html": svg_preview_html or "",
        "svg_code_text": svg_code_text or "",
        "metadata_text": metadata_text or "",
        "controls": {
            "prompt": prompt,
            "detect_method": detect_method,
            "min_score": min_score,
            "max_results": max_results,
            "mx1": mx1,
            "my1": my1,
            "mx2": mx2,
            "my2": my2,
            "mlabel": mlabel,
            "provider": provider,
            "model": model,
            "api_key": api_key,
            "use_z_order": use_z_order,
            "upscale_mode": upscale_mode,
            "upscale_quality": upscale_quality,
            "split_text_layers": split_text_layers,
            "svg_code_mode": svg_code_mode,
        },
    }


def save_current_state(
    state_name: str,
    input_image: Image.Image,
    detect_preview_image: Image.Image,
    selected_objects: List[Dict[str, Any]],
    processed_background_data: Optional[str],
    prepared_inpaint_key: Optional[str],
    inpaint_preview_image: Image.Image,
    z_order_text: str,
    inpaint_debug_gallery: Any,
    svg_preview_html: str,
    svg_code_text: str,
    metadata_text: str,
    prompt: str,
    detect_method: str,
    min_score: float,
    max_results: int,
    mx1: float,
    my1: float,
    mx2: float,
    my2: float,
    mlabel: str,
    provider: str,
    model: str,
    api_key: str,
    use_z_order: bool,
    upscale_mode: str,
    upscale_quality: str,
    split_text_layers: bool,
    svg_code_mode: str,
):
    _ensure_state_store()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    snapshot_id = str(uuid.uuid4())[:8]
    payload = {
        "schema_version": 1,
        "id": snapshot_id,
        "name": (state_name or "").strip() or f"state-{snapshot_id}",
        "saved_at": now,
        "data": _snapshot_data_from_inputs(
            input_image,
            detect_preview_image,
            selected_objects,
            processed_background_data,
            prepared_inpaint_key,
            inpaint_preview_image,
            z_order_text,
            inpaint_debug_gallery,
            svg_preview_html,
            svg_code_text,
            metadata_text,
            prompt,
            detect_method,
            min_score,
            max_results,
            mx1,
            my1,
            mx2,
            my2,
            mlabel,
            provider,
            model,
            api_key,
            use_z_order,
            upscale_mode,
            upscale_quality,
            split_text_layers,
            svg_code_mode,
        ),
    }
    snap_path = STATE_DIR / f"{snapshot_id}.json"
    snap_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    index_rows = _load_state_index()
    index_rows = [row for row in index_rows if row.get("id") != snapshot_id]
    index_rows.append({"id": snapshot_id, "name": payload["name"], "saved_at": now})
    _save_state_index(index_rows)

    choices = _state_choices()
    selected = next((c for c in choices if c.startswith(snapshot_id + " | ")), choices[0] if choices else None)
    return (
        f"State saved: {payload['name']} ({snapshot_id})",
        gr.update(choices=choices, value=selected),
        payload["name"],
    )


def delete_selected_state(choice: str):
    snapshot_id = _state_id_from_choice(choice)
    if not snapshot_id:
        return "Choose a saved state to delete.", gr.update(), gr.update()

    snap_path = STATE_DIR / f"{snapshot_id}.json"
    if snap_path.exists():
        try:
            snap_path.unlink()
        except Exception as exc:
            return f"Failed to delete state {snapshot_id}: {exc}", gr.update(), gr.update()

    index_rows = [row for row in _load_state_index() if row.get("id") != snapshot_id]
    _save_state_index(index_rows)
    choices = _state_choices()
    value = choices[0] if choices else None
    return (
        f"Deleted state: {snapshot_id}",
        gr.update(choices=choices, value=value),
        "",
    )


def overwrite_selected_state(
    choice: str,
    state_name: str,
    input_image: Image.Image,
    detect_preview_image: Image.Image,
    selected_objects: List[Dict[str, Any]],
    processed_background_data: Optional[str],
    prepared_inpaint_key: Optional[str],
    inpaint_preview_image: Image.Image,
    z_order_text: str,
    inpaint_debug_gallery: Any,
    svg_preview_html: str,
    svg_code_text: str,
    metadata_text: str,
    prompt: str,
    detect_method: str,
    min_score: float,
    max_results: int,
    mx1: float,
    my1: float,
    mx2: float,
    my2: float,
    mlabel: str,
    provider: str,
    model: str,
    api_key: str,
    use_z_order: bool,
    upscale_mode: str,
    upscale_quality: str,
    split_text_layers: bool,
    svg_code_mode: str,
):
    snapshot_id = _state_id_from_choice(choice)
    if not snapshot_id:
        return "Choose a saved state to overwrite.", gr.update(), gr.update()

    _ensure_state_store()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    existing_name = "state"
    for row in _load_state_index():
        if row.get("id") == snapshot_id:
            existing_name = row.get("name", existing_name)
            break
    final_name = (state_name or "").strip() or existing_name
    payload = {
        "schema_version": 1,
        "id": snapshot_id,
        "name": final_name,
        "saved_at": now,
        "data": _snapshot_data_from_inputs(
            input_image,
            detect_preview_image,
            selected_objects,
            processed_background_data,
            prepared_inpaint_key,
            inpaint_preview_image,
            z_order_text,
            inpaint_debug_gallery,
            svg_preview_html,
            svg_code_text,
            metadata_text,
            prompt,
            detect_method,
            min_score,
            max_results,
            mx1,
            my1,
            mx2,
            my2,
            mlabel,
            provider,
            model,
            api_key,
            use_z_order,
            upscale_mode,
            upscale_quality,
            split_text_layers,
            svg_code_mode,
        ),
    }
    (STATE_DIR / f"{snapshot_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    index_rows = [row for row in _load_state_index() if row.get("id") != snapshot_id]
    index_rows.append({"id": snapshot_id, "name": final_name, "saved_at": now})
    _save_state_index(index_rows)

    choices = _state_choices()
    selected = next((c for c in choices if c.startswith(snapshot_id + " | ")), choices[0] if choices else None)
    return (
        f"Overwrote state: {final_name} ({snapshot_id})",
        gr.update(choices=choices, value=selected),
        final_name,
    )


def load_saved_state(choice: str):
    snapshot_id = _state_id_from_choice(choice)
    if not snapshot_id:
        return (
            None, None, [], [], manager_dropdown_update([]), None, None, None, "", [], "", "", None, "",
            "", "yolo26l", 0.3, 5, 10, 10, 200, 200, "manual object", "big-lama", "", "", True, "none",
            "balanced", False, "Hide", "SVG code copy status.", "Choose a saved state first.",
        )

    path = STATE_DIR / f"{snapshot_id}.json"
    if not path.exists():
        return (
            None, None, [], [], manager_dropdown_update([]), None, None, None, "", [], "", "", None, "",
            "", "yolo26l", 0.3, 5, 10, 10, 200, 200, "manual object", "big-lama", "", "", True, "none",
            "balanced", False, "Hide", "SVG code copy status.", f"Saved state not found: {snapshot_id}",
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    data = payload.get("data", {})
    controls = data.get("controls", {})
    selected_objects = data.get("selected_objects", []) if isinstance(data.get("selected_objects"), list) else []

    input_image = _data_uri_to_image(data.get("input_image"))
    detect_preview = _data_uri_to_image(data.get("detect_preview_image"))
    inpaint_preview = _data_uri_to_image(data.get("inpaint_preview_image"))
    if inpaint_preview is None and data.get("processed_background_data"):
        inpaint_preview = _data_uri_to_image(data.get("processed_background_data"))

    prepared_payload = data.get("prepared_payload")
    prepared_key = None
    if isinstance(prepared_payload, dict) and prepared_payload.get("payload"):
        prepared_key = str(uuid.uuid4())
        _PREPARED_INPAINT_CACHE[prepared_key] = prepared_payload

    svg_code_text = str(data.get("svg_code_text", ""))
    download_svg = _svg_to_temp_file(svg_code_text)

    return (
        input_image,
        detect_preview,
        selected_objects,
        objects_to_table(selected_objects),
        manager_dropdown_update(selected_objects),
        data.get("processed_background_data"),
        prepared_key,
        inpaint_preview,
        str(data.get("z_order_text", "")),
        _gallery_from_serialized(data.get("inpaint_debug_gallery")),
        str(data.get("svg_preview_html", "")),
        svg_code_text,
        download_svg,
        str(data.get("metadata_text", "")),
        controls.get("prompt", ""),
        controls.get("detect_method", "yolo26l"),
        float(controls.get("min_score", 0.3)),
        int(controls.get("max_results", 5)),
        float(controls.get("mx1", 10)),
        float(controls.get("my1", 10)),
        float(controls.get("mx2", 200)),
        float(controls.get("my2", 200)),
        controls.get("mlabel", "manual object"),
        controls.get("provider", "big-lama"),
        controls.get("model", ""),
        controls.get("api_key", ""),
        bool(controls.get("use_z_order", True)),
        controls.get("upscale_mode", "none"),
        controls.get("upscale_quality", "balanced"),
        bool(controls.get("split_text_layers", False)),
        controls.get("svg_code_mode", "Hide"),
        "SVG code copy status.",
        f"Loaded state: {payload.get('name', snapshot_id)} ({snapshot_id})",
    )


def trace_and_assemble(
    image: Image.Image,
    processed_background_data: str,
    selected_objects: List[Dict[str, Any]],
    upscale_mode: str,
    upscale_quality: str,
    split_text_layers: bool,
):
    if image is None:
        empty_state = {"width": 0, "height": 0, "layers": []}
        return "", "", None, "Upload an image first.", gr.update(choices=[], value=[]), empty_state

    original_data = pil_to_data_uri(image)
    background_data = processed_background_data or original_data

    layers = [
        {
            "id": "background",
            "label": "background",
            "image_data": background_data,
        }
    ]

    trace_objects = _expand_objects_for_trace(selected_objects, image.size, split_text_layers)

    for obj in trace_objects:
        layer_image_data = obj["mask_data"]
        if _should_upscale(obj["label"], upscale_mode):
            layer_image_data = _upscale_data_uri(layer_image_data, upscale_quality)

        layers.append(
            {
                "id": obj["id"],
                "label": obj["label"],
                "image_data": layer_image_data,
                "source_image_data": original_data,
            }
        )

    try:
        traced = api_post("/trace-batch", {"layers": layers, "options": {"color_mode": "color", "mode": "spline"}})
    except requests.RequestException as exc:
        empty_state = {"width": 0, "height": 0, "layers": []}
        return "", "", None, f"Trace failed: {_api_error_text(exc)}", gr.update(choices=[], value=[]), empty_state

    assembled_layers = []
    total_paths = 0
    layer_path_rows = []
    for z, layer in enumerate(traced.get("layers", [])):
        svg_paths = layer.get("svg_paths", "")
        offset = layer.get("offset")
        if offset:
            tx = offset.get("x", 0)
            ty = offset.get("y", 0)
            svg_paths = f'<g transform="translate({tx} {ty})">{svg_paths}</g>'
        assembled_layers.append(
            {
                "id": layer["id"],
                "label": layer.get("label", ""),
                "svg_paths": svg_paths,
                "z_index": z,
                "hidden": False,
            }
        )
        label = layer.get("label", "")
        path_count = int(layer.get("stats", {}).get("path_count", 0))
        if path_count <= 0:
            path_count = len(re.findall(r"<path\b", svg_paths, re.IGNORECASE))
        total_paths += path_count
        layer_path_rows.append({"id": layer["id"], "label": label, "path_count": path_count, "svg_paths": svg_paths})

    w, h = image.size
    try:
        assembled = api_post("/assemble", {"width": w, "height": h, "layers": assembled_layers, "optimize": False})
    except requests.RequestException as exc:
        empty_state = {"width": 0, "height": 0, "layers": []}
        return "", "", None, f"Assemble failed: {_api_error_text(exc)}", gr.update(choices=[], value=[]), empty_state
    svg_text = assembled["svgText"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".svg") as f:
        f.write(svg_text.encode("utf-8"))
        svg_file = f.name

    metadata_lines = [f"Layers: {len(assembled_layers)} | Total Paths: {total_paths}", "Per-layer paths:"]
    for row in layer_path_rows:
        display_label = row["label"] or row["id"]
        metadata_lines.append(f"- {display_label} [{row['id']}]: {row['path_count']}")
    metadata = "\n".join(metadata_lines)

    choices = [_layer_choice(row) for row in layer_path_rows]
    trace_state = {"width": w, "height": h, "layers": layer_path_rows}
    preview_html = _render_svg_preview(trace_state, choices)
    return preview_html, svg_text, svg_file, metadata, gr.update(choices=choices, value=choices), trace_state


def _layer_choice(row: Dict[str, Any]) -> str:
    label = row.get("label") or row.get("id", "layer")
    return f"{row.get('id', '')} | {label} ({int(row.get('path_count', 0))} paths)"


def _layer_id_from_choice(choice: str) -> str:
    if not choice:
        return ""
    return choice.split(" | ", 1)[0].strip()


def _render_svg_preview(trace_state: Dict[str, Any], selected_choices: List[str]) -> str:
    if not isinstance(trace_state, dict):
        return ""
    width = int(trace_state.get("width") or 0)
    height = int(trace_state.get("height") or 0)
    layers = trace_state.get("layers") or []
    if width <= 0 or height <= 0 or not layers:
        return ""

    visible_ids = {_layer_id_from_choice(choice) for choice in (selected_choices or [])}
    groups = []
    for row in layers:
        layer_id = str(row.get("id", ""))
        if layer_id not in visible_ids:
            continue
        groups.append(str(row.get("svg_paths", "")))

    preview_svg = (
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' "
        f"width='100%' height='100%' preserveAspectRatio='xMidYMid meet'>"
        + "".join(groups)
        + "</svg>"
    )
    return (
        "<div style='border:1px solid #ccc; background:#fff; padding:8px;'>"
        f"<div style='height:300px; overflow:hidden; border:1px solid #ddd;'>{preview_svg}</div>"
        "</div>"
    )


def update_svg_preview_layers(layer_visibility: List[str], trace_preview_state: Dict[str, Any]) -> str:
    return _render_svg_preview(trace_preview_state, layer_visibility or [])


def show_all_layers(trace_preview_state: Dict[str, Any]):
    layers = trace_preview_state.get("layers") if isinstance(trace_preview_state, dict) else []
    choices = [_layer_choice(row) for row in layers]
    return gr.update(value=choices), _render_svg_preview(trace_preview_state, choices)


def hide_all_layers(trace_preview_state: Dict[str, Any]):
    return gr.update(value=[]), _render_svg_preview(trace_preview_state, [])


def reset_layer_controls():
    return gr.update(choices=[], value=[]), {"width": 0, "height": 0, "layers": []}, ""


def clear_all():
    _PREPARED_INPAINT_CACHE.clear()
    choices = _state_choices()
    return (
        None,
        None,
        None,
        [],
        [],
        manager_dropdown_update([]),
        "Cleared.",
        "",
        "",
        "",
        None,
        "",
        None,
        [],
        None,
        "Hide",
        "SVG code copy status.",
        gr.update(choices=choices, value=(choices[0] if choices else None)),
        "",
    )


def toggle_svg_code_visibility(mode: str):
    return gr.update(visible=(mode == "Show"))


with gr.Blocks(title="SVG Repair Colab Demo") as demo:
    gr.Markdown("## SVG Repair Colab Demo")

    selected_objects_state = gr.State([])
    processed_background_state = gr.State(None)
    prepared_inpaint_state = gr.State(None)

    with gr.Row():
        input_image = gr.Image(type="pil", label="Original", height=300)
        detect_preview_image = gr.Image(type="pil", label="Detection Preview (Overlay)", height=300)

    status_text = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        prompt_text = gr.Textbox(label="Prompt", placeholder="e.g. logo text")
        detect_method = gr.Dropdown(["yolo26l", "yolo26x", "gdino"], value="yolo26l", label="Method")
        detect_button = gr.Button("Detect + Add")

    with gr.Accordion("Advanced", open=False):
        with gr.Row():
            min_score = gr.Slider(0.01, 1.0, value=0.3, step=0.01, label="Min Score")
            max_results = gr.Slider(1, 10, value=5, step=1, label="Max Results")

        gr.Markdown("### Manual Box Fallback")
        with gr.Row():
            mx1 = gr.Number(label="x1", value=10)
            my1 = gr.Number(label="y1", value=10)
            mx2 = gr.Number(label="x2", value=200)
            my2 = gr.Number(label="y2", value=200)
            mlabel = gr.Textbox(label="Label", value="manual object")
            add_manual_btn = gr.Button("Add Manual Box")

        gr.Markdown("### Inpaint and Upscale")
        with gr.Row():
            provider = gr.Dropdown(["big-lama", "qualcomm-lama-dilated", "openrouter"], value="big-lama", label="Provider")
            model = gr.Textbox(label="Model (optional)")
            api_key = gr.Textbox(label="OpenRouter API Key (optional)", type="password")
            use_z_order = gr.Checkbox(value=True, label="Use Z-Order")

        with gr.Row():
            upscale_mode = gr.Dropdown(
                ["none", "text_only", "all_objects"],
                value="none",
                label="Upscale mode",
            )
            upscale_quality = gr.Dropdown(["fast", "balanced", "quality"], value="balanced", label="Upscale quality")
            split_text_layers = gr.Checkbox(value=False, label="Split text objects into separate sublayers")

    objects_table = gr.Dataframe(headers=["id", "label", "x1", "y1", "x2", "y2"], label="Selected Objects", interactive=False)
    with gr.Accordion("Manage Selected Objects", open=False):
        object_selector = gr.Dropdown(choices=[], label="Selected Object", interactive=True)
        relabel_text = gr.Textbox(label="New Label", placeholder="e.g. logo text")
        with gr.Row():
            remove_object_button = gr.Button("Remove Object")
            relabel_object_button = gr.Button("Relabel Object")
            clear_selected_button = gr.Button("Clear Selected Objects")

    with gr.Row():
        prepare_inpaint_button = gr.Button("Prepare Inpaint Inputs")
        run_inpaint_button = gr.Button("Run Inpaint")
        save_state_button = gr.Button("Save State")
        clear_button = gr.Button("Clear")

    with gr.Row():
        state_name_input = gr.Textbox(label="State Name (optional)", placeholder="e.g. after detect text")
        saved_states_dropdown = gr.Dropdown(choices=_state_choices(), label="Saved States")
        refresh_states_button = gr.Button("Refresh States")
        load_state_button = gr.Button("Load Selected State")
        overwrite_state_button = gr.Button("Overwrite Selected State")
        delete_state_button = gr.Button("Delete Selected State")

    inpaint_preview_image = gr.Image(type="pil", label="Inpaint Preview (Processed Background)", height=300)
    z_order_box = gr.Code(label="Z-Order Used", language="json")
    inpaint_debug_gallery = gr.Gallery(
        label="Inpaint Debug: Input / Mask / Holed Preview / Output",
        columns=4,
        rows=2,
        height="auto",
        object_fit="contain",
    )

    with gr.Row():
        trace_button = gr.Button("Trace + Assemble SVG")
    svg_preview = gr.HTML(label="SVG Preview")
    metadata = gr.Textbox(label="Metadata", interactive=False, lines=8)
    copy_svg_button = gr.Button("Copy SVG Code")
    copy_svg_status = gr.Textbox(label="Copy Status", interactive=False, value="SVG code copy status.")
    svg_code_mode = gr.Dropdown(["Hide", "Show"], value="Hide", label="SVG Code Display")
    svg_code = gr.Code(label="SVG Code", language="html", visible=False)
    download_svg = gr.File(label="Download SVG")

    input_image.change(
        fn=lambda img: img.convert("RGBA") if img is not None else None,
        inputs=[input_image],
        outputs=[detect_preview_image],
    )

    detect_button.click(
        fn=add_detected_objects,
        inputs=[input_image, prompt_text, detect_method, min_score, max_results, selected_objects_state],
        outputs=[detect_preview_image, selected_objects_state, objects_table, object_selector, status_text],
    )
    prompt_text.submit(
        fn=add_detected_objects,
        inputs=[input_image, prompt_text, detect_method, min_score, max_results, selected_objects_state],
        outputs=[detect_preview_image, selected_objects_state, objects_table, object_selector, status_text],
    )

    add_manual_btn.click(
        fn=add_manual_box,
        inputs=[input_image, mx1, my1, mx2, my2, mlabel, selected_objects_state],
        outputs=[detect_preview_image, selected_objects_state, objects_table, object_selector, status_text],
    )

    remove_object_button.click(
        fn=remove_selected_object,
        inputs=[input_image, selected_objects_state, object_selector],
        outputs=[detect_preview_image, selected_objects_state, objects_table, object_selector, status_text],
    )

    relabel_object_button.click(
        fn=relabel_selected_object,
        inputs=[input_image, selected_objects_state, object_selector, relabel_text],
        outputs=[detect_preview_image, selected_objects_state, objects_table, object_selector, status_text],
    )

    clear_selected_button.click(
        fn=clear_selected_objects,
        inputs=[input_image],
        outputs=[detect_preview_image, selected_objects_state, objects_table, object_selector, status_text],
    )

    prepare_inpaint_button.click(
        fn=prepare_inpaint,
        inputs=[input_image, selected_objects_state, provider, model, api_key, use_z_order],
        outputs=[inpaint_preview_image, processed_background_state, status_text, z_order_box, inpaint_debug_gallery, prepared_inpaint_state],
    )

    run_inpaint_button.click(
        fn=run_inpaint,
        inputs=[prepared_inpaint_state],
        outputs=[inpaint_preview_image, processed_background_state, status_text, z_order_box, inpaint_debug_gallery],
    )

    trace_button.click(
        fn=trace_and_assemble,
        inputs=[input_image, processed_background_state, selected_objects_state, upscale_mode, upscale_quality, split_text_layers],
        outputs=[svg_preview, svg_code, download_svg, metadata],
    )

    svg_code_mode.change(
        fn=toggle_svg_code_visibility,
        inputs=[svg_code_mode],
        outputs=[svg_code],
    )

    copy_svg_button.click(
        fn=None,
        inputs=[svg_code],
        outputs=[copy_svg_status],
        js="(svgText) => { if (!svgText) return 'No SVG code to copy.'; navigator.clipboard.writeText(svgText); return 'SVG code copied to clipboard.'; }",
    )

    save_state_button.click(
        fn=save_current_state,
        inputs=[
            state_name_input,
            input_image,
            detect_preview_image,
            selected_objects_state,
            processed_background_state,
            prepared_inpaint_state,
            inpaint_preview_image,
            z_order_box,
            inpaint_debug_gallery,
            svg_preview,
            svg_code,
            metadata,
            prompt_text,
            detect_method,
            min_score,
            max_results,
            mx1,
            my1,
            mx2,
            my2,
            mlabel,
            provider,
            model,
            api_key,
            use_z_order,
            upscale_mode,
            upscale_quality,
            split_text_layers,
            svg_code_mode,
        ],
        outputs=[status_text, saved_states_dropdown, state_name_input],
    )

    refresh_states_button.click(
        fn=refresh_saved_states,
        outputs=[saved_states_dropdown, status_text],
    )

    overwrite_state_button.click(
        fn=overwrite_selected_state,
        inputs=[
            saved_states_dropdown,
            state_name_input,
            input_image,
            detect_preview_image,
            selected_objects_state,
            processed_background_state,
            prepared_inpaint_state,
            inpaint_preview_image,
            z_order_box,
            inpaint_debug_gallery,
            svg_preview,
            svg_code,
            metadata,
            prompt_text,
            detect_method,
            min_score,
            max_results,
            mx1,
            my1,
            mx2,
            my2,
            mlabel,
            provider,
            model,
            api_key,
            use_z_order,
            upscale_mode,
            upscale_quality,
            split_text_layers,
            svg_code_mode,
        ],
        outputs=[status_text, saved_states_dropdown, state_name_input],
    )

    delete_state_button.click(
        fn=delete_selected_state,
        inputs=[saved_states_dropdown],
        outputs=[status_text, saved_states_dropdown, state_name_input],
    )

    load_state_button.click(
        fn=load_saved_state,
        inputs=[saved_states_dropdown],
        outputs=[
            input_image,
            detect_preview_image,
            selected_objects_state,
            objects_table,
            object_selector,
            processed_background_state,
            prepared_inpaint_state,
            inpaint_preview_image,
            z_order_box,
            inpaint_debug_gallery,
            svg_preview,
            svg_code,
            download_svg,
            metadata,
            prompt_text,
            detect_method,
            min_score,
            max_results,
            mx1,
            my1,
            mx2,
            my2,
            mlabel,
            provider,
            model,
            api_key,
            use_z_order,
            upscale_mode,
            upscale_quality,
            split_text_layers,
            svg_code_mode,
            copy_svg_status,
            status_text,
        ],
    ).then(
        fn=toggle_svg_code_visibility,
        inputs=[svg_code_mode],
        outputs=[svg_code],
    )

    clear_button.click(
        fn=clear_all,
        outputs=[
            input_image,
            detect_preview_image,
            inpaint_preview_image,
            selected_objects_state,
            objects_table,
            object_selector,
            status_text,
            z_order_box,
            svg_preview,
            svg_code,
            download_svg,
            metadata,
            processed_background_state,
            inpaint_debug_gallery,
            prepared_inpaint_state,
            svg_code_mode,
            copy_svg_status,
            saved_states_dropdown,
            state_name_input,
        ],
    ).then(
        fn=toggle_svg_code_visibility,
        inputs=[svg_code_mode],
        outputs=[svg_code],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
