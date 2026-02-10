from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import uuid
from typing import Any, Dict, List

import gradio as gr
import numpy as np
import requests
from PIL import Image, ImageDraw

API_BASE = os.getenv("COLAB_API_BASE", "http://127.0.0.1:5700")
VISION_SOC_URL = os.getenv("VISION_SOC_URL", "http://127.0.0.1:5050")

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
    return image, [], [], manager_dropdown_update([]), "Cleared selected objects."


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


def process_objects(
    image: Image.Image,
    selected_objects: List[Dict[str, Any]],
    provider: str,
    model: str,
    api_key: str,
    use_z_order: bool,
):
    if image is None:
        return None, None, "Upload an image first.", ""
    if not selected_objects:
        return image, pil_to_data_uri(image), "No selected objects.", ""

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

    payload = {
        "image_data": pil_to_data_uri(image),
        "objects": objects_payload,
        "provider": provider,
        "model": model or None,
        "api_key": api_key or None,
        "use_z_order": use_z_order,
    }

    try:
        result = api_post("/inpaint-sequential", payload)
    except requests.RequestException as exc:
        return image, None, f"Process failed: {_api_error_text(exc)}", ""
    bg_data = result["final_background"]
    bg_image = data_uri_to_pil(bg_data)
    status = f"Processed {len(result.get('layers', []))} object(s) in {result.get('total_processing_time', 0):.2f}s"
    return bg_image, bg_data, status, json.dumps(result.get("z_order_used", []), indent=2)


def trace_and_assemble(
    image: Image.Image,
    processed_background_data: str,
    selected_objects: List[Dict[str, Any]],
    upscale_mode: str,
    upscale_quality: str,
    split_text_layers: bool,
):
    if image is None:
        return "", "", None, "Upload an image first."

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
        return "", "", None, f"Trace failed: {_api_error_text(exc)}"

    assembled_layers = []
    total_paths = 0
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
        total_paths += int(layer.get("stats", {}).get("path_count", 0))

    w, h = image.size
    try:
        assembled = api_post("/assemble", {"width": w, "height": h, "layers": assembled_layers, "optimize": False})
    except requests.RequestException as exc:
        return "", "", None, f"Assemble failed: {_api_error_text(exc)}"
    svg_text = assembled["svgText"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".svg") as f:
        f.write(svg_text.encode("utf-8"))
        svg_file = f.name

    metadata = f"Layers: {len(assembled_layers)} | Paths: {total_paths}"
    preview_html = f'<div style="border:1px solid #ccc; background:white; min-height:300px">{svg_text}</div>'
    return preview_html, svg_text, svg_file, metadata


def clear_all():
    return None, None, [], [], manager_dropdown_update([]), "Cleared.", "", "", "", None, "", None


with gr.Blocks(title="SVG Repair Colab Demo") as demo:
    gr.Markdown("## SVG Repair Colab Demo")

    selected_objects_state = gr.State([])
    processed_background_state = gr.State(None)

    with gr.Row():
        input_image = gr.Image(type="pil", label="Original")

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
        process_button = gr.Button("Process Objects")
        clear_button = gr.Button("Clear")

    inpaint_preview_image = gr.Image(type="pil", label="Inpaint Preview (Processed Background)")
    z_order_box = gr.Code(label="Z-Order Used", language="json")

    with gr.Row():
        trace_button = gr.Button("Trace + Assemble SVG")
    svg_preview = gr.HTML(label="SVG Preview")
    svg_code = gr.Code(label="SVG Code", language="xml")
    metadata = gr.Textbox(label="Metadata", interactive=False)
    download_svg = gr.File(label="Download SVG")

    detect_button.click(
        fn=add_detected_objects,
        inputs=[input_image, prompt_text, detect_method, min_score, max_results, selected_objects_state],
        outputs=[input_image, selected_objects_state, objects_table, object_selector, status_text],
    )
    prompt_text.submit(
        fn=add_detected_objects,
        inputs=[input_image, prompt_text, detect_method, min_score, max_results, selected_objects_state],
        outputs=[input_image, selected_objects_state, objects_table, object_selector, status_text],
    )

    add_manual_btn.click(
        fn=add_manual_box,
        inputs=[input_image, mx1, my1, mx2, my2, mlabel, selected_objects_state],
        outputs=[input_image, selected_objects_state, objects_table, object_selector, status_text],
    )

    remove_object_button.click(
        fn=remove_selected_object,
        inputs=[input_image, selected_objects_state, object_selector],
        outputs=[input_image, selected_objects_state, objects_table, object_selector, status_text],
    )

    relabel_object_button.click(
        fn=relabel_selected_object,
        inputs=[input_image, selected_objects_state, object_selector, relabel_text],
        outputs=[input_image, selected_objects_state, objects_table, object_selector, status_text],
    )

    clear_selected_button.click(
        fn=clear_selected_objects,
        inputs=[input_image],
        outputs=[input_image, selected_objects_state, objects_table, object_selector, status_text],
    )

    process_button.click(
        fn=process_objects,
        inputs=[input_image, selected_objects_state, provider, model, api_key, use_z_order],
        outputs=[inpaint_preview_image, processed_background_state, status_text, z_order_box],
    )

    trace_button.click(
        fn=trace_and_assemble,
        inputs=[input_image, processed_background_state, selected_objects_state, upscale_mode, upscale_quality, split_text_layers],
        outputs=[svg_preview, svg_code, download_svg, metadata],
    )

    clear_button.click(
        fn=clear_all,
        outputs=[
            input_image,
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
        ],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
