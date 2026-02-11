from __future__ import annotations

import base64
import io
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from fastapi import HTTPException
from PIL import Image, ImageFilter

from .config import settings
from .contracts import (
    AssembleRequest,
    DetectRequest,
    DetectResponse,
    InpaintedLayer,
    InpaintProvider,
    SequentialInpaintRequest,
    SequentialInpaintResponse,
    TraceBatchRequest,
    TraceBatchResponse,
    TraceLayerResponse,
    ZOrderObject,
    ZOrderResponse,
    ZOrderResult,
)

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

SAM = None
YOLOE = None
load_gdino_model = None
gdino_predict = None
load_image = None

try:
    import vtracer
except Exception:
    vtracer = None

try:
    import potrace
except Exception:
    potrace = None


class ColabPipeline:
    def __init__(self) -> None:
        self.sam_model = None
        self.detectors: Dict[str, Any] = {}
        self.gdino_model = None
        self.gdino_predict = None
        self.gdino_load_image = None
        self.gdino_load_error: Optional[str] = None
        self.gdino_config_path: Optional[str] = None
        self.gdino_weights_path: Optional[str] = None
        self._loaded = False

    def _decode_image(self, image_data: str) -> Image.Image:
        if "," in image_data:
            _, b64 = image_data.split(",", 1)
        else:
            b64 = image_data
        try:
            raw = base64.b64decode(b64)
            return Image.open(io.BytesIO(raw)).convert("RGBA")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {exc}")

    def _encode_png(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    def _decode_data_uri_bytes(self, data_uri: str) -> bytes:
        if "," in data_uri:
            _, b64 = data_uri.split(",", 1)
        else:
            b64 = data_uri
        return base64.b64decode(b64)

    def _prepare_image_arrays(self, image: Image.Image) -> tuple:
        image_rgba = np.array(image.convert("RGBA"))
        image_rgb = np.array(image.convert("RGB"))
        image_rgb = np.ascontiguousarray(image_rgb)
        return image_rgba, image_rgb

    def _mask_to_png(self, mask: np.ndarray, image_rgba: np.ndarray) -> str:
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        mask_binary = (mask > 0.5).astype(np.uint8) * 255
        rgba = image_rgba.copy()
        rgba[..., 3] = mask_binary
        output = Image.fromarray(rgba, mode="RGBA")
        return self._encode_png(output)

    def _box_mask(self, box_xyxy: List[float], width: int, height: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
        return mask

    def _box_to_xywh(self, box_xyxy: List[float]) -> List[float]:
        x1, y1, x2, y2 = box_xyxy
        return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

    def _generate_sam_mask(self, box_xyxy: List[float], image_rgb: np.ndarray, image_rgba: np.ndarray) -> Optional[str]:
        if self.sam_model is None:
            return None
        try:
            result = self.sam_model.predict(image_rgb, bboxes=[box_xyxy], verbose=False)
            if result and result[0].masks is not None:
                masks = result[0].masks.data.cpu().numpy()
                if masks.size > 0:
                    return self._mask_to_png(masks[0], image_rgba)
        except Exception:
            return None
        return None

    def _is_text_like_label(self, label: str) -> bool:
        lower = (label or "").lower()
        return any(k in lower for k in ["text", "logo", "word", "letter", "title", "caption", "label", "watermark"])

    def _contrast_mask_from_box(
        self,
        box_xyxy: List[float],
        image_rgb: np.ndarray,
        image_rgba: np.ndarray,
    ) -> Optional[str]:
        h, w = image_rgb.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return None

        region_rgb = image_rgb[y1:y2, x1:x2].astype(np.float32)
        region_alpha = image_rgba[y1:y2, x1:x2, 3]
        if region_rgb.size == 0:
            return None

        border_pixels = np.concatenate(
            [
                region_rgb[0, :, :],
                region_rgb[-1, :, :],
                region_rgb[:, 0, :],
                region_rgb[:, -1, :],
            ],
            axis=0,
        )
        if border_pixels.size == 0:
            return None

        bg_color = np.median(border_pixels, axis=0)
        color_dist = np.linalg.norm(region_rgb - bg_color, axis=2)
        q70 = float(np.percentile(color_dist, 70))
        q85 = float(np.percentile(color_dist, 85))
        threshold = max(18.0, min(64.0, (q70 + q85) / 2.0))
        fg = (color_dist >= threshold) & (region_alpha > 0)

        if cv2 is not None and fg.any():
            fg_u8 = (fg.astype(np.uint8) * 255)
            kernel = np.ones((3, 3), np.uint8)
            fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_OPEN, kernel, iterations=1)
            fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_u8, connectivity=8)
            if n_labels > 1:
                min_area = max(8, int((x2 - x1) * (y2 - y1) * 0.001))
                filtered = np.zeros_like(fg_u8)
                for idx in range(1, n_labels):
                    area = int(stats[idx, cv2.CC_STAT_AREA])
                    if area >= min_area:
                        filtered[labels == idx] = 255
                fg_u8 = filtered
            fg = fg_u8 > 0

        if not fg.any():
            return None

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = fg.astype(np.uint8) * 255
        return self._mask_to_png(mask, image_rgba)

    def _resolve_model_path(self, env_key: str, fallback: str, expected_name: str) -> Optional[str]:
        raw = os.getenv(env_key, fallback)
        p = Path(raw).expanduser()
        if p.exists():
            return str(p)

        filename = p.name if p.name else expected_name
        candidates = [
            Path.cwd() / filename,
            Path("/content/svg-colab/.colab_models") / filename,
            Path("/content/.colab_models") / filename,
            Path("/content") / filename,
        ]
        for cand in candidates:
            if cand.exists():
                return str(cand)
        return None

    def _candidate_groundingdino_paths(self) -> List[Path]:
        paths: List[Path] = []
        env_path = os.getenv("GROUNDINGDINO_LOCAL_PATH", "").strip()
        if env_path:
            p = Path(env_path).expanduser()
            if (p / "groundingdino" / "util" / "inference.py").exists():
                paths.append(p)

        here = Path(__file__).resolve()
        for base in [Path.cwd(), *here.parents]:
            for rel in ("svg-repair/fastapi/GroundingDINO", "GroundingDINO"):
                cand = (base / rel).resolve()
                if (cand / "groundingdino" / "util" / "inference.py").exists():
                    paths.append(cand)

        # Deduplicate while preserving order.
        seen = set()
        out: List[Path] = []
        for p in paths:
            key = str(p)
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
        return out

    def _import_groundingdino_inference(self) -> tuple[Any, Any, Any]:
        try:
            from groundingdino.util.inference import (
                load_image as _load_image,
                load_model as _load_model,
                predict as _predict,
            )
            return _load_model, _predict, _load_image
        except Exception:
            pass

        for candidate in self._candidate_groundingdino_paths():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            try:
                from groundingdino.util.inference import (
                    load_image as _load_image,
                    load_model as _load_model,
                    predict as _predict,
                )
                return _load_model, _predict, _load_image
            except Exception:
                continue

        raise ImportError("Could not import groundingdino.util.inference from site-packages or local fallback paths.")

    def _groundingdino_custom_ops_available(self) -> bool:
        try:
            import groundingdino.models.GroundingDINO.ms_deform_attn as ms_deform_attn

            return hasattr(ms_deform_attn, "_C") and ms_deform_attn._C is not None
        except Exception:
            return False

    def _load_gdino_if_needed(self, force_reload: bool = False) -> None:
        if self.gdino_model is not None and not force_reload:
            return

        global load_gdino_model, gdino_predict, load_image

        self.gdino_load_error = None
        self.gdino_config_path = self._resolve_model_path("GDINO_CONFIG", settings.gdino_config, "groundingdino_swint_ogc.cfg.py")
        self.gdino_weights_path = self._resolve_model_path("GDINO_WEIGHTS", settings.gdino_weights, "groundingdino_swint_ogc.pth")

        if load_gdino_model is None or gdino_predict is None or load_image is None:
            try:
                _load_model, _predict, _load_image = self._import_groundingdino_inference()
                load_gdino_model = _load_model
                gdino_predict = _predict
                load_image = _load_image
            except Exception as exc:
                self.gdino_model = None
                self.gdino_predict = None
                self.gdino_load_image = None
                self.gdino_load_error = (
                    "GroundingDINO import failed. "
                    "Install inference deps (transformers/timm/addict/yapf/supervision) "
                    "and ensure either a working groundingdino package or local "
                    "`svg-repair/fastapi/GroundingDINO` source is present. "
                    f"(error: {exc})"
                )
                return

        self.gdino_predict = gdino_predict
        self.gdino_load_image = load_image

        missing = []
        if not self.gdino_config_path:
            missing.append("GDINO_CONFIG")
        if not self.gdino_weights_path:
            missing.append("GDINO_WEIGHTS")
        if missing:
            self.gdino_model = None
            self.gdino_load_error = (
                "Missing GroundingDINO model files. "
                f"Unavailable: {', '.join(missing)}. "
                f"Resolved paths: config={self.gdino_config_path}, weights={self.gdino_weights_path}"
            )
            return

        try:
            # If custom C++ ops are unavailable (common in Colab/Python 3.12), keep GDINO on CPU
            # to avoid CUDA path requiring groundingdino._C.
            device = "cpu"
            import torch

            if torch.cuda.is_available() and self._groundingdino_custom_ops_available():
                device = "cuda"
            self.gdino_model = load_gdino_model(self.gdino_config_path, self.gdino_weights_path, device=device)
        except Exception as exc:
            self.gdino_model = None
            self.gdino_load_error = (
                "Failed to initialize GroundingDINO model. "
                f"config={self.gdino_config_path}, weights={self.gdino_weights_path}, error={exc}"
            )

    def _load_models_if_needed(self) -> None:
        if self._loaded:
            return
        self._loaded = True

        global SAM, YOLOE, load_gdino_model, gdino_predict, load_image

        if SAM is None or YOLOE is None:
            try:
                from ultralytics import SAM as _SAM, YOLOE as _YOLOE

                SAM = _SAM
                YOLOE = _YOLOE
            except Exception:
                SAM = None
                YOLOE = None

        if load_gdino_model is None or gdino_predict is None or load_image is None:
            try:
                from groundingdino.util.inference import (
                    load_image as _load_image,
                    load_model as _load_model,
                    predict as _predict,
                )

                load_gdino_model = _load_model
                gdino_predict = _predict
                load_image = _load_image
                self.gdino_predict = _predict
                self.gdino_load_image = _load_image
            except Exception:
                self.gdino_predict = None
                self.gdino_load_image = None

        if SAM is not None:
            try:
                if os.path.exists(settings.sam_weights):
                    self.sam_model = SAM(settings.sam_weights)
            except Exception:
                self.sam_model = None

        if YOLOE is not None:
            for key, path in {
                "yolo26l": settings.yolo26l_weights,
                "yolo26x": settings.yolo26x_weights,
            }.items():
                try:
                    if os.path.exists(path):
                        self.detectors[key] = YOLOE(path)
                except Exception:
                    pass

        self._load_gdino_if_needed(force_reload=False)

    def detect(self, payload: DetectRequest) -> DetectResponse:
        self._load_models_if_needed()
        image = self._decode_image(payload.image_data)
        width, height = image.size
        image_rgba, image_rgb = self._prepare_image_arrays(image)

        if payload.method == "gdino":
            if self.gdino_model is None:
                self._load_gdino_if_needed(force_reload=True)
            if self.gdino_model is not None and self.gdino_predict is not None and self.gdino_load_image is not None:
                return self._detect_with_gdino(payload, image, image_rgba, image_rgb)
            raise HTTPException(
                status_code=503,
                detail=(
                    "GDINO was explicitly requested but GroundingDINO is not loaded. "
                    "Set GDINO_CONFIG and GDINO_WEIGHTS to valid files, then restart the API. "
                    f"Details: {self.gdino_load_error or 'unknown error'}"
                ),
            )
        if payload.method in ("yolo26l", "yolo26x"):
            return self._detect_with_yolo(payload, image_rgba, image_rgb, width, height)

        # Contract-safe fallback when no model is available.
        fallback_box = [0.0, 0.0, float(width), float(height)]
        mask_png = None
        if payload.return_masks:
            mask = self._box_mask([0, 0, width, height], width, height)
            mask_png = self._mask_to_png(mask, image_rgba)
        return DetectResponse(boxes=[{
            "box": fallback_box,
            "score": 0.01,
            "mask_png": mask_png,
            "label": payload.text,
        }])

    def _detect_with_yolo(
        self,
        payload: DetectRequest,
        image_rgba: np.ndarray,
        image_rgb: np.ndarray,
        width: int,
        height: int,
    ) -> DetectResponse:
        model_key = payload.method
        model = self.detectors.get(model_key) or self.detectors.get("yolo26x") or self.detectors.get("yolo26l")
        if model is None:
            fallback_box = [0.0, 0.0, float(width), float(height)]
            mask_png = None
            if payload.return_masks:
                mask_png = self._mask_to_png(self._box_mask([0, 0, width, height], width, height), image_rgba)
            return DetectResponse(boxes=[{"box": fallback_box, "score": 0.01, "mask_png": mask_png, "label": payload.text}])

        try:
            model.set_classes([payload.text], model.get_text_pe([payload.text]))
        except Exception:
            pass

        detection = model.predict(image_rgb, conf=payload.min_score, verbose=False)
        if not detection:
            return DetectResponse(boxes=[])
        boxes = detection[0].boxes
        if boxes is None or boxes.xyxy is None:
            return DetectResponse(boxes=[])

        xyxy = boxes.xyxy.cpu().numpy().tolist()
        scores = boxes.conf.cpu().numpy().tolist() if boxes.conf is not None else []

        indexed = list(zip(xyxy, scores if scores else [0.0] * len(xyxy)))
        indexed.sort(key=lambda it: it[1], reverse=True)
        indexed = indexed[: payload.max_results]

        out = []
        text_like = self._is_text_like_label(payload.text)
        for box, score in indexed:
            box_xyxy = [float(v) for v in box]
            mask_png = None
            if payload.return_masks:
                mask_png = self._generate_sam_mask(box_xyxy, image_rgb, image_rgba)
                refined_mask = self._contrast_mask_from_box(box_xyxy, image_rgb, image_rgba)
                if text_like and refined_mask is not None:
                    mask_png = refined_mask
                elif mask_png is None and refined_mask is not None:
                    mask_png = refined_mask
                if mask_png is None:
                    mask_png = self._mask_to_png(self._box_mask(box_xyxy, width, height), image_rgba)
            out.append(
                {
                    "box": self._box_to_xywh(box_xyxy),
                    "score": float(score),
                    "mask_png": mask_png,
                    "label": payload.text,
                }
            )

        return DetectResponse(boxes=out)

    def _detect_with_gdino(
        self,
        payload: DetectRequest,
        image: Image.Image,
        image_rgba: np.ndarray,
        image_rgb: np.ndarray,
    ) -> DetectResponse:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            path = tmp.name
            image.convert("RGB").save(path, "JPEG")
        try:
            image_source, image_tensor = self.gdino_load_image(path)
            _ = image_source
            boxes, logits, _phrases = self.gdino_predict(
                model=self.gdino_model,
                image=image_tensor,
                caption=payload.text,
                box_threshold=payload.min_score,
                text_threshold=0.25,
                device="cuda" if self._cuda_available() else "cpu",
            )

            if boxes is None or len(boxes) == 0:
                return DetectResponse(boxes=[])

            import torch

            boxes_np = boxes.cpu().numpy() if torch.is_tensor(boxes) else boxes
            logits_np = logits.cpu().numpy() if torch.is_tensor(logits) else logits
            h, w = image_rgb.shape[:2]
            out = []
            text_like = self._is_text_like_label(payload.text)
            for i, b in enumerate(boxes_np[: payload.max_results]):
                cx, cy, bw, bh = b
                x1 = float((cx - bw / 2) * w)
                y1 = float((cy - bh / 2) * h)
                x2 = float((cx + bw / 2) * w)
                y2 = float((cy + bh / 2) * h)
                box_xyxy = [x1, y1, x2, y2]
                mask_png = None
                if payload.return_masks:
                    mask_png = self._generate_sam_mask(box_xyxy, image_rgb, image_rgba)
                    refined_mask = self._contrast_mask_from_box(box_xyxy, image_rgb, image_rgba)
                    if text_like and refined_mask is not None:
                        mask_png = refined_mask
                    elif mask_png is None and refined_mask is not None:
                        mask_png = refined_mask
                    if mask_png is None:
                        mask_png = self._mask_to_png(self._box_mask(box_xyxy, w, h), image_rgba)
                out.append(
                    {
                        "box": self._box_to_xywh(box_xyxy),
                        "score": float(logits_np[i]) if i < len(logits_np) else 0.0,
                        "mask_png": mask_png,
                        "label": payload.text,
                    }
                )
            return DetectResponse(boxes=out)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def segment_manual_box(self, image_data: str, box_xyxy: List[float], label: str) -> Dict[str, Any]:
        self._load_models_if_needed()
        image = self._decode_image(image_data)
        width, height = image.size
        image_rgba, image_rgb = self._prepare_image_arrays(image)

        mask_png = self._generate_sam_mask(box_xyxy, image_rgb, image_rgba)
        refined_mask = self._contrast_mask_from_box(box_xyxy, image_rgb, image_rgba)
        if self._is_text_like_label(label) and refined_mask is not None:
            mask_png = refined_mask
        elif mask_png is None and refined_mask is not None:
            mask_png = refined_mask
        if mask_png is None:
            mask_png = self._mask_to_png(self._box_mask(box_xyxy, width, height), image_rgba)

        return {
            "box": box_xyxy,
            "label": label,
            "mask_png": mask_png,
            "width": width,
            "height": height,
        }

    def _compute_bbox_overlap(self, bbox1: List[float], bbox2: List[float]) -> float:
        x1_inter = max(bbox1[0], bbox2[0])
        y1_inter = max(bbox1[1], bbox2[1])
        x2_inter = min(bbox1[2], bbox2[2])
        y2_inter = min(bbox1[3], bbox2[3])
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = bbox1_area + bbox2_area - inter_area
        if union_area == 0:
            return 0.0
        return float(inter_area / union_area)

    def _compute_centrality_score(self, bbox: List[float], image_width: int, image_height: int) -> float:
        obj_center_x = (bbox[0] + bbox[2]) / 2
        obj_center_y = (bbox[1] + bbox[3]) / 2
        img_center_x = image_width / 2
        img_center_y = image_height / 2
        max_distance = np.sqrt((image_width / 2) ** 2 + (image_height / 2) ** 2)
        actual_distance = np.sqrt((obj_center_x - img_center_x) ** 2 + (obj_center_y - img_center_y) ** 2)
        return float(1.0 - min(actual_distance / max_distance, 1.0))

    def _apply_semantic_boost(self, label: str) -> float:
        lower = label.lower()
        if any(k in lower for k in ["text", "logo", "watermark", "caption", "title", "label"]):
            return 10.0
        if any(k in lower for k in ["background", "backdrop", "wall", "floor", "sky"]):
            return -10.0
        if any(k in lower for k in ["product", "person", "face", "character", "subject"]):
            return 3.0
        if any(k in lower for k in ["table", "plate", "surface", "shelf", "ground"]):
            return -3.0
        return 0.0

    def compute_z_order(self, objects: List[ZOrderObject], image_width: int, image_height: int) -> ZOrderResponse:
        if not objects:
            return ZOrderResponse(ordered_objects=[])

        image_area = image_width * image_height
        scored = []
        for obj in objects:
            z_score = 0.0
            reasoning = []

            semantic = self._apply_semantic_boost(obj.label)
            z_score += semantic
            if semantic != 0:
                reasoning.append(f"semantic({semantic:+.1f})")

            area = (obj.bbox[2] - obj.bbox[0]) * (obj.bbox[3] - obj.bbox[1])
            size_score = (1.0 - (area / image_area)) * 3.0
            z_score += size_score
            reasoning.append(f"size(+{size_score:.1f})")

            center = self._compute_centrality_score(obj.bbox, image_width, image_height) * 2.0
            z_score += center
            reasoning.append(f"center(+{center:.1f})")

            overlap_score = 0.0
            for other in objects:
                if other.id == obj.id:
                    continue
                iou = self._compute_bbox_overlap(obj.bbox, other.bbox)
                if iou > 0.1:
                    other_area = (other.bbox[2] - other.bbox[0]) * (other.bbox[3] - other.bbox[1])
                    overlap_score += 2.0 * iou if area < other_area else -1.0 * iou
            if overlap_score != 0:
                z_score += overlap_score
                reasoning.append(f"overlap({overlap_score:+.1f})")

            scored.append({"id": obj.id, "label": obj.label, "z_score": z_score, "reasoning": " ".join(reasoning)})

        scored.sort(key=lambda x: x["z_score"], reverse=True)
        ordered = [
            ZOrderResult(
                id=row["id"],
                label=row["label"],
                z_score=float(row["z_score"]),
                rank=i + 1,
                reasoning=row["reasoning"],
            )
            for i, row in enumerate(scored)
        ]
        return ZOrderResponse(ordered_objects=ordered)

    def _normalize_inpaint_provider(self, provider: Optional[str], model: Optional[str]) -> tuple[str, str]:
        normalized_provider = (provider or "").strip().lower()
        normalized_model = (model or "").strip()
        if normalized_provider:
            if normalized_provider == "openrouter" and normalized_model.startswith("openrouter:"):
                normalized_model = normalized_model.split(":", 1)[1]
            return normalized_provider, normalized_model
        if normalized_model.startswith("openrouter:"):
            return "openrouter", normalized_model.split(":", 1)[1]
        if normalized_model in {"big-lama", "big_lama", "lama"}:
            return "big-lama", ""
        if normalized_model in {"qualcomm-lama-dilated", "lama-dilated", "qualcomm-lama"}:
            return "qualcomm-lama-dilated", ""
        return "openrouter", normalized_model

    def _normalize_inpaint_mask(self, mask_bytes: bytes) -> Tuple[bytes, str]:
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert("RGBA")
        mask_arr = np.array(mask_img)
        alpha = mask_arr[:, :, 3]
        use_alpha = alpha.max() > 0 and alpha.min() < 255
        if use_alpha:
            mask_binary = (alpha > 0).astype(np.uint8) * 255
            source = "alpha"
        else:
            rgb = mask_arr[:, :, :3]
            gray = np.max(rgb, axis=2)
            mask_binary = (gray > 127).astype(np.uint8) * 255
            source = "luminance"
        output = Image.fromarray(mask_binary, mode="L")
        buf = io.BytesIO()
        output.save(buf, format="PNG")
        return buf.getvalue(), source

    def _extract_inpaint_image(self, payload: dict) -> Optional[str]:
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            images = message.get("images")
            if isinstance(images, list) and images:
                url = images[0].get("image_url", {}).get("url")
                if url:
                    return url
            content = message.get("content")
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url")
                        if url:
                            return url
                    if item.get("type") == "image" and item.get("data"):
                        return f"data:image/png;base64,{item['data']}"
            if isinstance(content, str) and content.startswith("data:image/"):
                return content
        return None

    def _run_inpaint_command(self, command_template: Optional[str], image_bytes: bytes, mask_bytes: bytes) -> str:
        if not command_template:
            raise RuntimeError("Inpaint command is not configured.")
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "image.png")
            mask_path = os.path.join(tmpdir, "mask.png")
            output_path = os.path.join(tmpdir, "output.png")
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            with open(mask_path, "wb") as f:
                f.write(mask_bytes)
            command = command_template.format(image=image_path, mask=mask_path, output=output_path)
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                err = (result.stderr or result.stdout or "").strip()
                raise RuntimeError(err or "Inpaint command failed")
            if not os.path.exists(output_path):
                raise RuntimeError("Inpaint command did not produce output")
            out = open(output_path, "rb").read()
        return "data:image/png;base64," + base64.b64encode(out).decode("ascii")

    def _resize_for_local_inpaint(self, image_bytes: bytes, mask_bytes: bytes) -> tuple[bytes, bytes, tuple[int, int], tuple[int, int]]:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
        orig_size = image.size
        w, h = orig_size
        max_side = max(w, h)
        limit = max(256, int(settings.local_inpaint_max_side))
        if max_side <= limit:
            return image_bytes, mask_bytes, orig_size, orig_size

        scale = float(limit) / float(max_side)
        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        image_small = image.resize(new_size, Image.Resampling.LANCZOS)
        mask_small = mask.resize(new_size, Image.Resampling.NEAREST)

        img_buf = io.BytesIO()
        image_small.save(img_buf, format="PNG")
        mask_buf = io.BytesIO()
        mask_small.save(mask_buf, format="PNG")
        return img_buf.getvalue(), mask_buf.getvalue(), orig_size, new_size

    def _resize_local_inpaint_output_back(self, inpainted_data_uri: str, output_size: tuple[int, int], target_size: tuple[int, int]) -> str:
        if output_size == target_size:
            return inpainted_data_uri
        image = self._decode_image(inpainted_data_uri)
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        return self._encode_png(image)

    def _inpaint(self, image_data: str, mask_data: str, provider: Optional[str], model: Optional[str], api_key: Optional[str], prompt: str = "") -> Dict[str, Any]:
        provider_name, model_name = self._normalize_inpaint_provider(provider, model)
        if provider_name == "openrouter":
            model_name = model_name or settings.inpaint_model_default

        image_bytes = self._decode_data_uri_bytes(image_data)
        mask_bytes = self._decode_data_uri_bytes(mask_data)
        mask_bytes, _ = self._normalize_inpaint_mask(mask_bytes)

        if provider_name == "openrouter":
            key = api_key or settings.openrouter_api_key
            if not key:
                return {"success": False, "error": "OPENROUTER_API_KEY not configured"}
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://colab.research.google.com",
                "X-Title": "svg-repair-colab-demo",
            }
            mask_uri = "data:image/png;base64," + base64.b64encode(mask_bytes).decode("ascii")
            full_prompt = (prompt or "Remove the masked object and fill naturally.").strip()
            if "mask" not in full_prompt.lower():
                full_prompt += " The second image is the mask where white pixels indicate the area to be inpainted/filled."
            payload = {
                "model": model_name,
                "modalities": ["image"],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": full_prompt},
                            {"type": "image_url", "image_url": {"url": image_data}},
                            {"type": "image_url", "image_url": {"url": mask_uri}},
                        ],
                    }
                ],
            }
            try:
                resp = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=settings.inpaint_timeout_s,
                )
            except requests.RequestException as exc:
                return {"success": False, "error": str(exc)}
            if resp.status_code >= 400:
                return {"success": False, "error": f"OpenRouter error {resp.status_code}: {resp.text[:300]}"}
            try:
                resp_json = resp.json()
            except ValueError:
                return {"success": False, "error": "OpenRouter response is not JSON"}
            image_out = self._extract_inpaint_image(resp_json)
            if not image_out:
                return {"success": False, "error": "No image in OpenRouter response"}
            return {"success": True, "inpainted_image": image_out}

        if provider_name == "big-lama":
            try:
                img_run, mask_run, orig_size, run_size = self._resize_for_local_inpaint(image_bytes, mask_bytes)
                image_out = self._run_inpaint_command(settings.big_lama_command, img_run, mask_run)
                image_out = self._resize_local_inpaint_output_back(image_out, run_size, orig_size)
                return {"success": True, "inpainted_image": image_out}
            except Exception as exc:
                return {"success": False, "error": str(exc)}

        if provider_name == "qualcomm-lama-dilated":
            try:
                img_run, mask_run, orig_size, run_size = self._resize_for_local_inpaint(image_bytes, mask_bytes)
                image_out = self._run_inpaint_command(settings.qualcomm_lama_dilated_command, img_run, mask_run)
                image_out = self._resize_local_inpaint_output_back(image_out, run_size, orig_size)
                return {"success": True, "inpainted_image": image_out}
            except Exception as exc:
                return {"success": False, "error": str(exc)}

        return {"success": False, "error": f"Unknown provider: {provider_name}"}

    def _dilate_mask_data_uri(self, mask_data: str, pixels: int) -> str:
        if pixels <= 0:
            return mask_data
        mask_bytes = self._decode_data_uri_bytes(mask_data)
        mask = Image.open(io.BytesIO(mask_bytes)).convert("RGBA")
        arr = np.array(mask)
        alpha = arr[:, :, 3]
        alpha_bin = (alpha > 0).astype(np.uint8) * 255
        m = Image.fromarray(alpha_bin, mode="L")
        m = m.filter(ImageFilter.MaxFilter(size=2 * pixels + 1))
        return self._encode_png(m.convert("RGBA"))

    def inpaint_sequential(self, payload: SequentialInpaintRequest) -> SequentialInpaintResponse:
        start = time.time()
        image = self._decode_image(payload.image_data)

        if payload.use_z_order:
            z_order = self.compute_z_order(payload.objects, image.width, image.height).ordered_objects
        else:
            z_order = [
                ZOrderResult(id=obj.id, label=obj.label, z_score=0.0, rank=i + 1, reasoning="user order")
                for i, obj in enumerate(payload.objects)
            ]

        object_map = {obj.id: obj for obj in payload.objects}
        layers: List[InpaintedLayer] = []
        current_image = payload.image_data

        for z in z_order:
            obj = object_map.get(z.id)
            if obj is None or not obj.mask_data:
                continue
            loop_start = time.time()
            prompt = ""
            if (payload.provider or InpaintProvider.big_lama.value) == InpaintProvider.openrouter.value:
                prompt = (
                    f"Inpaint and remove the {z.label} from this image. "
                    "Fill the masked area seamlessly with the surrounding background texture and colors. "
                    "Maintain the exact same style and quality of the rest of the image."
                )

            mask_data = obj.mask_data
            if any(k in z.label.lower() for k in ["text", "logo", "watermark", "caption", "title", "label"]):
                mask_data = self._dilate_mask_data_uri(mask_data, 6)

            result = self._inpaint(
                image_data=current_image,
                mask_data=mask_data,
                provider=payload.provider.value if payload.provider else None,
                model=payload.model,
                api_key=payload.api_key,
                prompt=prompt,
            )
            if not result.get("success"):
                raise HTTPException(status_code=500, detail=f"Inpaint failed for {z.label}: {result.get('error')}")

            current_image = result["inpainted_image"]
            layers.append(
                InpaintedLayer(
                    object_id=z.id,
                    label=z.label,
                    rank=z.rank,
                    inpainted_image=current_image,
                    processing_time=round(time.time() - loop_start, 3),
                )
            )

        return SequentialInpaintResponse(
            success=True,
            layers=layers,
            final_background=current_image,
            z_order_used=z_order,
            total_processing_time=round(time.time() - start, 3),
        )

    def _mask_bbox_from_alpha(self, mask_image: Image.Image) -> Optional[tuple]:
        arr = np.array(mask_image.convert("RGBA"))
        alpha = arr[:, :, 3]
        ys, xs = np.where(alpha > 0)
        if ys.size == 0 or xs.size == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    def _apply_mask_to_source(self, source_image: Image.Image, mask_image: Image.Image) -> Image.Image:
        src = np.array(source_image.convert("RGBA"))
        m = np.array(mask_image.convert("RGBA"))
        if src.shape[:2] != m.shape[:2]:
            mask_image = mask_image.resize((src.shape[1], src.shape[0]), Image.Resampling.LANCZOS)
            m = np.array(mask_image.convert("RGBA"))
        alpha = m[:, :, 3]
        if alpha.max() == 0:
            alpha = np.max(m[:, :, :3], axis=2)
        out = src.copy()
        out[:, :, 3] = alpha
        return Image.fromarray(out, mode="RGBA")

    def _extract_svg_inner(self, svg_text: str) -> str:
        match = re.search(r"<svg[^>]*>(.*)</svg>", svg_text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else svg_text

    def _count_svg_paths(self, svg_text: str) -> int:
        return len(re.findall(r"<path\\b", svg_text, re.IGNORECASE))

    def _trace_fallback(self, image: Image.Image) -> Tuple[str, str]:
        w, h = image.size
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'<rect x="0" y="0" width="{w}" height="{h}" fill="#000" /></svg>',
            "fallback",
        )

    def _build_vtracer_options(self, options: dict) -> dict:
        color_mode = options.get("color_mode", "color")
        return {
            "colormode": "binary" if color_mode == "binary" else "color",
            "mode": options.get("mode", "spline"),
            "filter_speckle": int(options.get("filter_speckle", 4)),
            "color_precision": int(options.get("color_precision", 6)),
            "corner_threshold": int(options.get("corner_threshold", 60)),
            "length_threshold": float(options.get("length_threshold", 4.0)),
            "max_iterations": int(options.get("max_iterations", 10)),
            "splice_threshold": int(options.get("splice_threshold", 45)),
            "path_precision": int(options.get("path_precision", 3)),
            "hierarchical": "stacked",
        }

    def _trace_image(self, image: Image.Image, options: dict) -> tuple[str, str]:
        errors = []
        if vtracer is not None:
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    input_path = os.path.join(tmpdir, "input.png")
                    output_path = os.path.join(tmpdir, "output.svg")
                    image.save(input_path, format="PNG")
                    tracer = getattr(vtracer, "convert_image_to_svg_py", None) or getattr(vtracer, "convert_image_to_svg", None)
                    if tracer is None:
                        raise RuntimeError("vtracer API not found")
                    result = tracer(input_path, output_path, **self._build_vtracer_options(options))
                    if isinstance(result, str) and result.strip().startswith("<svg"):
                        return result, "vtracer"
                    if os.path.exists(output_path):
                        return open(output_path, "r", encoding="utf-8").read(), "vtracer"
            except Exception as exc:
                errors.append(str(exc))

        if potrace is not None:
            try:
                rgba = np.array(image.convert("RGBA"))
                alpha = rgba[:, :, 3] > 0
                bitmap = potrace.Bitmap(alpha)
                path = bitmap.trace()
                paths = []
                for curve in path:
                    parts = [f"M {curve.start_point[0]} {curve.start_point[1]}"]
                    for seg in curve:
                        if seg.is_corner:
                            c, ep = seg.c, seg.end_point
                            parts.append(f"L {c[0]} {c[1]} L {ep[0]} {ep[1]}")
                        else:
                            c1, c2, ep = seg.c1, seg.c2, seg.end_point
                            parts.append(f"C {c1[0]} {c1[1]} {c2[0]} {c2[1]} {ep[0]} {ep[1]}")
                    parts.append("Z")
                    path_d = " ".join(parts)
                    paths.append(f'<path d="{path_d}" fill="#000000" />')
                w, h = image.size
                svg_open = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">".format(w, h, w, h)
                svg = svg_open + "".join(paths) + "</svg>"
                return svg, "potrace"
            except Exception as exc:
                errors.append(str(exc))

        if settings.trace_allow_fallback:
            return self._trace_fallback(image)
        raise HTTPException(status_code=500, detail=f"Trace failed: {' | '.join(errors) if errors else 'No backend'}")

    def trace_batch(self, payload: TraceBatchRequest) -> TraceBatchResponse:
        if not payload.layers:
            raise HTTPException(status_code=400, detail="layers are required")

        base_options = payload.options.model_dump() if payload.options else {}

        def trace_one(index: int, layer) -> Dict[str, Any]:
            options = {**base_options, **(layer.options.model_dump(exclude_unset=True) if layer.options else {})}
            offset = None

            if layer.source_image_data:
                source_image = self._decode_image(layer.source_image_data)
                mask_image = self._decode_image(layer.image_data)
                bbox = self._mask_bbox_from_alpha(mask_image)
                if bbox:
                    x_min, y_min, x_max, y_max = bbox
                    source_image = source_image.crop(bbox)
                    mask_image = mask_image.crop(bbox)
                    offset = {"x": x_min, "y": y_min, "width": x_max - x_min, "height": y_max - y_min}
                    if layer.input_offset:
                        offset["x"] += layer.input_offset.get("x", 0)
                        offset["y"] += layer.input_offset.get("y", 0)
                image = self._apply_mask_to_source(source_image, mask_image)
            else:
                image = self._decode_image(layer.image_data)
                if layer.input_offset:
                    offset = dict(layer.input_offset)

            t0 = time.time()
            svg_full, engine = self._trace_image(image, options)
            return {
                "index": index,
                "payload": TraceLayerResponse(
                    id=layer.id,
                    label=layer.label,
                    width=image.size[0],
                    height=image.size[1],
                    svg_paths=self._extract_svg_inner(svg_full),
                    svg_full=svg_full,
                    stats={
                        "path_count": self._count_svg_paths(svg_full),
                        "processing_time": round(time.time() - t0, 3),
                        "engine": engine,
                    },
                    offset=offset,
                ),
            }

        out = []
        max_workers = min(len(payload.layers), max(2, os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(trace_one, idx, layer) for idx, layer in enumerate(payload.layers)]
            for f in futures:
                out.append(f.result())
        out.sort(key=lambda x: x["index"])
        return TraceBatchResponse(layers=[x["payload"] for x in out])

    def _extract_svg_parts(self, svg_text: str) -> tuple:
        match = re.search(r"<svg[^>]*>(.*)</svg>", svg_text, re.DOTALL | re.IGNORECASE)
        inner = match.group(1) if match else svg_text
        defs_blocks = re.findall(r"<defs[^>]*>.*?</defs>", inner, re.DOTALL | re.IGNORECASE)
        defs_content = [re.sub(r"</?defs[^>]*>", "", block, flags=re.IGNORECASE).strip() for block in defs_blocks]
        body = re.sub(r"<defs[^>]*>.*?</defs>", "", inner, flags=re.DOTALL | re.IGNORECASE).strip()
        return defs_content, body

    def _escape_attr(self, value: str) -> str:
        return value.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")

    def _sanitize_id(self, value: str, fallback: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", (value or "").strip())
        cleaned = cleaned.strip("-")
        return cleaned or fallback

    def assemble(self, payload: AssembleRequest) -> dict:
        if payload.width <= 0 or payload.height <= 0:
            raise HTTPException(status_code=400, detail="width and height are required")
        if not payload.layers:
            raise HTTPException(status_code=400, detail="layers are required")

        ordered = sorted(list(enumerate(payload.layers)), key=lambda item: (item[1].z_index, item[0]))

        defs = []
        body = []
        for idx, layer in ordered:
            layer_id = self._sanitize_id(layer.id or f"layer-{idx+1}", f"layer-{idx+1}")
            label = layer.label or layer_id
            svg_body = ""
            if layer.svg_paths:
                svg_body = layer.svg_paths
            elif layer.svg_full:
                layer_defs, layer_body = self._extract_svg_parts(layer.svg_full)
                defs.extend([d for d in layer_defs if d])
                svg_body = layer_body
            if not svg_body:
                continue
            hidden_attr = ' display="none"' if layer.hidden else ""
            body.append(f'  <g id="{self._escape_attr(layer_id)}" data-label="{self._escape_attr(label)}"{hidden_attr}>{svg_body}</g>')

        defs_block = ""
        if defs:
            defs_block = "  <defs>\n" + "\n".join([f"    {d}" for d in defs]) + "\n  </defs>"

        svg = "\n".join(
            [
                '<?xml version="1.0" encoding="UTF-8"?>',
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{payload.width}" height="{payload.height}" viewBox="0 0 {payload.width} {payload.height}">',
                defs_block,
                "\n".join(body),
                "</svg>",
            ]
        )
        return {"svgText": svg}

    def upsample_via_visionsoc(self, image_data: str, mode: str = "balanced") -> Optional[str]:
        # VisionSoC branch is optional in local dev; silently return None on connectivity failure.
        if cv2 is None:
            return None
        try:
            image_bytes = self._decode_data_uri_bytes(image_data)
            files = {"image": ("input.png", image_bytes, "image/png")}
            resp = requests.post(f"{settings.vision_soc_url}/api/upscale", files=files, data={"mode": mode}, timeout=90)
            if resp.status_code >= 400:
                return None
            try:
                data = resp.json()
                out = data.get("output_image") or data.get("upscaled")
                if isinstance(out, str) and out.startswith("data:image/"):
                    return out
            except ValueError:
                pass
            return "data:image/jpeg;base64," + base64.b64encode(resp.content).decode("ascii")
        except Exception:
            return None

    def _cuda_available(self) -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False


pipeline = ColabPipeline()
