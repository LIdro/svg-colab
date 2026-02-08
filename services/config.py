from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str | None = os.getenv("OPENROUTER_API_KEY")

    sam_weights: str = os.getenv("SAM_WEIGHTS", "sam2.1_b.pt")
    yolo26l_weights: str = os.getenv("YOLO26L_WEIGHTS", "yoloe-26l-seg.pt")
    yolo26x_weights: str = os.getenv("YOLO26X_WEIGHTS", "yoloe-26x-seg.pt")
    gdino_config: str = os.getenv("GDINO_CONFIG", "groundingdino_swint_ogc.cfg.py")
    gdino_weights: str = os.getenv("GDINO_WEIGHTS", "groundingdino_swint_ogc.pth")

    inpaint_model_default: str = os.getenv("INPAINT_MODEL_DEFAULT", "bytedance-seed/seedream-4.5")
    inpaint_timeout_s: float = float(os.getenv("INPAINT_TIMEOUT", "60"))

    big_lama_command: str | None = os.getenv("BIG_LAMA_COMMAND")
    qualcomm_lama_dilated_command: str | None = os.getenv("QUALCOMM_LAMA_DILATED_COMMAND")

    vision_soc_url: str = os.getenv("VISION_SOC_URL", "http://127.0.0.1:5050")
    trace_allow_fallback: bool = os.getenv("TRACE_ALLOW_FALLBACK", "true").lower() == "true"


settings = Settings()
