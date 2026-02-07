# Colab Demo App Guide

This guide explains how to run and operate the `colab_demo` SVG repair app.

## What this demo does

The app provides a simplified flow:

1. Upload image
2. Detect/segment objects (prompt + manual fallback)
3. Sequential inpaint to remove selected objects from background
4. Trace layers into SVG
5. Assemble layered SVG
6. Preview and download

## Project paths

- Notebook: `colab_demo/notebook/svg_repair_colab_demo.ipynb`
- API service: `colab_demo/services/api.py`
- UI app: `colab_demo/ui/gradio_app.py`
- Dependencies: `colab_demo/services/requirements-colab.txt`

## 1) Run in Google Colab

### A. Runtime requirements

- Use a GPU runtime in Colab when possible.
- Python 3.10+ recommended.

### B. Install dependencies

In a Colab cell:

```bash
%cd /content/svg-colab
!pip -q install --upgrade pip
!pip -q install -r colab_demo/services/requirements-colab.txt
```

### C. Configure Colab Secrets (recommended)

In Colab, open the **Secrets** panel and add:

- `OPENROUTER_API_KEY` (required only for OpenRouter provider)
- `HF_TOKEN` (optional; used for some Hugging Face model downloads)
- `SAM2_URL` (optional direct URL for `sam2.1_b.pt`)
- `BIG_LAMA_URL` (optional direct URL for `big-lama.pt` if HF download fails)

The notebook reads these automatically via `google.colab.userdata`.

### D. Model downloads during notebook run

The notebook now downloads and configures these at runtime:

- SAM2.1-B (`sam2.1_b.pt`, same version used by `svg-repair`)
- YOLO26L (`openvision/yoloe-26l-seg`, `model.pt`)
- YOLO26X (`openvision/yoloe-26x-seg`, `model.pt`)
- GroundingDINO config + weights
- Big-LaMa checkpoint (`big-lama.pt`, with fallback logic)

Downloaded files are stored under `/content/svg-colab/.colab_models/` and exported to env vars:

- `SAM_WEIGHTS`
- `YOLO26L_WEIGHTS`
- `YOLO26X_WEIGHTS`
- `GDINO_CONFIG`
- `GDINO_WEIGHTS`
- `BIG_LAMA_CHECKPOINT`
- `BIG_LAMA_COMMAND`
- `QUALCOMM_LAMA_DILATED_COMMAND`

### E. Optional manual environment variables

Set only what you need:

```python
import os

# Required only for OpenRouter inpaint provider
# os.environ['OPENROUTER_API_KEY'] = '...'

# Optional model/weight paths
# os.environ['SAM_WEIGHTS'] = '/content/sam2.1_b.pt'
# os.environ['YOLO26L_WEIGHTS'] = '/content/yoloe-26l-seg.pt'
# os.environ['YOLO26X_WEIGHTS'] = '/content/yoloe-26x-seg.pt'
# os.environ['GDINO_CONFIG'] = '/content/groundingdino_swint_ogc.cfg.py'
# os.environ['GDINO_WEIGHTS'] = '/content/groundingdino_swint_ogc.pth'

# Optional VisionSoC endpoint for upscaling branch
# os.environ['VISION_SOC_URL'] = 'http://127.0.0.1:5050'
```

### F. Start API service

Run in one cell/session:

```bash
!python -m uvicorn colab_demo.services.api:app --host 0.0.0.0 --port 5700
```

Health check (new cell):

```bash
!curl -s http://127.0.0.1:5700/health
```

Expected:

```json
{"status":"ok"}
```

### G. Start Gradio UI

Run in another cell/session:

```bash
!python colab_demo/ui/gradio_app.py
```

Open the Gradio link shown in output.

## 2) How to operate the app (UI workflow)

## Primary flow (4 actions)

1. Upload image (left panel).
2. Enter prompt (for example `logo text`) and click `Detect + Add`.
3. Click `Process Objects`.
4. Click `Trace + Assemble SVG`, then download from `Download SVG`.

## Manual fallback for missed objects

Use the `Advanced` panel:

1. Enter `x1, y1, x2, y2` and label.
2. Click `Add Manual Box`.
3. Continue with `Process Objects`.

## Provider and processing settings

Inside `Advanced`:

- `Provider`:
  - `big-lama`
  - `qualcomm-lama-dilated`
  - `openrouter` (requires API key)
- `Model`: optional override
- `Use Z-Order`: recommended enabled
- Upscale mode:
  - `none`
  - `text_only`
  - `all_objects`

## Outputs you get

- Right panel processed image (background after sequential inpaint)
- SVG preview
- SVG code text
- Metadata (`Layers`, `Paths`)
- Downloadable `.svg` file

## 3) Run locally (non-Colab)

From repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r colab_demo/services/requirements-colab.txt
python -m uvicorn colab_demo.services.api:app --host 0.0.0.0 --port 5700
```

In a second terminal:

```bash
python colab_demo/ui/gradio_app.py
```

## 4) Troubleshooting

- API not reachable from UI:
  - Confirm API is running on port `5700`.
  - Set `COLAB_API_BASE` if using a different host/port.
- OpenRouter inpaint fails:
  - Set `OPENROUTER_API_KEY` in Colab Secrets.
  - Verify selected model is available.
- SAM2.1-B download fails:
  - For SAM2.1-B, set `SAM2_URL` in Colab Secrets to a direct `sam2.1_b.pt` URL.
- Local LaMa providers fail:
  - Ensure command env vars are configured:
    - `BIG_LAMA_COMMAND`
    - `QUALCOMM_LAMA_DILATED_COMMAND`
  - If `big-lama.pt` is not found, provide `BIG_LAMA_URL` secret to a direct model file URL.
- VisionSoC upscaling does nothing:
  - Confirm `VISION_SOC_URL` points to a running VisionSoC service.
- Slow detection/tracing:
  - Use GPU runtime in Colab.
  - Lower `max_results` and keep prompts specific.

## 5) Current implementation notes

- Manual fallback currently uses numeric box entry (not drag-box UI yet).
- Full model quality depends on installed weights and runtime setup.
