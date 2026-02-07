#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
from PIL import Image
import torch


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask(path: str) -> Image.Image:
    return Image.open(path).convert("L")


def to_tensor(image: Image.Image) -> "torch.Tensor":
    array = np.asarray(image).astype("float32") / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor


def to_mask_tensor(mask: Image.Image) -> "torch.Tensor":
    array = np.asarray(mask).astype("float32") / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
    return (tensor > 0.5).float()


def infer(model, image_tensor, mask_tensor):
    with torch.no_grad():
        try:
            return model(image_tensor, mask_tensor)
        except TypeError:
            return model({"image": image_tensor, "mask": mask_tensor})


def normalize_output(output) -> np.ndarray:
    if isinstance(output, (list, tuple)):
        output = output[0]
    if hasattr(output, "detach"):
        output = output.detach()
    array = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    if array.min() < 0:
        array = (array + 1.0) / 2.0
    array = np.clip(array, 0.0, 1.0)
    return (array * 255.0).astype("uint8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Qualcomm LaMa Dilated inpainting.")
    parser.add_argument("--image", required=True, help="Path to input image PNG.")
    parser.add_argument("--mask", required=True, help="Path to input mask PNG (white=hole).")
    parser.add_argument("--output", required=True, help="Path to write output PNG.")
    args = parser.parse_args()

    try:
        from qai_hub_models.models.lama_dilated import Model
    except Exception as exc:
        print(f"Failed to import qai_hub_models for LaMa Dilated: {exc}", file=sys.stderr)
        return 2

    size = int(os.getenv("QUALCOMM_LAMA_DILATED_SIZE", "512"))
    image = load_image(args.image).resize((size, size), resample=Image.BICUBIC)
    mask = load_mask(args.mask).resize((size, size), resample=Image.NEAREST)

    image_tensor = to_tensor(image)
    mask_tensor = to_mask_tensor(mask)

    try:
        model = Model.from_pretrained()
    except Exception as exc:
        print(f"Failed to load LaMa Dilated model: {exc}", file=sys.stderr)
        return 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    image_tensor = image_tensor.to(device)
    mask_tensor = mask_tensor.to(device)

    try:
        output = infer(model, image_tensor, mask_tensor)
    except Exception as exc:
        print(f"LaMa Dilated inference failed: {exc}", file=sys.stderr)
        return 1

    output_array = normalize_output(output)
    Image.fromarray(output_array).save(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
