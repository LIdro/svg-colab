#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import cv2
import torch
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Big-LaMa inpainting using JIT model.")
    parser.add_argument("--image", required=True, help="Path to input image PNG.")
    parser.add_argument("--mask", required=True, help="Path to input mask PNG (white=hole).")
    parser.add_argument("--output", required=True, help="Path to write output PNG.")
    args = parser.parse_args()

    # Get model path
    repo_dir = os.getenv("BIG_LAMA_DIR", os.path.join(os.path.dirname(__file__), "third_party", "lama"))
    model_path = os.getenv("BIG_LAMA_CHECKPOINT", os.path.join(repo_dir, "big-lama.pt"))
    
    if not os.path.exists(model_path):
        print(f"Big-LaMa model not found at {model_path}", file=sys.stderr)
        return 2

    try:
        # Load images
        image = cv2.imread(args.image)
        if image is None:
            print(f"Could not read image: {args.image}", file=sys.stderr)
            return 1
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Could not read mask: {args.mask}", file=sys.stderr)
            return 1
        
        # Store original dimensions for final resize
        original_h, original_w = image.shape[:2]
        
        # Pad to make dimensions divisible by 8 (required by Big-LaMa)
        def pad_to_multiple(img, multiple=8):
            h, w = img.shape[:2]
            pad_h = (multiple - h % multiple) % multiple
            pad_w = (multiple - w % multiple) % multiple
            
            if len(img.shape) == 3:
                padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            else:
                padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
            
            return padded, (h, w)
        
        image_padded, (orig_h, orig_w) = pad_to_multiple(image)
        mask_padded, _ = pad_to_multiple(mask)
        
        print(f"Original size: {original_h}x{original_w}, Padded size: {image_padded.shape[0]}x{image_padded.shape[1]}", file=sys.stderr)
        
        # Normalize to [0, 1] with strict clipping
        image_norm = image_padded.astype(np.float32) / 255.0
        image_norm = np.clip(image_norm, 0.0, 1.0)
        
        # Mask: ensure strict binary values where 1 = area to inpaint
        mask_norm = mask_padded.astype(np.float32) / 255.0
        mask_norm = (mask_norm > 0.5).astype(np.float32)
        
        # Convert to tensors (B, C, H, W)
        image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0).contiguous()
        mask_tensor = torch.from_numpy(mask_norm).unsqueeze(0).unsqueeze(0).contiguous()
        
        # Ensure float32 dtype explicitly
        image_tensor = image_tensor.float()
        mask_tensor = mask_tensor.float()
        
        # Verify tensor properties
        print(f"Image tensor: shape={image_tensor.shape}, dtype={image_tensor.dtype}, "
              f"range=[{image_tensor.min():.4f}, {image_tensor.max():.4f}]", file=sys.stderr)
        print(f"Mask tensor: shape={mask_tensor.shape}, dtype={mask_tensor.dtype}, "
              f"unique_values={torch.unique(mask_tensor).tolist()}", file=sys.stderr)
        
        # Load JIT model
        print(f"Loading Big-LaMa model from {model_path}...", file=sys.stderr)
        model = torch.jit.load(model_path, map_location='cpu')
        model.eval()
        
        # Run inference
        print("Running inpainting...", file=sys.stderr)
        with torch.no_grad():
            result = model(image_tensor, mask_tensor)
        
        # Convert back to image
        result_image = result[0].permute(1, 2, 0).cpu().numpy()
        result_image = np.clip(result_image * 255, 0, 255).astype(np.uint8)
        
        # Crop back to original size
        result_image = result_image[:orig_h, :orig_w]
        
        # Convert color space
        result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        
        # Save result
        cv2.imwrite(args.output, result_image)
        print(f"✓ Inpainting complete: {args.output}", file=sys.stderr)
        return 0
        
    except Exception as e:
        print(f"Big-LaMa inpainting failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
