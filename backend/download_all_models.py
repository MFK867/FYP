"""
Script to download ALL required AI model weights in one go.
"""

import os
import sys

# Set encoding for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Enable high-speed Rust multi-threaded parallel download engine
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import snapshot_download, hf_hub_download
import open_clip

print("==================================================")
print("   SKETCH-TO-GAME PIPELINE: ALL MODELS DOWNLOADER  ")
print("==================================================")

try:
    print("\n[1/4] Downloading Florence-2 Base Model (microsoft/Florence-2-base)...")
    florence_path = snapshot_download(repo_id="microsoft/Florence-2-base")
    print(f"[OK] Florence-2 Base ready at: {florence_path}")

    print("\n[2/4] Downloading SDXL Base Model 1.0 (stabilityai/stable-diffusion-xl-base-1.0)...")
    sdxl_path = snapshot_download(
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        allow_patterns=[
            "*.json", "*.txt", "*.png", "*.md",
            "*fp16*", "*scheduler*", "*tokenizer*"
        ]
    )
    print(f"[OK] SDXL Base 1.0 ready at: {sdxl_path}")

    print("\n[3/4] Downloading Pixel-Art LoRA (nerijs/pixel-art-xl)...")
    lora_path = hf_hub_download(repo_id="nerijs/pixel-art-xl", filename="pixel-art-xl.safetensors")
    print(f"[OK] Pixel-Art LoRA ready at: {lora_path}")

    print("\n[4/4] Downloading CLIP Model (ViT-B-32 / laion2b_s34b_b79k)...")
    open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    print("[OK] CLIP ViT-B-32 model ready!")

    print("\n==================================================")
    print("SUCCESS: ALL AI MODEL WEIGHTS SUCCESSFULLY DOWNLOADED AND CACHED!")
    print("==================================================")

except Exception as e:
    print(f"\n[ERROR] Error during model download: {e}")
    sys.exit(1)
