"""
Live terminal progress downloader for all AI models.
"""

import os
import sys
import time
import threading

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from huggingface_hub import snapshot_download, hf_hub_download
import open_clip

stop_monitor = False

def monitor_progress():
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    target_bytes = 6_850_000_000  # ~6.38 GB
    last_bytes = 0
    last_time = time.time()

    while not stop_monitor:
        total_bytes = 0
        latest_file = "model_weights.safetensors"
        if os.path.exists(cache_dir):
            for root, dirs, files in os.walk(cache_dir):
                for f in files:
                    fp = os.path.join(root, f)
                    try:
                        sz = os.path.getsize(fp)
                        total_bytes += sz
                        if f.endswith(".incomplete") or f.endswith(".safetensors"):
                            latest_file = f[:35]
                    except Exception:
                        pass

        now = time.time()
        dt = max(0.5, now - last_time)
        speed_mb = max(0, ((total_bytes - last_bytes) / dt) / (1024 * 1024))
        last_bytes = total_bytes
        last_time = now

        gb_down = total_bytes / (1024 ** 3)
        gb_target = target_bytes / (1024 ** 3)
        percent = min(100, int((total_bytes / target_bytes) * 100))

        bar_len = 25
        filled = int(bar_len * percent / 100)
        bar = "=" * filled + "-" * (bar_len - filled)

        sys.stdout.write(f"\r[{bar}] {percent}% | {gb_down:.2f}GB / {gb_target:.2f}GB | Speed: {speed_mb:.2f} MB/s | File: {latest_file}")
        sys.stdout.flush()
        time.sleep(1.0)

t = threading.Thread(target=monitor_progress, daemon=True)
t.start()

print("==================================================")
print("   LIVE TERMINAL DOWNLOADER (ALL MODELS)         ")
print("==================================================\n")

try:
    print("\n>>> Downloading Florence-2 Base Model...")
    snapshot_download(repo_id="microsoft/Florence-2-base")
    print("\n[OK] Florence-2 ready!")

    print("\n>>> Downloading SDXL Base Model 1.0 (fp16 weights)...")
    snapshot_download(
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        allow_patterns=["*.json", "*.txt", "*.png", "*.md", "*fp16*", "*scheduler*", "*tokenizer*"]
    )
    print("\n[OK] SDXL Base 1.0 ready!")

    print("\n>>> Downloading Pixel-Art LoRA...")
    hf_hub_download(repo_id="nerijs/pixel-art-xl", filename="pixel-art-xl.safetensors")
    print("\n[OK] Pixel-Art LoRA ready!")

    print("\n>>> Downloading CLIP Model...")
    open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    print("\n[OK] CLIP Model ready!")

    stop_monitor = True
    print("\n\n==================================================")
    print("SUCCESS: ALL 6.8 GB MODEL WEIGHTS DOWNLOADED & CACHED!")
    print("==================================================")

except Exception as e:
    stop_monitor = True
    print(f"\n\n[ERROR] Download error: {e}")
    sys.exit(1)
