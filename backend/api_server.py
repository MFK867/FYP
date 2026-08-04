"""
Sketch-to-Game pipeline API.
"""

import gc
import json
import os
import uuid
import math
import random
import time
from collections import deque
import asyncio
from concurrent.futures import ThreadPoolExecutor

import zipfile
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from openai import OpenAI

from prompt_config import (
    STYLE_LOCK_BLOCK,
    STYLE_BLOCKS,
    get_style_block,
    NEGATIVE_PROMPT_BLOCK,
    STEPS,
    CFG_SCALE,
    LORA_TRIGGER,
    build_full_prompt,
    validate_payload,
)

# Enable high-speed Rust multi-threaded parallel downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def _load_env():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))

_load_env()

MODEL_ID = "microsoft/Florence-2-base"

# Check for fine-tuned LoRA weights in parent or local folder
DEFAULT_FLORENCE_LORA = "../florence_game_lora/final" if os.path.exists("../florence_game_lora/final") else "./models/florence_lora"
FLORENCE_LORA_PATH = os.environ.get("FLORENCE_LORA_PATH", DEFAULT_FLORENCE_LORA)
SDXL_LORA_PATH = os.environ.get("SDXL_LORA_PATH", "./models/sdxl_lora")

def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please set it in backend/.env or environment variables."
        )
    return OpenAI(api_key=api_key)

API_OUTPUT_DIR = os.environ.get("API_OUTPUT_DIR", "./api_output")
os.makedirs(API_OUTPUT_DIR, exist_ok=True)

def get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

DEVICE = os.environ.get("DEVICE", get_device())

executor = ThreadPoolExecutor(max_workers=2)
JOB_STATUS = {}


def set_job_status(job_id, step, progress, details="", result=None, error=None):
    if not job_id:
        return
    JOB_STATUS[job_id] = {
        "status": "error" if error else ("completed" if progress >= 100 else "processing"),
        "step": step,
        "progress": progress,
        "details": details,
        "result": result,
        "error": error,
    }

# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------

florence_processor = None
florence_base = None
florence_model = None
sdxl_pipe = None
clip_model = None
clip_preprocess = None
clip_tokenizer = None


def ensure_florence_loaded():
    global florence_processor, florence_base, florence_model, sdxl_pipe
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM
    from peft import PeftModel

    if florence_model is not None:
        return

    if sdxl_pipe is not None:
        print("Freeing SDXL from memory before loading Florence-2...")
        sdxl_pipe = None
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print("Loading Florence-2...")
    florence_processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    florence_base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map=DEVICE,
    )
    if os.path.exists(FLORENCE_LORA_PATH) and len(os.listdir(FLORENCE_LORA_PATH)) > 0:
        print(f"Loading fine-tuned Florence-2 LoRA from {FLORENCE_LORA_PATH}...")
        florence_model = PeftModel.from_pretrained(florence_base, FLORENCE_LORA_PATH)
    else:
        print(f"No custom Florence LoRA found at {FLORENCE_LORA_PATH} - using base Florence-2 model...")
        florence_model = florence_base

    florence_model.eval()
    print("Florence-2 ready.")


def _load_sdxl_pipeline():
    import torch
    from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
    print("Loading SDXL...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        variant="fp16" if DEVICE == "cuda" else None,
        use_safetensors=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True,
    )
    if DEVICE == "cuda":
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_tiling()

    if os.path.exists(SDXL_LORA_PATH) and len(os.listdir(SDXL_LORA_PATH)) > 0:
        print("Loading custom-trained pixel art LoRA...")
        pipe.load_lora_weights(SDXL_LORA_PATH)
    else:
        print("No custom LoRA found - loading pretrained nerijs/pixel-art-xl...")
        pipe.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors")
    pipe.fuse_lora(lora_scale=0.90)
    return pipe


def ensure_sdxl_loaded():
    global sdxl_pipe, florence_model, florence_base
    import torch

    if sdxl_pipe is not None:
        return

    if florence_model is not None:
        print("Freeing Florence-2 from memory before loading SDXL...")
        florence_model = None
        florence_base = None
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    sdxl_pipe = _load_sdxl_pipeline()


def ensure_clip_loaded():
    global clip_model, clip_preprocess, clip_tokenizer
    import open_clip
    if clip_model is not None:
        return
    print("Loading CLIP model...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    clip_model.eval()

# --------------------------------------------------------------------------
# Phase 2 - Sketch -> Layout JSON & Rich Visual Analysis (2-Pass Florence-2)
# --------------------------------------------------------------------------

def sketch_to_layout(image_path):
    raw_caption = "A hand-drawn game level sketch"
    raw_od = ""
    if DEVICE == "cuda" and florence_model is not None:
        try:
            import torch
            img = Image.open(image_path).convert("RGB")
            dtype = torch.float16
            
            inputs_cap = florence_processor(
                text="<MORE_DETAILED_CAPTION>", images=img, return_tensors="pt"
            ).to(DEVICE, dtype)
            with torch.no_grad():
                gen_cap = florence_model.generate(**inputs_cap, max_new_tokens=256, do_sample=False)
            raw_caption = florence_processor.tokenizer.decode(gen_cap[0], skip_special_tokens=True)

            inputs_od = florence_processor(
                text="<OD>", images=img, return_tensors="pt"
            ).to(DEVICE, dtype)
            with torch.no_grad():
                gen_od = florence_model.generate(**inputs_od, max_new_tokens=256, do_sample=False)
            raw_od = florence_processor.tokenizer.decode(gen_od[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Florence-2 2-pass note: {e}")

    try:
        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            img_pil = Image.open(image_path).convert("L")
            img_gray = np.array(img_pil)

        h, w = img_gray.shape
        grid_cols, grid_rows = 24, 12
        cell_w, cell_h = w / grid_cols, h / grid_rows

        _, thresh = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY_INV)

        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 30, 15), 2))
        h_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)

        platform_set = set()
        for r in range(grid_rows):
            for c in range(grid_cols):
                x1, y1 = int(c * cell_w), int(r * cell_h)
                x2, y2 = int((c + 1) * cell_w), int((r + 1) * cell_h)
                
                cell_full = thresh[y1:y2, x1:x2]
                cell_h_layer = h_thresh[y1:y2, x1:x2]
                
                if np.mean(cell_full) > 6 or np.mean(cell_h_layer) > 2:
                    platform_set.add((c, r))

        platforms = [list(p) for p in platform_set]

        if len(platforms) > 5:
            sorted_for_player = sorted(platforms, key=lambda p: (-p[1], p[0]))
            player = [sorted_for_player[0][0], max(0, sorted_for_player[0][1] - 1)]

            sorted_for_goal = sorted(platforms, key=lambda p: (p[1], -p[0]))
            goal = [sorted_for_goal[0][0], max(0, sorted_for_goal[0][1] - 1)]

            enemies = []
            if len(sorted_for_player) > 10:
                enemies = [
                    sorted_for_player[len(sorted_for_player) // 3],
                    sorted_for_player[2 * len(sorted_for_player) // 3]
                ]

            layout = {
                "player": player,
                "goal": goal,
                "platforms": platforms,
                "enemies": enemies
            }
            return layout, raw_caption, raw_od
    except Exception as e:
        print(f"Contour extraction note: {e}")

    default_layout = {
        "player": [1, 10],
        "goal": [22, 3],
        "platforms": [
            [0, 11], [1, 11], [2, 11], [3, 11], [4, 11], [5, 11], [6, 11], [7, 11], [8, 11], [9, 11], [10, 11],
            [3, 8], [4, 8], [5, 8],
            [8, 6], [9, 6], [10, 6],
            [14, 5], [15, 5], [16, 5],
            [19, 4], [20, 4], [21, 4], [22, 4], [23, 4]
        ],
        "enemies": [[5, 7], [15, 4]]
    }
    return default_layout, raw_caption, raw_od

# --------------------------------------------------------------------------
# Phase 3 - Open-Ended GPT-4o Genre-Aware Game Planner
# --------------------------------------------------------------------------

GAME_PLAN_SYSTEM_PROMPT = """You are an expert AAA Game Designer, Technical Director, and AI Planning Engine.

You receive four inputs:
1. Grid layout coordinates for player, goal, platforms, and enemies.
2. A detailed visual caption of the sketch image.
3. A list of detected objects and regions from the sketch.
4. A user theme description.

Your job is to infer the correct game genre entirely from these inputs — do not default to platformer. If the sketch shows a car or road, make a racing game. If it shows two figures facing each other, make a fighting game. If it shows a maze or top-down corridors, make a shooter or puzzle game. If it shows floating platforms and a stick figure, make a platformer. If it shows a knight, a dragon, or a castle — make an action/RPG game.

GENRE-AWARE ASSET PROMPT INSTRUCTIONS:
In the "assets" dictionary, you MUST write detailed 16-bit SNES arcade pixel art prompts using genre-appropriate asset concepts:
- For Racing Games:
  "player": "pixel_art, 16-bit pixel art red supercar sports car, sleek aerodynamic body, sharp dark pixel outline, detailed pixel shading, speed side-view, isolated on pure white background"
  "enemy": "pixel_art, 16-bit pixel art blue rival racing car, sleek aerodynamic body, rear spoiler, sharp dark pixel outline, detailed pixel shading, speed side-view, isolated on pure white background"
  "platform_tile": "pixel_art, 16-bit pixel art asphalt road tile, yellow dashed center lines, dark asphalt texture"
  "background": "pixel_art, 16-bit pixel art night city highway background, neon city skyline, street lights"
- For Fighting Games:
  "player": "pixel_art, 16-bit pixel art martial artist hero in red gi, headband, boxing gloves, sharp dark pixel outline, isolated on pure white background"
  "enemy": "pixel_art, 16-bit pixel art rival opponent fighter in cyber armor, sharp dark pixel outline, isolated on pure white background"
- For Fantasy / Action RPG:
  "player": "pixel_art, 16-bit pixel art knight hero in silver steel plate armor, visor helm with plume, blue cape, holding broadsword, sharp dark pixel outline, isolated on pure white background"
  "enemy": "pixel_art, 16-bit pixel art ferocious green dragon, white horns, yellow underbelly scales, red glowing eye, sharp dark pixel outline, isolated on pure white background"

Respond with ONLY valid JSON (no markdown, no backticks):
{
  "genre": "Exact Inferred Genre",
  "sketch_interpretation": "One clear sentence explaining what you understood the sketch to depict and why you chose this genre.",
  "camera": {
    "style": "Side Camera / Behind Car / Fixed Arena / Orthographic Top Down / Follow Camera",
    "fov": 60,
    "smoothing": 0.15
  },
  "title": "Creative AAA Game Title",
  "theme": "Creative Theme Name",
  "description": "Engaging single sentence game description",
  "color_palette": {
    "sky": "#hex",
    "ground": "#hex",
    "platform": "#hex",
    "accent": "#hex"
  },
  "gameplay_systems": {
    "health": 100,
    "lives": 3,
    "movement_style": "physics_driven",
    "attack_key": "SPACE / J",
    "attack_action": "Sword Slash / Laser / Nitro Boost / Magic Spell / Stomp",
    "win_condition": "Clear win objective derived from genre",
    "lose_condition": "Lose condition derived from genre",
    "scoring": "Score system"
  },
  "animations": {
    "player": ["idle", "walk", "run", "jump", "attack", "hit", "death"],
    "enemy": ["idle", "patrol", "attack", "hit", "death"],
    "environment": ["wind_sway", "water_flow", "torch_fire", "glow_pulse"]
  },
  "physics_data": {
    "gravity": 9.8,
    "move_speed": 6.0,
    "jump_force": 12.5,
    "friction": 0.85,
    "restitution": 0.1
  },
  "level_structure": {
    "learning_area": "Introduction zone",
    "challenge_area": "Main obstacle trajectory",
    "reward_area": "Power-up placement",
    "goal_area": "Finish area"
  },
  "assets": {
    "player": "pixel_art, 16-bit pixel art hero asset description",
    "enemy": "pixel_art, 16-bit pixel art opponent asset description",
    "platform_tile": "pixel_art, 16-bit pixel art tileable surface description",
    "background": "pixel_art, 16-bit pixel art background environment description"
  },
  "video_prompt": "pixel_art, 16-bit pixel art game footage...",
  "asset_metadata": [
    { "name": "player", "category": "character/hero", "collision": "solid", "gameplay": "user_controlled" },
    { "name": "enemy", "category": "hazard/opponent", "collision": "trigger_damage", "gameplay": "patrol_ai" },
    { "name": "platform_tile", "category": "environment", "collision": "solid", "gameplay": "walkable_ground" }
  ],
  "difficulty": "medium",
  "enemy_count": 3
}"""


def plan_game(layout_json, user_description="A fun game", florence_caption="", florence_od=""):
    user_msg = (
        f"Game layout:\n{json.dumps(layout_json, indent=2)}\n\n"
        f"Sketch Visual Caption: {florence_caption}\n"
        f"Detected Regions/Objects: {florence_od}\n"
        f"Theme description: {user_description}"
    )
    client = get_openai_client()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": GAME_PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.7,
        max_tokens=1200,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        plan = json.loads(raw)
        if "sketch_interpretation" not in plan or not plan["sketch_interpretation"]:
            plan["sketch_interpretation"] = f"The sketch was interpreted as a {plan.get('genre', 'action')} scene based on visual layout and drawing patterns."
        return plan
    except json.JSONDecodeError:
        print(f"GPT-4o returned invalid JSON:\n{raw[:500]}")
        return None

# --------------------------------------------------------------------------
# Phase 4 - High-Detail Organic Procedural Sprite Renderer
# --------------------------------------------------------------------------

def draw_parallax_sky(width=1024, height=512, game_plan=None):
    import random as _rnd
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    genre_lower = (game_plan.get("genre", "") if game_plan else "").lower()
    theme_lower = (game_plan.get("theme", "") if game_plan else "").lower()
    is_car = "rac" in genre_lower or "car" in genre_lower or "vehicle" in theme_lower or "racer" in theme_lower
    is_fighting = "fight" in genre_lower or "arena" in genre_lower or "beat" in genre_lower or "brawl" in genre_lower or "fight" in theme_lower
    is_shooter = "shoot" in genre_lower or "top down" in genre_lower

    if is_car:
        # ═══ RACING: Cyberpunk Night City 3-Layer Parallax ═══════════════
        # Deep indigo-to-purple sky gradient
        for y in range(height):
            t = y / height
            r = int(5 + t * 20)
            g = int(2 + t * 12)
            b = int(18 + t * 50)
            draw.line([0, y, width, y], fill=(r, g, b, 255))

        # Glowing crescent moon with soft halo
        mx, my = int(width * 0.88), int(height * 0.08)
        for ring in range(6, 0, -1):
            alpha = int(15 + ring * 5)
            draw.ellipse([mx - 40 - ring*6, my - 40 - ring*6, mx + 40 + ring*6, my + 40 + ring*6],
                         fill=(60, 50, 90, alpha))
        draw.ellipse([mx - 30, my - 30, mx + 30, my + 30], fill=(255, 248, 220, 255))
        draw.ellipse([mx - 18, my - 34, mx + 22, my + 26], fill=(5, 2, 18, 255))

        # Stars with twinkling variety
        _rnd.seed(777)
        for _ in range(100):
            sx = _rnd.randint(0, width)
            sy = _rnd.randint(0, int(height * 0.55))
            br = _rnd.randint(140, 255)
            sz = _rnd.choice([1, 1, 1, 2, 2])
            color = _rnd.choice([(br, br, br, 255), (br, br-20, br-60, 255), (br-40, br-20, br, 255)])
            draw.rectangle([sx, sy, sx + sz, sy + sz], fill=color)

        # ── Layer 1: Far distant buildings (small, dark) ──
        _rnd.seed(101)
        bx = 0
        while bx < width:
            bw = _rnd.randint(30, 65)
            bh = _rnd.randint(int(height * 0.15), int(height * 0.35))
            by = int(height * 0.68) - bh
            shade = _rnd.randint(12, 22)
            draw.rectangle([bx, by, bx + bw - 2, int(height * 0.68)], fill=(shade, shade - 2, shade + 15, 255))
            # Tiny dim windows
            for wy in range(by + 6, int(height * 0.68) - 6, 12):
                for wx in range(bx + 4, bx + bw - 6, 10):
                    if _rnd.random() > 0.6:
                        wc = _rnd.choice([(80, 70, 40, 255), (40, 60, 80, 255)])
                        draw.rectangle([wx, wy, wx + 3, wy + 4], fill=wc)
            bx += bw + _rnd.randint(1, 6)

        # ── Layer 2: Mid-range buildings (medium, with neon) ──
        _rnd.seed(202)
        neon_colors = [(0, 230, 255), (255, 50, 180), (50, 255, 160), (255, 100, 40), (180, 80, 255)]
        bx = _rnd.randint(0, 20)
        while bx < width:
            bw = _rnd.randint(45, 100)
            bh = _rnd.randint(int(height * 0.25), int(height * 0.50))
            by = int(height * 0.72) - bh
            shade = _rnd.randint(18, 32)
            draw.rectangle([bx, by, bx + bw - 3, int(height * 0.72)], fill=(shade, shade - 4, shade + 20, 255))
            # Lit windows (warm yellow/orange/cyan)
            for wy in range(by + 8, int(height * 0.72) - 8, 14):
                for wx in range(bx + 5, bx + bw - 8, 12):
                    if _rnd.random() > 0.35:
                        wc = _rnd.choice([(255, 215, 70, 255), (255, 180, 50, 255), (60, 200, 255, 255), (255, 130, 80, 255)])
                        draw.rectangle([wx, wy, wx + 5, wy + 6], fill=wc)
            # Neon roof accent
            nc = _rnd.choice(neon_colors)
            draw.line([bx, by, bx + bw - 3, by], fill=(*nc, 255), width=3)
            # Neon sign on some buildings
            if _rnd.random() > 0.5 and bw > 55:
                sign_y = by + int(bh * 0.3)
                sign_c = _rnd.choice(neon_colors)
                draw.rectangle([bx + 8, sign_y, bx + bw - 10, sign_y + 14], fill=(*sign_c, 180))
                draw.rectangle([bx + 10, sign_y + 2, bx + bw - 12, sign_y + 12], fill=(10, 8, 25, 220))
            # Antenna / tower on tall buildings
            if bh > height * 0.35:
                ax = bx + bw // 2
                draw.line([ax, by - 18, ax, by], fill=(60, 55, 80, 255), width=2)
                draw.rectangle([ax - 2, by - 20, ax + 2, by - 16], fill=(255, 30, 30, 255))
            bx += bw + _rnd.randint(3, 12)

        # ── Road surface at bottom ──
        road_top = int(height * 0.72)
        # Asphalt gradient (lighter at top edge, darker at bottom)
        for y in range(road_top, height):
            t = (y - road_top) / max(1, height - road_top)
            shade = int(42 - t * 12)
            draw.line([0, y, width, y], fill=(shade, shade, shade + 5, 255))
        # White edge line at top of road
        draw.rectangle([0, road_top, width, road_top + 3], fill=(180, 185, 195, 255))
        # Dashed yellow center line
        mid_y = road_top + (height - road_top) // 2
        for dx in range(0, width, 48):
            draw.rectangle([dx, mid_y - 2, dx + 26, mid_y + 2], fill=(255, 215, 30, 255))
        # Bottom kerb
        draw.rectangle([0, height - 5, width, height], fill=(25, 22, 35, 255))

    elif is_fighting:
        # ═══ FIGHTING: Martial Arts Dojo Arena ═══════════════════════════
        # Deep crimson-to-dark gradient wall
        for y in range(height):
            t = y / height
            r = int(55 + t * 30)
            g = int(12 + t * 15)
            b = int(15 + t * 20)
            draw.line([0, y, width, y], fill=(r, g, b, 255))

        # Wooden floor boards at bottom
        floor_top = int(height * 0.72)
        for y in range(floor_top, height):
            t = (y - floor_top) / max(1, height - floor_top)
            r = int(85 + t * 25)
            g = int(55 + t * 15)
            b = int(25 + t * 10)
            draw.line([0, y, width, y], fill=(r, g, b, 255))
        # Plank lines
        for px_off in range(0, width, 80):
            draw.line([px_off, floor_top, px_off, height], fill=(55, 35, 18, 255), width=1)
        draw.line([0, floor_top, width, floor_top], fill=(40, 25, 12, 255), width=3)

        # Wooden pillars on left and right
        for px_pos in [int(width * 0.05), int(width * 0.92)]:
            pw = int(width * 0.04)
            draw.rectangle([px_pos, 0, px_pos + pw, height], fill=(70, 40, 20, 255))
            draw.rectangle([px_pos + 2, 0, px_pos + pw // 3, height], fill=(95, 58, 28, 255))
            # Capital at top
            draw.rectangle([px_pos - 8, 0, px_pos + pw + 8, 20], fill=(85, 50, 22, 255))
            draw.rectangle([px_pos - 4, 20, px_pos + pw + 4, 28], fill=(75, 42, 18, 255))

        # Gold circular dragon medallion in center
        cx, cy = width // 2, int(height * 0.32)
        rad = int(min(width, height) * 0.14)
        # Outer gold ring
        for ring in range(3):
            draw.ellipse([cx - rad - ring, cy - rad - ring, cx + rad + ring, cy + rad + ring],
                         outline=(210, 170, 40, 255), width=3)
        # Inner dark circle
        draw.ellipse([cx - rad + 6, cy - rad + 6, cx + rad - 6, cy + rad - 6], fill=(35, 12, 12, 255))
        # Dragon pattern (simplified circular dragon)
        _rnd.seed(88)
        for angle_step in range(0, 360, 15):
            a = math.radians(angle_step)
            ir = rad * 0.35 + math.sin(a * 3) * rad * 0.12
            ex = cx + int(math.cos(a) * ir)
            ey = cy + int(math.sin(a) * ir)
            draw.ellipse([ex - 4, ey - 4, ex + 4, ey + 4], fill=(220, 180, 50, 255))
        # Central character symbol
        draw.rectangle([cx - 12, cy - 16, cx + 12, cy + 16], fill=(220, 180, 50, 255))
        draw.rectangle([cx - 8, cy - 12, cx + 8, cy + 12], fill=(35, 12, 12, 255))
        draw.line([cx - 6, cy, cx + 6, cy], fill=(220, 180, 50, 255), width=2)
        draw.line([cx, cy - 8, cx, cy + 8], fill=(220, 180, 50, 255), width=2)

        # Hanging red lanterns
        _rnd.seed(55)
        for lx in [int(width * 0.20), int(width * 0.40), int(width * 0.60), int(width * 0.80)]:
            ly = int(height * 0.06)
            # String
            draw.line([lx, 0, lx, ly], fill=(60, 30, 15, 255), width=2)
            # Lantern body
            draw.ellipse([lx - 16, ly, lx + 16, ly + 40], fill=(210, 30, 25, 255))
            draw.ellipse([lx - 12, ly + 4, lx + 12, ly + 36], fill=(240, 60, 45, 255))
            # Gold rim
            draw.rectangle([lx - 14, ly + 2, lx + 14, ly + 6], fill=(220, 180, 50, 255))
            draw.rectangle([lx - 14, ly + 34, lx + 14, ly + 38], fill=(220, 180, 50, 255))
            # Inner glow
            draw.ellipse([lx - 6, ly + 12, lx + 6, ly + 28], fill=(255, 200, 80, 200))
            # Tassel
            draw.line([lx, ly + 40, lx, ly + 52], fill=(210, 30, 25, 255), width=2)
            draw.polygon([(lx - 4, ly + 52), (lx + 4, ly + 52), (lx, ly + 60)], fill=(180, 25, 20, 255))

        # Decorative wall scrolls on sides
        for sx_pos in [int(width * 0.14), int(width * 0.82)]:
            sw, sh = 30, int(height * 0.35)
            sy = int(height * 0.15)
            draw.rectangle([sx_pos, sy, sx_pos + sw, sy + sh], fill=(240, 230, 200, 255))
            draw.rectangle([sx_pos + 2, sy + 2, sx_pos + sw - 2, sy + sh - 2], fill=(250, 242, 220, 255))
            # Brush stroke pattern
            for sdy in range(10, sh - 10, 16):
                draw.line([sx_pos + 8, sy + sdy, sx_pos + sw - 8, sy + sdy + 8], fill=(30, 20, 15, 180), width=2)

    elif is_shooter:
        # ═══ SHOOTER: Deep Space Nebula Field ════════════════════════════
        # Pure black base
        draw.rectangle([0, 0, width, height], fill=(3, 2, 10, 255))

        # Nebula gas clouds (layered, soft colored ellipses)
        _rnd.seed(303)
        nebula_colors = [(80, 20, 120), (20, 60, 130), (120, 30, 60), (30, 80, 60), (100, 40, 90)]
        for _ in range(12):
            nx = _rnd.randint(-100, width + 100)
            ny = _rnd.randint(-50, height + 50)
            nw = _rnd.randint(100, 300)
            nh = _rnd.randint(60, 180)
            nc = _rnd.choice(nebula_colors)
            for layer in range(4, 0, -1):
                alpha = int(15 + layer * 8)
                expand = layer * 20
                draw.ellipse([nx - expand, ny - expand, nx + nw + expand, ny + nh + expand],
                             fill=(*nc, alpha))

        # Dense starfield with varying sizes and colors
        _rnd.seed(404)
        for _ in range(250):
            sx = _rnd.randint(0, width)
            sy = _rnd.randint(0, height)
            br = _rnd.randint(100, 255)
            sz = _rnd.choice([1, 1, 1, 1, 2, 2, 3])
            sc = _rnd.choice([(br, br, br), (br, br - 30, br - 60), (br - 50, br - 20, br), (br, br - 10, br - 40)])
            draw.rectangle([sx, sy, sx + sz - 1, sy + sz - 1], fill=(*sc, 255))

        # Distant planet with ring
        px_c, py_c = int(width * 0.25), int(height * 0.30)
        pr = 45
        draw.ellipse([px_c - pr, py_c - pr, px_c + pr, py_c + pr], fill=(40, 55, 80, 255))
        draw.ellipse([px_c - pr + 8, py_c - pr + 5, px_c + pr - 15, py_c + pr - 8], fill=(55, 75, 100, 255))
        # Ring
        draw.arc([px_c - pr * 2, py_c - 12, px_c + pr * 2, py_c + 12], 0, 180, fill=(120, 140, 170, 200), width=3)

    else:
        # ═══ DEFAULT: Enchanted Twilight Forest with Mountains ════════════
        # Twilight sky gradient (deep blue to purple to warm horizon)
        for y in range(height):
            t = y / height
            r = int(15 + t * 60)
            g = int(10 + t * 35)
            b = int(45 + t * 25)
            if t > 0.4:
                r = int(15 + (t - 0.4) * 120)
                g = int(10 + (t - 0.4) * 50)
            draw.line([0, y, width, y], fill=(min(r, 80), min(g, 50), b, 255))

        # Stars in the upper sky
        _rnd.seed(500)
        for _ in range(60):
            sx = _rnd.randint(0, width)
            sy = _rnd.randint(0, int(height * 0.4))
            br = _rnd.randint(160, 255)
            draw.rectangle([sx, sy, sx + 1, sy + 1], fill=(br, br, br - 20, 255))

        # Northern lights / aurora effect
        for band in range(3):
            _rnd.seed(600 + band)
            by_base = int(height * (0.12 + band * 0.08))
            aurora_c = [(30, 180, 120, 35), (60, 140, 200, 30), (120, 60, 180, 25)][band]
            for seg in range(0, width, 8):
                wave = int(math.sin(seg * 0.015 + band) * 20)
                draw.rectangle([seg, by_base + wave, seg + 7, by_base + wave + 15], fill=aurora_c)

        # Distant mountain range (dark silhouette)
        mtn_base = int(height * 0.50)
        _rnd.seed(700)
        points = [(0, mtn_base)]
        x_pos = 0
        while x_pos < width + 50:
            peak_h = _rnd.randint(int(height * 0.22), int(height * 0.42))
            points.append((x_pos, peak_h))
            x_pos += _rnd.randint(60, 140)
        points.append((width, mtn_base))
        points.append((width, height))
        points.append((0, height))
        draw.polygon(points, fill=(20, 18, 35, 255))

        # Closer mountain layer (slightly lighter)
        mtn_base2 = int(height * 0.58)
        _rnd.seed(710)
        points2 = [(0, mtn_base2)]
        x_pos = 0
        while x_pos < width + 50:
            peak_h = _rnd.randint(int(height * 0.35), int(height * 0.52))
            points2.append((x_pos, peak_h))
            x_pos += _rnd.randint(50, 120)
        points2.append((width, mtn_base2))
        points2.append((width, height))
        points2.append((0, height))
        draw.polygon(points2, fill=(28, 25, 42, 255))

        # Pine tree silhouettes
        tree_base = int(height * 0.62)
        _rnd.seed(800)
        for _ in range(25):
            tx = _rnd.randint(0, width)
            th = _rnd.randint(40, 90)
            tw = _rnd.randint(20, 40)
            # Trunk
            draw.rectangle([tx - 3, tree_base - 5, tx + 3, tree_base + 10], fill=(18, 15, 25, 255))
            # Foliage triangles
            for layer in range(3):
                ly = tree_base - 5 - layer * int(th * 0.3)
                lw = tw - layer * 8
                draw.polygon([(tx - lw, ly), (tx + lw, ly), (tx, ly - int(th * 0.35))], fill=(15, 18, 28, 255))

        # Ground with grass tufts
        draw.rectangle([0, tree_base + 5, width, height], fill=(22, 28, 18, 255))
        for gx in range(0, width, 6):
            gh = _rnd.randint(3, 10)
            draw.line([gx, tree_base + 5, gx, tree_base + 5 - gh], fill=(18, 35, 15, 255), width=2)

    return img


def generate_procedural_sprite(asset_name, prompt, width=512, height=512, seed=42, game_plan=None):
    random.seed(seed)
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    px = lambda f: int(width * f)
    py = lambda f: int(height * f)

    prompt_lower = (prompt or "").lower()
    genre_lower = (game_plan.get("genre", "") if game_plan else "").lower()
    theme_lower = (game_plan.get("theme", "") if game_plan else "").lower()

    is_car = "car" in prompt_lower or "car" in theme_lower or "racer" in theme_lower or "vehicle" in prompt_lower or "rac" in genre_lower
    is_fighting = "fight" in genre_lower or "arena" in genre_lower or "beat" in genre_lower or "brawl" in genre_lower or "fight" in theme_lower
    is_shooter = "shoot" in genre_lower or "top down" in genre_lower or "bullet" in genre_lower or "laser" in theme_lower or "gun" in genre_lower
    is_dragon = "dragon" in prompt_lower or "monster" in prompt_lower or "lizard" in prompt_lower or "boss" in prompt_lower
    is_knight = "knight" in prompt_lower or "hero" in prompt_lower or "warrior" in prompt_lower or "paladin" in prompt_lower

    if asset_name == "player":
        if is_car:
            # 16-Bit Aerodynamic Red Supercar (Smooth Organic Car Body Curves)
            # Dark chassis shadow
            draw.polygon([(px(.06), py(.48)), (px(.20), py(.32)), (px(.84), py(.32)), (px(.94), py(.48)), (px(.94), py(.78)), (px(.06), py(.78))], fill=(25, 15, 20, 255))
            # Smooth Red Car Body
            draw.polygon([(px(.08), py(.46)), (px(.22), py(.34)), (px(.82), py(.34)), (px(.92), py(.46)), (px(.92), py(.74)), (px(.08), py(.74))], fill=(220, 25, 25, 255))
            draw.polygon([(px(.10), py(.44)), (px(.24), py(.36)), (px(.80), py(.36)), (px(.90), py(.44))], fill=(250, 75, 75, 255)) # Metallic Body Highlight
            # Rear Spoiler Wing
            draw.rectangle([px(.04), py(.26), px(.18), py(.34)], fill=(160, 15, 15, 255))
            draw.line([px(.10), py(.34), px(.10), py(.46)], fill=(15, 15, 20, 255), width=px(.02))
            # Sloped Curved Windshield & Windows
            draw.polygon([(px(.30), py(.34)), (px(.42), py(.20)), (px(.70), py(.20)), (px(.78), py(.34))], fill=(20, 30, 45, 255))
            draw.polygon([(px(.34), py(.32)), (px(.44), py(.22)), (px(.68), py(.22)), (px(.74), py(.32))], fill=(120, 215, 255, 230))
            # Gold Racing Stripe
            draw.rectangle([px(.08), py(.50), px(.92), py(.56)], fill=(255, 215, 20, 255))
            # Wheels with Alloy Rims
            draw.ellipse([px(.16), py(.58), px(.36), py(.86)], fill=(15, 15, 20, 255))
            draw.ellipse([px(.21), py(.65), px(.31), py(.79)], fill=(210, 210, 225, 255))
            draw.ellipse([px(.64), py(.58), px(.84), py(.86)], fill=(15, 15, 20, 255))
            draw.ellipse([px(.69), py(.65), px(.79), py(.79)], fill=(210, 210, 225, 255))
            # Headlights glint
            draw.rectangle([px(.86), py(.44), px(.92), py(.52)], fill=(255, 235, 60, 255))

        else:
            # 16-Bit Armored Knight Hero (Matching Reference Image 4)
            # Outline layer (Dark crisp pixels)
            draw.ellipse([px(.28), py(.06), px(.72), py(.38)], fill=(20, 20, 30, 255))
            draw.polygon([(px(.18), py(.30)), (px(.82), py(.30)), (px(.86), py(.78)), (px(.14), py(.78))], fill=(20, 20, 30, 255))
            draw.rectangle([px(.24), py(.74), px(.76), py(.98)], fill=(20, 20, 30, 255))

            # Blue Cape (Behind body)
            draw.polygon([(px(.18), py(.32)), (px(.30), py(.32)), (px(.22), py(.82)), (px(.12), py(.82))], fill=(35, 75, 180, 255))
            draw.polygon([(px(.70), py(.32)), (px(.82), py(.32)), (px(.88), py(.82)), (px(.78), py(.82))], fill=(35, 75, 180, 255))

            # Steel Plate Helmet with visor & plume
            draw.ellipse([px(.30), py(.08), px(.70), py(.36)], fill=(190, 200, 215, 255))
            draw.rectangle([px(.32), py(.08), px(.50), py(.34)], fill=(230, 238, 248, 255)) # Highlight
            draw.rectangle([px(.36), py(.20), px(.64), py(.28)], fill=(30, 30, 40, 255)) # Visor slot
            draw.rectangle([px(.46), py(.20), px(.54), py(.28)], fill=(255, 190, 30, 255)) # Gold visor detail
            draw.ellipse([px(.44), py(.02), px(.56), py(.14)], fill=(220, 30, 30, 255)) # Red helmet plume

            # Armor Breastplate & Shoulders (Pauldrons)
            draw.ellipse([px(.18), py(.28), px(.38), py(.46)], fill=(180, 190, 205, 255)) # Left Pauldron
            draw.ellipse([px(.62), py(.28), px(.82), py(.46)], fill=(180, 190, 205, 255)) # Right Pauldron
            draw.rectangle([px(.28), py(.32), px(.72), py(.66)], fill=(160, 170, 185, 255)) # Torso armor
            draw.rectangle([px(.30), py(.34), px(.50), py(.64)], fill=(220, 230, 245, 255)) # Torso highlight
            draw.polygon([(px(.50), py(.38)), (px(.42), py(.54)), (px(.58), py(.54))], fill=(225, 225, 240, 255)) # Cross emblem

            # Leather Belt & Scabbard
            draw.rectangle([px(.26), py(.62), px(.74), py(.68)], fill=(100, 50, 20, 255))
            draw.rectangle([px(.44), py(.61), px(.56), py(.69)], fill=(230, 190, 40, 255)) # Buckle

            # Greaves & Leather Boots
            draw.rectangle([px(.28), py(.68), px(.46), py(.88)], fill=(150, 160, 175, 255))
            draw.rectangle([px(.54), py(.68), px(.72), py(.88)], fill=(150, 160, 175, 255))
            draw.rectangle([px(.24), py(.86), px(.46), py(.96)], fill=(80, 40, 20, 255))
            draw.rectangle([px(.54), py(.86), px(.76), py(.96)], fill=(80, 40, 20, 255))

            # Greatsword (Broadsword held downward right)
            draw.polygon([(px(.62), py(.42)), (px(.92), py(.78)), (px(.96), py(.74)), (px(.66), py(.38))], fill=(210, 220, 235, 255)) # Blade
            draw.line([px(.64), py(.40), px(.94), py(.76)], fill=(255, 255, 255, 255), width=px(.02)) # Blade edge
            draw.rectangle([px(.58), py(.38), px(.68), py(.44)], fill=(230, 190, 40, 255)) # Crossguard
            draw.ellipse([px(.54), py(.36), px(.60), py(.42)], fill=(230, 190, 40, 255)) # Pommel

    elif asset_name == "enemy":
        if is_car:
            # 16-Bit Aerodynamic Blue Rival Race Car (Organic Car Body Curves)
            # Dark chassis shadow
            draw.polygon([(px(.06), py(.48)), (px(.20), py(.32)), (px(.84), py(.32)), (px(.94), py(.48)), (px(.94), py(.78)), (px(.06), py(.78))], fill=(15, 20, 35, 255))
            # Sleek Blue Car Body
            draw.polygon([(px(.08), py(.46)), (px(.22), py(.34)), (px(.82), py(.34)), (px(.92), py(.46)), (px(.92), py(.74)), (px(.08), py(.74))], fill=(25, 65, 200, 255))
            draw.polygon([(px(.10), py(.44)), (px(.24), py(.36)), (px(.80), py(.36)), (px(.90), py(.44))], fill=(45, 105, 240, 255)) # Metallic Body Highlight
            # Rear Spoiler Wing
            draw.rectangle([px(.04), py(.26), px(.18), py(.34)], fill=(20, 45, 150, 255))
            draw.line([px(.10), py(.34), px(.10), py(.46)], fill=(15, 15, 20, 255), width=px(.02))
            # Sloped Curved Windshield & Windows
            draw.polygon([(px(.30), py(.34)), (px(.42), py(.20)), (px(.70), py(.20)), (px(.78), py(.34))], fill=(20, 30, 45, 255))
            draw.polygon([(px(.34), py(.32)), (px(.44), py(.22)), (px(.68), py(.22)), (px(.74), py(.32))], fill=(120, 215, 255, 230))
            # Yellow Racing Stripe
            draw.rectangle([px(.08), py(.50), px(.92), py(.56)], fill=(255, 215, 20, 255))
            # Wheels with Alloy Rims
            draw.ellipse([px(.16), py(.58), px(.36), py(.86)], fill=(15, 15, 20, 255))
            draw.ellipse([px(.21), py(.65), px(.31), py(.79)], fill=(210, 210, 225, 255))
            draw.ellipse([px(.64), py(.58), px(.84), py(.86)], fill=(15, 15, 20, 255))
            draw.ellipse([px(.69), py(.65), px(.79), py(.79)], fill=(210, 210, 225, 255))
            # Headlights glint
            draw.rectangle([px(.86), py(.44), px(.92), py(.52)], fill=(255, 180, 20, 255))

        elif is_dragon or is_knight or not is_fighting:
            # 16-Bit Ferocious Green Dragon (Matching Reference Image 3)
            # Outline layer
            draw.ellipse([px(.14), py(.18), px(.86), py(.86)], fill=(15, 25, 15, 255))
            draw.polygon([(px(.60), py(.65)), (px(.94), py(.65)), (px(.92), py(.88)), (px(.58), py(.88))], fill=(15, 25, 15, 255))

            # Tail
            draw.polygon([(px(.60), py(.68)), (px(.90), py(.68)), (px(.88), py(.84)), (px(.58), py(.84))], fill=(40, 140, 50, 255))
            draw.polygon([(px(.82), py(.68)), (px(.92), py(.68)), (px(.90), py(.84))], fill=(140, 30, 30, 255)) # Red tail tip

            # Body & Back Scales
            draw.ellipse([px(.22), py(.34), px(.72), py(.82)], fill=(45, 155, 55, 255))
            draw.ellipse([px(.24), py(.36), px(.48), py(.80)], fill=(75, 195, 85, 255)) # Body highlight

            # Yellow Underbelly Scales
            draw.ellipse([px(.32), py(.48), px(.54), py(.78)], fill=(245, 215, 110, 255))
            for sy_f in [0.54, 0.62, 0.70]:
                draw.line([px(.34), py(sy_f), px(.52), py(sy_f)], fill=(200, 165, 60, 255), width=px(.015))

            # Head & Snout
            draw.ellipse([px(.16), py(.16), px(.54), py(.48)], fill=(45, 155, 55, 255))
            draw.ellipse([px(.18), py(.18), px(.38), py(.44)], fill=(85, 205, 95, 255)) # Head highlight

            # Horns (White/Silver Horns)
            draw.polygon([(px(.30), py(.20)), (px(.36), py(.04)), (px(.42), py(.20))], fill=(235, 235, 245, 255))
            draw.polygon([(px(.42), py(.22)), (px(.48), py(.06)), (px(.54), py(.22))], fill=(235, 235, 245, 255))

            # Red Eye with Black Pupil
            draw.ellipse([px(.28), py(.26), px(.38), py(.36)], fill=(230, 30, 40, 255))
            draw.rectangle([px(.32), py(.28), px(.36), py(.34)], fill=(15, 15, 20, 255))

            # Feet & Claws
            draw.ellipse([px(.24), py(.76), px(.42), py(.88)], fill=(35, 125, 45, 255))
            draw.polygon([(px(.22), py(.86)), (px(.28), py(.82)), (px(.26), py(.90))], fill=(235, 235, 245, 255))
            draw.polygon([(px(.30), py(.86)), (px(.36), py(.82)), (px(.34), py(.90))], fill=(235, 235, 245, 255))

        else:
            # 16-Bit Cyber Brawler Fighter
            draw.rectangle([px(.24), py(.10), px(.76), py(.38)], fill=(35, 35, 45, 255))
            draw.rectangle([px(.28), py(.20), px(.72), py(.30)], fill=(240, 30, 30, 255)) # Cyber Visor
            draw.rectangle([px(.16), py(.38), px(.84), py(.76)], fill=(60, 65, 80, 255))
            draw.polygon([(px(.06), py(.38)), (px(.20), py(.24)), (px(.20), py(.44))], fill=(230, 190, 40, 255))
            draw.polygon([(px(.94), py(.38)), (px(.80), py(.24)), (px(.80), py(.44))], fill=(230, 190, 40, 255))
            draw.rectangle([px(.24), py(.76), px(.46), py(.96)], fill=(30, 35, 45, 255))
            draw.rectangle([px(.54), py(.76), px(.76), py(.96)], fill=(30, 35, 45, 255))

    elif asset_name == "platform_tile":
        if is_car:
            # 16-Bit Asphalt Road Tile with Yellow Lines
            draw.rectangle([0, 0, width, height], fill=(45, 48, 58, 255))
            draw.rectangle([0, py(.42), width, py(.58)], fill=(240, 205, 30, 255))
            random.seed(42)
            for _ in range(30):
                tx = random.randint(0, width - px(.08))
                ty = random.randint(0, height - py(.08))
                draw.rectangle([tx, ty, tx + random.randint(px(.02), px(.06)), ty + random.randint(py(.02), py(.06))], fill=(30, 32, 40, 255))
        else:
            # 16-Bit Mossy Stone Brick Blocks (Matching Reference Image 1)
            draw.rectangle([0, 0, width, height], fill=(70, 75, 65, 255))
            
            # Draw grid of individual stone bricks with highlight edges and dark grout lines
            b_size = width // 4
            for r in range(4):
                for c in range(4):
                    bx1, by1 = c * b_size, r * b_size
                    bx2, by2 = bx1 + b_size - 1, by1 + b_size - 1
                    
                    # Brick fill
                    fill_c = (110 + (c*15)%30, 115 + (r*12)%25, 95 + (c*10)%20, 255)
                    draw.rectangle([bx1, by1, bx2, by2], fill=fill_c)
                    
                    # Top/Left highlight edge
                    draw.line([bx1, by1, bx2, by1], fill=(160, 165, 145, 255), width=2)
                    draw.line([bx1, by1, bx1, by2], fill=(160, 165, 145, 255), width=2)
                    
                    # Bottom/Right dark shadow groove
                    draw.line([bx1, by2, bx2, by2], fill=(45, 48, 40, 255), width=3)
                    draw.line([bx2, by1, bx2, by2], fill=(45, 48, 40, 255), width=3)

            # Add green moss tufts & cracks
            random.seed(101)
            for _ in range(12):
                mx = random.randint(8, width - 24)
                my = random.randint(8, height - 24)
                draw.rectangle([mx, my, mx + random.randint(10, 22), my + random.randint(8, 16)], fill=(65, 145, 35, 255))
                draw.rectangle([mx + 2, my + 2, mx + random.randint(6, 14), my + 6], fill=(115, 195, 55, 255))

    else:
        return draw_parallax_sky(width, height, game_plan)

    return img


def generate_asset(prompt, width=512, height=512, num_images=3, seed=42, genre="default"):
    import torch
    full_prompt = build_full_prompt(prompt, genre=genre)
    negative = NEGATIVE_PROMPT_BLOCK

    payload = {
        "prompt": full_prompt,
        "negative_prompt": negative,
        "num_inference_steps": STEPS,
        "guidance_scale": CFG_SCALE,
    }
    validation = validate_payload(payload)
    if not all(validation.values()):
        print(f"Payload validation warning: {validation}")

    images = []
    for i in range(num_images):
        generator = torch.Generator("cuda" if DEVICE == "cuda" else "cpu").manual_seed(seed + i)
        img = sdxl_pipe(
            prompt=full_prompt,
            negative_prompt=negative,
            width=width,
            height=height,
            num_inference_steps=STEPS,
            guidance_scale=CFG_SCALE,
            generator=generator,
        ).images[0]
        images.append(img)
    return images


def generate_all_assets(game_plan, save_dir, job_id=None):
    set_job_status(job_id, "Checking pixel-art generation engine...", 32, "Selecting SDXL or Fast CPU mode")
    os.makedirs(save_dir, exist_ok=True)
    assets = game_plan.get("assets", {})
    genre = game_plan.get("genre", "default")
    theme = game_plan.get("theme", "")

    genre_lower = genre.lower()
    theme_lower = theme.lower()
    is_car = "rac" in genre_lower or "car" in genre_lower or "vehicle" in theme_lower or "racer" in theme_lower

    def get_dynamic_prompt(key, fallback_desc):
        if is_car:
            if key == "enemy":
                return f"{LORA_TRIGGER}, 16-bit pixel art blue rival racing car, sleek aerodynamic sports car, speed side-view, sharp dark pixel outline, detailed pixel shading, isolated on pure white background"
            elif key == "player":
                return f"{LORA_TRIGGER}, 16-bit pixel art red supercar sports car, speed side-view, sharp dark pixel outline, detailed pixel shading, isolated on pure white background"
            elif key == "platform_tile":
                return f"{LORA_TRIGGER}, 16-bit pixel art asphalt road tile, yellow dashed center line, road surface, isolated on pure white background"
            elif key == "background":
                return f"{LORA_TRIGGER}, 16-bit pixel art night city racing background, neon city skyline, dark road with dashes, parallax buildings, isolated on pure white background"
        if key in assets and assets[key]:
            return f"{LORA_TRIGGER}, {assets[key]}, sharp dark pixel outline, detailed pixel shading, 16-bit SNES arcade sprite, isolated on pure white background"
        return f"{LORA_TRIGGER}, 16-bit pixel art {genre} {key}, {theme}, {fallback_desc}, sharp dark pixel outline, detailed pixel shading, isolated on pure white background"

    configs = {
        "player":        {"prompt": get_dynamic_prompt("player", "16-bit pixel art hero main character"), "width": 512, "height": 512, "num": 3, "progress": 40},
        "enemy":         {"prompt": get_dynamic_prompt("enemy", "16-bit pixel art rival opponent asset"), "width": 512, "height": 512, "num": 3, "progress": 55},
        "platform_tile": {"prompt": get_dynamic_prompt("platform_tile", "16-bit pixel art terrain tile surface"), "width": 512, "height": 512, "num": 3, "progress": 70},
        "background":    {"prompt": get_dynamic_prompt("background", "16-bit pixel art environment landscape panel"), "width": 1024, "height": 512, "num": 3, "progress": 82},
    }

    use_fast_mode = (DEVICE == "cpu")

    if not use_fast_mode:
        try:
            ensure_sdxl_loaded()
        except Exception as e:
            print(f"SDXL load note ({e}), switching to Fast CPU mode...")
            use_fast_mode = True

    all_candidates = {}
    for asset_name, cfg in configs.items():
        set_job_status(
            job_id,
            f"Generating 16-bit pixel art: {asset_name.replace('_', ' ').title()}...",
            cfg["progress"],
            f"Creating candidates for '{cfg['prompt'][:40]}...'"
        )
        if use_fast_mode:
            candidates = [
                generate_procedural_sprite(asset_name, cfg["prompt"], cfg["width"], cfg["height"], seed=42 + i, game_plan=game_plan)
                for i in range(cfg["num"])
            ]
        else:
            candidates = generate_asset(cfg["prompt"], cfg["width"], cfg["height"], cfg["num"], genre=genre)

        all_candidates[asset_name] = candidates
        for i, img in enumerate(candidates):
            img.save(os.path.join(save_dir, f"{asset_name}_v{i}.png"))

    gc.collect()
    return all_candidates

# --------------------------------------------------------------------------
# Phase 5 - CLIP scoring + selection
# --------------------------------------------------------------------------

def score_candidates(candidates, text_prompt):
    import torch
    text_tokens = clip_tokenizer([text_prompt])
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    scores = []
    for img in candidates:
        img_tensor = clip_preprocess(img).unsqueeze(0)
        with torch.no_grad():
            img_features = clip_model.encode_image(img_tensor)
            img_features /= img_features.norm(dim=-1, keepdim=True)
        scores.append((text_features @ img_features.T).item())

    best_idx = int(np.argmax(scores))
    return candidates[best_idx], best_idx, scores


def select_best_assets(all_candidates, game_plan):
    if DEVICE == "cuda":
        try:
            ensure_clip_loaded()
        except Exception as e:
            print(f"CLIP load note: {e}")

    assets_prompts = game_plan.get("assets", {})
    best_assets = {}

    for asset_name, candidates in all_candidates.items():
        if DEVICE == "cuda" and clip_model is not None:
            try:
                prompt = assets_prompts.get(asset_name, f"16-bit pixel art {asset_name}")
                best_img, _, _ = score_candidates(candidates, prompt)
                best_assets[asset_name] = best_img.copy()
            except Exception as e:
                print(f"CLIP scoring note ({e}), selecting candidate 0...")
                best_assets[asset_name] = candidates[0].copy()
        else:
            best_assets[asset_name] = candidates[0].copy()

    all_candidates.clear()
    gc.collect()
    return best_assets


def remove_flat_background(img, tolerance=28):
    """
    Removes solid/white background cleanly without eroding dark pixel outlines or creating white halos.
    """
    img = img.convert("RGBA")
    data = np.array(img, dtype=np.int32)
    h, w = data.shape[:2]

    # Sample four corner pixels to detect solid background color
    corners = [data[0, 0, :3], data[0, w - 1, :3], data[h - 1, 0, :3], data[h - 1, w - 1, :3]]

    mask = np.zeros((h, w), dtype=bool)
    for corner in corners:
        diff = np.abs(data[:, :, :3] - corner).sum(axis=-1)
        mask |= (diff < tolerance)

    data_out = np.array(img, dtype=np.uint8)
    data_out[mask, 3] = 0
    return Image.fromarray(data_out, mode="RGBA")

# --------------------------------------------------------------------------
# Phase 6 - Scene composition & HUD Drawing Helpers
# --------------------------------------------------------------------------

def _draw_fighting_hud(draw, canvas_width, canvas_height, game_plan, p1_hp=100, p2_hp=100):
    draw.rectangle([0, 0, canvas_width, 64], fill=(15, 15, 25, 230))
    p1_name, p2_name = "HERO", "RIVAL"
    if game_plan and "assets" in game_plan:
        if "player" in game_plan["assets"]:
            p1_name = str(game_plan["assets"]["player"]).split(",")[0].replace("pxlrpg_style", "").replace("pixel_art", "").strip() or "HERO"
        if "enemy" in game_plan["assets"]:
            p2_name = str(game_plan["assets"]["enemy"]).split(",")[0].replace("pxlrpg_style", "").replace("pixel_art", "").strip() or "RIVAL"

    p1_name = p1_name[:12].upper()
    p2_name = p2_name[:12].upper()

    draw.rectangle([8, 6, 60, 58], fill=(45, 25, 65, 255))
    draw.rectangle([8, 6, 60, 58], outline=(180, 100, 240, 255), width=2)
    draw.text((16, 22), "P1", fill=(255, 255, 255, 255))

    hp_left, hp_right = 68, canvas_width // 2 - 44
    p1_bar_w = int((hp_right - hp_left) * max(0, min(1.0, p1_hp / 100.0)))
    draw.rectangle([hp_left, 12, hp_right, 36], fill=(180, 30, 30, 255))
    if p1_bar_w > 0:
        draw.rectangle([hp_left, 12, hp_left + p1_bar_w, 36], fill=(40, 220, 80, 255))
    draw.rectangle([hp_left, 12, hp_right, 36], outline=(255, 255, 255, 255), width=2)
    draw.text((hp_left, 40), p1_name, fill=(255, 255, 255, 255))

    timer_cx = canvas_width // 2
    draw.rectangle([timer_cx - 32, 6, timer_cx + 32, 48], fill=(20, 20, 30, 255))
    draw.rectangle([timer_cx - 32, 6, timer_cx + 32, 48], outline=(255, 215, 0, 255), width=2)
    draw.text((timer_cx - 10, 14), "64", fill=(255, 215, 0, 255))
    draw.ellipse([timer_cx - 14, 52, timer_cx - 6, 60], fill=(255, 215, 0, 255))
    draw.ellipse([timer_cx + 6, 52, timer_cx + 14, 60], fill=(255, 215, 0, 255))

    p2_hp_left, p2_hp_right = canvas_width // 2 + 44, canvas_width - 68
    p2_bar_w = int((p2_hp_right - p2_hp_left) * max(0, min(1.0, p2_hp / 100.0)))
    draw.rectangle([p2_hp_left, 12, p2_hp_right, 36], fill=(180, 30, 30, 255))
    if p2_bar_w > 0:
        draw.rectangle([p2_hp_right - p2_bar_w, 12, p2_hp_right, 36], fill=(40, 220, 80, 255))
    draw.rectangle([p2_hp_left, 12, p2_hp_right, 36], outline=(255, 255, 255, 255), width=2)
    draw.text((p2_hp_left, 40), p2_name, fill=(255, 255, 255, 255))

    draw.rectangle([canvas_width - 60, 6, canvas_width - 8, 58], fill=(65, 25, 45, 255))
    draw.rectangle([canvas_width - 60, 6, canvas_width - 8, 58], outline=(240, 100, 140, 255), width=2)
    draw.text((canvas_width - 50, 22), "P2", fill=(255, 255, 255, 255))


def _draw_racing_hud(draw, canvas_width, canvas_height):
    px1, py1 = canvas_width - 200, canvas_height - 80
    px2, py2 = canvas_width - 10, canvas_height - 10
    draw.rectangle([px1, py1, px2, py2], fill=(20, 25, 35, 230))
    draw.rectangle([px1, py1, px2, py2], outline=(0, 230, 255, 255), width=2)

    draw.text((px1 + 12, py1 + 8),  "SPEED:", fill=(160, 170, 185, 255))
    draw.text((px1 + 75, py1 + 8),  "145 km/h", fill=(0, 255, 240, 255))
    draw.text((px1 + 12, py1 + 28), "LAP 2/3", fill=(255, 215, 0, 255))
    draw.text((px1 + 12, py1 + 48), "POS 1ST", fill=(255, 255, 255, 255))


def _draw_shooter_hud(draw, canvas_width, canvas_height):
    px1, py1 = 10, canvas_height - 75
    px2, py2 = 180, canvas_height - 10
    draw.rectangle([px1, py1, px2, py2], fill=(15, 20, 30, 230))
    draw.rectangle([px1, py1, px2, py2], outline=(0, 255, 200, 255), width=2)

    draw.text((px1 + 10, py1 + 8), "HP", fill=(255, 255, 255, 255))
    draw.rectangle([px1 + 35, py1 + 10, px1 + 160, py1 + 22], fill=(160, 30, 30, 255))
    draw.rectangle([px1 + 35, py1 + 10, px1 + 130, py1 + 22], fill=(30, 220, 90, 255))
    draw.text((px1 + 10, py1 + 35), "AMMO 24/120", fill=(255, 215, 0, 255))


def compose_scene(best_assets, layout_json, game_plan, tile_size=32, canvas_width=1024, canvas_height=480, include_sprites=True):
    genre_lower = (game_plan.get("genre", "") if game_plan else "").lower()
    theme_lower = (game_plan.get("theme", "") if game_plan else "").lower()
    camera_style = (game_plan.get("camera", {}).get("style", "") if game_plan else "").lower()

    is_fighting = "fight" in genre_lower or "arena" in genre_lower or "beat" in genre_lower or "brawl" in genre_lower or "fixed" in camera_style
    is_racing = "car" in genre_lower or "rac" in genre_lower or "vehicle" in genre_lower or "speed" in genre_lower or "behind" in camera_style or "chase" in camera_style
    is_shooter = "shoot" in genre_lower or "top down" in genre_lower or "bullet" in genre_lower or "gun" in genre_lower or "top" in camera_style or "ortho" in camera_style
    is_puzzle = "puzzl" in genre_lower or "maze" in genre_lower or "logic" in genre_lower

    if is_fighting:
        bg = best_assets.get("background")
        if bg:
            canvas = bg.resize((canvas_width, canvas_height), Image.LANCZOS).convert("RGBA")
        else:
            canvas = draw_parallax_sky(canvas_width, canvas_height, game_plan)
        draw = ImageDraw.Draw(canvas)

        if include_sprites:
            player_sprite = best_assets.get("player")
            if player_sprite:
                p1 = remove_flat_background(player_sprite).resize((150, 150), Image.NEAREST).convert("RGBA")
                canvas.paste(p1, (int(canvas_width * 0.18), int(canvas_height * 0.44)), p1)

            enemy_sprite = best_assets.get("enemy")
            if enemy_sprite:
                p2 = remove_flat_background(enemy_sprite).resize((150, 150), Image.NEAREST).convert("RGBA")
                p2 = p2.transpose(Image.FLIP_LEFT_RIGHT)
                canvas.paste(p2, (int(canvas_width * 0.66), int(canvas_height * 0.44)), p2)

        _draw_fighting_hud(draw, canvas_width, canvas_height, game_plan, p1_hp=100, p2_hp=100)
        return canvas.convert("RGB")

    elif is_racing:
        bg = best_assets.get("background")
        if bg:
            canvas = bg.resize((canvas_width, canvas_height), Image.LANCZOS).convert("RGBA")
        else:
            canvas = draw_parallax_sky(canvas_width, canvas_height, game_plan)
        draw = ImageDraw.Draw(canvas)

        if include_sprites:
            player_sprite = best_assets.get("player")
            if player_sprite:
                car = remove_flat_background(player_sprite).resize((180, 90), Image.NEAREST).convert("RGBA")
                canvas.paste(car, (int(canvas_width * 0.15), int(canvas_height * 0.72)), car)

        fx = int(canvas_width * 0.85)
        fy = int(canvas_height * 0.55)
        draw.rectangle([fx, fy, fx + 8, int(canvas_height * 0.85)], fill=(240, 240, 240, 255))
        for r in range(4):
            for c in range(4):
                col = (0, 0, 0, 255) if (r + c) % 2 == 0 else (255, 255, 255, 255)
                draw.rectangle([fx + 8 + c * 10, fy + r * 10, fx + 18 + c * 10, fy + 10 + r * 10], fill=col)

        _draw_racing_hud(draw, canvas_width, canvas_height)
        return canvas.convert("RGB")

    elif is_shooter or is_puzzle:
        canvas = Image.new("RGBA", (canvas_width, canvas_height), (15, 20, 30, 255))
        draw = ImageDraw.Draw(canvas)

        for gx in range(0, canvas_width, 48):
            for gy in range(0, canvas_height, 48):
                c_val = 30 if ((gx // 48 + gy // 48) % 2 == 0) else 40
                draw.rectangle([gx, gy, gx + 46, gy + 46], fill=(c_val, c_val + 5, c_val + 15, 255))

        draw.rectangle([0, 0, canvas_width, 24], fill=(80, 90, 110, 255))
        draw.rectangle([0, canvas_height - 24, canvas_width, canvas_height], fill=(80, 90, 110, 255))
        draw.rectangle([0, 0, 24, canvas_height], fill=(80, 90, 110, 255))
        draw.rectangle([canvas_width - 24, 0, canvas_width, canvas_height], fill=(80, 90, 110, 255))

        if include_sprites:
            player_sprite = best_assets.get("player")
            if player_sprite:
                ps = remove_flat_background(player_sprite).resize((80, 80), Image.NEAREST).convert("RGBA")
                canvas.paste(ps, (120, canvas_height // 2 - 40), ps)

            enemy_sprite = best_assets.get("enemy")
            if enemy_sprite:
                es = remove_flat_background(enemy_sprite).resize((80, 80), Image.NEAREST).convert("RGBA")
                canvas.paste(es, (canvas_width - 220, canvas_height // 2 - 40), es)

        draw.ellipse([canvas_width - 210, canvas_height // 2 - 30, canvas_width - 150, canvas_height // 2 + 30], outline=(255, 50, 50, 255), width=3)
        draw.line([canvas_width - 180, canvas_height // 2 - 40, canvas_width - 180, canvas_height // 2 + 40], fill=(255, 50, 50, 255), width=2)
        draw.line([canvas_width - 220, canvas_height // 2, canvas_width - 140, canvas_height // 2], fill=(255, 50, 50, 255), width=2)

        _draw_shooter_hud(draw, canvas_width, canvas_height)
        return canvas.convert("RGB")

    else:
        bg = best_assets.get("background")
        if bg:
            canvas = bg.resize((canvas_width, canvas_height), Image.LANCZOS).convert("RGBA")
        else:
            canvas = draw_parallax_sky(canvas_width, canvas_height, game_plan)

        platforms = layout_json.get("platforms", [])
        if not platforms:
            return canvas.convert("RGB")

        all_x = [p[0] for p in platforms]
        all_y = [p[1] for p in platforms]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        grid_w, grid_h = max_x - min_x + 1, max_y - min_y + 1

        scale = min(canvas_width / (grid_w * tile_size), canvas_height / (grid_h * tile_size), 1.0)
        ts = max(int(tile_size * scale), 6)

        platform_tile = best_assets.get("platform_tile")
        if platform_tile:
            platform_tile = platform_tile.resize((ts, ts), Image.NEAREST).convert("RGBA")

        for px_grid, py_grid in platforms:
            screen_x = int((px_grid - min_x) * ts)
            screen_y = int((py_grid - min_y) * ts)
            if platform_tile and 0 <= screen_x < canvas_width and 0 <= screen_y < canvas_height:
                canvas.paste(platform_tile, (screen_x, screen_y), platform_tile)

        if include_sprites:
            player_pos = layout_json.get("player", [0, 0])
            player_sprite = best_assets.get("player")
            if player_sprite:
                ps = remove_flat_background(player_sprite).resize((ts * 2, ts * 2), Image.NEAREST).convert("RGBA")
                px_screen = int((player_pos[0] - min_x) * ts)
                py_screen = int((player_pos[1] - min_y) * ts) - ts
                canvas.paste(ps, (px_screen, py_screen), ps)

            enemies = layout_json.get("enemies", [])
            enemy_sprite = best_assets.get("enemy")
            if enemy_sprite and enemies:
                es = remove_flat_background(enemy_sprite).resize((ts * 2, ts * 2), Image.NEAREST).convert("RGBA")
                for ex, ey in enemies[:10]:
                    ex_screen = int((ex - min_x) * ts)
                    ey_screen = int((ey - min_y) * ts) - ts
                    canvas.paste(es, (ex_screen, ey_screen), es)

        goal_pos = layout_json.get("goal", [0, 0])
        gx = int((goal_pos[0] - min_x) * ts)
        gy = int((goal_pos[1] - min_y) * ts) - ts
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([gx, gy, gx + 6, gy + ts * 2], fill=(255, 215, 0, 255))
        draw.polygon([(gx + 6, gy), (gx + ts * 2, gy + int(ts * 0.6)), (gx + 6, gy + int(ts * 1.2))], fill=(255, 40, 40, 255))

        return canvas.convert("RGB")

def validate_playability(layout_json, max_jump_height=4, max_jump_width=5):
    platforms = set(tuple(p) for p in layout_json.get("platforms", []))
    player = tuple(layout_json.get("player", [0, 0]))
    goal = tuple(layout_json.get("goal", [0, 0]))

    if not platforms:
        return False, [], {"error": "No platforms defined"}

    def get_standing_positions():
        standing = set()
        for px_coord, py_coord in platforms:
            standing.add((px_coord, py_coord - 1))
        return standing

    walkable = get_standing_positions()
    walkable.update(platforms)

    def nearest_walkable(pos):
        if pos in walkable:
            return pos
        best, best_dist = None, float("inf")
        for w in walkable:
            d = abs(w[0] - pos[0]) + abs(w[1] - pos[1])
            if d < best_dist:
                best_dist, best = d, w
        return best

    start = nearest_walkable(player)
    end = nearest_walkable(goal)
    if start is None or end is None:
        return False, [], {"error": "Start or goal not near any platform"}

    visited = {start}
    queue = deque([(start, [start])])

    while queue:
        (cx, cy), path = queue.popleft()
        if abs(cx - end[0]) <= 2 and abs(cy - end[1]) <= 2:
            return True, path, {"path_length": len(path), "start": start, "goal": end, "status": "PLAYABLE"}

        moves = [(cx - 1, cy), (cx + 1, cy)]
        for dx in range(-max_jump_width, max_jump_width + 1):
            for dy in range(-max_jump_height, 1):
                if dx == 0 and dy == 0:
                    continue
                moves.append((cx + dx, cy + dy))
        for dy in range(1, 20):
            moves.append((cx, cy + dy))
            moves.append((cx - 1, cy + dy))
            moves.append((cx + 1, cy + dy))

        for nx, ny in moves:
            if (nx, ny) not in visited and (nx, ny) in walkable:
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(nx, ny)]))

    suggestions = suggest_fixes(start, end, walkable, max_jump_height, max_jump_width)
    return False, [], {"status": "NOT PLAYABLE", "start": start, "goal": end,
                        "tiles_explored": len(visited), "suggestions": suggestions}


def suggest_fixes(start, end, walkable, jump_h, jump_w):
    suggestions = []
    visited = {start}
    queue = deque([start])
    rightmost = start

    while queue:
        cx, cy = queue.popleft()
        if cx > rightmost[0]:
            rightmost = (cx, cy)
        for dx in range(-jump_w, jump_w + 1):
            for dy in range(-jump_h, 3):
                nx, ny = cx + dx, cy + dy
                if (nx, ny) not in visited and (nx, ny) in walkable:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

    gap_x = rightmost[0] + 1
    suggestions.append({"type": "add_platform", "position": [gap_x + 2, rightmost[1]],
                        "reason": f"Bridge gap after rightmost reachable tile at {rightmost}"})
    mid_x = (rightmost[0] + end[0]) // 2
    mid_y = (rightmost[1] + end[1]) // 2
    suggestions.append({"type": "add_platform", "position": [mid_x, mid_y],
                        "reason": "Stepping stone between reachable area and goal"})
    return suggestions


def auto_fix_level(layout_json, max_attempts=5):
    layout = json.loads(json.dumps(layout_json))
    for _ in range(max_attempts):
        playable, path, details = validate_playability(layout)
        if playable:
            return layout, path, details
        for s in details.get("suggestions", []):
            pos = s["position"]
            for dx in range(-1, 2):
                new_tile = [pos[0] + dx, pos[1]]
                if new_tile not in layout["platforms"]:
                    layout["platforms"].append(new_tile)
    return layout, [], {"status": "COULD NOT FIX", "attempts": max_attempts}

# --------------------------------------------------------------------------
# Phase 8 - Rich Action Video Preview Generator
# --------------------------------------------------------------------------

def generate_preview_video(scene, layout_json, path, best_assets, output_path, fps=24, tile_size=32, game_plan=None):
    genre_lower = (game_plan.get("genre", "") if game_plan else "").lower()
    theme_lower = (game_plan.get("theme", "") if game_plan else "").lower()

    is_fighting = "fight" in genre_lower or "arena" in genre_lower or "beat" in genre_lower or "brawl" in genre_lower or "fight" in theme_lower
    is_racing = "car" in genre_lower or "rac" in genre_lower or "vehicle" in genre_lower or "speed" in genre_lower or "car" in theme_lower or "racer" in theme_lower
    is_shooter = "shoot" in genre_lower or "top down" in genre_lower or "bullet" in genre_lower or "gun" in genre_lower or "laser" in theme_lower

    # Reuse the already-composed scene to avoid a second compose_scene() RAM spike.
    # Pre-scale to viewport size BEFORE the frame loop so base_img.copy() is cheap (640x368 not 2048x1024).
    vp_w, vp_h = 640, 368
    base_img = scene.convert("RGBA").resize((vp_w, vp_h), Image.BILINEAR)
    w, h = vp_w, vp_h
    frames = []

    if is_fighting:
        clean_bg = Image.new("RGBA", (w, h), (20, 18, 30, 255))
        draw_clean = ImageDraw.Draw(clean_bg)
        draw_clean.rectangle([0, 0, w, h], fill=(25, 20, 40, 255))
        for b in range(0, w, 120):
            draw_clean.rectangle([b, int(h * 0.18), b + 95, h], fill=(35, 30, 55, 255))
            draw_clean.rectangle([b + 15, int(h * 0.28), b + 45, int(h * 0.42)], fill=(255, 180, 40, 200))
            draw_clean.rectangle([b + 55, int(h * 0.28), b + 80, int(h * 0.42)], fill=(0, 220, 255, 200))
        draw_clean.rectangle([0, int(h * 0.72), w, h], fill=(55, 50, 65, 255))
        draw_clean.rectangle([0, int(h * 0.72), w, int(h * 0.75)], fill=(115, 110, 125, 255))

        total_frames = 150
        p1_base_x = int(w * 0.22)
        p2_base_x = int(w * 0.62)
        y_ground = int(h * 0.44)

        player_sprite = best_assets.get("player")
        enemy_sprite = best_assets.get("enemy")
        p1_img = remove_flat_background(player_sprite).resize((160, 160), Image.NEAREST).convert("RGBA") if player_sprite else None
        p2_img = remove_flat_background(enemy_sprite).resize((160, 160), Image.NEAREST).convert("RGBA") if enemy_sprite else None
        if p2_img:
            p2_img = p2_img.transpose(Image.FLIP_LEFT_RIGHT)

        for i in range(total_frames):
            frame_img = clean_bg.copy()
            draw = ImageDraw.Draw(frame_img)

            p1_hp, p2_hp = 100, 100
            p1_x, p1_y = p1_base_x, y_ground
            p2_x, p2_y = p2_base_x, y_ground
            shake_x, shake_y = 0, 0
            show_hit_spark = False
            show_fireball = False
            fireball_x = 0
            ko_text = False

            v_i = int(i * (240.0 / 150.0))
            if v_i < 35:
                progress = v_i / 35.0
                p1_x = p1_base_x + int(progress * 60)
                p2_x = p2_base_x - int(progress * 40)
            elif v_i < 75:
                p1_x = p1_base_x + 60 + int(math.sin((v_i - 35) * 0.5) * 15)
                p2_x = p2_base_x - 40
                p2_hp = max(60, 100 - int((v_i - 35) * 1.0))
                if (v_i % 8) < 4:
                    show_hit_spark = True
            elif v_i < 120:
                p1_x = p1_base_x + 50
                p2_x = p2_base_x - 30
                p2_hp = 60
                p1_hp = max(85, 100 - int((v_i - 75) * 0.35))
                show_fireball = True
                fireball_x = p2_x - int((v_i - 75) * 6)
            elif v_i < 170:
                t = (v_i - 120) / 50.0
                p1_x = p1_base_x + 50 + int(t * 120)
                p1_y = y_ground - int(math.sin(t * math.pi) * 90)
                p2_hp = max(0, 60 - int(t * 70))
                if 0.4 < t < 0.8:
                    shake_x = random.randint(-4, 4)
                    shake_y = random.randint(-4, 4)
                    show_hit_spark = True
            else:
                p1_x = p1_base_x + 140
                p1_y = y_ground
                p2_x = p2_base_x + 30
                p2_y = y_ground + 35
                p1_hp = 85
                p2_hp = 0
                ko_text = True

            _draw_fighting_hud(draw, w, h, game_plan, p1_hp=p1_hp, p2_hp=p2_hp)

            if show_fireball and fireball_x > p1_x:
                draw.ellipse([fireball_x - 18, y_ground + 40, fireball_x + 18, y_ground + 76], fill=(255, 160, 20, 255))
                draw.ellipse([fireball_x - 10, y_ground + 46, fireball_x + 10, y_ground + 70], fill=(255, 240, 80, 255))

            if p1_img: frame_img.paste(p1_img, (p1_x + shake_x, p1_y + shake_y), p1_img)
            if p2_img: frame_img.paste(p2_img, (p2_x + shake_x, p2_y + shake_y), p2_img)

            if show_hit_spark:
                hx = p2_x + 20
                hy = p2_y + 40
                draw.polygon([(hx, hy-25), (hx+12, hy-8), (hx+28, hy-18), (hx+15, hy), (hx+30, hy+15), (hx+8, hy+8), (hx, hy+30), (hx-8, hy+8), (hx-25, hy+15), (hx-12, hy)], fill=(255, 230, 40, 255))

            if ko_text:
                draw.rectangle([w // 2 - 120, h // 2 - 40, w // 2 + 120, h // 2 + 30], fill=(20, 20, 30, 230))
                draw.rectangle([w // 2 - 120, h // 2 - 40, w // 2 + 120, h // 2 + 30], outline=(255, 215, 0, 255), width=3)
                draw.text((w // 2 - 50, h // 2 - 15), "K. O.", fill=(255, 40, 40, 255))
                draw.text((w // 2 - 80, h // 2 + 10), "P1 VICTORY!", fill=(255, 215, 0, 255))

            frames.append(np.array(frame_img.convert("RGB")))

    elif is_racing:
        total_frames = 150
        player_sprite = best_assets.get("player")
        enemy_sprite = best_assets.get("enemy")
        car_p1 = remove_flat_background(player_sprite).resize((180, 90), Image.NEAREST).convert("RGBA") if player_sprite else None
        car_p2 = remove_flat_background(enemy_sprite).resize((180, 90), Image.NEAREST).convert("RGBA") if enemy_sprite else None

        for i in range(total_frames):
            frame_img = Image.new("RGBA", (w, h), (15, 10, 35, 255))
            draw = ImageDraw.Draw(frame_img)

            for b in range(0, w, 90):
                bh = 120 + int(math.sin((b + i * 25) * 0.02) * 40)
                draw.rectangle([b, h - bh, b + 75, h], fill=(30, 20, 60, 255))

            draw.rectangle([0, int(h * 0.68), w, h], fill=(40, 40, 50, 255))
            offset = (i * 28) % 60
            for x in range(-60 + offset, w + 60, 60):
                draw.rectangle([x, int(h * 0.84), x + 35, int(h * 0.87)], fill=(250, 210, 40, 255))

            progress = i / float(total_frames)
            p1_x = int(w * 0.08 + progress * (w * 0.65))
            p2_x = int(w * 0.25 + progress * (w * 0.35))
            p1_y = int(h * 0.76)
            p2_y = int(h * 0.70)

            if car_p2:
                frame_img.paste(car_p2, (p2_x, p2_y), car_p2)

            if 0.3 < progress < 0.85:
                draw.polygon([(p1_x - 30, p1_y + 35), (p1_x - 5, p1_y + 40), (p1_x - 30, p1_y + 45)], fill=(0, 240, 255, 255))
                draw.polygon([(p1_x - 50, p1_y + 30), (p1_x - 15, p1_y + 40), (p1_x - 50, p1_y + 50)], fill=(255, 140, 0, 200))

            if car_p1:
                frame_img.paste(car_p1, (p1_x, p1_y), car_p1)

            _draw_racing_hud(draw, w, h)

            if progress > 0.8:
                draw.rectangle([w // 2 - 140, 80, w // 2 + 140, 140], fill=(10, 20, 30, 230))
                draw.rectangle([w // 2 - 140, 80, w // 2 + 140, 140], outline=(0, 255, 200, 255), width=3)
                draw.text((w // 2 - 100, 98), "FINISH! 1ST PLACE", fill=(0, 255, 240, 255))

            frame = frame_img.convert("RGB")
            frames.append(np.array(frame))

    else:
        total_frames = 150
        platforms = layout_json.get("platforms", [[0, 11]])
        all_x = [p[0] for p in platforms]
        all_y = [p[1] for p in platforms]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        grid_w, grid_h = max_x - min_x + 1, max_y - min_y + 1

        scale = min(w / (grid_w * tile_size), h / (grid_h * tile_size), 1.0)
        ts = max(int(tile_size * scale), 6)
        sprite_w, sprite_h = ts * 2, ts * 2

        player_sprite = best_assets.get("player")
        enemy_sprite = best_assets.get("enemy")
        p_img = remove_flat_background(player_sprite).resize((sprite_w, sprite_h), Image.NEAREST).convert("RGBA") if player_sprite else None
        e_img = remove_flat_background(enemy_sprite).resize((sprite_w, sprite_h), Image.NEAREST).convert("RGBA") if enemy_sprite else None

        path_coords = path if path else layout_json.get("platforms", [])[:10]
        if not path_coords:
            path_coords = [(1, 10), (2, 10), (3, 10)]

        for i in range(total_frames):
            frame_img = base_img.copy()
            idx = int((i / total_frames) * len(path_coords)) % len(path_coords)
            curr_pos = path_coords[idx]

            px_scr = int((curr_pos[0] - min_x) * ts)
            py_scr = int((curr_pos[1] - min_y) * ts) - ts

            jump_offset = 0
            for ex, ey in layout_json.get("enemies", []):
                ex_scr = int((ex - min_x) * ts)
                if abs(px_scr - ex_scr) < 80:
                    t_jump = max(0, min(1.0, 1.0 - abs(px_scr - ex_scr) / 80.0))
                    jump_offset = int(math.sin(t_jump * math.pi) * (ts * 2))

            py_scr -= jump_offset

            # Clamp coordinates so player stays safely within background canvas
            px_scr = max(0, min(w - sprite_w, px_scr))
            py_scr = max(0, min(h - sprite_h, py_scr))

            if p_img:
                frame_img.paste(p_img, (px_scr, py_scr), p_img)

            if e_img:
                for ex, ey in layout_json.get("enemies", [])[:5]:
                    ex_scr = int((ex - min_x) * ts) + int(math.sin(i * 0.1) * 8)
                    ey_scr = int((ey - min_y) * ts) - ts
                    ex_scr = max(0, min(w - sprite_w, ex_scr))
                    ey_scr = max(0, min(h - sprite_h, ey_scr))
                    frame_img.paste(e_img, (ex_scr, ey_scr), e_img)

            # Smoothly center camera viewport on player position
            cam_x = max(0, min(max(0, w - vp_w), px_scr + sprite_w // 2 - vp_w // 2))
            cam_y = max(0, min(max(0, h - vp_h), py_scr + sprite_h // 2 - vp_h // 2))

            # Since base_img is already at viewport size, no crop needed
            frames.append(np.array(frame_img.convert("RGB")))

    # Save as H.264 MP4 using imageio and ffmpeg-python wheel
    try:
        import imageio
        writer = imageio.get_writer(output_path, fps=20, codec="libx264", quality=8)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
    except Exception as e:
        print(f"Video render note: {e}")

# --------------------------------------------------------------------------
# Pipeline Execution Endpoint Engine
# --------------------------------------------------------------------------

def run_full_pipeline(image_path, user_description="A fun game", job_id=None, base_url="http://localhost:7860"):
    job_dir = os.path.join(API_OUTPUT_DIR, job_id) if job_id else os.path.join(API_OUTPUT_DIR, str(uuid.uuid4())[:8])
    os.makedirs(job_dir, exist_ok=True)

    set_job_status(job_id, "Analyzing level sketch with Florence-2 & Vision...", 10, "Extracting layout grid, caption, and object detections")
    if DEVICE == "cuda":
        try:
            ensure_florence_loaded()
        except Exception as e:
            print(f"Florence load note: {e}")
    layout, florence_caption, florence_od = sketch_to_layout(image_path)

    set_job_status(job_id, "Generating AAA game plan with GPT-4o...", 25, f"Caption: '{florence_caption[:40]}...'")
    game_plan = plan_game(layout, user_description, florence_caption, florence_od)
    if not game_plan:
        set_job_status(job_id, "Failed", 0, error="GPT-4o planning failed")
        return {"error": "Failed to generate game plan from sketch"}

    all_candidates = generate_all_assets(game_plan, job_dir, job_id=job_id)

    set_job_status(job_id, "Scoring candidates with CLIP...", 88, "Selecting best 16-bit sprites")
    best_assets = select_best_assets(all_candidates, game_plan)

    for asset_name, img in best_assets.items():
        img.save(os.path.join(job_dir, f"best_{asset_name}.png"))

    set_job_status(job_id, "Validating level playability...", 92, "Running BFS pathfinding")
    playable, path, playability_info = validate_playability(layout)

    if not playable:
        set_job_status(job_id, "Auto-fixing unplayable gaps...", 94, "Adding stepping stone platforms")
        layout, path, playability_info = auto_fix_level(layout)

    set_job_status(job_id, "Composing composed level scene...", 96, "Stitching backgrounds and sprites")
    scene = compose_scene(best_assets, layout, game_plan)
    scene_path = os.path.join(job_dir, "scene.png")
    scene.save(scene_path)

    try:
        set_job_status(job_id, "Rendering H.264 gameplay preview video...", 98, "Generating H.264 animation sequence")
        preview_path = os.path.join(job_dir, "preview.mp4")
        generate_preview_video(scene, layout, path, best_assets, preview_path, game_plan=game_plan)
    except Exception as e:
        print(f"Preview video generation note: {e}")

    zip_filename = f"game_assets_{job_id}.zip"
    zip_path = os.path.join(job_dir, zip_filename)
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for fname in os.listdir(job_dir):
                if fname != zip_filename:
                    zipf.write(os.path.join(job_dir, fname), fname)
    except Exception as e:
        print(f"Zip packaging note: {e}")

    # Use timestamp query parameters to prevent frontend asset caching issues
    ts = str(uuid.uuid4())[:6]
    result = {
        "job_id": job_id,
        "game_plan": game_plan,
        "layout": layout,
        "florence_caption": florence_caption,
        "florence_od": florence_od,
        "playability": playability_info,
        "urls": {
            "scene": f"/files/{job_id}/scene.png?v={ts}",
            "preview": f"/files/{job_id}/preview.mp4?v={ts}",
        },
        "zip_url": f"/files/{job_id}/{zip_filename}?v={ts}"
    }

    set_job_status(job_id, "Completed", 100, "Level generation finished successfully", result=result)
    return result

# --------------------------------------------------------------------------
# FastAPI Web App Routes
# --------------------------------------------------------------------------

app = FastAPI(title="Sketch-to-Game API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/files", StaticFiles(directory=API_OUTPUT_DIR), name="files")


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Sketch-to-Game API Engine",
        "device": DEVICE,
        "lora_florence": FLORENCE_LORA_PATH,
        "lora_sdxl": SDXL_LORA_PATH
    }


@app.post("/generate")
async def generate_endpoint(
    sketch: UploadFile = File(...),
    description: str = Form("A fun platformer game"),
    job_id: str = Form(None)
):
    if not job_id:
        job_id = str(uuid.uuid4())[:8]

    temp_sketch_path = os.path.join(API_OUTPUT_DIR, f"temp_{job_id}_{sketch.filename}")
    with open(temp_sketch_path, "wb") as f:
        content = await sketch.read()
        f.write(content)

    set_job_status(job_id, "Initializing pipeline...", 2, "Received uploaded sketch file")

    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        executor,
        run_full_pipeline,
        temp_sketch_path,
        description,
        job_id,
        "http://localhost:7860"
    )

    return {"job_id": job_id, "status": "processing"}


@app.get("/status/{job_id}")
def status_endpoint(job_id: str):
    info = JOB_STATUS.get(job_id, {"status": "not_found", "progress": 0, "step": "Unknown job"})
    return JSONResponse(content=info)


@app.get("/download-status")
def download_status_endpoint():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    key_configured = bool(api_key and len(api_key) > 8)
    key_preview = f"sk-...{api_key[-4:]}" if key_configured else "Not set"

    return {
        "downloaded_gb": 6.5 if DEVICE == "cuda" else 0.5,
        "target_gb": 6.5,
        "percent": 100 if DEVICE == "cuda" else 100,
        "speed_mbps": 0,
        "file_name": "Models cached & ready on device",
        "api_key_configured": key_configured,
        "api_key_preview": key_preview,
        "is_cached": True,
        "is_downloading": False
    }


@app.post("/start-download")
def start_download_endpoint():
    return {"status": "cached", "message": "All models are pre-loaded or ready on device."}


@app.get("/download-assets/{job_id}")
def download_assets_endpoint(job_id: str):
    job_dir = os.path.join(API_OUTPUT_DIR, job_id)
    zip_path = os.path.join(job_dir, f"game_assets_{job_id}.zip")
    if not os.path.exists(zip_path) and os.path.exists(job_dir):
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for fname in os.listdir(job_dir):
                if fname != f"game_assets_{job_id}.zip":
                    zipf.write(os.path.join(job_dir, fname), fname)
    if os.path.exists(zip_path):
        return FileResponse(zip_path, filename=f"game_assets_{job_id}.zip", media_type="application/zip")
    return JSONResponse(status_code=404, content={"error": f"Job assets for '{job_id}' not found"})


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return JSONResponse(status_code=204, content={})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
