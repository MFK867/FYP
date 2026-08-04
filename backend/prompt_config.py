"""
Master Generation Prompt Kit configuration and prompt builder utilities.
"""

# Technical parameters
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
LORA_NAME = "pxlrpg_style"
LORA_TRIGGER = "pixel_art"
LORA_WEIGHT = 0.85
SAMPLER = "DPM++ 2M Karras"
STEPS = 35
CFG_SCALE = 7.5
SPRITE_RESOLUTION = (512, 512)
BACKGROUND_RESOLUTION = (1024, 512)

# Master Style-Lock Blocks by Genre (Enhanced for SNES/SNK 16-Bit Arcade Quality)
STYLE_BLOCKS = {
    "platformer": (
        f"{LORA_TRIGGER}, 16-bit pixel art SNES sprite, Celeste and Hollow Knight style, "
        "hand-placed pixels, sharp dark pixel outline, detailed pixel shading, "
        "vibrant 16-bit color palette, crisp readable silhouette, "
        "consistent top-left light source, soft ambient occlusion, "
        "isolated on solid white background, game-ready sprite, high quality indie game art"
    ),
    "fighting": (
        f"{LORA_TRIGGER}, 16-bit pixel art SNK arcade sprite, Street Fighter II and Capcom style, "
        "hand-placed pixels, bold dark pixel outline, muscular detailed shading, "
        "vibrant arcade palette, sharp highlights, isolated on solid white background, "
        "game-ready fighting sprite, master 90s arcade art quality"
    ),
    "racing": (
        f"{LORA_TRIGGER}, 16-bit pixel art arcade sprite, F-Zero and OutRun aesthetic, retro racing, "
        "hand-placed pixels, sharp dark pixel outline, vibrant neon highlights, sleek vehicle shading, "
        "isolated on solid white background, game-ready racing asset"
    ),
    "shooter": (
        f"{LORA_TRIGGER}, 16-bit pixel art top-down sprite, Hotline Miami and Enter the Gungeon style, "
        "hand-placed pixels, sharp dark pixel outline, high contrast action game palette, "
        "distinct top-down silhouette, isolated on solid white background, game-ready action asset"
    ),
    "puzzle": (
        f"{LORA_TRIGGER}, 16-bit pixel art isometric sprite, Monument Valley and Fez look, "
        "hand-placed pixels, sharp clean pixel lines, pastel geometric color palette, "
        "isolated on solid white background, game-ready puzzle asset"
    ),
    "default": (
        f"{LORA_TRIGGER}, 16-bit pixel art SNES arcade sprite, retro game asset, "
        "hand-placed pixels, sharp dark pixel outline, detailed pixel shading, "
        "limited 16-bit color palette, crisp readable silhouette, "
        "isolated on solid white background, game-ready sprite, professional game studio quality"
    ),
}

# Backward compatibility reference
STYLE_LOCK_BLOCK = STYLE_BLOCKS["default"]

def get_style_block(genre=""):
    """
    Returns genre-specific style block via substring matching.
    """
    genre_lower = (genre or "").lower()
    if "platform" in genre_lower:
        return STYLE_BLOCKS["platformer"]
    elif "fight" in genre_lower or "arena" in genre_lower or "brawl" in genre_lower or "beat" in genre_lower:
        return STYLE_BLOCKS["fighting"]
    elif "rac" in genre_lower or "car" in genre_lower or "vehicle" in genre_lower or "speed" in genre_lower:
        return STYLE_BLOCKS["racing"]
    elif "shoot" in genre_lower or "bullet" in genre_lower or "top down" in genre_lower or "gun" in genre_lower:
        return STYLE_BLOCKS["shooter"]
    elif "puzzl" in genre_lower or "maze" in genre_lower or "logic" in genre_lower or "iso" in genre_lower:
        return STYLE_BLOCKS["puzzle"]
    return STYLE_BLOCKS["default"]

# Master Negative Prompt Block
NEGATIVE_PROMPT_BLOCK = (
    "blurry, 3d render, photorealistic, vector art, smooth gradient, "
    "anti-aliased, noise, messy, low resolution, jpeg artifacts, watermark, "
    "text, extra limbs, bad anatomy, distorted proportions, cut off, cropped, "
    "flat simple rectangle, childish drawing"
)

def build_character_prompt(character_name, role, pose, species_design, key_colors, scale_note, facing="right", genre="default"):
    """
    Builds character/enemy sprite prompt following Master Generation Prompt Kit spec.
    """
    style_block = get_style_block(genre)
    asset_details = (
        f"{character_name}, {role}, {pose}, {species_design}, "
        f"key colors: {key_colors}, scale: {scale_note}, facing {facing}, "
        "isolated on pure solid white background, full body visible, no cropping"
    )
    return f"{style_block}, {asset_details}"

def build_tile_prompt(material, variation_notes, view="side-view", genre="default"):
    """
    Builds tileable environment tile prompt following Master Generation Prompt Kit spec.
    """
    style_block = get_style_block(genre)
    asset_details = (
        f"seamless tileable texture, {material}, {variation_notes}, "
        f"{view}, 16-bit pixel art tileset, crisp brick lines, highlight edges"
    )
    return f"{style_block}, {asset_details}"

def build_full_prompt(prompt, genre="default"):
    """
    Combines input prompt with genre style block.
    """
    style_block = get_style_block(genre)
    return f"{style_block}, {prompt}, isolated on solid white background, high quality 16-bit pixel art"

def validate_payload(payload):
    """
    Validates generation payload dictionary.
    """
    return {
        "has_prompt": bool(payload.get("prompt")),
        "has_negative": bool(payload.get("negative_prompt")),
        "steps_valid": 20 <= payload.get("num_inference_steps", 0) <= 50,
        "cfg_valid": 5.0 <= payload.get("guidance_scale", 0.0) <= 15.0,
    }
