import glob
import os
import random

from PIL import Image

from .config import QCHConfig


def build_mosaic_cover(folder: str | None, cfg: QCHConfig, logger) -> Image.Image:
    if not folder or not os.path.isdir(folder):
        logger.info("[COVER] Using noise cover.")
        return Image.frombytes("RGB", (cfg.width, cfg.height), os.urandom(cfg.width * cfg.height * 3))
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        paths += glob.glob(os.path.join(folder, "**", ext), recursive=True)
    if not paths:
        logger.info("[COVER] No images found; falling back to noise.")
        return Image.frombytes("RGB", (cfg.width, cfg.height), os.urandom(cfg.width * cfg.height * 3))
    tile = 60
    cols = cfg.width // tile
    rows = cfg.height // tile
    mosaic = Image.new("RGB", (cols * tile, rows * tile))
    rnd = random.Random(hash(folder) & 0xFFFFFFFF)
    for r in range(rows):
        for c in range(cols):
            p = paths[rnd.randrange(0, len(paths))]
            try:
                im = Image.open(p).convert("RGB").resize((tile, tile), Image.LANCZOS)
            except Exception:
                im = Image.frombytes("RGB", (tile, tile), os.urandom(tile * tile * 3))
            mosaic.paste(im, (c * tile, r * tile))
    mosaic = mosaic.resize((cfg.width, cfg.height), Image.LANCZOS)
    logger.info(f"[COVER] Mosaic from {len(paths)} images.")
    return mosaic
