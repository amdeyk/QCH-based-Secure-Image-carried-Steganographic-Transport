from PIL import Image

from ..config import QCHConfig
from ..utils import prng


def interp_embed(img: Image.Image, bits, cfg: QCHConfig, seed: bytes, logger):
    base_w = cfg.width // 2
    base_h = cfg.height // 2
    base = img.convert("RGB").resize((base_w, base_h), Image.LANCZOS)
    up = base.resize((cfg.width, cfg.height), Image.BICUBIC)
    data = bytearray(up.tobytes())
    r = prng(seed)
    coords = []
    for y in range(cfg.height):
        for x in range(cfg.width):
            if (x & 1) or (y & 1):
                idx = (y * cfg.width + x) * 3 + 2
                coords.append(idx)
    r.shuffle(coords)
    if len(bits) > len(coords):
        raise ValueError(f"[INTERP] capacity {len(coords)} bits < needed {len(bits)}")
    for i, idx in enumerate(coords[: len(bits)]):
        data[idx] = (data[idx] & 0xFE) | (bits[i] & 1)
    return Image.frombytes("RGB", (cfg.width, cfg.height), bytes(data))


def interp_extract(img: Image.Image, cfg: QCHConfig, seed: bytes, logger):
    base_w = cfg.width // 2
    base_h = cfg.height // 2
    base = img.convert("RGB").resize((base_w, base_h), Image.LANCZOS)
    up = base.resize((cfg.width, cfg.height), Image.BICUBIC)
    data = up.tobytes()
    r = prng(seed)
    coords = []
    for y in range(cfg.height):
        for x in range(cfg.width):
            if (x & 1) or (y & 1):
                idx = (y * cfg.width + x) * 3 + 2
                coords.append(idx)
    r.shuffle(coords)
    for idx in coords:
        yield (data[idx] & 1)
