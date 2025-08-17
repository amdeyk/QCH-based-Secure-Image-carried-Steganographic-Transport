from typing import List

from PIL import Image

from ..config import QCHConfig
from ..utils import prng


def lsb_positions(cfg: QCHConfig, r, channel: str = "B") -> List[int]:
    total_bytes = cfg.width * cfg.height * 3
    ch_idx = {"R": 0, "G": 1, "B": 2}[channel]
    byte_indices = [pix * 3 + ch_idx for pix in range(cfg.width * cfg.height)]
    r.shuffle(byte_indices)
    return byte_indices


def embed_lsb_scatter(
    img: Image.Image,
    bits: List[int],
    cfg: QCHConfig,
    seed: bytes,
    logger,
    bpc: int | None = None,
) -> Image.Image:
    bpc = bpc or cfg.bpc
    img = img.convert("RGB").resize((cfg.width, cfg.height), Image.LANCZOS)
    pix = bytearray(img.tobytes())
    r = prng(seed)
    byte_indices = lsb_positions(cfg, r, "B")
    capacity_bits = len(byte_indices) * bpc
    if len(bits) > capacity_bits:
        raise ValueError(f"LSB capacity {capacity_bits} bits < needed {len(bits)}")
    bi = 0
    for idx in byte_indices:
        for k in range(bpc):
            if bi >= len(bits):
                break
            mask = ~(1 << k) & 0xFF
            pix[idx] = (pix[idx] & mask) | ((bits[bi] & 1) << k)
            bi += 1
        if bi >= len(bits):
            break
    return Image.frombytes("RGB", (cfg.width, cfg.height), bytes(pix))


def extract_lsb_scatter(
    img: Image.Image,
    bit_count: int,
    cfg: QCHConfig,
    seed: bytes,
    logger,
    bpc: int | None = None,
):
    bpc = bpc or cfg.bpc
    img = img.convert("RGB").resize((cfg.width, cfg.height), Image.LANCZOS)
    data = img.tobytes()
    r = prng(seed)
    byte_indices = lsb_positions(cfg, r, "B")
    capacity_bits = len(byte_indices) * bpc
    if bit_count > capacity_bits:
        bit_count = capacity_bits
    out = []
    bi = 0
    for idx in byte_indices:
        for k in range(bpc):
            if bi >= bit_count:
                break
            yield (data[idx] >> k) & 1
            bi += 1
        if bi >= bit_count:
            break
