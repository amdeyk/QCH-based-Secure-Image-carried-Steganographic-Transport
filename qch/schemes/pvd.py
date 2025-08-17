import math
from typing import List

from PIL import Image

from ..config import QCHConfig
from ..utils import prng

PVD_RANGES = [(0, 7), (8, 15), (16, 31), (32, 63), (64, 127)]


def pvd_capacity_estimate(img: Image.Image, cfg: QCHConfig, seed: bytes) -> int:
    img = img.convert("RGB").resize((cfg.width, cfg.height), Image.LANCZOS)
    B = img.tobytes()[2::3]
    r = prng(seed)
    pairs = [(i, i + 1) for i in range(0, len(B) - 1, 2)]
    r.shuffle(pairs)
    cap = 0
    for i, j in pairs:
        d = abs(B[i] - B[j])
        for a, b in PVD_RANGES:
            if a <= d <= b:
                w = b - a + 1
                t = int(math.floor(math.log2(w)))
                cap += t
                break
    return cap


def pvd_embed(img: Image.Image, bits: List[int], cfg: QCHConfig, seed: bytes, logger) -> Image.Image:
    img = img.convert("RGB").resize((cfg.width, cfg.height), Image.LANCZOS)
    R = bytearray(img.tobytes())
    B_idx = 2
    r = prng(seed)
    blue_indices = [i for i in range(B_idx, len(R), 3)]
    pairs = [(blue_indices[k], blue_indices[k + 1]) for k in range(0, len(blue_indices) - 1, 2)]
    r.shuffle(pairs)
    bi = 0
    for i, j in pairs:
        if bi >= len(bits):
            break
        p1 = R[i]
        p2 = R[j]
        d = abs(p1 - p2)
        for a, b in PVD_RANGES:
            if a <= d <= b:
                w = b - a + 1
                t = int(math.floor(math.log2(w)))
                if t <= 0:
                    break
                val_bits = 0
                for _ in range(t):
                    val_bits = (val_bits << 1) | (bits[bi] if bi < len(bits) else 0)
                    bi += 1
                new_d = a + val_bits
                if new_d > b:
                    new_d = b
                if p1 >= p2:
                    delta = new_d - d
                    p1n = max(0, min(255, p1 + delta))
                    if abs(p1n - p2) != new_d:
                        diff = new_d - abs(p1n - p2)
                        p2n = max(0, min(255, p2 - diff if p1n >= p2 else p2 + diff))
                    else:
                        p2n = p2
                else:
                    delta = new_d - d
                    p2n = max(0, min(255, p2 + delta))
                    if abs(p1 - p2n) != new_d:
                        diff = new_d - abs(p1 - p2n)
                        p1n = max(0, min(255, p1 - diff if p2n >= p1 else p1 + diff))
                    else:
                        p1n = p1
                R[i] = p1n
                R[j] = p2n
                break
    if bi < len(bits):
        logger.warning(f"[PVD] Ran out of capacity; embedded {bi}/{len(bits)} bits")
        raise ValueError("PVD capacity insufficient for payload")
    return Image.frombytes("RGB", (cfg.width, cfg.height), bytes(R))


def pvd_extract(img: Image.Image, cfg: QCHConfig, seed: bytes, logger):
    img = img.convert("RGB").resize((cfg.width, cfg.height), Image.LANCZOS)
    R = img.tobytes()
    B_idx = 2
    r = prng(seed)
    blue_indices = [i for i in range(B_idx, len(R), 3)]
    pairs = [(blue_indices[k], blue_indices[k + 1]) for k in range(0, len(blue_indices) - 1, 2)]
    r.shuffle(pairs)
    for i, j in pairs:
        p1 = R[i]
        p2 = R[j]
        d = abs(p1 - p2)
        for a, b in PVD_RANGES:
            if a <= d <= b:
                w = b - a + 1
                t = int(math.floor(math.log2(w)))
                if t <= 0:
                    break
                d_clamped = min(max(d, a), b)
                val = d_clamped - a
                for k in range(t - 1, -1, -1):
                    yield (val >> k) & 1
                break
