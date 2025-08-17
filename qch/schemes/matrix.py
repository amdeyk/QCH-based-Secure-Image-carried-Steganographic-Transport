from typing import List

from PIL import Image

from ..config import QCHConfig
from ..utils import prng
from .lsb import lsb_positions

H = [
    [1, 1, 1, 1, 0, 0, 0],
    [1, 1, 0, 0, 1, 1, 0],
    [1, 0, 1, 0, 1, 0, 1],
]


def hamming_syndrome(bits7: List[int]) -> List[int]:
    s = []
    for row in H:
        v = 0
        for b, c in zip(bits7, row):
            v ^= (b & c)
        s.append(v & 1)
    return s


def matrix_embed(img: Image.Image, bits: List[int], cfg: QCHConfig, seed: bytes, logger) -> Image.Image:
    img = img.convert("RGB").resize((cfg.width, cfg.height), Image.LANCZOS)
    data = bytearray(img.tobytes())
    r = prng(seed)
    indices = lsb_positions(cfg, r, "B")
    groups = [indices[i:i + 7] for i in range(0, len(indices), 7)]
    bi = 0
    for g in groups:
        if len(g) < 7:
            break
        c7 = [(data[idx] & 1) for idx in g]
        if bi >= len(bits):
            break
        m0 = bits[bi] if bi < len(bits) else 0
        m1 = bits[bi + 1] if bi + 1 < len(bits) else 0
        m2 = bits[bi + 2] if bi + 2 < len(bits) else 0
        bi += 3
        s = hamming_syndrome(c7)
        e2 = (m0 ^ s[0])
        e1 = (m1 ^ s[1])
        e0 = (m2 ^ s[2])
        err_pos = (e2 << 2) | (e1 << 1) | e0
        if err_pos != 0:
            flip = err_pos - 1
            c7[flip] ^= 1
        for k, idx in enumerate(g):
            data[idx] = (data[idx] & 0xFE) | (c7[k] & 1)
        if bi >= len(bits):
            break
    if bi < len(bits):
        logger.warning(f"[MATRIX] Ran out of capacity; embedded {bi}/{len(bits)} bits")
        raise ValueError("Matrix-coding capacity insufficient for payload")
    return Image.frombytes("RGB", (cfg.width, cfg.height), bytes(data))


def matrix_extract(img: Image.Image, cfg: QCHConfig, seed: bytes, logger):
    img = img.convert("RGB").resize((cfg.width, cfg.height), Image.LANCZOS)
    data = img.tobytes()
    r = prng(seed)
    indices = lsb_positions(cfg, r, "B")
    groups = [indices[i:i + 7] for i in range(0, len(indices), 7)]
    for g in groups:
        if len(g) < 7:
            break
        c7 = [(data[idx] & 1) for idx in g]
        s = hamming_syndrome(c7)
        for b in s:
            yield b
