from typing import Tuple

from PIL import Image

from .config import QCHConfig
from .utils import prng, bytes_to_bits
from .blob import parse_blob_progressive
from .schemes import lsb, pvd, matrix, interp


def embed_one(img_cover: Image.Image, blob: bytes, scheme: str, seed: bytes, cfg: QCHConfig, logger) -> Image.Image:
    bits = bytes_to_bits(blob)  # quick conversion
    if scheme == "lsb":
        return lsb.embed_lsb_scatter(img_cover, bits, cfg, seed, logger)
    if scheme == "pvd":
        return pvd.pvd_embed(img_cover, bits, cfg, seed, logger)
    if scheme == "matrix":
        return matrix.matrix_embed(img_cover, bits, cfg, seed, logger)
    if scheme == "interp":
        return interp.interp_embed(img_cover, bits, cfg, seed, logger)
    raise ValueError("Unknown scheme")


def extract_one(img: Image.Image, scheme: str, cfg: QCHConfig, seed: bytes, logger) -> Tuple[dict, bytes]:
    if scheme == "lsb":
        cap_bits = len(lsb.lsb_positions(cfg, prng(seed), "B")) * cfg.bpc
        bits_iter = lsb.extract_lsb_scatter(img, cap_bits, cfg, seed, logger)
    elif scheme == "pvd":
        bits_iter = pvd.pvd_extract(img, cfg, seed, logger)
    elif scheme == "matrix":
        bits_iter = matrix.matrix_extract(img, cfg, seed, logger)
    elif scheme == "interp":
        bits_iter = interp.interp_extract(img, cfg, seed, logger)
    else:
        raise ValueError("Unknown scheme")
    meta, slc, _ = parse_blob_progressive(bits_iter)
    return meta, slc
