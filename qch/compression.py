import io
import json
import lzma
import os
import struct
from typing import List, Tuple

try:
    import zstandard as zstd
    HAVE_ZSTD = True
except Exception:  # pragma: no cover - optional dependency
    HAVE_ZSTD = False

from .utils import now_ts


def pack_files(paths: List[str], logger) -> bytes:
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, fnames in os.walk(p):
                for fn in fnames:
                    fp = os.path.join(root, fn)
                    rel = os.path.relpath(fp, start=os.path.dirname(paths[0]) or ".")
                    files.append((rel, open(fp, "rb").read()))
        else:
            files.append((os.path.basename(p), open(p, "rb").read()))
    manifest = [{"path": rel, "size": len(b)} for (rel, b) in files]
    logger.info(f"[PACK] {len(files)} files; building container with manifest.")
    buf = io.BytesIO()
    mjson = json.dumps({"manifest": manifest, "created_ts": now_ts()}, separators=(",", ":")).encode("utf-8")
    buf.write(struct.pack(">I", len(mjson)))
    buf.write(mjson)
    buf.write(struct.pack(">I", len(files)))
    for rel, b in files:
        rel_b = rel.encode("utf-8")
        buf.write(struct.pack(">I", len(rel_b)))
        buf.write(rel_b)
        buf.write(struct.pack(">I", len(b)))
        buf.write(b)
    raw = buf.getvalue()
    logger.info(f"[PACK] Raw size={len(raw)} bytes")
    return raw


def compress_bytes(raw: bytes, method: str, level: int, logger) -> Tuple[str, bytes]:
    method = (method or "lzma").lower()
    if method == "zstd" and HAVE_ZSTD:
        c = zstd.ZstdCompressor(level=level if level else 10)
        out = c.compress(raw)
        logger.info(f"[CMP] Zstandard level={level or 10}: {len(raw)} -> {len(out)} bytes")
        return "zstd", out
    preset = level if level else 9
    try:
        out = lzma.compress(raw, preset=preset | lzma.PRESET_EXTREME)
    except Exception:  # pragma: no cover - lzma may not support extreme flag
        out = lzma.compress(raw, preset=preset)
    logger.info(f"[CMP] LZMA preset={preset} extreme: {len(raw)} -> {len(out)} bytes")
    return "lzma", out


def decompress_bytes(tag: str, comp: bytes, logger) -> bytes:
    if tag == "zstd" and HAVE_ZSTD:
        d = zstd.ZstdDecompressor()
        out = d.decompress(comp)
        logger.info("[DCMP] Zstandard OK.")
        return out
    out = lzma.decompress(comp)
    logger.info("[DCMP] LZMA OK.")
    return out


def unpack_files(comp: bytes, out_dir: str, logger) -> None:
    s = io.BytesIO(comp)
    mlen = struct.unpack(">I", s.read(4))[0]
    mjson = json.loads(s.read(mlen).decode("utf-8"))
    count = struct.unpack(">I", s.read(4))[0]
    logger.info(f"[UNPACK] Restoring {count} files; created_ts={mjson.get('created_ts')}")
    os.makedirs(out_dir, exist_ok=True)
    for _ in range(count):
        nlen = struct.unpack(">I", s.read(4))[0]
        rel = s.read(nlen).decode("utf-8")
        blen = struct.unpack(">I", s.read(4))[0]
        b = s.read(blen)
        dst = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            f.write(b)
        logger.debug(f"[UNPACK] Wrote {dst} ({blen} bytes)")
