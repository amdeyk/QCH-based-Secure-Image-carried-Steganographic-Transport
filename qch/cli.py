import argparse
import json
import os
import struct
from typing import Dict, List

from PIL import Image
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .blob import build_embedded_blob
from .compression import pack_files, compress_bytes, decompress_bytes, unpack_files
from .config import QCHConfig
from .cover import build_mosaic_cover
from .crypto import (
    write_private_key,
    write_public_key,
    read_private_key,
    encrypt_and_sign,
    verify_and_decrypt,
)
from .logger import setup_logger
from .redundancy import xor_parity
from .stego import embed_one, extract_one
from .utils import b64d, sha256


# ----- Commands -----

def cmd_init_keys(args) -> None:
    logger, xid = setup_logger(args.verbose)
    logger.info(f"[START] INIT-KEYS xid={xid}")
    os.makedirs(args.out_dir, exist_ok=True)
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()
    write_private_key(os.path.join(args.out_dir, "qch_ed25519_private.pem"), priv)
    write_public_key(os.path.join(args.out_dir, "qch_ed25519_public.pem"), pub)
    logger.info("[END] INIT-KEYS done.")


def embed_slices(args, cfg: QCHConfig, ct: bytes, meta: dict, logger):
    half = len(ct) // 2
    ov = int(half * cfg.overlap_ratio)
    sliceA = ct[: half + ov]
    sliceB = ct[half - ov :]
    replicas = max(1, args.replicas)
    groups_A = [sliceA for _ in range(replicas)]
    groups_B = [sliceB for _ in range(replicas)]
    if args.xor_parity:
        parityA = xor_parity(groups_A)
        parityB = xor_parity(groups_B)
        groups_A.append(parityA)
        groups_B.append(parityB)
        logger.info("[REDUNDANCY] Added XOR parity replica (can recover 1 missing replica).")
    total_reps = len(groups_A)
    os.makedirs(args.out_dir, exist_ok=True)
    for r_id in range(total_reps):
        coverA = build_mosaic_cover(args.cover_folder, cfg, logger)
        coverB = build_mosaic_cover(args.cover_folder, cfg, logger)
        blobA = build_embedded_blob(meta, args.scheme, r_id, 0, 2, groups_A[r_id])
        blobB = build_embedded_blob(meta, args.scheme, r_id, 1, 2, groups_B[r_id])
        seedA = b64d(meta["nonce_b64"]) + meta["run_id"].encode() + b"A" + struct.pack(">I", meta["timecode"]) + struct.pack(">I", r_id)
        seedB = b64d(meta["nonce_b64"]) + meta["run_id"].encode() + b"B" + struct.pack(">I", meta["timecode"]) + struct.pack(">I", r_id)
        imgA = embed_one(coverA, blobA, args.scheme, seedA, cfg, logger)
        imgB = embed_one(coverB, blobB, args.scheme, seedB, cfg, logger)
        tag = f"{meta['run_id']}_rep{r_id}"
        pathA = os.path.join(args.out_dir, f"{tag}_A.png")
        pathB = os.path.join(args.out_dir, f"{tag}_B.png")
        imgA.save(pathA, optimize=True)
        imgB.save(pathB, optimize=True)
        logger.info(f"[EMBED] Wrote {pathA} / {pathB}")


def cmd_send(args) -> None:
    logger, xid = setup_logger(args.verbose)
    cfg = QCHConfig(
        width=args.width,
        height=args.height,
        bpc=args.bpc,
        overlap_ratio=args.overlap,
        timestep_sec=args.timestep,
    )
    logger.info(
        f"[START] SEND xid={xid} scheme={args.scheme} replicas={args.replicas} xor_parity={args.xor_parity}"
    )
    raw = pack_files(args.inputs, logger)
    ctag, comp = compress_bytes(raw, args.compress, args.level, logger)
    priv = read_private_key(args.privkey)
    ct, meta = encrypt_and_sign(
        struct.pack(">I", len(ctag)) + ctag.encode("ascii") + comp,
        args.passphrase,
        args.library,
        priv,
        cfg,
        logger,
    )
    embed_slices(args, cfg, ct, meta, logger)
    logger.info("[END] SEND complete.")


def cmd_recv(args) -> None:
    logger, xid = setup_logger(args.verbose)
    cfg = QCHConfig(
        width=args.width,
        height=args.height,
        bpc=args.bpc,
        overlap_ratio=args.overlap,
        timestep_sec=args.timestep,
    )
    logger.info(f"[START] RECV xid={xid}")
    if len(args.images) < 1:
        raise SystemExit("Provide at least 1 embedded PNG.")
    candidates = []
    for path in args.images:
        img = Image.open(path).convert("RGB").resize((cfg.width, cfg.height), Image.LANCZOS)
        ok = False
        for label in (b"A", b"B"):
            for scheme in ("pvd", "matrix", "interp", "lsb"):
                try:
                    img_bytes = img.tobytes()
                    seed_hint = sha256(img_bytes)[:12] + label + b"\x00" * 8
                    meta_tmp, slc_tmp = extract_one(img, scheme, cfg, seed_hint, logger)
                    nonce = b64d(meta_tmp["nonce_b64"])
                    tcode = meta_tmp["timecode"]
                    run_id = meta_tmp["run_id"].encode()
                    rep = meta_tmp.get("replica", 0)
                    seed_true = nonce + run_id + label + struct.pack(">I", tcode) + struct.pack(">I", rep)
                    meta, slc = extract_one(img, scheme, cfg, seed_true, logger)
                    candidates.append((path, label.decode(), scheme, meta, slc))
                    logger.info(
                        f"[EXTRACT] {path}: scheme={scheme} label={label.decode()} rep={meta.get('replica')}"
                    )
                    ok = True
                    break
                except Exception:
                    continue
            if ok:
                break
        if not ok:
            logger.warning(f"[WARN] Could not parse: {path}")
    if not candidates:
        raise SystemExit("No valid embedded images parsed.")
    metas: List[dict] = []
    slices_by_index: Dict[int, List[bytes]] = {0: [], 1: []}
    meta_ref = None
    replicas_seen = set()
    for path, label, scheme, meta, slc in candidates:
        if meta_ref is None:
            meta_ref = meta
        if meta["run_id"] != meta_ref["run_id"]:
            logger.warning(f"[SKIP] Different run_id in {path}")
            continue
        if scheme != meta.get("scheme", scheme):
            logger.warning(f"[SKIP] Scheme mismatch in {path}")
            continue
        idx = meta["slice"]["index"]
        slices_by_index[idx].append(slc)
        replicas_seen.add(meta.get("replica", 0))
        metas.append(meta)
    recA = slices_by_index.get(0, [])
    recB = slices_by_index.get(1, [])
    if (not recA or not recB) and len(replicas_seen) >= 2:
        max_rep = max(replicas_seen)
        if not recA:
            data_reps = [slc for (_, _, _, m, slc) in candidates if m["slice"]["index"] == 0 and m.get("replica") < max_rep]
            parity = [slc for (_, _, _, m, slc) in candidates if m["slice"]["index"] == 0 and m.get("replica") == max_rep]
            if data_reps and parity:
                recA = [xor_parity(data_reps + parity)]
                logger.info("[RECOVER] Rebuilt slice A via XOR parity.")
        if not recB:
            data_reps = [slc for (_, _, _, m, slc) in candidates if m["slice"]["index"] == 1 and m.get("replica") < max_rep]
            parity = [slc for (_, _, _, m, slc) in candidates if m["slice"]["index"] == 1 and m.get("replica") == max_rep]
            if data_reps and parity:
                recB = [xor_parity(data_reps + parity)]
                logger.info("[RECOVER] Rebuilt slice B via XOR parity.")
    if not recA:
        raise SystemExit("Missing slice A after attempts.")
    if not recB:
        raise SystemExit("Missing slice B after attempts.")
    a = recA[0]
    b = recB[0]
    ov = min(len(a), len(b), 256_000)
    join = None
    for k in range(ov, 0, -1024):
        if a[-k:] == b[:k]:
            join = k
            break
    ct = a + b[join:] if join else a + b
    logger.info(f"[RECV] Reconstructed ciphertext; size={len(ct)} bytes")
    pt = verify_and_decrypt(ct, meta_ref, args.passphrase, logger)
    tag_len = struct.unpack(">I", pt[:4])[0]
    ctag = pt[4 : 4 + tag_len].decode("ascii")
    comp = pt[4 + tag_len :]
    raw = decompress_bytes(ctag, comp, logger)
    out_dir = args.out_dir or f"qch_out_{meta_ref['run_id']}"
    unpack_files(raw, out_dir, logger)
    logger.info(f"[END] RECV done. Restored to: {out_dir}")


# ----- CLI -----

def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="QCH-Advanced: modular stego toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init-keys")
    p_init.add_argument("--out-dir", default="qch_keys")
    p_init.add_argument("-v", "--verbose", action="store_true")
    p_init.set_defaults(func=cmd_init_keys)

    p_send = sub.add_parser("send")
    p_send.add_argument("--inputs", nargs="+", required=True)
    p_send.add_argument("--passphrase", required=True)
    p_send.add_argument("--privkey", required=True)
    p_send.add_argument("--library")
    p_send.add_argument("--cover-folder")
    p_send.add_argument("--out-dir", default="qch_tx")
    p_send.add_argument("--scheme", choices=["pvd", "matrix", "interp", "lsb"], default="pvd")
    p_send.add_argument("--compress", choices=["lzma", "zstd"], default="lzma")
    p_send.add_argument("--level", type=int, help="Compression level")
    p_send.add_argument("--width", type=int, default=1920)
    p_send.add_argument("--height", type=int, default=1080)
    p_send.add_argument("--bpc", type=int, default=2)
    p_send.add_argument("--overlap", type=float, default=0.10)
    p_send.add_argument("--timestep", type=int, default=30)
    p_send.add_argument("--replicas", type=int, default=2)
    p_send.add_argument("--xor-parity", action="store_true")
    p_send.add_argument("-v", "--verbose", action="store_true")
    p_send.set_defaults(func=cmd_send)

    p_recv = sub.add_parser("recv")
    p_recv.add_argument("--images", nargs="+", required=True)
    p_recv.add_argument("--passphrase", required=True)
    p_recv.add_argument("--out-dir")
    p_recv.add_argument("--width", type=int, default=1920)
    p_recv.add_argument("--height", type=int, default=1080)
    p_recv.add_argument("--bpc", type=int, default=2)
    p_recv.add_argument("--overlap", type=float, default=0.10)
    p_recv.add_argument("--timestep", type=int, default=30)
    p_recv.add_argument("-v", "--verbose", action="store_true")
    p_recv.set_defaults(func=cmd_recv)

    return p


def main() -> None:
    cli = build_cli()
    args = cli.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
