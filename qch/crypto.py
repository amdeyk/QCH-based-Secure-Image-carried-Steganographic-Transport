import json
import os
import struct
import uuid
from typing import Optional, Tuple, Dict

from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives import serialization

from .utils import now_ts, sha256, b64e, b64d, secure_randbytes
from .config import QCHConfig


def write_private_key(path: str, priv: Ed25519PrivateKey) -> None:
    pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    with open(path, "wb") as f:
        f.write(pem)


def write_public_key(path: str, pub: Ed25519PublicKey) -> None:
    pem = pub.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    with open(path, "wb") as f:
        f.write(pem)


def read_private_key(path: str) -> Ed25519PrivateKey:
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def derive_key_scrypt(passphrase: str, salt: bytes, logger) -> bytes:
    kdf = Scrypt(salt=salt, length=32, n=2 ** 15, r=8, p=1)
    key = kdf.derive(passphrase.encode("utf-8"))
    logger.debug(f"[KDF] Scrypt derived; salt={b64e(salt)}")
    return key


def library_fingerprint(library_path: Optional[str], logger) -> bytes:
    if library_path and os.path.isfile(library_path):
        data = open(library_path, "rb").read()
        logger.info(f"[LIB] Using provided library file: {library_path}")
    else:
        env = json.dumps(
            {
                "platform": os.sys.platform,
                "py": os.sys.version.split()[0],
                "tz": os.time.tzname,
            },
            sort_keys=True,
        ).encode("utf-8")
        data = env
        logger.info("[LIB] Using environment fingerprint as salt (no library provided).")
    return sha256(data)


def encrypt_and_sign(
    plaintext: bytes,
    passphrase: str,
    library_path: Optional[str],
    priv: Ed25519PrivateKey,
    cfg: QCHConfig,
    logger,
) -> Tuple[bytes, dict]:
    salt = library_fingerprint(library_path, logger)
    key = derive_key_scrypt(passphrase, salt, logger)
    aes = AESGCM(key)
    nonce = secure_randbytes(12)
    ct = aes.encrypt(nonce, plaintext, associated_data=None)
    run_id = str(uuid.uuid4())
    created_ts = now_ts()
    timecode = created_ts // cfg.timestep_sec
    pub = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    header = {
        "version": 2,
        "run_id": run_id,
        "created_ts": created_ts,
        "timestep_sec": cfg.timestep_sec,
        "timecode": timecode,
        "nonce_b64": b64e(nonce),
        "salt_b64": b64e(salt),
        "cipher_sha256_b64": b64e(sha256(ct)),
        "pubkey_b64": b64e(pub),
        "cfg": {"w": cfg.width, "h": cfg.height, "bpc": cfg.bpc, "overlap": cfg.overlap_ratio},
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    sig = priv.sign(header_bytes + ct)
    meta = {**header, "sig_b64": b64e(sig)}
    logger.info(f"[CRYPTO] AES-GCM encrypt + Ed25519 sign; ct={len(ct)} bytes")
    return ct, meta


def verify_and_decrypt(ciphertext: bytes, meta: dict, passphrase: str, logger) -> bytes:
    req = ["nonce_b64", "salt_b64", "cipher_sha256_b64", "pubkey_b64", "sig_b64"]
    for k in req:
        if k not in meta:
            raise ValueError(f"Missing meta field: {k}")
    if b64e(sha256(ciphertext)) != meta["cipher_sha256_b64"]:
        raise ValueError("Cipher SHA-256 mismatch")
    pub = Ed25519PublicKey.from_public_bytes(b64d(meta["pubkey_b64"]))
    header_wo_sig = json.dumps({k: meta[k] for k in meta if k != "sig_b64"}, separators=(",", ":")).encode("utf-8")
    pub.verify(b64d(meta["sig_b64"]), header_wo_sig + ciphertext)
    logger.info("[CRYPTO] Signature + hash OK.")
    key = derive_key_scrypt(passphrase, b64d(meta["salt_b64"]), logger)
    aes = AESGCM(key)
    pt = aes.decrypt(b64d(meta["nonce_b64"]), ciphertext, associated_data=None)
    logger.info("[CRYPTO] Decrypt OK.")
    return pt
