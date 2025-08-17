import base64
import hashlib
import secrets
import time
import zlib
from typing import List
import random


def now_ts() -> int:
    return int(time.time())


def sha256(data: bytes) -> bytes:
    h = hashlib.sha256()
    h.update(data)
    return h.digest()


def crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def b64e(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64d(txt: str) -> bytes:
    return base64.b64decode(txt.encode("ascii"))


def secure_randbytes(n: int) -> bytes:
    return secrets.token_bytes(n)


def bytes_to_bits(b: bytes) -> List[int]:
    out: List[int] = []
    for ch in b:
        for i in range(8):
            out.append((ch >> (7 - i)) & 1)
    return out


def bits_to_bytes(bits: List[int]) -> bytes:
    out = bytearray()
    for i in range(0, len(bits), 8):
        val = 0
        for j in range(8):
            if i + j < len(bits):
                val = (val << 1) | (bits[i + j] & 1)
            else:
                val <<= 1
        out.append(val)
    return bytes(out)


def prng(seed_bytes: bytes) -> random.Random:
    return random.Random(int.from_bytes(sha256(seed_bytes), "big"))
