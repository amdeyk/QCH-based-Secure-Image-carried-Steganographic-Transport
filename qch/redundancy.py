from typing import List


def xor_parity(chunks: List[bytes]) -> bytes:
    m = max(len(c) for c in chunks)
    acc = bytearray(m)
    for c in chunks:
        cc = c + b"\x00" * (m - len(c))
        for i in range(m):
            acc[i] ^= cc[i]
    return bytes(acc)
