import io
import json
import struct
from typing import Iterator, Tuple

from .utils import crc32


def build_embedded_blob(
    meta: dict,
    scheme: str,
    replica_id: int,
    slice_idx: int,
    slice_total: int,
    ct_slice: bytes,
) -> bytes:
    m = dict(meta)
    m["scheme"] = scheme
    m["replica"] = replica_id
    m["slice"] = {"index": slice_idx, "total": slice_total, "length": len(ct_slice)}
    mj = json.dumps(m, separators=(",", ":")).encode("utf-8")
    buf = io.BytesIO()
    buf.write(struct.pack(">I", len(mj)))
    buf.write(mj)
    buf.write(struct.pack(">I", len(ct_slice)))
    buf.write(ct_slice)
    buf.write(struct.pack(">I", crc32(ct_slice)))
    return buf.getvalue()


def parse_blob_progressive(bit_reader_iter: Iterator[int]) -> Tuple[dict, bytes, int]:
    """Parse a variable-length blob from an iterator of bits."""

    def bits_to_bytes(bits):
        out = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                if i + j < len(bits):
                    byte = (byte << 1) | (bits[i + j] & 1)
                else:
                    byte <<= 1
            out.append(byte)
        return bytes(out)

    bits = []
    need = 32
    for bit in bit_reader_iter:
        bits.append(bit)
        if len(bits) < need:
            continue
        raw = bits_to_bytes(bits[:4])
        mlen = struct.unpack(">I", raw)[0]
        need = (4 + mlen + 4) * 8
        if len(bits) < need:
            continue
        raw2 = bits_to_bytes(bits[: 4 + mlen])
        mjson = raw2[4:4 + mlen]
        meta = json.loads(mjson.decode("utf-8"))
        prefix_bits = (4 + mlen + 4) * 8
        if len(bits) < prefix_bits:
            continue
        raw3 = bits_to_bytes(bits[: 4 + mlen + 4])
        slen = struct.unpack(">I", raw3[-4:])[0]
        total_bits_needed = (4 + mlen + 4 + slen + 4) * 8
        if len(bits) < total_bits_needed:
            continue
        raw_all = bits_to_bytes(bits[: 4 + mlen + 4 + slen + 4])
        offset = 4 + mlen + 4
        slc = raw_all[offset: offset + slen]
        c = struct.unpack(">I", raw_all[offset + slen: offset + slen + 4])[0]
        if crc32(slc) != c:
            raise ValueError("CRC mismatch in parsed blob")
        return meta, slc, total_bits_needed
    raise ValueError("Not enough bits to parse blob")
