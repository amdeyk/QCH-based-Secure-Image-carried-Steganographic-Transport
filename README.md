# QCH-based Secure Image-carried Steganographic Transport

This project provides a modular implementation of the "QCH" steganographic
transport system.  Payloads are encrypted (AES-256-GCM, Ed25519 signatures) and
embedded into PNG images using one of several schemes:

- **pvd** – pixel value differencing on the blue channel
- **matrix** – Hamming (7,3) matrix coding on LSBs
- **interp** – interpolation based embedding during 2× upscaling
- **lsb** – baseline scattered LSBs

The toolkit also supports optional LZMA/Zstandard compression, redundant
replicas with XOR parity, and detailed logging with unique run identifiers.

## Installation

```bash
python -m pip install cryptography pillow
# Optional for faster/better compression
python -m pip install zstandard
```

## Quick start

### Generate keys
```bash
python -m qch.cli init-keys --out-dir keys -v
```

### Send payload
```bash
python -m qch.cli send \
  --inputs ./docs ./media/trailer.mp4 ./notes.txt \
  --passphrase "My#StrongPass!2025" \
  --privkey keys/qch_ed25519_private.pem \
  --library ./char_font_library.txt \
  --cover-folder ./covers \
  --scheme pvd --compress lzma --level 9 \
  --replicas 2 --xor-parity \
  --out-dir tx -v
```

### Receive payload
```bash
python -m qch.cli recv \
  --images tx/<run_id>_rep0_A.png tx/<run_id>_rep1_B.png tx/<run_id>_rep2_A.png \
  --passphrase "My#StrongPass!2025" \
  --out-dir rx -v
```

See `python -m qch.cli --help` for all options.
