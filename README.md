# QCH-based Secure Image-carried Steganographic Transport

This project provides a modular implementation of the "QCH" steganographic
transport system. Payloads are encrypted (AES-256-GCM, Ed25519 signatures) and
embedded into PNG images using one of several schemes.

## Key Features

- **Classic schemes** – `pvd`, `matrix`, `interp`, `lsb`
- **Neural compression** – Transformer VAE for learned byte compression
- **Adaptive multi-scale embedding** – CNN-guided capacity analysis
- **Hardware acceleration** – CUDA/Numba/Intel XPU helpers
- **Experimental quantum crypto** – optional key generation and flags

## Installation

```bash
python -m pip install -r requirements.txt
```

## Quick start

### Generate classical keys
```bash
python -m qch.cli init-keys --out-dir keys -v
```

### Generate quantum-resistant keys (experimental)
```bash
python -m qch.cli init-quantum-keys --out-dir qkeys -v
```

### Send payload with neural compression and adaptive embedding
```bash
python -m qch.cli send \
  --inputs ./docs ./media/trailer.mp4 ./notes.txt \
  --passphrase "My#StrongPass!2025" \
  --privkey keys/qch_ed25519_private.pem \
  --cover-folder ./covers \
  --scheme pvd --compress neural --neural-model compressor.pth \
  --adaptive-embed --adaptive-model embedder.pth \
  --replicas 2 --xor-parity \
  --out-dir tx -v
```

### Receive payload
```bash
python -m qch.cli recv \
  --images tx/<run_id>_rep0_A.png tx/<run_id>_rep1_B.png \
  --passphrase "My#StrongPass!2025" \
  --neural-model compressor.pth \
  --out-dir rx -v
```

See `python -m qch.cli --help` for all options. Quantum and hardware
acceleration flags are included for future work and may require additional
setup.
