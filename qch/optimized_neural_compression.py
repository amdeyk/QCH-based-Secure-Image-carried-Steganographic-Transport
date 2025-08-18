import torch
import torch.nn as nn
import numpy as np
import lzma
import zstd
from typing import Optional, Tuple, Dict, Any
import io
from pathlib import Path


class LightweightVAE(nn.Module):
    """Optimized VAE for consumer hardware"""

    def __init__(self, latent_dim: int = 128, input_size: int = 1024):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim

        # Encoder - minimal layers for speed
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim * 2),  # mu and logvar
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, input_size),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class OptimizedNeuralCompressor:
    """Production-ready neural compressor for consumer hardware"""

    def __init__(self, model_path: Optional[str] = None, device: str = "auto", chunk_size: int = 8192):
        self.chunk_size = chunk_size
        self.device = self._setup_device(device)
        self.model: Optional[LightweightVAE] = None
        self.quantized_model = None

        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            self.model = LightweightVAE(input_size=self.chunk_size).to(self.device)

    def _setup_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")  # Apple Silicon
            return torch.device("cpu")
        return torch.device(device)

    def _preprocess_data(self, data: bytes) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Convert bytes to tensor chunks with deterministic padding"""
        original_length = len(data)
        remainder = len(data) % self.chunk_size
        if remainder:
            pad_length = self.chunk_size - remainder
            padding = bytes([(256 - pad_length + i) % 256 for i in range(pad_length)])
            data = data + padding
        else:
            pad_length = 0
            padding = b""

        arr = np.frombuffer(data, dtype=np.uint8)
        arr = arr.reshape(-1, self.chunk_size)
        tensor = torch.from_numpy(arr.astype(np.float32) / 255.0)

        metadata = {
            "original_length": original_length,
            "pad_length": pad_length,
            "padding_pattern": padding.hex() if padding else "",
            "num_chunks": tensor.shape[0],
            "chunk_size": self.chunk_size,
        }

        return tensor.to(self.device), metadata

    def _postprocess_data(self, tensor: torch.Tensor, metadata: Dict[str, Any]) -> bytes:
        """Convert tensor back to bytes using stored metadata"""
        arr = torch.round(tensor * 255.0).clamp(0, 255).byte().cpu().numpy()
        data = arr.flatten().tobytes()
        if metadata["pad_length"] > 0:
            data = data[:-metadata["pad_length"]]
        return data[: metadata["original_length"]]

    def compress(self, data: bytes, logger=None, fallback_classical: bool = True) -> bytes:
        """Compress data with neural network + fallback"""

        try:
            if self.model is None:
                raise RuntimeError("No model loaded")

            self.model.eval()
            with torch.no_grad():
                x, metadata = self._preprocess_data(data)
                mu, _ = self.model.encode(x)

                # Use high-precision quantization
                mu_quantized = torch.round(mu * 32767) / 32767
                mu_quantized = mu_quantized.clamp(-1, 1)

                compressed_dict = {
                    "mu": mu_quantized.cpu().numpy().astype(np.float16),
                    "metadata": np.array([metadata], dtype=object),
                    "method": "neural_precision",
                }

                buffer = io.BytesIO()
                np.savez_compressed(buffer, **compressed_dict)
                neural_compressed = buffer.getvalue()

                if fallback_classical:
                    classical_compressed = lzma.compress(data, preset=1)
                    if len(neural_compressed) < len(classical_compressed):
                        if logger:
                            ratio = len(data) / len(neural_compressed)
                            logger.info(
                                f"[NEURAL] Compressed {len(data)} -> {len(neural_compressed)} bytes (ratio: {ratio:.2f}x)"
                            )
                        return neural_compressed
                    if logger:
                        ratio = len(data) / len(classical_compressed)
                        logger.info(
                            f"[CLASSICAL] Neural failed, using LZMA {len(data)} -> {len(classical_compressed)} bytes (ratio: {ratio:.2f}x)"
                        )
                    return classical_compressed

                return neural_compressed
        except Exception as e:
            if logger:
                logger.warning(f"[NEURAL] Compression failed: {e}, falling back to classical")
            if fallback_classical:
                return lzma.compress(data, preset=1)
            raise

    def decompress(self, compressed_data: bytes, logger=None) -> bytes:
        """Decompress data with automatic method detection"""
        try:
            buffer = io.BytesIO(compressed_data)
            data_dict = np.load(buffer, allow_pickle=True)
            if "method" in data_dict and data_dict["method"] == "neural_precision":
                return self._decompress_neural(data_dict, logger)
            raise ValueError("Not neural compressed")
        except Exception:
            try:
                result = lzma.decompress(compressed_data)
                if logger:
                    logger.info(
                        f"[CLASSICAL] Decompressed {len(compressed_data)} -> {len(result)} bytes"
                    )
                return result
            except Exception:
                result = zstd.decompress(compressed_data)
                if logger:
                    logger.info(
                        f"[ZSTD] Decompressed {len(compressed_data)} -> {len(result)} bytes"
                    )
                return result

    def _decompress_neural(self, data_dict, logger=None) -> bytes:
        if self.model is None:
            raise RuntimeError("No model loaded for neural decompression")
        self.model.eval()
        with torch.no_grad():
            mu = torch.from_numpy(data_dict["mu"].astype(np.float32)).to(self.device)
            metadata = data_dict["metadata"].item()
            reconstructed = self.model.decode(mu)
            result = self._postprocess_data(reconstructed, metadata)
            if logger:
                logger.info(
                    f"[NEURAL] Decompressed {len(mu)} tensors -> {len(result)} bytes"
                )
            return result

    def quantize_model(self) -> None:
        """Quantize model for faster inference"""
        if self.model is None:
            return
        self.model.eval()
        self.quantized_model = torch.quantization.quantize_dynamic(
            self.model, {nn.Linear}, dtype=torch.qint8
        )
        self.model = self.quantized_model

    def save_model(self, path: str) -> None:
        if self.model is None:
            return
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "chunk_size": self.chunk_size,
                "device": str(self.device),
            },
            path,
        )

    def load_model(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.chunk_size = checkpoint.get("chunk_size", 8192)
        self.model = LightweightVAE(input_size=self.chunk_size).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()


def create_lightweight_compressor(model_path: Optional[str] = None, enable_gpu: bool = True) -> OptimizedNeuralCompressor:
    """Factory function for creating optimized compressor"""
    device = "auto" if enable_gpu else "cpu"
    compressor = OptimizedNeuralCompressor(model_path=model_path, device=device)
    if compressor.device.type == "cpu":
        compressor.quantize_model()
    return compressor


def benchmark_compression(data_size_mb: int = 10, compressor: Optional[OptimizedNeuralCompressor] = None):
    """Benchmark compression performance"""
    import time

    if compressor is None:
        compressor = create_lightweight_compressor()

    test_data = np.random.bytes(data_size_mb * 1024 * 1024)

    start_time = time.time()
    compressed = compressor.compress(test_data)
    neural_time = time.time() - start_time

    start_time = time.time()
    decompressed = compressor.decompress(compressed)
    decomp_time = time.time() - start_time

    start_time = time.time()
    classical = lzma.compress(test_data, preset=1)
    classical_time = time.time() - start_time

    print(f"Original size: {len(test_data):,} bytes")
    print(f"Neural compressed: {len(compressed):,} bytes in {neural_time:.2f}s")
    print(f"Classical compressed: {len(classical):,} bytes in {classical_time:.2f}s")
    print(f"Decompression time: {decomp_time:.2f}s")
    print(f"Data integrity: {'PASS' if test_data == decompressed else 'FAIL'}")

    return {
        "neural_ratio": len(test_data) / len(compressed),
        "classical_ratio": len(test_data) / len(classical),
        "neural_time": neural_time,
        "classical_time": classical_time,
        "decomp_time": decomp_time,
    }
