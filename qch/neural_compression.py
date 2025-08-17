import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import onnxruntime as ort
from dataclasses import dataclass

@dataclass
class CompressionConfig:
    latent_dim: int = 256
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    chunk_size: int = 8192
    quantize: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class TransformerVAE(nn.Module):
    def __init__(self, config: CompressionConfig):
        super().__init__()
        self.config = config
        
        # Encoder
        self.input_projection = nn.Linear(256, config.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.num_layers)
        
        # VAE bottleneck
        self.fc_mu = nn.Linear(config.hidden_dim, config.latent_dim)
        self.fc_var = nn.Linear(config.hidden_dim, config.latent_dim)
        
        # Decoder
        self.latent_projection = nn.Linear(config.latent_dim, config.hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, config.num_layers)
        self.output_projection = nn.Linear(config.hidden_dim, 256)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x):
        x = self.input_projection(x)
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def decode(self, z):
        x = self.latent_projection(z)
        x = self.decoder(x, x)
        return torch.sigmoid(self.output_projection(x))
    
    def forward(self, x):
        z, mu, logvar = self.encode(x)
        return self.decode(z), mu, logvar

class NeuralCompressor:
    def __init__(self, model_path: Optional[str] = None, config: Optional[CompressionConfig] = None):
        self.config = config or CompressionConfig()
        self.model = TransformerVAE(self.config).to(self.config.device)
        
        if model_path:
            self.load_model(model_path)
        
        self.ort_session = None
        
    def bytes_to_tensor(self, data: bytes) -> torch.Tensor:
        # Convert bytes to normalized float tensor
        arr = np.frombuffer(data, dtype=np.uint8)
        # Pad to chunk size
        pad_len = (self.config.chunk_size - len(arr) % self.config.chunk_size) % self.config.chunk_size
        arr = np.pad(arr, (0, pad_len), 'constant')
        # Reshape to chunks
        arr = arr.reshape(-1, self.config.chunk_size // 32, 32)
        # Normalize to [0, 1]
        tensor = torch.from_numpy(arr.astype(np.float32) / 255.0)
        # Flatten last dimension for transformer
        tensor = tensor.reshape(tensor.shape[0], tensor.shape[1], -1)
        return tensor.to(self.config.device)
    
    def tensor_to_bytes(self, tensor: torch.Tensor, original_len: int) -> bytes:
        # Denormalize and convert back to bytes
        arr = (tensor.cpu().numpy() * 255).astype(np.uint8)
        arr = arr.flatten()[:original_len]
        return arr.tobytes()
    
    def compress(self, data: bytes, logger) -> bytes:
        self.model.eval()
        original_len = len(data)
        
        with torch.no_grad():
            # Convert to tensor
            x = self.bytes_to_tensor(data)
            
            # Encode
            z, mu, _ = self.model.encode(x)
            
            # Quantize if enabled
            if self.config.quantize:
                z = torch.round(z * 127) / 127  # INT8 quantization
            
            # Pack compressed data
            compressed = {
                'latent': z.cpu().numpy().astype(np.float16),  # Half precision
                'shape': x.shape,
                'original_len': original_len
            }
            
            # Serialize with numpy
            import io
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **compressed)
            compressed_bytes = buffer.getvalue()
            
        ratio = len(data) / len(compressed_bytes)
        logger.info(f"[NEURAL] Compressed {len(data)} -> {len(compressed_bytes)} bytes (ratio: {ratio:.2f}x)")
        return compressed_bytes
    
    def decompress(self, compressed_data: bytes, logger) -> bytes:
        self.model.eval()
        
        with torch.no_grad():
            # Unpack compressed data
            import io
            buffer = io.BytesIO(compressed_data)
            data = np.load(buffer)
            
            z = torch.from_numpy(data['latent'].astype(np.float32)).to(self.config.device)
            original_len = int(data['original_len'])
            
            # Decode
            reconstructed = self.model.decode(z)
            
            # Convert back to bytes
            result = self.tensor_to_bytes(reconstructed, original_len)
            
        logger.info(f"[NEURAL] Decompressed {len(compressed_data)} -> {len(result)} bytes")
        return result
    
    def export_onnx(self, path: str):
        """Export model to ONNX for faster inference"""
        dummy_input = torch.randn(1, 32, 256).to(self.config.device)
        torch.onnx.export(
            self.model,
            dummy_input,
            path,
            export_params=True,
            opset_version=11,
            input_names=['input'],
            output_names=['output', 'mu', 'logvar'],
            dynamic_axes={'input': {0: 'batch_size'}}
        )
    
    def load_onnx(self, path: str):
        """Load ONNX model for inference"""
        self.ort_session = ort.InferenceSession(path)
    
    def save_model(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, path)
    
    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
