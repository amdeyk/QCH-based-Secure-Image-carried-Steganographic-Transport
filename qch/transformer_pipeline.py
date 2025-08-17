import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
import struct

class BinaryTokenizer:
    """Tokenizer for binary data using byte-pair encoding approach"""
    
    def __init__(self, vocab_size: int = 4096):
        self.vocab_size = vocab_size
        self.byte_to_token = {}
        self.token_to_byte = {}
        self._build_vocab()
    
    def _build_vocab(self):
        # Start with single bytes
        for i in range(256):
            self.byte_to_token[bytes([i])] = i
            self.token_to_byte[i] = bytes([i])
        
        # Add common byte pairs
        next_token = 256
        common_pairs = [
            b'\x00\x00', b'\xff\xff', b'\x00\x01', b'\x01\x00',
            b'\x0a\x0d', b'\x0d\x0a', b'\x20\x20', b'\x00\xff'
        ]
        
        for pair in common_pairs:
            if next_token < self.vocab_size:
                self.byte_to_token[pair] = next_token
                self.token_to_byte[next_token] = pair
                next_token += 1
    
    def tokenize(self, data: bytes) -> List[int]:
        tokens = []
        i = 0
        while i < len(data):
            # Try to match longer sequences first
            matched = False
            for length in [2, 1]:
                if i + length <= len(data):
                    seq = data[i:i+length]
                    if seq in self.byte_to_token:
                        tokens.append(self.byte_to_token[seq])
                        i += length
                        matched = True
                        break
            
            if not matched:
                tokens.append(data[i])
                i += 1
        
        return tokens
    
    def detokenize(self, tokens: List[int]) -> bytes:
        result = bytearray()
        for token in tokens:
            if token in self.token_to_byte:
                result.extend(self.token_to_byte[token])
            elif token < 256:
                result.append(token)
        return bytes(result)

class TransformerPipeline(nn.Module):
    def __init__(self, vocab_size: int = 4096, d_model: int = 512, nhead: int = 8, num_layers: int = 6):
        super().__init__()
        self.d_model = d_model
        self.tokenizer = BinaryTokenizer(vocab_size)
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self._create_positional_encoding(10000, d_model)
        
        # Transformer encoder for compression
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Compression head
        self.compress_head = nn.Linear(d_model, d_model // 4)
        
        # Transformer decoder for decompression
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            activation='gelu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Decompression head
        self.decompress_head = nn.Linear(d_model // 4, d_model)
        self.output_head = nn.Linear(d_model, vocab_size)
        
        # Error correction layers
        self.error_correction = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * 2, batch_first=True),
            num_layers=2
        )
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def encode_data(self, data: bytes) -> Tuple[torch.Tensor, dict]:
        # Tokenize
        tokens = self.tokenizer.tokenize(data)
        tokens_tensor = torch.tensor(tokens).unsqueeze(0)
        
        # Embed
        x = self.token_embedding(tokens_tensor)
        x += self.positional_encoding[:, :x.size(1), :]
        
        # Encode
        encoded = self.encoder(x)
        
        # Compress
        compressed = self.compress_head(encoded)
        
        metadata = {
            'original_length': len(data),
            'num_tokens': len(tokens),
            'compressed_shape': compressed.shape
        }
        
        return compressed, metadata
    
    def decode_data(self, compressed: torch.Tensor, metadata: dict) -> bytes:
        # Decompress
        x = self.decompress_head(compressed)
        
        # Add positional encoding
        x += self.positional_encoding[:, :x.size(1), :]
        
        # Decode
        decoded = self.decoder(x, x)
        
        # Error correction
        corrected = self.error_correction(decoded)
        
        # Output tokens
        output = self.output_head(corrected)
        tokens = torch.argmax(output, dim=-1)
        
        # Detokenize
        tokens_list = tokens.squeeze(0).tolist()
        data = self.tokenizer.detokenize(tokens_list)
        
        # Truncate to original length
        return data[:metadata['original_length']]
    
    def progressive_decode(self, compressed: torch.Tensor, metadata: dict, progress_callback=None):
        """Progressive decoding with intermediate results"""
        x = self.decompress_head(compressed)
        
        chunk_size = 128
        reconstructed = []
        
        for i in range(0, x.size(1), chunk_size):
            chunk = x[:, i:i+chunk_size, :]
            chunk += self.positional_encoding[:, :chunk.size(1), :]
            
            decoded_chunk = self.decoder(chunk, chunk)
            corrected_chunk = self.error_correction(decoded_chunk)
            output_chunk = self.output_head(corrected_chunk)
            
            tokens = torch.argmax(output_chunk, dim=-1)
            tokens_list = tokens.squeeze(0).tolist()
            chunk_data = self.tokenizer.detokenize(tokens_list)
            
            reconstructed.append(chunk_data)
            
            if progress_callback:
                progress = min(100, (i + chunk_size) * 100 // x.size(1))
                progress_callback(progress, b''.join(reconstructed))
        
        return b''.join(reconstructed)[:metadata['original_length']]
