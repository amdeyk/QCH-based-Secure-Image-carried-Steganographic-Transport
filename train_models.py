import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
import argparse

from qch.neural_compression import TransformerVAE, CompressionConfig
from qch.adaptive_embedding import RegionAnalyzer
from qch.transformer_pipeline import TransformerPipeline

class CompressionDataset(Dataset):
    """Dataset for training compression models"""
    
    def __init__(self, data_dir: str, chunk_size: int = 8192):
        self.files = list(Path(data_dir).glob("**/*"))
        self.chunk_size = chunk_size
    
    def __len__(self):
        return len(self.files) * 10  # Multiple chunks per file
    
    def __getitem__(self, idx):
        file_idx = idx // 10
        chunk_idx = idx % 10
        with open(self.files[file_idx], 'rb') as f:
            data = f.read()
        start = chunk_idx * self.chunk_size
        chunk = data[start:start + self.chunk_size]
        if len(chunk) < self.chunk_size:
            chunk = chunk + b'\x00' * (self.chunk_size - len(chunk))
        arr = np.frombuffer(chunk, dtype=np.uint8)
        tensor = torch.from_numpy(arr.astype(np.float32) / 255.0)
        return tensor.reshape(self.chunk_size // 32, 32)

def train_compression_model(args):
    """Train the neural compression model"""
    config = CompressionConfig()
    model = TransformerVAE(config).to(config.device)
    dataset = CompressionDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(config.device)
            reconstructed, mu, logvar = model(batch)
            recon_loss = nn.MSELoss()(reconstructed, batch)
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.001 * kld_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss/len(dataloader):.4f}")
    torch.save({'model_state_dict': model.state_dict(), 'config': config}, args.output)
    print(f"Model saved to {args.output}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["compression", "adaptive", "pipeline"], required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    if args.mode == "compression":
        train_compression_model(args)
    # Additional modes can be implemented later

if __name__ == "__main__":
    main()
