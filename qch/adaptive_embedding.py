import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Optional
import cv2

class RegionAnalyzer(nn.Module):
    """CNN to analyze image regions and predict embedding capacity"""
    
    def __init__(self, patch_size: int = 32):
        super().__init__()
        self.patch_size = patch_size
        
        # Lightweight CNN for region analysis
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Attention mechanism for identifying safe zones
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Capacity predictor
        self.capacity = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 3 capacity levels: low, medium, high
        )
    
    def forward(self, x):
        features = self.features(x)
        features_flat = features.view(features.size(0), -1)
        
        attention_score = self.attention(features_flat)
        capacity_level = torch.softmax(self.capacity(features_flat), dim=1)
        
        return attention_score, capacity_level

class AdaptiveEmbedder:
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.analyzer = RegionAnalyzer().to(self.device)
        self.patch_size = 32
        
        if model_path:
            self.load_model(model_path)
        
        # Capacity multipliers for each level
        self.capacity_multipliers = [0.5, 1.0, 2.0]  # low, medium, high
    
    def analyze_image(self, img: Image.Image) -> Dict[str, np.ndarray]:
        """Analyze image and return capacity map"""
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        
        # Divide image into patches
        patches = []
        positions = []
        
        for y in range(0, h - self.patch_size + 1, self.patch_size):
            for x in range(0, w - self.patch_size + 1, self.patch_size):
                patch = img_np[y:y+self.patch_size, x:x+self.patch_size]
                patches.append(patch)
                positions.append((x, y))
        
        if not patches:
            return {'capacity_map': np.zeros((h, w)), 'attention_map': np.zeros((h, w)), 'total_capacity': 0}
        
        # Convert to tensor
        patches_tensor = torch.stack([
            torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
            for patch in patches
        ]).to(self.device)
        
        # Analyze patches
        with torch.no_grad():
            attention_scores, capacity_levels = self.analyzer(patches_tensor)
        
        # Create capacity map
        capacity_map = np.zeros((h, w))
        attention_map = np.zeros((h, w))
        
        for i, (x, y) in enumerate(positions):
            capacity_idx = torch.argmax(capacity_levels[i]).item()
            capacity_mult = self.capacity_multipliers[capacity_idx]
            attention_val = attention_scores[i].item()
            
            capacity_map[y:y+self.patch_size, x:x+self.patch_size] = capacity_mult * attention_val
            attention_map[y:y+self.patch_size, x:x+self.patch_size] = attention_val
        
        return {
            'capacity_map': capacity_map,
            'attention_map': attention_map,
            'total_capacity': int(np.sum(capacity_map))
        }
    
    def adaptive_embed(self, img: Image.Image, data: bytes, scheme: str, cfg, seed: bytes, logger):
        """Embed data adaptively based on region analysis"""
        analysis = self.analyze_image(img)
        capacity_map = analysis['capacity_map']
        
        logger.info(f"[ADAPTIVE] Total capacity: {analysis['total_capacity']} bits")
        
        # Convert image to numpy
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        
        # Create priority queue of regions sorted by capacity
        regions = []
        for y in range(0, h, 16):
            for x in range(0, w, 16):
                region_capacity = np.mean(capacity_map[y:min(y+16, h), x:min(x+16, w)])
                regions.append((region_capacity, x, y))
        
        regions.sort(reverse=True)
        
        # Embed data in high-capacity regions first
        from .utils import bytes_to_bits
        bits = bytes_to_bits(data)
        bit_idx = 0
        
        modified_img = img_np.copy()
        
        for capacity, x, y in regions:
            if bit_idx >= len(bits):
                break
            
            # Calculate bits to embed in this region
            region_bits = int(capacity * 256)  # 256 bits base capacity per region
            end_idx = min(bit_idx + region_bits, len(bits))
            
            # Embed bits in this region
            region_data = bits[bit_idx:end_idx]
            if len(region_data) > 0:
                self._embed_in_region(modified_img, x, y, 16, 16, region_data, scheme)
                bit_idx = end_idx
        
        if bit_idx < len(bits):
            logger.warning(f"[ADAPTIVE] Only embedded {bit_idx}/{len(bits)} bits")
        
        return Image.fromarray(modified_img)
    
    def _embed_in_region(self, img_np: np.ndarray, x: int, y: int, w: int, h: int, bits: List[int], scheme: str):
        """Embed bits in a specific region"""
        # Simple LSB embedding in region (can be replaced with other schemes)
        bit_idx = 0
        for dy in range(h):
            for dx in range(w):
                if bit_idx >= len(bits):
                    return
                px, py = x + dx, y + dy
                if px < img_np.shape[1] and py < img_np.shape[0]:
                    # Embed in blue channel
                    img_np[py, px, 2] = (img_np[py, px, 2] & 0xFE) | (bits[bit_idx] & 1)
                    bit_idx += 1
    
    def save_model(self, path: str):
        torch.save(self.analyzer.state_dict(), path)
    
    def load_model(self, path: str):
        self.analyzer.load_state_dict(torch.load(path, map_location=self.device))
