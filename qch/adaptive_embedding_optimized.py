import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
from typing import List, Tuple, Dict, Optional
import time
from dataclasses import dataclass

try:  # Optional imports
    from skimage.feature import local_binary_pattern
    from skimage.filters import sobel
    HAVE_SKIMAGE = True
except Exception:  # pragma: no cover - optional dependency
    HAVE_SKIMAGE = False


@dataclass
class EmbeddingConfig:
    patch_size: int = 32
    min_texture_threshold: float = 0.1
    max_texture_threshold: float = 0.9
    edge_weight: float = 0.3
    texture_weight: float = 0.4
    smoothness_weight: float = 0.3
    device: str = "auto"


class LightweightRegionAnalyzer(nn.Module):
    """Lightweight CNN for real-time region analysis on consumer hardware"""

    def __init__(self, patch_size: int = 32):
        super().__init__()
        self.patch_size = patch_size
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.capacity_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3),
        )
        self.safety_head = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        features = self.features(x)
        features_flat = features.view(features.size(0), -1)
        capacity = torch.softmax(self.capacity_head(features_flat), dim=1)
        safety = self.safety_head(features_flat)
        return capacity, safety


class FastRegionAnalyzer:
    def __init__(self, config: EmbeddingConfig):
        self.config = config

    def analyze_texture(self, patch: np.ndarray) -> float:
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        kernel = np.ones((3, 3), np.float32) / 9
        mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        sqr_mean = cv2.filter2D((gray.astype(np.float32)) ** 2, -1, kernel)
        texture = np.sqrt(sqr_mean - mean ** 2)
        return np.mean(texture) / 255.0

    def analyze_edges(self, patch: np.ndarray) -> float:
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        return np.mean(edges) / 255.0

    def analyze_smoothness(self, patch: np.ndarray) -> float:
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gx ** 2 + gy ** 2)
        smoothness = 1.0 / (1.0 + np.mean(gradient_magnitude) / 255.0)
        return smoothness

    def calculate_capacity_score(self, patch: np.ndarray) -> float:
        texture = self.analyze_texture(patch)
        edges = self.analyze_edges(patch)
        smoothness = self.analyze_smoothness(patch)
        score = (
            self.config.texture_weight * texture
            + self.config.edge_weight * edges
            + self.config.smoothness_weight * smoothness
        )
        return np.clip(score, 0.0, 1.0)


class OptimizedAdaptiveEmbedder:
    def __init__(self, config: Optional[EmbeddingConfig] = None, use_neural: bool = True):
        self.config = config or EmbeddingConfig()
        self.use_neural = use_neural
        if self.config.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
                self.use_neural = False
        else:
            self.device = torch.device(self.config.device)
        if self.use_neural:
            self.neural_analyzer = LightweightRegionAnalyzer(self.config.patch_size).to(self.device)
            self.neural_analyzer.eval()
        self.fast_analyzer = FastRegionAnalyzer(self.config)
        self.capacity_levels = {
            'low': 0.3,
            'medium': 0.6,
            'high': 1.0,
            'ultra': 1.5,
        }

    def analyze_image_fast(self, img: Image.Image) -> Dict[str, np.ndarray]:
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        capacity_map = np.zeros((h, w), dtype=np.float32)
        safety_map = np.zeros((h, w), dtype=np.float32)
        patch_size = self.config.patch_size
        for y in range(0, h - patch_size + 1, patch_size // 2):
            for x in range(0, w - patch_size + 1, patch_size // 2):
                patch = img_np[y:y + patch_size, x:x + patch_size]
                if patch.shape[:2] == (patch_size, patch_size):
                    capacity_score = self.fast_analyzer.calculate_capacity_score(patch)
                    safety_score = 1.0 - capacity_score
                    capacity_map[y:y + patch_size, x:x + patch_size] = capacity_score
                    safety_map[y:y + patch_size, x:x + patch_size] = safety_score
        capacity_map = cv2.GaussianBlur(capacity_map, (5, 5), 1.0)
        safety_map = cv2.GaussianBlur(safety_map, (5, 5), 1.0)
        return {
            'capacity_map': capacity_map,
            'safety_map': safety_map,
            'total_capacity': int(np.sum(capacity_map) * 1000),
            'analysis_method': 'fast_classical',
        }

    def analyze_image_neural(self, img: Image.Image) -> Dict[str, np.ndarray]:
        if not self.use_neural:
            return self.analyze_image_fast(img)
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        capacity_map = np.zeros((h, w), dtype=np.float32)
        safety_map = np.zeros((h, w), dtype=np.float32)
        patch_size = self.config.patch_size
        patches = []
        positions = []
        for y in range(0, h - patch_size + 1, patch_size // 2):
            for x in range(0, w - patch_size + 1, patch_size // 2):
                patch = img_np[y:y + patch_size, x:x + patch_size]
                if patch.shape[:2] == (patch_size, patch_size):
                    patches.append(patch)
                    positions.append((x, y))
        if not patches:
            return self.analyze_image_fast(img)
        batch_size = 32
        with torch.no_grad():
            for i in range(0, len(patches), batch_size):
                batch_patches = patches[i:i + batch_size]
                batch_positions = positions[i:i + batch_size]
                batch_tensor = torch.stack([
                    torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
                    for patch in batch_patches
                ]).to(self.device)
                capacity_logits, safety_scores = self.neural_analyzer(batch_tensor)
                capacity_probs = torch.softmax(capacity_logits, dim=1)
                capacity_scores = (capacity_probs[:, 1] + 2 * capacity_probs[:, 2]) / 2
                for j, (x, y) in enumerate(batch_positions):
                    capacity_score = capacity_scores[j].cpu().item()
                    safety_score = safety_scores[j].cpu().item()
                    capacity_map[y:y + patch_size, x:x + patch_size] = capacity_score
                    safety_map[y:y + patch_size, x:x + patch_size] = safety_score
        return {
            'capacity_map': capacity_map,
            'safety_map': safety_map,
            'total_capacity': int(np.sum(capacity_map) * 1200),
            'analysis_method': 'neural_network',
        }

    def create_embedding_plan(self, capacity_map: np.ndarray, data_size: int) -> Dict[str, any]:
        h, w = capacity_map.shape
        regions = []
        block_size = 16
        for y in range(0, h - block_size + 1, block_size):
            for x in range(0, w - block_size + 1, block_size):
                region_capacity = np.mean(capacity_map[y:y + block_size, x:x + block_size])
                regions.append({'x': x, 'y': y, 'width': block_size, 'height': block_size,
                                'capacity': region_capacity, 'priority': region_capacity})
        regions.sort(key=lambda r: r['priority'], reverse=True)
        total_capacity = sum(r['capacity'] for r in regions)
        bits_needed = data_size * 8
        if total_capacity == 0:
            raise ValueError("No suitable regions found for embedding")
        allocated_bits = 0
        for region in regions:
            if allocated_bits >= bits_needed:
                region['bits_to_embed'] = 0
            else:
                region_bits = int((region['capacity'] / total_capacity) * bits_needed)
                region_bits = min(region_bits, bits_needed - allocated_bits)
                region['bits_to_embed'] = region_bits
                allocated_bits += region_bits
        return {
            'regions': regions,
            'total_bits': bits_needed,
            'allocated_bits': allocated_bits,
            'efficiency': allocated_bits / bits_needed if bits_needed > 0 else 0,
        }

    def adaptive_embed(self, img: Image.Image, data: bytes, scheme: str, cfg, seed: bytes, logger) -> Image.Image:
        start_time = time.time()
        if self.use_neural and self.device.type in ['cuda', 'mps']:
            analysis = self.analyze_image_neural(img)
        else:
            analysis = self.analyze_image_fast(img)
        analysis_time = time.time() - start_time
        logger.info(f"[ADAPTIVE] Image analysis completed in {analysis_time:.2f}s using {analysis['analysis_method']}")
        logger.info(f"[ADAPTIVE] Estimated capacity: {analysis['total_capacity']} bits")
        plan = self.create_embedding_plan(analysis['capacity_map'], len(data))
        logger.info(
            f"[ADAPTIVE] Embedding plan: {plan['allocated_bits']}/{plan['total_bits']} bits (efficiency: {plan['efficiency']:.1%})"
        )
        from ..utils import bytes_to_bits
        bits = bytes_to_bits(data)
        if plan['allocated_bits'] < len(bits):
            logger.warning(f"[ADAPTIVE] Insufficient capacity: need {len(bits)}, have {plan['allocated_bits']}")
            return self._fallback_embed(img, data, scheme, cfg, seed, logger)
        img_np = np.array(img)
        modified_img = img_np.copy()
        bit_idx = 0
        embedded_regions = 0
        for region in plan['regions']:
            if bit_idx >= len(bits) or region['bits_to_embed'] == 0:
                break
            end_idx = min(bit_idx + region['bits_to_embed'], len(bits))
            region_bits = bits[bit_idx:end_idx]
            if len(region_bits) > 0:
                success = self._embed_in_region(
                    modified_img, region['x'], region['y'], region['width'], region['height'], region_bits, scheme
                )
                if success:
                    bit_idx = end_idx
                    embedded_regions += 1
                else:
                    logger.warning(
                        f"[ADAPTIVE] Failed to embed in region at ({region['x']}, {region['y']})"
                    )
        total_time = time.time() - start_time
        logger.info(
            f"[ADAPTIVE] Embedded {bit_idx}/{len(bits)} bits in {embedded_regions} regions (total time: {total_time:.2f}s)"
        )
        return Image.fromarray(modified_img)

    def _embed_in_region(self, img_np: np.ndarray, x: int, y: int, width: int, height: int, bits: List[int], scheme: str) -> bool:
        try:
            bit_idx = 0
            for dy in range(height):
                for dx in range(width):
                    if bit_idx >= len(bits):
                        return True
                    px, py = x + dx, y + dy
                    if 0 <= px < img_np.shape[1] and 0 <= py < img_np.shape[0]:
                        img_np[py, px, 2] = (img_np[py, px, 2] & 0xFE) | (bits[bit_idx] & 1)
                        bit_idx += 1
            return True
        except Exception:
            return False

    def _fallback_embed(self, img: Image.Image, data: bytes, scheme: str, cfg, seed: bytes, logger) -> Image.Image:
        logger.info("[ADAPTIVE] Falling back to standard embedding")
        from ..stego import embed_one
        from ..utils import bytes_to_bits
        bits = bytes_to_bits(data)
        if scheme == "lsb":
            from ..schemes.lsb import embed_lsb_scatter
            return embed_lsb_scatter(img, bits, cfg, seed, logger)
        elif scheme == "pvd":
            from ..schemes.pvd import pvd_embed
            return pvd_embed(img, bits, cfg, seed, logger)
        elif scheme == "matrix":
            from ..schemes.matrix import matrix_embed
            return matrix_embed(img, bits, cfg, seed, logger)
        elif scheme == "interp":
            from ..schemes.interp import interp_embed
            return interp_embed(img, bits, cfg, seed, logger)
        else:
            raise ValueError(f"Unknown embedding scheme: {scheme}")

    def save_model(self, path: str) -> None:
        if self.use_neural and hasattr(self, 'neural_analyzer'):
            torch.save({
                'model_state_dict': self.neural_analyzer.state_dict(),
                'config': self.config,
                'device': str(self.device),
            }, path)

    def load_model(self, path: str) -> None:
        if not self.use_neural:
            return
        checkpoint = torch.load(path, map_location=self.device)
        if not hasattr(self, 'neural_analyzer'):
            self.neural_analyzer = LightweightRegionAnalyzer(self.config.patch_size).to(self.device)
        self.neural_analyzer.load_state_dict(checkpoint['model_state_dict'])
        self.neural_analyzer.eval()


def create_adaptive_embedder(use_neural: Optional[bool] = None, device: str = "auto") -> OptimizedAdaptiveEmbedder:
    if use_neural is None:
        use_neural = torch.cuda.is_available() or (
            hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        )
    config = EmbeddingConfig(device=device)
    return OptimizedAdaptiveEmbedder(config, use_neural=use_neural)


def benchmark_adaptive_embedding():
    import time
    test_img = Image.new('RGB', (512, 512))
    test_img_np = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    test_img = Image.fromarray(test_img_np)
    test_data = b"Hello, this is test data for adaptive embedding!" * 10
    embedder_neural = create_adaptive_embedder(use_neural=True)
    embedder_fast = create_adaptive_embedder(use_neural=False)
    results = {}
    if embedder_neural.use_neural:
        start_time = time.time()
        analysis_neural = embedder_neural.analyze_image_neural(test_img)
        neural_time = time.time() - start_time
        results['neural'] = {
            'time': neural_time,
            'capacity': analysis_neural['total_capacity'],
            'method': analysis_neural['analysis_method'],
        }
    start_time = time.time()
    analysis_fast = embedder_fast.analyze_image_fast(test_img)
    fast_time = time.time() - start_time
    results['fast'] = {
        'time': fast_time,
        'capacity': analysis_fast['total_capacity'],
        'method': analysis_fast['analysis_method'],
    }
    print("Adaptive Embedding Benchmark:")
    for method, result in results.items():
        print(f"  {method.capitalize()} method:")
        print(f"    Analysis time: {result['time']*1000:.1f} ms")
        print(f"    Estimated capacity: {result['capacity']} bits")
        print(f"    Analysis method: {result['method']}")
    return results
