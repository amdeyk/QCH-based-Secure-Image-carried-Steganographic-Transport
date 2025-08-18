import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
from typing import List, Tuple, Dict, Optional
import math
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
                capacity_probs, safety_scores = self.neural_analyzer(batch_tensor)
                capacity_scores = (capacity_probs[:, 1] + 2 * capacity_probs[:, 2]) / 3
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

    def create_embedding_plan(
        self,
        capacity_map: np.ndarray,
        safety_map: np.ndarray,
        data_size: int,
        scheme: str,
    ) -> Dict[str, any]:
        h, w = capacity_map.shape
        region_sizes = {
            'lsb': 16,
            'pvd': 20,
            'matrix': 24,
            'interp': 18,
        }
        region_size = region_sizes.get(scheme, 16)
        regions = []
        for y in range(0, h - region_size + 1, region_size):
            for x in range(0, w - region_size + 1, region_size):
                region_capacity = np.mean(capacity_map[y:y + region_size, x:x + region_size])
                region_safety = np.mean(safety_map[y:y + region_size, x:x + region_size])
                if scheme == 'lsb':
                    bpp = min(3, max(1, int(region_capacity * 3)))
                elif scheme == 'pvd':
                    bpp = max(1, int(region_capacity * 4))
                elif scheme == 'matrix':
                    bpp = 3 if region_capacity > 0.5 else 2
                else:
                    bpp = max(1, int(region_capacity * 2))
                pixels = region_size * region_size
                if scheme == 'matrix':
                    region_bits = (pixels // 7) * 3
                else:
                    region_bits = pixels * bpp
                region_bits = int(region_bits * region_safety)
                regions.append(
                    {
                        'x': x,
                        'y': y,
                        'width': region_size,
                        'height': region_size,
                        'capacity_score': region_capacity,
                        'safety_score': region_safety,
                        'estimated_bits': region_bits,
                        'bpp': bpp,
                        'priority': region_capacity * region_safety,
                    }
                )
        regions.sort(key=lambda r: r['priority'], reverse=True)
        bits_needed = data_size * 8
        allocated_bits = 0
        for region in regions:
            if allocated_bits >= bits_needed:
                region['bits_to_embed'] = 0
            else:
                available_bits = min(region['estimated_bits'], bits_needed - allocated_bits)
                region['bits_to_embed'] = available_bits
                allocated_bits += available_bits
        return {
            'regions': regions,
            'total_bits_needed': bits_needed,
            'total_bits_allocated': allocated_bits,
            'efficiency': allocated_bits / bits_needed if bits_needed > 0 else 0,
            'estimated_capacity': sum(r['estimated_bits'] for r in regions),
        }

    def adaptive_embed(self, img: Image.Image, data: bytes, scheme: str, cfg, seed: bytes, logger) -> Image.Image:
        start_time = time.time()
        if self.use_neural and self.device.type in ['cuda', 'mps']:
            analysis = self.analyze_image_neural(img)
        else:
            analysis = self.analyze_image_fast(img)
        analysis_time = time.time() - start_time
        logger.info(
            f"[ADAPTIVE] Image analysis completed in {analysis_time:.2f}s using {analysis['analysis_method']}"
        )
        logger.info(f"[ADAPTIVE] Estimated capacity: {analysis['total_capacity']} bits")
        plan = self.create_embedding_plan(
            analysis['capacity_map'], analysis['safety_map'], len(data), scheme
        )
        logger.info(
            f"[ADAPTIVE] Plan: {plan['total_bits_allocated']}/{plan['total_bits_needed']} bits (efficiency: {plan['efficiency']:.1%})"
        )
        if plan['efficiency'] < 0.8:
            logger.warning("[ADAPTIVE] Insufficient capacity, falling back to standard embedding")
            return self._fallback_embed(img, data, scheme, cfg, seed, logger)
        from .utils import bytes_to_bits
        bits = bytes_to_bits(data)
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
                success = self._embed_in_region_adaptive(
                    modified_img,
                    region['x'],
                    region['y'],
                    region['width'],
                    region['height'],
                    region_bits,
                    scheme,
                    region['bpp'],
                    seed,
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

    def _embed_in_region_adaptive(
        self,
        img_np: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        bits: List[int],
        scheme: str,
        bpp: int,
        seed: bytes,
    ) -> bool:
        try:
            if scheme == 'lsb':
                return self._embed_lsb_region(img_np, x, y, width, height, bits, bpp, seed)
            elif scheme == 'pvd':
                return self._embed_pvd_region(img_np, x, y, width, height, bits, seed)
            elif scheme == 'matrix':
                return self._embed_matrix_region(img_np, x, y, width, height, bits, seed)
            elif scheme == 'interp':
                return self._embed_interp_region(img_np, x, y, width, height, bits, seed)
            else:
                return self._embed_lsb_region(img_np, x, y, width, height, bits, 1, seed)
        except Exception:
            return False

    def _embed_lsb_region(
        self,
        img_np: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        bits: List[int],
        bpp: int,
        seed: bytes,
    ) -> bool:
        from .utils import prng
        positions = []
        for dy in range(height):
            for dx in range(width):
                px, py = x + dx, y + dy
                if 0 <= px < img_np.shape[1] and 0 <= py < img_np.shape[0]:
                    positions.append((py, px, 2))
        r = prng(seed + f"{x}_{y}".encode())
        r.shuffle(positions)
        bit_idx = 0
        for py, px, channel in positions:
            for bit_pos in range(bpp):
                if bit_idx >= len(bits):
                    return True
                mask = ~(1 << bit_pos) & 0xFF
                img_np[py, px, channel] = (img_np[py, px, channel] & mask) | ((bits[bit_idx] & 1) << bit_pos)
                bit_idx += 1
        return True

    def _embed_pvd_region(
        self,
        img_np: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        bits: List[int],
        seed: bytes,
    ) -> bool:
        from .schemes.pvd import PVD_RANGES
        from .utils import prng
        pairs = []
        for dy in range(height):
            for dx in range(0, width - 1, 2):
                px1, py1 = x + dx, y + dy
                px2, py2 = x + dx + 1, y + dy
                if 0 <= px2 < img_np.shape[1] and 0 <= py2 < img_np.shape[0]:
                    pairs.append(((py1, px1, 2), (py2, px2, 2)))
        r = prng(seed + f"{x}_{y}".encode())
        r.shuffle(pairs)
        bit_idx = 0
        for (py1, px1, c1), (py2, px2, c2) in pairs:
            if bit_idx >= len(bits):
                break
            p1 = img_np[py1, px1, c1]
            p2 = img_np[py2, px2, c2]
            d = abs(int(p1) - int(p2))
            for a, b in PVD_RANGES:
                if a <= d <= b:
                    w = b - a + 1
                    t = int(math.floor(math.log2(w))) if w > 1 else 0
                    if t <= 0:
                        break
                    val = 0
                    for _ in range(min(t, len(bits) - bit_idx)):
                        val = (val << 1) | (bits[bit_idx] & 1)
                        bit_idx += 1
                    new_d = a + val
                    if new_d > b:
                        new_d = b
                    if p1 >= p2:
                        delta = new_d - d
                        p1_new = max(0, min(255, p1 + delta // 2))
                        p2_new = max(0, min(255, p2 - delta // 2))
                    else:
                        delta = new_d - d
                        p2_new = max(0, min(255, p2 + delta // 2))
                        p1_new = max(0, min(255, p1 - delta // 2))
                    img_np[py1, px1, c1] = p1_new
                    img_np[py2, px2, c2] = p2_new
                    break
        return True

    def _embed_matrix_region(
        self,
        img_np: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        bits: List[int],
        seed: bytes,
    ) -> bool:
        from .schemes.matrix import hamming_syndrome
        from .utils import prng
        positions = []
        for dy in range(height):
            for dx in range(width):
                px, py = x + dx, y + dy
                if 0 <= px < img_np.shape[1] and 0 <= py < img_np.shape[0]:
                    positions.append((py, px, 2))
        r = prng(seed + f"{x}_{y}".encode())
        r.shuffle(positions)
        groups = [positions[i:i + 7] for i in range(0, len(positions), 7)]
        bit_idx = 0
        for group in groups:
            if len(group) < 7 or bit_idx >= len(bits):
                break
            current_bits = [img_np[py, px, c] & 1 for py, px, c in group]
            m0 = bits[bit_idx] if bit_idx < len(bits) else 0
            m1 = bits[bit_idx + 1] if bit_idx + 1 < len(bits) else 0
            m2 = bits[bit_idx + 2] if bit_idx + 2 < len(bits) else 0
            bit_idx += 3
            syndrome = hamming_syndrome(current_bits)
            e2 = m0 ^ syndrome[0]
            e1 = m1 ^ syndrome[1]
            e0 = m2 ^ syndrome[2]
            error_pos = (e2 << 2) | (e1 << 1) | e0
            if error_pos != 0 and error_pos - 1 < len(group):
                py, px, c = group[error_pos - 1]
                img_np[py, px, c] ^= 1
        return True

    def _embed_interp_region(
        self,
        img_np: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        bits: List[int],
        seed: bytes,
    ) -> bool:
        from .utils import prng
        positions = []
        for dy in range(height):
            for dx in range(width):
                px, py = x + dx, y + dy
                if 0 <= px < img_np.shape[1] and 0 <= py < img_np.shape[0] and ((px & 1) or (py & 1)):
                    positions.append((py, px, 2))
        r = prng(seed + f"{x}_{y}".encode())
        r.shuffle(positions)
        for i, (py, px, c) in enumerate(positions):
            if i >= len(bits):
                break
            img_np[py, px, c] = (img_np[py, px, c] & 0xFE) | (bits[i] & 1)
        return True

    def _fallback_embed(self, img: Image.Image, data: bytes, scheme: str, cfg, seed: bytes, logger) -> Image.Image:
        logger.info("[ADAPTIVE] Falling back to standard embedding")
        from .stego import embed_one
        from .utils import bytes_to_bits
        bits = bytes_to_bits(data)
        if scheme == "lsb":
            from .schemes.lsb import embed_lsb_scatter
            return embed_lsb_scatter(img, bits, cfg, seed, logger)
        elif scheme == "pvd":
            from .schemes.pvd import pvd_embed
            return pvd_embed(img, bits, cfg, seed, logger)
        elif scheme == "matrix":
            from .schemes.matrix import matrix_embed
            return matrix_embed(img, bits, cfg, seed, logger)
        elif scheme == "interp":
            from .schemes.interp import interp_embed
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
