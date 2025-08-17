import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
import os
from typing import List, Optional, Callable, Any
import time
import psutil
import platform

# Optional imports with fallbacks
try:
    import numba
    from numba import jit, prange, cuda
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - optional dependency
    HAVE_NUMBA = False

    def jit(*args, **kwargs):  # type: ignore
        def decorator(func):
            return func
        return decorator

try:  # pragma: no cover - optional dependency
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

try:  # pragma: no cover - optional dependency
    import cupy as cp
    HAVE_CUPY = True
except ImportError:
    HAVE_CUPY = False


class HardwareCapabilities:
    """Detect and manage hardware capabilities"""

    def __init__(self) -> None:
        self.cpu_count = mp.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        self.platform = platform.system()
        self.arch = platform.machine()

        self.has_avx = self._check_avx()
        self.has_cuda = self._check_cuda()
        self.has_mps = self._check_mps()
        self.has_opencl = self._check_opencl()

        self.max_threads = min(self.cpu_count, 16)
        self.chunk_size = self._calculate_optimal_chunk_size()

    def _check_avx(self) -> bool:
        try:
            import cpuinfo  # type: ignore
            info = cpuinfo.get_cpu_info()
            return any(flag in info.get('flags', []) for flag in ['avx', 'avx2', 'avx512f'])
        except Exception:
            return False

    def _check_cuda(self) -> bool:
        return HAVE_TORCH and torch.cuda.is_available()

    def _check_mps(self) -> bool:
        return HAVE_TORCH and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

    def _check_opencl(self) -> bool:
        try:  # pragma: no cover - optional dependency
            import pyopencl as cl  # type: ignore
            return bool(cl.get_platforms())
        except Exception:
            return False

    def _calculate_optimal_chunk_size(self) -> int:
        if self.memory_gb >= 16:
            return 64 * 1024
        if self.memory_gb >= 8:
            return 32 * 1024
        return 16 * 1024

    def get_best_device(self) -> str:
        if self.has_cuda:
            return 'cuda'
        if self.has_mps:
            return 'mps'
        return 'cpu'

    def print_capabilities(self) -> None:  # pragma: no cover - debug helper
        print("Hardware Capabilities:")
        print(f"  CPU cores: {self.cpu_count}")
        print(f"  Memory: {self.memory_gb:.1f} GB")
        print(f"  Platform: {self.platform} ({self.arch})")
        print(f"  AVX support: {self.has_avx}")
        print(f"  CUDA support: {self.has_cuda}")
        print(f"  MPS support: {self.has_mps}")
        print(f"  OpenCL support: {self.has_opencl}")
        print(f"  Max threads: {self.max_threads}")
        print(f"  Optimal chunk size: {self.chunk_size}")


class AcceleratedProcessor:
    """Hardware-accelerated processing for QCH operations"""

    def __init__(self, capabilities: Optional[HardwareCapabilities] = None) -> None:
        self.caps = capabilities or HardwareCapabilities()
        self.device = self.caps.get_best_device()
        self._setup_acceleration()

    def _setup_acceleration(self) -> None:
        if self.device == 'cuda' and HAVE_TORCH:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
        if HAVE_TORCH:
            torch.set_num_threads(self.caps.max_threads)
        np.seterr(all='ignore')

    @staticmethod
    @jit(nopython=True, parallel=True) if HAVE_NUMBA else (lambda func: func)
    def _numba_xor_parallel(data1, data2, result):  # type: ignore
        for i in prange(len(data1)):
            result[i] = data1[i] ^ data2[i]
        return result

    def accelerated_xor(self, data1: bytes, data2: bytes) -> bytes:
        if len(data1) != len(data2):
            raise ValueError("Data lengths must match")

        if self.device == 'cuda' and HAVE_CUPY:
            arr1 = cp.frombuffer(data1, dtype=cp.uint8)
            arr2 = cp.frombuffer(data2, dtype=cp.uint8)
            result = cp.bitwise_xor(arr1, arr2)
            return result.get().tobytes()
        elif HAVE_TORCH and self.device in ['cuda', 'mps']:
            t1 = torch.frombuffer(bytearray(data1), dtype=torch.uint8).to(self.device)
            t2 = torch.frombuffer(bytearray(data2), dtype=torch.uint8).to(self.device)
            result = torch.bitwise_xor(t1, t2)
            return result.cpu().numpy().tobytes()
        elif HAVE_NUMBA:
            arr1 = np.frombuffer(data1, dtype=np.uint8)
            arr2 = np.frombuffer(data2, dtype=np.uint8)
            result = np.empty_like(arr1)
            self._numba_xor_parallel(arr1, arr2, result)
            return result.tobytes()
        else:
            arr1 = np.frombuffer(data1, dtype=np.uint8)
            arr2 = np.frombuffer(data2, dtype=np.uint8)
            return np.bitwise_xor(arr1, arr2).tobytes()

    @staticmethod
    @jit(nopython=True, parallel=True) if HAVE_NUMBA else (lambda func: func)
    def _numba_embed_bits(image_data, bit_data, positions):  # type: ignore
        for i in prange(min(len(bit_data), len(positions))):
            pos = positions[i]
            if pos < len(image_data):
                image_data[pos] = (image_data[pos] & 0xFE) | (bit_data[i] & 1)
        return image_data

    def accelerated_embed(self, image_data: np.ndarray, bits: List[int], positions: List[int]) -> np.ndarray:
        image_flat = image_data.flatten()

        if self.device == 'cuda' and HAVE_CUPY:
            img_gpu = cp.asarray(image_flat)
            bits_gpu = cp.asarray(bits, dtype=cp.uint8)
            pos_gpu = cp.asarray(positions[:len(bits)], dtype=cp.int32)
            kernel = cp.ElementwiseKernel(
                'uint8 img, uint8 bit, int32 pos',
                'uint8 result',
                'if (i < pos.size() && pos < img.size()) { result = (img & 0xFE) | (bit & 1); } else { result = img; }',
                'embed_kernel',
            )
            for i, pos in enumerate(positions[:len(bits)]):
                if pos < len(img_gpu):
                    img_gpu[pos] = kernel(img_gpu[pos], bits_gpu[i], pos)
            return img_gpu.get().reshape(image_data.shape)
        elif HAVE_NUMBA:
            bits_array = np.array(bits, dtype=np.uint8)
            pos_array = np.array(positions, dtype=np.int32)
            result = self._numba_embed_bits(image_flat.copy(), bits_array, pos_array)
            return result.reshape(image_data.shape)
        else:
            image_copy = image_flat.copy()
            valid_positions = [pos for pos in positions[:len(bits)] if pos < len(image_copy)]
            valid_bits = bits[:len(valid_positions)]
            image_copy[valid_positions] = (image_copy[valid_positions] & 0xFE) | np.array(valid_bits, dtype=np.uint8)
            return image_copy.reshape(image_data.shape)

    def parallel_chunk_processing(self, data: bytes, chunk_processor: Callable, chunk_size: Optional[int] = None) -> List[Any]:
        chunk_size = chunk_size or self.caps.chunk_size
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        results: List[Any] = [None] * len(chunks)

        with ThreadPoolExecutor(max_workers=self.caps.max_threads) as executor:
            future_to_index = {executor.submit(chunk_processor, i, chunk): i for i, chunk in enumerate(chunks)}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:  # pragma: no cover - error path
                    print(f"Chunk {index} processing failed: {e}")
                    results[index] = None
        return [r for r in results if r is not None]

    def streaming_process_large_file(self, file_path: str, processor: Callable, chunk_size: Optional[int] = None) -> List[Any]:
        chunk_size = chunk_size or self.caps.chunk_size

        def chunk_reader():
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk

        results: List[Any] = []
        for idx, chunk in enumerate(chunk_reader()):
            results.append(processor(idx, chunk))
        return results


def setup_optimal_hardware() -> AcceleratedProcessor:
    """Convenience factory used by ProductionQCH"""
    return AcceleratedProcessor()


def benchmark_hardware_performance() -> dict:
    caps = HardwareCapabilities()
    processor = AcceleratedProcessor(caps)

    data1 = os.urandom(1024 * 1024)
    data2 = os.urandom(1024 * 1024)

    start = time.time()
    processor.accelerated_xor(data1, data2)
    duration = time.time() - start

    return {
        'cpu_cores': caps.cpu_count,
        'memory_gb': caps.memory_gb,
        'xor_time_s': duration,
    }
