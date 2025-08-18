import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
import os
import warnings
from typing import List, Optional, Callable, Any, Dict, Tuple
import time
import platform
from dataclasses import dataclass

# Optional imports with comprehensive fallbacks
try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False
    
try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
    torch = None

try:
    import cupy as cp
    HAVE_CUPY = True
except ImportError:
    HAVE_CUPY = False
    cp = None

try:
    import numba
    from numba import jit, prange, cuda
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
    # Create dummy decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def prange(*args, **kwargs):
        return range(*args, **kwargs)


@dataclass
class HardwareCapabilities:
    """Hardware capability detection with safe fallbacks"""
    cpu_count: int
    memory_gb: float
    platform: str
    arch: str
    has_cuda: bool
    has_mps: bool
    has_avx: bool
    max_threads: int
    optimal_chunk_size: int

    @classmethod
    def detect(cls) -> 'HardwareCapabilities':
        """Detect hardware capabilities safely"""
        # Basic CPU info
        cpu_count = mp.cpu_count()
        
        # Memory detection with fallback
        if HAVE_PSUTIL:
            try:
                memory_gb = psutil.virtual_memory().total / (1024 ** 3)
            except Exception:
                memory_gb = 8.0  # Conservative fallback
        else:
            memory_gb = 8.0  # Default assumption
        
        # Platform info
        platform_name = platform.system()
        arch = platform.machine()
        
        # GPU detection
        has_cuda = cls._check_cuda()
        has_mps = cls._check_mps()
        
        # CPU features
        has_avx = cls._check_avx()
        
        # Compute optimal settings
        max_threads = min(cpu_count, 16)  # Reasonable limit
        optimal_chunk_size = cls._calculate_chunk_size(memory_gb)
        
        return cls(
            cpu_count=cpu_count,
            memory_gb=memory_gb,
            platform=platform_name,
            arch=arch,
            has_cuda=has_cuda,
            has_mps=has_mps,
            has_avx=has_avx,
            max_threads=max_threads,
            optimal_chunk_size=optimal_chunk_size
        )

    @staticmethod
    def _check_cuda() -> bool:
        """Safely check CUDA availability"""
        if not HAVE_TORCH:
            return False
        
        try:
            if torch.cuda.is_available():
                # Actually test CUDA works
                test_tensor = torch.tensor([1.0]).cuda()
                _ = test_tensor.cpu()
                return True
        except Exception:
            pass
        
        return False

    @staticmethod
    def _check_mps() -> bool:
        """Safely check Apple MPS availability"""
        if not HAVE_TORCH:
            return False
        
        try:
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # Test MPS actually works
                test_tensor = torch.tensor([1.0]).to('mps')
                _ = test_tensor.cpu()
                return True
        except Exception:
            pass
        
        return False

    @staticmethod
    def _check_avx() -> bool:
        """Check AVX support"""
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            return any(flag in info.get('flags', []) 
                      for flag in ['avx', 'avx2', 'avx512f'])
        except Exception:
            # Assume modern CPU has AVX
            return True

    @staticmethod
    def _calculate_chunk_size(memory_gb: float) -> int:
        """Calculate optimal chunk size based on memory"""
        if memory_gb >= 32:
            return 128 * 1024
        elif memory_gb >= 16:
            return 64 * 1024
        elif memory_gb >= 8:
            return 32 * 1024
        else:
            return 16 * 1024

    def get_best_device(self) -> str:
        """Get the best available compute device"""
        if self.has_cuda:
            return 'cuda'
        elif self.has_mps:
            return 'mps'
        else:
            return 'cpu'

    def print_info(self) -> None:
        """Print hardware information"""
        print("Hardware Capabilities:")
        print(f"  CPU cores: {self.cpu_count}")
        print(f"  Memory: {self.memory_gb:.1f} GB")
        print(f"  Platform: {self.platform} ({self.arch})")
        print(f"  CUDA support: {self.has_cuda}")
        print(f"  MPS support: {self.has_mps}")
        print(f"  AVX support: {self.has_avx}")
        print(f"  Max threads: {self.max_threads}")
        print(f"  Optimal chunk size: {self.optimal_chunk_size}")


class SafeHardwareAccelerator:
    """Hardware accelerator with comprehensive error handling and fallbacks"""

    def __init__(self, capabilities: Optional[HardwareCapabilities] = None):
        self.caps = capabilities or HardwareCapabilities.detect()
        self.device = self.caps.get_best_device()
        
        # Initialize acceleration backends
        self.torch_available = HAVE_TORCH and self.device in ['cuda', 'mps']
        self.cupy_available = HAVE_CUPY and self.caps.has_cuda
        self.numba_available = HAVE_NUMBA
        
        self._setup_acceleration()
        self._test_backends()

    def _setup_acceleration(self) -> None:
        """Setup acceleration with error handling"""
        # Setup PyTorch
        if HAVE_TORCH:
            try:
                if self.device == 'cuda':
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.enabled = True
                torch.set_num_threads(self.caps.max_threads)
            except Exception as e:
                warnings.warn(f"PyTorch setup failed: {e}")
                self.torch_available = False
        
        # Setup NumPy
        try:
            os.environ['OMP_NUM_THREADS'] = str(self.caps.max_threads)
            os.environ['MKL_NUM_THREADS'] = str(self.caps.max_threads)
            np.seterr(all='ignore')
        except Exception:
            pass

    def _test_backends(self) -> None:
        """Test that backends actually work"""
        # Test PyTorch
        if self.torch_available:
            try:
                test = torch.tensor([1.0]).to(self.device)
                _ = test.cpu()
            except Exception:
                self.torch_available = False
                warnings.warn("PyTorch backend test failed, disabling")
        
        # Test CuPy
        if self.cupy_available:
            try:
                test = cp.array([1.0])
                _ = test.get()
            except Exception:
                self.cupy_available = False
                warnings.warn("CuPy backend test failed, disabling")

    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about available backends"""
        return {
            'device': self.device,
            'torch_available': self.torch_available,
            'cupy_available': self.cupy_available,
            'numba_available': self.numba_available,
            'cpu_fallback': not (self.torch_available or self.cupy_available)
        }

    def accelerated_xor(self, data1: bytes, data2: bytes) -> bytes:
        """Hardware-accelerated XOR with fallbacks"""
        if len(data1) != len(data2):
            raise ValueError("Data lengths must match for XOR operation")
        
        if len(data1) == 0:
            return b''
        
        # Try GPU acceleration first for large data
        if len(data1) > 1024 * 1024:  # Only for data > 1MB
            if self.cupy_available:
                try:
                    return self._xor_cupy(data1, data2)
                except Exception as e:
                    warnings.warn(f"CuPy XOR failed: {e}, falling back")
            
            if self.torch_available:
                try:
                    return self._xor_torch(data1, data2)
                except Exception as e:
                    warnings.warn(f"PyTorch XOR failed: {e}, falling back")
        
        # CPU fallback with optional Numba acceleration
        return self._xor_cpu(data1, data2)

    def _xor_cupy(self, data1: bytes, data2: bytes) -> bytes:
        """XOR using CuPy"""
        arr1 = cp.frombuffer(data1, dtype=cp.uint8)
        arr2 = cp.frombuffer(data2, dtype=cp.uint8)
        result = cp.bitwise_xor(arr1, arr2)
        return result.get().tobytes()

    def _xor_torch(self, data1: bytes, data2: bytes) -> bytes:
        """XOR using PyTorch"""
        # Convert to tensors
        t1 = torch.frombuffer(bytearray(data1), dtype=torch.uint8).to(self.device)
        t2 = torch.frombuffer(bytearray(data2), dtype=torch.uint8).to(self.device)
        
        # Perform XOR
        result = torch.bitwise_xor(t1, t2)
        
        # Convert back
        return result.cpu().numpy().tobytes()

    def _xor_cpu(self, data1: bytes, data2: bytes) -> bytes:
        """CPU XOR with optional Numba acceleration"""
        if self.numba_available and len(data1) > 10000:
            try:
                return self._xor_numba(data1, data2)
            except Exception:
                pass
        
        # Pure NumPy fallback
        arr1 = np.frombuffer(data1, dtype=np.uint8)
        arr2 = np.frombuffer(data2, dtype=np.uint8)
        return np.bitwise_xor(arr1, arr2).tobytes()

    @staticmethod
    @jit(nopython=True, parallel=True) if HAVE_NUMBA else (lambda func: func)
    def _xor_numba_impl(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
        """Numba-accelerated XOR implementation"""
        result = np.empty_like(arr1)
        for i in prange(len(arr1)):
            result[i] = arr1[i] ^ arr2[i]
        return result

    def _xor_numba(self, data1: bytes, data2: bytes) -> bytes:
        """XOR using Numba if available"""
        arr1 = np.frombuffer(data1, dtype=np.uint8)
        arr2 = np.frombuffer(data2, dtype=np.uint8)
        result = self._xor_numba_impl(arr1, arr2)
        return result.tobytes()

    def accelerated_embed(self, image_data: np.ndarray, bits: List[int], 
                         positions: List[int]) -> np.ndarray:
        """Hardware-accelerated embedding with fallbacks"""
        if len(bits) == 0 or len(positions) == 0:
            return image_data
        
        # Validate inputs
        max_pos = max(positions) if positions else 0
        if max_pos >= image_data.size:
            raise ValueError(f"Position {max_pos} exceeds image size {image_data.size}")
        
        # Flatten image for processing
        image_flat = image_data.flatten().copy()
        
        # Try GPU acceleration for large operations
        if len(bits) > 10000:
            if self.torch_available:
                try:
                    return self._embed_torch(image_data, bits, positions)
                except Exception as e:
                    warnings.warn(f"PyTorch embedding failed: {e}, falling back")
            
            if self.cupy_available:
                try:
                    return self._embed_cupy(image_data, bits, positions)
                except Exception as e:
                    warnings.warn(f"CuPy embedding failed: {e}, falling back")
        
        # CPU fallback
        return self._embed_cpu(image_data, bits, positions)

    def _embed_torch(self, image_data: np.ndarray, bits: List[int], 
                    positions: List[int]) -> np.ndarray:
        """Embedding using PyTorch"""
        image_flat = image_data.flatten()
        image_tensor = torch.from_numpy(image_flat.copy()).to(self.device)
        
        # Process in chunks to avoid memory issues
        chunk_size = min(10000, len(bits))
        
        for i in range(0, len(bits), chunk_size):
            end_idx = min(i + chunk_size, len(bits))
            chunk_bits = bits[i:end_idx]
            chunk_positions = positions[i:end_idx]
            
            if len(chunk_positions) > 0:
                pos_tensor = torch.tensor(chunk_positions, device=self.device)
                bits_tensor = torch.tensor(chunk_bits, device=self.device, dtype=torch.uint8)
                
                # Get current values and modify
                current_values = image_tensor[pos_tensor]
                new_values = (current_values & 0xFE) | bits_tensor
                image_tensor[pos_tensor] = new_values
        
        return image_tensor.cpu().numpy().reshape(image_data.shape)

    def _embed_cupy(self, image_data: np.ndarray, bits: List[int], 
                   positions: List[int]) -> np.ndarray:
        """Embedding using CuPy"""
        image_flat = image_data.flatten()
        image_gpu = cp.array(image_flat)
        
        # Process in chunks
        chunk_size = min(10000, len(bits))
        
        for i in range(0, len(bits), chunk_size):
            end_idx = min(i + chunk_size, len(bits))
            chunk_bits = bits[i:end_idx]
            chunk_positions = positions[i:end_idx]
            
            if len(chunk_positions) > 0:
                pos_gpu = cp.array(chunk_positions)
                bits_gpu = cp.array(chunk_bits, dtype=cp.uint8)
                
                # Embed bits
                current_values = image_gpu[pos_gpu]
                new_values = (current_values & 0xFE) | bits_gpu
                image_gpu[pos_gpu] = new_values
        
        return image_gpu.get().reshape(image_data.shape)

    def _embed_cpu(self, image_data: np.ndarray, bits: List[int], 
                  positions: List[int]) -> np.ndarray:
        """CPU embedding with optional Numba acceleration"""
        if self.numba_available and len(bits) > 1000:
            try:
                return self._embed_numba(image_data, bits, positions)
            except Exception:
                pass
        
        # Pure NumPy fallback
        image_flat = image_data.flatten().copy()
        
        # Vectorized embedding for better performance
        valid_indices = [i for i, pos in enumerate(positions[:len(bits)]) if pos < len(image_flat)]
        
        if valid_indices:
            valid_positions = np.array([positions[i] for i in valid_indices])
            valid_bits = np.array([bits[i] for i in valid_indices], dtype=np.uint8)
            
            image_flat[valid_positions] = (image_flat[valid_positions] & 0xFE) | valid_bits
        
        return image_flat.reshape(image_data.shape)

    @staticmethod
    @jit(nopython=True, parallel=True) if HAVE_NUMBA else (lambda func: func)
    def _embed_numba_impl(image_flat: np.ndarray, bits: np.ndarray, 
                         positions: np.ndarray) -> np.ndarray:
        """Numba-accelerated embedding implementation"""
        for i in prange(min(len(bits), len(positions))):
            pos = positions[i]
            if pos < len(image_flat):
                image_flat[pos] = (image_flat[pos] & 0xFE) | (bits[i] & 1)
        return image_flat

    def _embed_numba(self, image_data: np.ndarray, bits: List[int], 
                    positions: List[int]) -> np.ndarray:
        """Embedding using Numba"""
        image_flat = image_data.flatten().copy()
        bits_array = np.array(bits, dtype=np.uint8)
        positions_array = np.array(positions, dtype=np.int32)
        
        result = self._embed_numba_impl(image_flat, bits_array, positions_array)
        return result.reshape(image_data.shape)

    def parallel_process_chunks(self, data: bytes, processor: Callable[[int, bytes], Any], 
                              chunk_size: Optional[int] = None) -> List[Any]:
        """Process data in parallel chunks with error handling"""
        if len(data) == 0:
            return []
        
        chunk_size = chunk_size or self.caps.optimal_chunk_size
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        if len(chunks) == 1:
            # Single chunk, no need for threading
            try:
                return [processor(0, chunks[0])]
            except Exception as e:
                warnings.warn(f"Chunk processing failed: {e}")
                return []
        
        results = [None] * len(chunks)
        failed_chunks = []
        
        try:
            with ThreadPoolExecutor(max_workers=self.caps.max_threads) as executor:
                # Submit all tasks
                future_to_index = {
                    executor.submit(processor, i, chunk): i 
                    for i, chunk in enumerate(chunks)
                }
                
                # Collect results
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        results[index] = future.result()
                    except Exception as e:
                        warnings.warn(f"Chunk {index} processing failed: {e}")
                        failed_chunks.append(index)
                        results[index] = None
        
        except Exception as e:
            warnings.warn(f"Parallel processing failed: {e}, falling back to sequential")
            # Sequential fallback
            for i, chunk in enumerate(chunks):
                try:
                    results[i] = processor(i, chunk)
                except Exception as e:
                    warnings.warn(f"Sequential chunk {i} failed: {e}")
                    results[i] = None
        
        # Filter out failed results
        successful_results = [r for r in results if r is not None]
        
        if failed_chunks:
            warnings.warn(f"{len(failed_chunks)} chunks failed processing")
        
        return successful_results

    def streaming_file_processor(self, file_path: str, processor: Callable[[int, bytes], Any], 
                                chunk_size: Optional[int] = None) -> List[Any]:
        """Process large files in streaming fashion"""
        chunk_size = chunk_size or self.caps.optimal_chunk_size
        results = []
        
        try:
            with open(file_path, 'rb') as f:
                chunk_idx = 0
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    
                    try:
                        result = processor(chunk_idx, chunk)
                        results.append(result)
                    except Exception as e:
                        warnings.warn(f"File chunk {chunk_idx} processing failed: {e}")
                    
                    chunk_idx += 1
        
        except Exception as e:
            warnings.warn(f"File streaming failed: {e}")
        
        return results

    def memory_efficient_operation(self, data: np.ndarray, operation: Callable[[np.ndarray], np.ndarray], 
                                 max_memory_mb: int = 512) -> np.ndarray:
        """Perform operations with memory constraints"""
        data_size_mb = data.nbytes / (1024 * 1024)
        
        if data_size_mb <= max_memory_mb:
            # Data fits in memory, process normally
            try:
                return operation(data)
            except Exception as e:
                warnings.warn(f"Direct operation failed: {e}")
                return data
        
        # Process in chunks
        total_elements = data.size
        elements_per_chunk = int((max_memory_mb * 1024 * 1024) / data.itemsize)
        
        if elements_per_chunk <= 0:
            warnings.warn("Data too large for memory-efficient processing")
            return data
        
        # Flatten for processing
        original_shape = data.shape
        data_flat = data.flatten()
        result_flat = np.empty_like(data_flat)
        
        for i in range(0, total_elements, elements_per_chunk):
            end_idx = min(i + elements_per_chunk, total_elements)
            chunk = data_flat[i:end_idx]
            
            try:
                # Reshape chunk to reasonable shape for operation
                chunk_shape = (len(chunk),) if len(chunk.shape) == 1 else chunk.shape
                chunk_reshaped = chunk.reshape(chunk_shape)
                result_chunk = operation(chunk_reshaped)
                result_flat[i:end_idx] = result_chunk.flatten()[:end_idx-i]
            except Exception as e:
                warnings.warn(f"Chunk operation failed: {e}, using original data")
                result_flat[i:end_idx] = chunk
        
        return result_flat.reshape(original_shape)

    def benchmark_operations(self) -> Dict[str, float]:
        """Benchmark different acceleration methods"""
        test_size = 1024 * 1024  # 1MB test
        test_data1 = os.urandom(test_size)
        test_data2 = os.urandom(test_size)
        
        results = {}
        
        # Benchmark XOR operations
        for method in ['cpu', 'numba', 'torch', 'cupy']:
            if method == 'numba' and not self.numba_available:
                continue
            if method == 'torch' and not self.torch_available:
                continue
            if method == 'cupy' and not self.cupy_available:
                continue
            
            try:
                start_time = time.time()
                
                if method == 'cpu':
                    self._xor_cpu(test_data1, test_data2)
                elif method == 'numba':
                    self._xor_numba(test_data1, test_data2)
                elif method == 'torch':
                    self._xor_torch(test_data1, test_data2)
                elif method == 'cupy':
                    self._xor_cupy(test_data1, test_data2)
                
                duration = time.time() - start_time
                results[f'xor_{method}'] = duration
                
            except Exception as e:
                results[f'xor_{method}'] = float('inf')  # Mark as failed
        
        # Benchmark embedding
        test_image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        test_bits = [1, 0] * 1000
        test_positions = list(range(2000))
        
        try:
            start_time = time.time()
            self._embed_cpu(test_image, test_bits, test_positions)
            results['embed_cpu'] = time.time() - start_time
        except Exception:
            results['embed_cpu'] = float('inf')
        
        if self.torch_available:
            try:
                start_time = time.time()
                self._embed_torch(test_image, test_bits, test_positions)
                results['embed_torch'] = time.time() - start_time
            except Exception:
                results['embed_torch'] = float('inf')
        
        return results


def create_hardware_accelerator() -> SafeHardwareAccelerator:
    """Factory function for creating hardware accelerator"""
    return SafeHardwareAccelerator()


def benchmark_hardware_performance() -> Dict[str, Any]:
    """Comprehensive hardware benchmark"""
    print("Running hardware performance benchmark...")
    
    caps = HardwareCapabilities.detect()
    accelerator = SafeHardwareAccelerator(caps)
    
    print(f"Detected capabilities:")
    caps.print_info()
    
    print(f"\nBackend availability:")
    backend_info = accelerator.get_backend_info()
    for backend, available in backend_info.items():
        print(f"  {backend}: {available}")
    
    # Run benchmarks
    benchmark_results = accelerator.benchmark_operations()
    
    print(f"\nBenchmark results (seconds):")
    for operation, duration in benchmark_results.items():
        if duration == float('inf'):
            print(f"  {operation}: FAILED")
        else:
            print(f"  {operation}: {duration:.4f}s")
    
    # Test parallel processing
    test_data = os.urandom(1024 * 1024)  # 1MB
    
    def dummy_processor(idx: int, chunk: bytes) -> int:
        return len(chunk)
    
    start_time = time.time()
    results = accelerator.parallel_process_chunks(test_data, dummy_processor)
    parallel_time = time.time() - start_time
    
    print(f"\nParallel processing test:")
    print(f"  Processed {len(results)} chunks in {parallel_time:.4f}s")
    print(f"  Total bytes: {sum(results)}")
    
    return {
        'capabilities': caps,
        'backend_info': backend_info,
        'benchmark_results': benchmark_results,
        'parallel_test': {
            'chunks': len(results),
            'time': parallel_time,
            'total_bytes': sum(results)
        }
    }


def test_acceleration_fallbacks():
    """Test that all fallbacks work correctly"""
    print("Testing acceleration fallbacks...")
    
    accelerator = SafeHardwareAccelerator()
    
    # Test XOR with different data sizes
    for size in [100, 10000, 1000000]:
        data1 = os.urandom(size)
        data2 = os.urandom(size)
        
        try:
            result = accelerator.accelerated_xor(data1, data2)
            
            # Verify result
            expected = bytes(a ^ b for a, b in zip(data1, data2))
            assert result == expected, f"XOR failed for size {size}"
            print(f"  XOR test passed for {size} bytes")
        
        except Exception as e:
            print(f"  XOR test failed for {size} bytes: {e}")
    
    # Test embedding
    test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    test_bits = [1, 0, 1, 1, 0]
    test_positions = [10, 50, 100, 200, 500]
    
    try:
        result = accelerator.accelerated_embed(test_image, test_bits, test_positions)
        print(f"  Embedding test passed for {test_image.size} pixels")
    except Exception as e:
        print(f"  Embedding test failed: {e}")
    
    # Test parallel processing
    test_data = b"Hello, World!" * 1000
    
    def test_processor(idx: int, chunk: bytes) -> str:
        return f"Processed chunk {idx} with {len(chunk)} bytes"
    
    try:
        results = accelerator.parallel_process_chunks(test_data, test_processor)
        print(f"  Parallel processing test passed with {len(results)} results")
    except Exception as e:
        print(f"  Parallel processing test failed: {e}")
    
    print("Fallback testing completed!")


def setup_optimal_hardware() -> SafeHardwareAccelerator:
    """Convenience factory used by ProductionQCH"""
    return SafeHardwareAccelerator()


if __name__ == "__main__":
    # Run comprehensive tests
    benchmark_hardware_performance()
    print("\n" + "="*50 + "\n")
    test_acceleration_fallbacks()
