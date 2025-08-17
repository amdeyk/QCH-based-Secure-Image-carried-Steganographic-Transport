import time
import os
import argparse

from qch.neural_compression import NeuralCompressor
from qch.compression import compress_bytes

class SimpleLogger:
    def info(self, msg):
        print(msg)


def benchmark_compression(size_mb: int, neural_model: str = None):
    data = os.urandom(size_mb * 1024 * 1024)
    logger = SimpleLogger()
    start = time.time()
    ctag, comp = compress_bytes(data, 'lzma', None, logger)
    lzma_time = time.time() - start
    print(f"LZMA compressed to {len(comp)} bytes in {lzma_time:.3f}s")
    if neural_model:
        compressor = NeuralCompressor(model_path=neural_model)
        start = time.time()
        comp_n = compressor.compress(data, logger)
        n_time = time.time() - start
        print(f"Neural compressed to {len(comp_n)} bytes in {n_time:.3f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=10, help="Data size in MB")
    parser.add_argument("--neural-model", help="Path to neural model", default=None)
    args = parser.parse_args()
    benchmark_compression(args.size, args.neural_model)

if __name__ == "__main__":
    main()
