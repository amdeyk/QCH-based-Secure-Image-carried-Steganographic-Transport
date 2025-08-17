# Multi-stage build for optimized image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as builder

# Install Python and build dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install liboqs
RUN git clone https://github.com/open-quantum-safe/liboqs.git && \
    cd liboqs && \
    mkdir build && cd build && \
    cmake -DBUILD_SHARED_LIBS=ON .. && \
    make -j$(nproc) && \
    make install

WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install additional optimizations
RUN pip3 install --no-cache-dir \
    onnxruntime-gpu \
    tensorrt \
    flash-attn

# Production stage
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy from builder
COPY --from=builder /usr/local /usr/local
COPY --from=builder /usr/lib/python3.10/dist-packages /usr/lib/python3.10/dist-packages

WORKDIR /app

# Copy application
COPY qch/ ./qch/
COPY *.py ./

# Set environment variables for optimization
ENV CUDA_VISIBLE_DEVICES=0
ENV OMP_NUM_THREADS=8
ENV MKL_NUM_THREADS=8
ENV NUMBA_NUM_THREADS=8

ENTRYPOINT ["python3", "-m", "qch.cli"]
