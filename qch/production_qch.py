import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import time
import psutil
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class ProductionQCH:
    """Production-ready QCH with all SOTA features integrated"""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.hardware = None
        self.compressor = None
        self.quantum_crypto = None
        self.adaptive_embedder = None
        self._initialize_components()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        default_config = {
            'hardware': {
                'enable_gpu': True,
                'max_memory_gb': 8,
                'max_threads': None,
                'chunk_size': None,
            },
            'neural_compression': {
                'enabled': True,
                'model_path': None,
                'fallback_classical': True,
                'quantize_on_cpu': True,
            },
            'adaptive_embedding': {
                'enabled': True,
                'use_neural': None,
                'model_path': None,
                'patch_size': 32,
            },
            'quantum_crypto': {
                'enabled': False,
                'hybrid_mode': True,
                'fallback_classical': True,
            },
            'fountain_codes': {
                'enabled': False,
                'redundancy_factor': 1.5,
                'symbol_size': 1024,
            },
            'performance': {
                'target_fps': 1.0,
                'quality_levels': 5,
                'auto_adjust_quality': True,
            },
            'logging': {
                'level': 'INFO',
                'file': None,
                'console': True,
            },
        }
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            def merge(base, update):
                for k, v in update.items():
                    if isinstance(v, dict) and k in base:
                        merge(base[k], v)
                    else:
                        base[k] = v
            merge(default_config, user_config)
        return default_config

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('ProductionQCH')
        logger.setLevel(getattr(logging, self.config['logging']['level']))
        logger.handlers.clear()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        if self.config['logging']['console']:
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        if self.config['logging']['file']:
            fh = logging.FileHandler(self.config['logging']['file'])
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        return logger

    def _initialize_components(self) -> None:
        self.logger.info("Initializing Production QCH components...")
        try:
            from .hardware_accelerated_processing import setup_optimal_hardware
            self.hardware = setup_optimal_hardware()
            self.logger.info("Hardware acceleration initialized")
        except Exception as e:
            self.logger.warning(f"Hardware acceleration failed: {e}")
            self.hardware = None
        if self.config['neural_compression']['enabled']:
            try:
                from .optimized_neural_compression import create_lightweight_compressor
                self.compressor = create_lightweight_compressor(
                    model_path=self.config['neural_compression']['model_path'],
                    enable_gpu=self.config['hardware']['enable_gpu'],
                )
                self.logger.info("Neural compression initialized")
            except Exception as e:
                self.logger.warning(f"Neural compression failed: {e}")
                self.compressor = None
        if self.config['adaptive_embedding']['enabled']:
            try:
                from .adaptive_embedding_optimized import create_adaptive_embedder
                use_neural = self.config['adaptive_embedding']['use_neural']
                self.adaptive_embedder = create_adaptive_embedder(
                    use_neural=use_neural,
                    device='auto' if self.config['hardware']['enable_gpu'] else 'cpu',
                )
                model_path = self.config['adaptive_embedding']['model_path']
                if model_path and os.path.exists(model_path):
                    self.adaptive_embedder.load_model(model_path)
                self.logger.info("Adaptive embedding initialized")
            except Exception as e:
                self.logger.warning(f"Adaptive embedding failed: {e}")
                self.adaptive_embedder = None
        if self.config['quantum_crypto']['enabled']:
            try:
                from .quantum_crypto_integration import HybridQuantumCrypto, QuantumConfig
                qconfig = QuantumConfig(
                    hybrid_mode=self.config['quantum_crypto']['hybrid_mode'],
                    fallback_classical=self.config['quantum_crypto']['fallback_classical'],
                )
                self.quantum_crypto = HybridQuantumCrypto(qconfig)
                self.logger.info("Quantum-resistant crypto initialized")
            except Exception as e:
                self.logger.warning(f"Quantum crypto failed: {e}")
                self.quantum_crypto = None

    # send and receive methods as in snippet
    def process_send(self, inputs: List[str], passphrase: str, privkey_path: str, output_dir: str, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        try:
            from .compression import pack_files
            from .config import QCHConfig
            from .crypto import read_private_key, encrypt_and_sign
            from .logger import setup_logger
            logger, xid = setup_logger(verbose=kwargs.get('verbose', False))
            cfg = QCHConfig(
                width=kwargs.get('width', 1920),
                height=kwargs.get('height', 1080),
                bpc=kwargs.get('bpc', 2),
                overlap_ratio=kwargs.get('overlap', 0.10),
                timestep_sec=kwargs.get('timestep', 30),
            )
            logger.info(f"[PRODUCTION-SEND] Starting with xid={xid}")
            raw = pack_files(inputs, logger)
            logger.info(f"[PRODUCTION-SEND] Packed {len(inputs)} inputs -> {len(raw)} bytes")
            compress_method = kwargs.get('compress', 'lzma')
            if compress_method == 'neural' and self.compressor:
                logger.info("[PRODUCTION-SEND] Using neural compression")
                ctag = 'neural'
                comp = self.compressor.compress(raw, logger)
            else:
                from .compression import compress_bytes
                ctag, comp = compress_bytes(raw, compress_method, kwargs.get('level'), logger)
            priv = read_private_key(privkey_path)
            import struct
            from .crypto import encrypt_and_sign
            ct, meta = encrypt_and_sign(
                struct.pack(">I", len(ctag)) + ctag.encode("ascii") + comp,
                passphrase,
                kwargs.get('library'),
                priv,
                cfg,
                logger,
            )
            scheme = kwargs.get('scheme', 'pvd')
            replicas = kwargs.get('replicas', 2)
            if self.config['adaptive_embedding']['enabled'] and self.adaptive_embedder:
                logger.info("[PRODUCTION-SEND] Using adaptive embedding")
                self._embed_adaptive(ct, meta, cfg, output_dir, scheme, replicas,
                                   kwargs.get('xor_parity', False),
                                   kwargs.get('cover_folder'), logger)
            else:
                logger.info("[PRODUCTION-SEND] Using standard embedding")
                self._embed_standard(ct, meta, cfg, output_dir, scheme, replicas,
                                   kwargs.get('xor_parity', False),
                                   kwargs.get('cover_folder'), logger)
            total_time = time.time() - start_time
            logger.info(f"[PRODUCTION-SEND] Completed in {total_time:.2f}s")
            return {
                'success': True,
                'processing_time': total_time,
                'output_dir': output_dir,
                'run_id': meta['run_id'],
                'compression_method': ctag,
                'embedding_method': 'adaptive' if self.adaptive_embedder else 'standard',
            }
        except Exception as e:
            self.logger.error(f"Send operation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time,
            }

    def process_recv(self, images: List[str], passphrase: str, output_dir: str, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        try:
            from .config import QCHConfig
            from .logger import setup_logger
            from .stego import extract_one
            from .crypto import verify_and_decrypt
            from .compression import decompress_bytes, unpack_files
            from .utils import b64d, sha256
            from .redundancy import xor_parity
            from PIL import Image
            import struct
            logger, xid = setup_logger(verbose=kwargs.get('verbose', False))
            cfg = QCHConfig(
                width=kwargs.get('width', 1920),
                height=kwargs.get('height', 1080),
                bpc=kwargs.get('bpc', 2),
                overlap_ratio=kwargs.get('overlap', 0.10),
                timestep_sec=kwargs.get('timestep', 30),
            )
            logger.info(f"[PRODUCTION-RECV] Starting with xid={xid}")
            logger.info(f"[PRODUCTION-RECV] Processing {len(images)} images")
            candidates = []
            for path in images:
                img = Image.open(path).convert("RGB").resize((cfg.width, cfg.height), Image.LANCZOS)
                for label in (b"A", b"B"):
                    for scheme in ("pvd", "matrix", "interp", "lsb"):
                        try:
                            img_bytes = img.tobytes()
                            seed_hint = sha256(img_bytes)[:12] + label + b"\x00" * 8
                            meta_tmp, slc_tmp = extract_one(img, scheme, cfg, seed_hint, logger)
                            nonce = b64d(meta_tmp["nonce_b64"])
                            tcode = meta_tmp["timecode"]
                            run_id = meta_tmp["run_id"].encode()
                            rep = meta_tmp.get("replica", 0)
                            seed_true = nonce + run_id + label + struct.pack(">I", tcode) + struct.pack(">I", rep)
                            meta, slc = extract_one(img, scheme, cfg, seed_true, logger)
                            candidates.append((path, label.decode(), scheme, meta, slc))
                            logger.info(f"[PRODUCTION-RECV] Extracted from {path}: {scheme}/{label.decode()}")
                            break
                        except Exception:
                            continue
                    else:
                        continue
                    break
            if not candidates:
                raise ValueError("No valid embedded data found in provided images")
            slices_by_index = {0: [], 1: []}
            meta_ref = None
            for path, label, scheme, meta, slc in candidates:
                if meta_ref is None:
                    meta_ref = meta
                if meta["run_id"] != meta_ref["run_id"]:
                    continue
                idx = meta["slice"]["index"]
                slices_by_index[idx].append(slc)
            recA = slices_by_index.get(0, [])
            recB = slices_by_index.get(1, [])
            if not recA or not recB:
                raise ValueError("Missing required slice data")
            a = recA[0]
            b = recB[0]
            ov = min(len(a), len(b), 256_000)
            join = None
            for k in range(ov, 0, -1024):
                if a[-k:] == b[:k]:
                    join = k
                    break
            ct = a + b[join:] if join else a + b
            logger.info(f"[PRODUCTION-RECV] Reconstructed {len(ct)} bytes of ciphertext")
            pt = verify_and_decrypt(ct, meta_ref, passphrase, logger)
            tag_len = struct.unpack(">I", pt[:4])[0]
            ctag = pt[4:4 + tag_len].decode("ascii")
            comp = pt[4 + tag_len:]
            if ctag == 'neural' and self.compressor:
                logger.info("[PRODUCTION-RECV] Using neural decompression")
                raw = self.compressor.decompress(comp, logger)
            else:
                raw = decompress_bytes(ctag, comp, logger)
            final_output_dir = output_dir or f"qch_out_{meta_ref['run_id']}"
            unpack_files(raw, final_output_dir, logger)
            total_time = time.time() - start_time
            logger.info(f"[PRODUCTION-RECV] Completed in {total_time:.2f}s")
            logger.info(f"[PRODUCTION-RECV] Output: {final_output_dir}")
            return {
                'success': True,
                'processing_time': total_time,
                'output_dir': final_output_dir,
                'run_id': meta_ref['run_id'],
                'images_processed': len(candidates),
                'decompression_method': ctag,
            }
        except Exception as e:
            self.logger.error(f"Receive operation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time,
            }

    # embedding helper methods as snippet
    def _embed_adaptive(self, ct, meta, cfg, output_dir, scheme, replicas, xor_parity, cover_folder, logger):
        from .blob import build_embedded_blob
        from .cover import build_mosaic_cover
        from .redundancy import xor_parity as xor_func
        from .utils import b64d
        import struct
        half = len(ct) // 2
        ov = int(half * cfg.overlap_ratio)
        sliceA = ct[: half + ov]
        sliceB = ct[half - ov:]
        groups_A = [sliceA for _ in range(replicas)]
        groups_B = [sliceB for _ in range(replicas)]
        if xor_parity:
            parityA = xor_func(groups_A)
            parityB = xor_func(groups_B)
            groups_A.append(parityA)
            groups_B.append(parityB)
        os.makedirs(output_dir, exist_ok=True)
        for r_id in range(len(groups_A)):
            coverA = build_mosaic_cover(cover_folder, cfg, logger)
            coverB = build_mosaic_cover(cover_folder, cfg, logger)
            blobA = build_embedded_blob(meta, scheme, r_id, 0, 2, groups_A[r_id])
            blobB = build_embedded_blob(meta, scheme, r_id, 1, 2, groups_B[r_id])
            seedA = b64d(meta["nonce_b64"]) + meta["run_id"].encode() + b"A" + struct.pack(">I", meta["timecode"]) + struct.pack(">I", r_id)
            seedB = b64d(meta["nonce_b64"]) + meta["run_id"].encode() + b"B" + struct.pack(">I", meta["timecode"]) + struct.pack(">I", r_id)
            imgA = self.adaptive_embedder.adaptive_embed(coverA, blobA, scheme, cfg, seedA, logger)
            imgB = self.adaptive_embedder.adaptive_embed(coverB, blobB, scheme, cfg, seedB, logger)
            tag = f"{meta['run_id']}_rep{r_id}"
            pathA = os.path.join(output_dir, f"{tag}_A.png")
            pathB = os.path.join(output_dir, f"{tag}_B.png")
            imgA.save(pathA, optimize=True)
            imgB.save(pathB, optimize=True)
            logger.info(f"[ADAPTIVE-EMBED] Saved {pathA} / {pathB}")

    def _embed_standard(self, ct, meta, cfg, output_dir, scheme, replicas, xor_parity, cover_folder, logger):
        from .blob import build_embedded_blob
        from .cover import build_mosaic_cover
        from .stego import embed_one
        from .redundancy import xor_parity as xor_func
        from .utils import b64d
        import struct
        half = len(ct) // 2
        ov = int(half * cfg.overlap_ratio)
        sliceA = ct[: half + ov]
        sliceB = ct[half - ov:]
        groups_A = [sliceA for _ in range(replicas)]
        groups_B = [sliceB for _ in range(replicas)]
        if xor_parity:
            parityA = xor_func(groups_A)
            parityB = xor_func(groups_B)
            groups_A.append(parityA)
            groups_B.append(parityB)
        os.makedirs(output_dir, exist_ok=True)
        for r_id in range(len(groups_A)):
            coverA = build_mosaic_cover(cover_folder, cfg, logger)
            coverB = build_mosaic_cover(cover_folder, cfg, logger)
            blobA = build_embedded_blob(meta, scheme, r_id, 0, 2, groups_A[r_id])
            blobB = build_embedded_blob(meta, scheme, r_id, 1, 2, groups_B[r_id])
            seedA = b64d(meta["nonce_b64"]) + meta["run_id"].encode() + b"A" + struct.pack(">I", meta["timecode"]) + struct.pack(">I", r_id)
            seedB = b64d(meta["nonce_b64"]) + meta["run_id"].encode() + b"B" + struct.pack(">I", meta["timecode"]) + struct.pack(">I", r_id)
            imgA = embed_one(coverA, blobA, scheme, seedA, cfg, logger)
            imgB = embed_one(coverB, blobB, scheme, seedB, cfg, logger)
            tag = f"{meta['run_id']}_rep{r_id}"
            pathA = os.path.join(output_dir, f"{tag}_A.png")
            pathB = os.path.join(output_dir, f"{tag}_B.png")
            imgA.save(pathA, optimize=True)
            imgB.save(pathB, optimize=True)
            logger.info(f"[STANDARD-EMBED] Saved {pathA} / {pathB}")

    def benchmark_system(self) -> Dict[str, Any]:
        self.logger.info("Starting comprehensive system benchmark...")
        results = {
            'timestamp': time.time(),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / (1024 ** 3),
                'platform': sys.platform,
            },
        }
        if self.hardware:
            try:
                from .hardware_accelerated_processing import benchmark_hardware_performance
                results['hardware'] = benchmark_hardware_performance()
            except Exception as e:
                results['hardware'] = {'error': str(e)}
        if self.compressor:
            try:
                from .optimized_neural_compression import benchmark_compression
                results['compression'] = benchmark_compression(data_size_mb=5, compressor=self.compressor)
            except Exception as e:
                results['compression'] = {'error': str(e)}
        if self.adaptive_embedder:
            try:
                from .adaptive_embedding_optimized import benchmark_adaptive_embedding
                results['adaptive_embedding'] = benchmark_adaptive_embedding()
            except Exception as e:
                results['adaptive_embedding'] = {'error': str(e)}
        if self.quantum_crypto:
            try:
                from .quantum_crypto_integration import benchmark_quantum_crypto
                results['quantum_crypto'] = benchmark_quantum_crypto()
            except Exception as e:
                results['quantum_crypto'] = {'error': str(e)}
        self.logger.info("System benchmark completed")
        return results

    def get_system_status(self) -> Dict[str, Any]:
        return {
            'components': {
                'hardware_acceleration': self.hardware is not None,
                'neural_compression': self.compressor is not None,
                'adaptive_embedding': self.adaptive_embedder is not None,
                'quantum_crypto': self.quantum_crypto is not None,
            },
            'config': self.config,
            'memory_usage': psutil.virtual_memory().percent,
            'cpu_usage': psutil.cpu_percent(),
        }


def create_production_config(output_path: str = "qch_config.json"):
    config = {
        'hardware': {
            'enable_gpu': True,
            'max_memory_gb': 8,
            'max_threads': None,
            'chunk_size': None,
        },
        'neural_compression': {
            'enabled': True,
            'model_path': None,
            'fallback_classical': True,
            'quantize_on_cpu': True,
        },
        'adaptive_embedding': {
            'enabled': True,
            'use_neural': None,
            'model_path': None,
            'patch_size': 32,
        },
        'quantum_crypto': {
            'enabled': False,
            'hybrid_mode': True,
            'fallback_classical': True,
        },
        'fountain_codes': {
            'enabled': False,
            'redundancy_factor': 1.5,
            'symbol_size': 1024,
        },
        'performance': {
            'target_fps': 1.0,
            'quality_levels': 5,
            'auto_adjust_quality': True,
        },
        'logging': {
            'level': 'INFO',
            'file': None,
            'console': True,
        },
    }
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {output_path}")
    return config


def main():
    parser = argparse.ArgumentParser(description="Production QCH with SOTA features")
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--benchmark', action='store_true', help='Run system benchmark')
    parser.add_argument('--status', action='store_true', help='Show system status')
    parser.add_argument('--create-config', help='Create configuration file')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    send_parser = subparsers.add_parser('send', help='Send (embed) operation')
    send_parser.add_argument('--inputs', nargs='+', required=True)
    send_parser.add_argument('--passphrase', required=True)
    send_parser.add_argument('--privkey', required=True)
    send_parser.add_argument('--output-dir', default='qch_output')
    send_parser.add_argument('--scheme', choices=['pvd', 'matrix', 'interp', 'lsb'], default='pvd')
    send_parser.add_argument('--compress', choices=['lzma', 'zstd', 'neural'], default='lzma')
    send_parser.add_argument('--replicas', type=int, default=2)
    send_parser.add_argument('--xor-parity', action='store_true')
    send_parser.add_argument('--cover-folder')
    send_parser.add_argument('--verbose', action='store_true')
    recv_parser = subparsers.add_parser('recv', help='Receive (extract) operation')
    recv_parser.add_argument('--images', nargs='+', required=True)
    recv_parser.add_argument('--passphrase', required=True)
    recv_parser.add_argument('--output-dir')
    recv_parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    if args.create_config:
        create_production_config(args.create_config)
        return
    qch = ProductionQCH(args.config)
    if args.benchmark:
        results = qch.benchmark_system()
        print(json.dumps(results, indent=2))
        return
    if args.status:
        status = qch.get_system_status()
        print(json.dumps(status, indent=2))
        return
    if args.command == 'send':
        result = qch.process_send(
            inputs=args.inputs,
            passphrase=args.passphrase,
            privkey_path=args.privkey,
            output_dir=args.output_dir,
            scheme=args.scheme,
            compress=args.compress,
            replicas=args.replicas,
            xor_parity=args.xor_parity,
            cover_folder=args.cover_folder,
            verbose=args.verbose,
        )
        print(json.dumps(result, indent=2))
    elif args.command == 'recv':
        result = qch.process_recv(
            images=args.images,
            passphrase=args.passphrase,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
