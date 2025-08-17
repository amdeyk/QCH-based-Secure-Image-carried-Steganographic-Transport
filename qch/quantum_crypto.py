import os
from typing import Tuple, Optional
from dataclasses import dataclass

try:
    import oqs
    HAVE_OQS = True
except ImportError:
    HAVE_OQS = False
    
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

@dataclass
class QuantumConfig:
    kem_algorithm: str = "Kyber1024"  # NIST Level 5 security
    sig_algorithm: str = "Dilithium5"  # NIST Level 5 security
    hybrid_mode: bool = True  # Use hybrid classical-quantum

class QuantumCrypto:
    """Quantum-resistant cryptography implementation"""
    
    def __init__(self, config: Optional[QuantumConfig] = None):
        self.config = config or QuantumConfig()
        
        if not HAVE_OQS:
            raise ImportError("liboqs-python required for quantum crypto. Install with: pip install liboqs-python")
        
        self.kem = oqs.KeyEncapsulation(self.config.kem_algorithm)
        self.sig = oqs.Signature(self.config.sig_algorithm)
    
    def generate_keypair(self) -> Tuple[bytes, bytes]:
        kem_public = self.kem.generate_keypair()
        kem_secret = self.kem.export_secret_key()
        sig_public = self.sig.generate_keypair()
        sig_secret = self.sig.export_secret_key()
        public_key = kem_public + b'|' + sig_public
        secret_key = kem_secret + b'|' + sig_secret
        return public_key, secret_key
    
    def encapsulate_key(self, public_key: bytes) -> Tuple[bytes, bytes]:
        kem_public, _ = public_key.split(b'|')
        ciphertext, shared_secret = self.kem.encap_secret(kem_public)
        if self.config.hybrid_mode:
            from cryptography.hazmat.primitives.asymmetric import x25519
            private_key = x25519.X25519PrivateKey.generate()
            public_key_dh = private_key.public_key()
            classical_component = public_key_dh.public_bytes_raw()
            hkdf = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=b'hybrid-kem'
            )
            hybrid_secret = hkdf.derive(shared_secret + classical_component)
            return ciphertext + b'|' + classical_component, hybrid_secret
        return ciphertext, shared_secret
    
    def decapsulate_key(self, ciphertext: bytes, secret_key: bytes) -> bytes:
        kem_secret, _ = secret_key.split(b'|')
        if self.config.hybrid_mode:
            kem_ciphertext, classical_component = ciphertext.split(b'|')
            shared_secret = self.kem.decap_secret(kem_ciphertext)
            hkdf = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=b'hybrid-kem'
            )
            hybrid_secret = hkdf.derive(shared_secret + classical_component)
            return hybrid_secret
        return self.kem.decap_secret(ciphertext)
    
    def sign(self, message: bytes, secret_key: bytes) -> bytes:
        _, sig_secret = secret_key.split(b'|')
        self.sig.import_secret_key(sig_secret)
        signature = self.sig.sign(message)
        if self.config.hybrid_mode:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
            classical_key = Ed25519PrivateKey.generate()
            classical_sig = classical_key.sign(message)
            return signature + b'|' + classical_sig
        return signature
    
    def verify(self, message: bytes, signature: bytes, public_key: bytes) -> bool:
        _, sig_public = public_key.split(b'|')
        if self.config.hybrid_mode:
            quantum_sig, classical_sig = signature.split(b'|')
            quantum_valid = self.sig.verify(message, quantum_sig, sig_public)
            return quantum_valid
        return self.sig.verify(message, signature, sig_public)

def quantum_encrypt(plaintext: bytes, passphrase: str, public_key: bytes) -> Tuple[bytes, dict]:
    qc = QuantumCrypto()
    ciphertext_kem, shared_secret = qc.encapsulate_key(public_key)
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=passphrase.encode(),
        info=b'quantum-encryption'
    )
    key = hkdf.derive(shared_secret)
    aes = AESGCM(key)
    nonce = os.urandom(12)
    ciphertext = aes.encrypt(nonce, plaintext, associated_data=None)
    metadata = {
        'algorithm': 'Kyber1024-AES256-GCM',
        'kem_ciphertext': ciphertext_kem.hex(),
        'nonce': nonce.hex(),
        'hybrid': True
    }
    return ciphertext, metadata

def quantum_decrypt(ciphertext: bytes, metadata: dict, passphrase: str, secret_key: bytes) -> bytes:
    qc = QuantumCrypto()
    kem_ciphertext = bytes.fromhex(metadata['kem_ciphertext'])
    shared_secret = qc.decapsulate_key(kem_ciphertext, secret_key)
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=passphrase.encode(),
        info=b'quantum-encryption'
    )
    key = hkdf.derive(shared_secret)
    aes = AESGCM(key)
    nonce = bytes.fromhex(metadata['nonce'])
    plaintext = aes.decrypt(nonce, ciphertext, associated_data=None)
    return plaintext
