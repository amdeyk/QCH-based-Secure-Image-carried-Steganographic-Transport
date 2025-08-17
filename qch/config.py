from dataclasses import dataclass


@dataclass
class QCHConfig:
    width: int = 1920
    height: int = 1080
    channels: int = 3
    bpc: int = 2
    overlap_ratio: float = 0.10
    timestep_sec: int = 30
