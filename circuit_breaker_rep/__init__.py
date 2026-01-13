"""
Representation-Based Circuit Breakers

This module implements circuit breakers using representation engineering,
based on the GraySwanAI approach (arXiv:2406.04313).

Unlike the traditional circuit breaker pattern, this approach:
- Modifies internal model representations during training
- Intervenes at the activation level (middle layers)
- Uses LoRA adapters to make harmful representations orthogonal
- Has zero runtime overhead after training
"""

from circuit_breaker_rep.config import CircuitBreakerConfig
from circuit_breaker_rep.model import CircuitBreakerModel
from circuit_breaker_rep.trainer import CircuitBreakerTrainer
from circuit_breaker_rep.dataset import CircuitBreakerDataset, prepare_datasets

__all__ = [
    "CircuitBreakerConfig",
    "CircuitBreakerModel",
    "CircuitBreakerTrainer",
    "CircuitBreakerDataset",
    "prepare_datasets",
]

__version__ = "0.2.0"
