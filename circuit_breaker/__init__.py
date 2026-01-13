"""
Circuit Breaker for Jailbreak Prevention

A circuit breaker implementation designed to detect and prevent jailbreak attacks
in Large Language Models (LLMs).
"""

from .breaker import CircuitBreaker, CircuitState, CircuitBreakerConfig, CircuitBreakerOpenError
from .detector import JailbreakDetector, DetectionResult

__version__ = "0.1.0"
__all__ = [
    "CircuitBreaker",
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitBreakerOpenError",
    "JailbreakDetector",
    "DetectionResult",
]

