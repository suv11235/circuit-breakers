"""
Core Circuit Breaker Implementation

Implements the circuit breaker pattern with three states:
- Closed: Normal operation
- Open: Blocking requests after threshold exceeded
- Half-Open: Testing recovery
"""

import time
from enum import Enum
from typing import Callable, Optional, Any
from dataclasses import dataclass, field


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5  # Number of failures before opening
    success_threshold: int = 2  # Number of successes in half-open to close
    timeout: float = 60.0  # Time in seconds before transitioning from open to half-open
    failure_rate_threshold: float = 0.5  # Failure rate threshold (0.0 to 1.0)
    sliding_window_size: int = 100  # Number of recent requests to consider


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker"""
    total_requests: int = 0
    total_failures: int = 0
    total_successes: int = 0
    state_changes: list[tuple[float, CircuitState]] = field(default_factory=list)
    recent_requests: list[bool] = field(default_factory=list)  # True = success, False = failure


class CircuitBreaker:
    """
    Circuit breaker for jailbreak prevention.
    
    Monitors requests and opens the circuit when suspicious patterns are detected.
    """
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self.opened_at: Optional[float] = None
        self.half_open_successes: int = 0
        self.half_open_failures: int = 0
    
    def call(self, func: Callable[[], Any], *args, **kwargs) -> Any:
        """
        Execute a function call through the circuit breaker.
        
        Args:
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Result of func execution
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN. Will retry after {self._time_until_reset():.2f} seconds"
                )
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.opened_at is None:
            return False
        return time.time() - self.opened_at >= self.config.timeout
    
    def _time_until_reset(self) -> float:
        """Calculate time remaining until reset attempt"""
        if self.opened_at is None:
            return 0.0
        elapsed = time.time() - self.opened_at
        return max(0.0, self.config.timeout - elapsed)
    
    def _record_success(self):
        """Record a successful request"""
        self.stats.total_requests += 1
        self.stats.total_successes += 1
        self.stats.recent_requests.append(True)
        
        # Maintain sliding window
        if len(self.stats.recent_requests) > self.config.sliding_window_size:
            self.stats.recent_requests.pop(0)
        
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_successes += 1
            self.half_open_failures = 0
            if self.half_open_successes >= self.config.success_threshold:
                self._transition_to_closed()
        elif self.state == CircuitState.OPEN:
            # This shouldn't happen, but handle gracefully
            self._transition_to_half_open()
    
    def _record_failure(self):
        """Record a failed request"""
        self.stats.total_requests += 1
        self.stats.total_failures += 1
        self.stats.recent_requests.append(False)
        
        # Maintain sliding window
        if len(self.stats.recent_requests) > self.config.sliding_window_size:
            self.stats.recent_requests.pop(0)
        
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_failures += 1
            self.half_open_successes = 0
            if self.half_open_failures > 0:
                self._transition_to_open()
        elif self.state == CircuitState.CLOSED:
            # Check if we should open the circuit
            if self._should_open_circuit():
                self._transition_to_open()
    
    def _should_open_circuit(self) -> bool:
        """Determine if circuit should be opened based on failure patterns"""
        if len(self.stats.recent_requests) < self.config.failure_threshold:
            return False
        
        # Check consecutive failures
        recent_failures = sum(1 for r in self.stats.recent_requests[-self.config.failure_threshold:] if not r)
        if recent_failures >= self.config.failure_threshold:
            return True
        
        # Check failure rate in sliding window
        if len(self.stats.recent_requests) >= self.config.sliding_window_size:
            failure_rate = sum(1 for r in self.stats.recent_requests if not r) / len(self.stats.recent_requests)
            if failure_rate >= self.config.failure_rate_threshold:
                return True
        
        return False
    
    def _transition_to_open(self):
        """Transition circuit to OPEN state"""
        if self.state != CircuitState.OPEN:
            self.state = CircuitState.OPEN
            self.opened_at = time.time()
            self.half_open_successes = 0
            self.half_open_failures = 0
            self.stats.state_changes.append((time.time(), CircuitState.OPEN))
    
    def _transition_to_half_open(self):
        """Transition circuit to HALF_OPEN state"""
        if self.state != CircuitState.HALF_OPEN:
            self.state = CircuitState.HALF_OPEN
            self.half_open_successes = 0
            self.half_open_failures = 0
            self.stats.state_changes.append((time.time(), CircuitState.HALF_OPEN))
    
    def _transition_to_closed(self):
        """Transition circuit to CLOSED state"""
        if self.state != CircuitState.CLOSED:
            self.state = CircuitState.CLOSED
            self.opened_at = None
            self.half_open_successes = 0
            self.half_open_failures = 0
            self.stats.state_changes.append((time.time(), CircuitState.CLOSED))
    
    def get_state(self) -> CircuitState:
        """Get current circuit state"""
        return self.state
    
    def get_stats(self) -> CircuitBreakerStats:
        """Get circuit breaker statistics"""
        return self.stats
    
    def reset(self):
        """Manually reset circuit breaker to CLOSED state"""
        self._transition_to_closed()
        self.stats = CircuitBreakerStats()


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and request is blocked"""
    pass

