"""
Evaluation Metrics for Circuit Breaker Performance

Metrics for assessing the effectiveness of jailbreak prevention.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from .detector import DetectionResult, ThreatLevel


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker performance"""
    total_requests: int = 0
    blocked_requests: int = 0
    allowed_requests: int = 0
    true_positives: int = 0  # Correctly blocked jailbreaks
    false_positives: int = 0  # Incorrectly blocked safe requests
    true_negatives: int = 0  # Correctly allowed safe requests
    false_negatives: int = 0  # Incorrectly allowed jailbreaks
    circuit_opens: int = 0
    average_detection_score: float = 0.0
    threat_level_distribution: Dict[ThreatLevel, int] = None
    
    def __post_init__(self):
        if self.threat_level_distribution is None:
            self.threat_level_distribution = {level: 0 for level in ThreatLevel}
    
    @property
    def precision(self) -> float:
        """Precision: TP / (TP + FP)"""
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)
    
    @property
    def recall(self) -> float:
        """Recall: TP / (TP + FN)"""
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)
    
    @property
    def f1_score(self) -> float:
        """F1 score: 2 * (precision * recall) / (precision + recall)"""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)
    
    @property
    def accuracy(self) -> float:
        """Accuracy: (TP + TN) / (TP + TN + FP + FN)"""
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        if total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / total
    
    @property
    def false_positive_rate(self) -> float:
        """False positive rate: FP / (FP + TN)"""
        if self.false_positives + self.true_negatives == 0:
            return 0.0
        return self.false_positives / (self.false_positives + self.true_negatives)
    
    @property
    def false_negative_rate(self) -> float:
        """False negative rate: FN / (FN + TP)"""
        if self.false_negatives + self.true_positives == 0:
            return 0.0
        return self.false_negatives / (self.false_negatives + self.true_positives)
    
    @property
    def block_rate(self) -> float:
        """Percentage of requests that were blocked"""
        if self.total_requests == 0:
            return 0.0
        return self.blocked_requests / self.total_requests
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "allowed_requests": self.allowed_requests,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "circuit_opens": self.circuit_opens,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "accuracy": self.accuracy,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "block_rate": self.block_rate,
            "average_detection_score": self.average_detection_score,
            "threat_level_distribution": {
                level.value: count 
                for level, count in self.threat_level_distribution.items()
            }
        }
    
    def __str__(self) -> str:
        """String representation of metrics"""
        return f"""Circuit Breaker Metrics:
  Total Requests: {self.total_requests}
  Blocked: {self.blocked_requests} ({self.block_rate:.2%})
  Allowed: {self.allowed_requests}
  
  True Positives: {self.true_positives}
  False Positives: {self.false_positives}
  True Negatives: {self.true_negatives}
  False Negatives: {self.false_negatives}
  
  Precision: {self.precision:.3f}
  Recall: {self.recall:.3f}
  F1 Score: {self.f1_score:.3f}
  Accuracy: {self.accuracy:.3f}
  
  False Positive Rate: {self.false_positive_rate:.3f}
  False Negative Rate: {self.false_negative_rate:.3f}
  
  Circuit Opens: {self.circuit_opens}
  Avg Detection Score: {self.average_detection_score:.3f}
"""


class MetricsCollector:
    """Collects and aggregates metrics during experiments"""
    
    def __init__(self):
        self.metrics = CircuitBreakerMetrics()
        self.detection_scores: List[float] = []
    
    def record_request(self, was_blocked: bool, detection_result: DetectionResult, 
                      is_actual_jailbreak: bool):
        """
        Record a request and its outcome.
        
        Args:
            was_blocked: Whether the request was blocked by circuit breaker
            detection_result: Result from jailbreak detector
            is_actual_jailbreak: Ground truth - was this actually a jailbreak?
        """
        self.metrics.total_requests += 1
        
        if was_blocked:
            self.metrics.blocked_requests += 1
            if is_actual_jailbreak:
                self.metrics.true_positives += 1
            else:
                self.metrics.false_positives += 1
        else:
            self.metrics.allowed_requests += 1
            if is_actual_jailbreak:
                self.metrics.false_negatives += 1
            else:
                self.metrics.true_negatives += 1
        
        # Record detection score
        self.detection_scores.append(detection_result.score)
        self.metrics.average_detection_score = sum(self.detection_scores) / len(self.detection_scores)
        
        # Record threat level
        self.metrics.threat_level_distribution[detection_result.threat_level] += 1
    
    def record_circuit_open(self):
        """Record that the circuit was opened"""
        self.metrics.circuit_opens += 1
    
    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get current metrics"""
        return self.metrics
    
    def reset(self):
        """Reset all metrics"""
        self.metrics = CircuitBreakerMetrics()
        self.detection_scores = []

