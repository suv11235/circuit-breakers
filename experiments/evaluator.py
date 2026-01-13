"""
Experiment Evaluator

Evaluates the effectiveness of the circuit breaker in preventing jailbreaks.
"""

import sys
from typing import List, Dict, Any
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from circuit_breaker import CircuitBreaker, CircuitBreakerConfig, JailbreakDetector, CircuitBreakerOpenError
from circuit_breaker.metrics import MetricsCollector
from experiments.test_cases import (
    get_all_test_cases,
    get_multi_turn_test_cases,
    TestCase
)


class CircuitBreakerEvaluator:
    """Evaluates circuit breaker performance"""
    
    def __init__(self, config: CircuitBreakerConfig = None):
        self.config = config or CircuitBreakerConfig()
        self.circuit_breaker = CircuitBreaker(self.config)
        self.detector = JailbreakDetector()
        self.metrics_collector = MetricsCollector()
        self.conversation_history: List[Dict[str, str]] = []
    
    def evaluate_single_turn(self, test_cases: List[TestCase] = None) -> Dict[str, Any]:
        """
        Evaluate circuit breaker on single-turn test cases.
        
        Args:
            test_cases: List of test cases. If None, uses all test cases.
            
        Returns:
            Dictionary with evaluation results
        """
        if test_cases is None:
            test_cases = get_all_test_cases()
        
        self.metrics_collector.reset()
        self.conversation_history = []
        
        for test_case in test_cases:
            # Detect jailbreak
            detection_result = self.detector.detect(
                user_input=test_case.user_input,
                conversation_history=self.conversation_history
            )
            
            # Try to process through circuit breaker
            was_blocked = False
            try:
                def process_request():
                    # Simulate processing - in real scenario, this would call the LLM
                    if detection_result.is_jailbreak:
                        raise ValueError("Jailbreak detected")
                    return "Response"
                
                self.circuit_breaker.call(process_request)
            except CircuitBreakerOpenError:
                was_blocked = True
                self.metrics_collector.record_circuit_open()
            except ValueError:
                # Jailbreak detected - record as failure for circuit breaker
                was_blocked = True
            
            # Record metrics
            self.metrics_collector.record_request(
                was_blocked=was_blocked,
                detection_result=detection_result,
                is_actual_jailbreak=test_case.is_jailbreak
            )
            
            # Update conversation history
            self.conversation_history.append({
                "user": test_case.user_input,
                "assistant": "Response" if not was_blocked else "[BLOCKED]"
            })
        
        return {
            "metrics": self.metrics_collector.get_metrics().to_dict(),
            "circuit_breaker_stats": {
                "state": self.circuit_breaker.get_state().value,
                "total_requests": self.circuit_breaker.get_stats().total_requests,
                "total_failures": self.circuit_breaker.get_stats().total_failures,
                "state_changes": len(self.circuit_breaker.get_stats().state_changes)
            }
        }
    
    def evaluate_multi_turn(self, test_cases: List[Dict] = None) -> Dict[str, Any]:
        """
        Evaluate circuit breaker on multi-turn conversations.
        
        Args:
            test_cases: List of multi-turn test cases. If None, uses all multi-turn cases.
            
        Returns:
            Dictionary with evaluation results
        """
        if test_cases is None:
            test_cases = get_multi_turn_test_cases()
        
        self.metrics_collector.reset()
        
        for test_case in test_cases:
            conversation = test_case["conversation"]
            is_jailbreak = test_case["is_jailbreak"]
            
            # Reset conversation history for each test case
            self.conversation_history = []
            
            for turn in conversation:
                user_input = turn["user"]
                
                # Detect jailbreak with conversation context
                detection_result = self.detector.detect(
                    user_input=user_input,
                    conversation_history=self.conversation_history
                )
                
                # Try to process through circuit breaker
                was_blocked = False
                try:
                    def process_request():
                        if detection_result.is_jailbreak:
                            raise ValueError("Jailbreak detected")
                        return turn.get("assistant", "Response")
                    
                    self.circuit_breaker.call(process_request)
                except CircuitBreakerOpenError:
                    was_blocked = True
                    self.metrics_collector.record_circuit_open()
                except ValueError:
                    was_blocked = True
                
                # Record metrics
                self.metrics_collector.record_request(
                    was_blocked=was_blocked,
                    detection_result=detection_result,
                    is_actual_jailbreak=is_jailbreak
                )
                
                # Update conversation history
                self.conversation_history.append({
                    "user": user_input,
                    "assistant": turn.get("assistant", "[BLOCKED]" if was_blocked else "Response")
                })
        
        return {
            "metrics": self.metrics_collector.get_metrics().to_dict(),
            "circuit_breaker_stats": {
                "state": self.circuit_breaker.get_state().value,
                "total_requests": self.circuit_breaker.get_stats().total_requests,
                "total_failures": self.circuit_breaker.get_stats().total_failures,
                "state_changes": len(self.circuit_breaker.get_stats().state_changes)
            }
        }
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """Run full evaluation on all test cases"""
        print("Running single-turn evaluation...")
        single_turn_results = self.evaluate_single_turn()
        
        print("\nRunning multi-turn evaluation...")
        multi_turn_results = self.evaluate_multi_turn()
        
        return {
            "single_turn": single_turn_results,
            "multi_turn": multi_turn_results,
            "summary": {
                "single_turn_accuracy": single_turn_results["metrics"]["accuracy"],
                "single_turn_f1": single_turn_results["metrics"]["f1_score"],
                "multi_turn_accuracy": multi_turn_results["metrics"]["accuracy"],
                "multi_turn_f1": multi_turn_results["metrics"]["f1_score"],
            }
        }


def main():
    """Run evaluation experiments"""
    print("=" * 60)
    print("Circuit Breaker Evaluation for Jailbreak Prevention")
    print("=" * 60)
    print()
    
    # Test with default configuration
    print("Configuration:")
    config = CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout=30.0,
        failure_rate_threshold=0.5
    )
    print(f"  Failure Threshold: {config.failure_threshold}")
    print(f"  Success Threshold: {config.success_threshold}")
    print(f"  Timeout: {config.timeout}s")
    print(f"  Failure Rate Threshold: {config.failure_rate_threshold}")
    print()
    
    evaluator = CircuitBreakerEvaluator(config)
    results = evaluator.run_full_evaluation()
    
    print("\n" + "=" * 60)
    print("SINGLE-TURN RESULTS")
    print("=" * 60)
    print(f"Accuracy: {results['single_turn']['metrics']['accuracy']:.3f}")
    print(f"Precision: {results['single_turn']['metrics']['precision']:.3f}")
    print(f"Recall: {results['single_turn']['metrics']['recall']:.3f}")
    print(f"F1 Score: {results['single_turn']['metrics']['f1_score']:.3f}")
    print(f"False Positive Rate: {results['single_turn']['metrics']['false_positive_rate']:.3f}")
    print(f"False Negative Rate: {results['single_turn']['metrics']['false_negative_rate']:.3f}")
    print(f"Block Rate: {results['single_turn']['metrics']['block_rate']:.3f}")
    
    print("\n" + "=" * 60)
    print("MULTI-TURN RESULTS")
    print("=" * 60)
    print(f"Accuracy: {results['multi_turn']['metrics']['accuracy']:.3f}")
    print(f"Precision: {results['multi_turn']['metrics']['precision']:.3f}")
    print(f"Recall: {results['multi_turn']['metrics']['recall']:.3f}")
    print(f"F1 Score: {results['multi_turn']['metrics']['f1_score']:.3f}")
    print(f"False Positive Rate: {results['multi_turn']['metrics']['false_positive_rate']:.3f}")
    print(f"False Negative Rate: {results['multi_turn']['metrics']['false_negative_rate']:.3f}")
    print(f"Block Rate: {results['multi_turn']['metrics']['block_rate']:.3f}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Single-turn F1: {results['summary']['single_turn_f1']:.3f}")
    print(f"Multi-turn F1: {results['summary']['multi_turn_f1']:.3f}")
    
    return results


if __name__ == "__main__":
    main()

