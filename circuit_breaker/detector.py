"""
Jailbreak Detection Mechanisms

Implements various detection strategies inspired by:
- Active Honeypot Guardrail System
- EEG-Defender (Early Exit Generation)
- AdaSteer (Adaptive detection)
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum


class ThreatLevel(Enum):
    """Threat level classification"""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DetectionResult:
    """Result of jailbreak detection"""
    is_jailbreak: bool
    threat_level: ThreatLevel
    score: float  # 0.0 to 1.0, higher = more suspicious
    reasons: List[str]
    matched_patterns: List[str] = None


class JailbreakDetector:
    """
    Detects potential jailbreak attempts using multiple strategies.
    
    Combines pattern matching, keyword detection, and behavioral analysis.
    """
    
    def __init__(self):
        self.jailbreak_patterns = self._load_jailbreak_patterns()
        self.suspicious_keywords = self._load_suspicious_keywords()
        self.bypass_techniques = self._load_bypass_techniques()
    
    def detect(self, user_input: str, model_output: Optional[str] = None, 
               conversation_history: Optional[List[Dict[str, str]]] = None) -> DetectionResult:
        """
        Detect potential jailbreak attempt.
        
        Args:
            user_input: Current user input
            model_output: Model's response (if available)
            conversation_history: Previous conversation turns
            
        Returns:
            DetectionResult with threat assessment
        """
        reasons = []
        matched_patterns = []
        score = 0.0
        
        # Pattern-based detection
        pattern_score, pattern_reasons, pattern_matches = self._detect_patterns(user_input)
        score += pattern_score * 0.4
        reasons.extend(pattern_reasons)
        matched_patterns.extend(pattern_matches)
        
        # Keyword-based detection
        keyword_score, keyword_reasons, keyword_matches = self._detect_keywords(user_input)
        score += keyword_score * 0.3
        reasons.extend(keyword_reasons)
        matched_patterns.extend(keyword_matches)
        
        # Bypass technique detection
        bypass_score, bypass_reasons, bypass_matches = self._detect_bypass_techniques(user_input)
        score += bypass_score * 0.2
        reasons.extend(bypass_reasons)
        matched_patterns.extend(bypass_matches)
        
        # Multi-turn analysis (inspired by Active Honeypot)
        if conversation_history:
            multi_turn_score, multi_turn_reasons = self._detect_multi_turn_pattern(
                user_input, conversation_history
            )
            score += multi_turn_score * 0.1
            reasons.extend(multi_turn_reasons)
        
        # Normalize score to [0, 1]
        score = min(1.0, score)
        
        # Determine threat level
        threat_level = self._classify_threat_level(score)
        
        # Determine if it's a jailbreak
        # Lower threshold for initial detection - can be tuned based on false positive/negative rates
        is_jailbreak = threat_level in [ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL] or score >= 0.3
        
        return DetectionResult(
            is_jailbreak=is_jailbreak,
            threat_level=threat_level,
            score=score,
            reasons=reasons,
            matched_patterns=matched_patterns
        )
    
    def _detect_patterns(self, text: str) -> tuple:
        """Detect known jailbreak patterns"""
        score = 0.0
        reasons = []
        matched = []
        
        text_lower = text.lower()
        
        for pattern_name, pattern in self.jailbreak_patterns.items():
            if re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE):
                matched.append(pattern_name)
                score += 0.3  # Increased weight for pattern matches
                reasons.append(f"Matched jailbreak pattern: {pattern_name}")
        
        return min(1.0, score), reasons, matched
    
    def _detect_keywords(self, text: str) -> tuple:
        """Detect suspicious keywords"""
        score = 0.0
        reasons = []
        matched = []
        
        text_lower = text.lower()
        
        for keyword in self.suspicious_keywords:
            if keyword in text_lower:
                matched.append(keyword)
                score += 0.15  # Increased weight for keywords
                reasons.append(f"Found suspicious keyword: {keyword}")
        
        return min(1.0, score), reasons, matched
    
    def _detect_bypass_techniques(self, text: str) -> tuple:
        """Detect common bypass techniques"""
        score = 0.0
        reasons = []
        matched = []
        
        text_lower = text.lower()
        
        for technique, pattern in self.bypass_techniques.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                matched.append(technique)
                score += 0.15
                reasons.append(f"Detected bypass technique: {technique}")
        
        return min(1.0, score), reasons, matched
    
    def _detect_multi_turn_pattern(self, current_input: str, 
                                   history: List[Dict[str, str]]) -> tuple:
        """
        Detect multi-turn jailbreak patterns (inspired by Active Honeypot Guardrail System).
        
        Looks for patterns where user gradually escalates requests or uses
        conversational techniques to bypass safety measures.
        """
        score = 0.0
        reasons = []
        
        if len(history) < 2:
            return score, reasons
        
        # Check for gradual escalation
        suspicious_count = 0
        for turn in history[-3:]:  # Check last 3 turns
            user_msg = turn.get("user", "").lower()
            if any(keyword in user_msg for keyword in ["ignore", "forget", "pretend", "act as"]):
                suspicious_count += 1
        
        if suspicious_count >= 2:
            score += 0.3
            reasons.append("Detected multi-turn escalation pattern")
        
        # Check for role-playing attempts
        role_play_keywords = ["you are", "act as", "pretend to be", "roleplay"]
        if any(keyword in current_input.lower() for keyword in role_play_keywords):
            if len(history) > 0:
                score += 0.2
                reasons.append("Detected role-playing in multi-turn context")
        
        return min(1.0, score), reasons
    
    def _classify_threat_level(self, score: float) -> ThreatLevel:
        """Classify threat level based on detection score"""
        if score >= 0.8:
            return ThreatLevel.CRITICAL
        elif score >= 0.6:
            return ThreatLevel.HIGH
        elif score >= 0.4:
            return ThreatLevel.MEDIUM
        elif score >= 0.2:
            return ThreatLevel.LOW
        else:
            return ThreatLevel.SAFE
    
    def _load_jailbreak_patterns(self) -> Dict[str, str]:
        """Load known jailbreak patterns"""
        return {
            "ignore_previous": r"ignore\s+(previous|all|above|earlier)",
            "forget_instructions": r"forget\s+(your|all|previous)\s+(instructions|rules|guidelines)",
            "new_instructions": r"(new|different)\s+(instructions|rules|guidelines|system)",
            "developer_mode": r"(developer|debug|admin)\s+mode",
            "jailbreak_dan": r"dan\s+mode|do\s+anything\s+now",
            "hypothetical": r"hypothetically|in\s+a\s+hypothetical|imagine",
            "role_play": r"roleplay|role\s+play|pretend\s+to\s+be",
            "override": r"override|bypass|circumvent",
        }
    
    def _load_suspicious_keywords(self) -> List[str]:
        """Load suspicious keywords"""
        return [
            "jailbreak",
            "bypass",
            "override",
            "ignore safety",
            "remove restrictions",
            "unrestricted",
            "no limits",
            "ignore guidelines",
            "forget rules",
            "developer mode",
            "admin access",
            "system prompt",
            "hidden instructions",
        ]
    
    def _load_bypass_techniques(self) -> Dict[str, str]:
        """Load common bypass technique patterns"""
        return {
            "base64_encoding": r"[a-z0-9+/]{20,}={0,2}",
            "unicode_obfuscation": r"[\u0000-\u001f]|[\u200b-\u200f]",
            "repetition": r"(.{1,20})\1{5,}",
            "instruction_injection": r"(system|user|assistant):\s*",
            "xml_injection": r"<[^>]+>(ignore|forget|override)",
        }

