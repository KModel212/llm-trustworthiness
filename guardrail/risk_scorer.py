from config import RiskLevel
from guardrail.normalize import TextNormalizer
from guardrail.prompt_injection import PromptInjectionDetector
from guardrail.jailbreak_detector import JailbreakDetector
from guardrail.intent_classifier import IntentClassifier
from guardrail.context import SafetyContext
from guardrail.semantic_signals import SemanticSignalExtractor
from guardrail.thai_safety_classifier import get_thai_safety_classifier
from typing import Dict, Any
import config
from utils.logger import get_logger

logger = get_logger(__name__)


class RiskScorer:
    """
    Aggregates signals from all detectors and classifies the query into a risk tier.

    Decision logic (in priority order):
      1. Clear harmful intent + actionable payload -> HIGH_RISK
      2. Prompt injection / jailbreak framing alone -> MEDIUM_RISK
      3. Ambiguous sensitive or dual-use requests -> MEDIUM_RISK
      4. Everything else -> SAFE

    MEDIUM and SAFE queries both proceed to generation - only HIGH_RISK is refused
    pre-generation. This maximises helpfulness for the benchmark.
    """

    def __init__(self):
        self.normalizer = TextNormalizer()
        self.injection_detector = PromptInjectionDetector()
        self.jailbreak_detector = JailbreakDetector()
        self.intent_classifier = IntentClassifier()
        self.context = SafetyContext()
        self.semantic_signals = SemanticSignalExtractor()
        self.thai_safety = get_thai_safety_classifier()

    @staticmethod
    def _has_malicious_composition(evidence) -> bool:
        action = bool(evidence.action_signals)
        objective = bool(evidence.objective_signals)
        target = bool(evidence.target_signals)
        evasion = bool(evidence.evasion_signals)
        family = bool(evidence.risk_families)
        high_context_family = bool(
            set(evidence.risk_families)
            & {"Weapons", "Violence", "Malware", "Fraud", "Privacy"}
        )

        return (
            (action and objective and target)
            or (action and objective and evasion)
            or (action and target and evasion)
            or (objective and target and evasion)
            or (family and action and objective)
            or (family and action and evasion)
            or (high_context_family and objective)
            or (high_context_family and action and target)
            or (high_context_family and objective and target)
            or (action and evasion and evidence.malicious_signal_count >= 3)
        )

    @staticmethod
    def _has_jailbreak_composition(evidence, is_injection: bool, is_jailbreak: bool) -> bool:
        jailbreak_count = len(evidence.jailbreak_signals)
        malicious_payload = bool(
            evidence.risk_families
            or evidence.objective_signals
            or evidence.target_signals
        )
        jailbreak_signals = set(evidence.jailbreak_signals)
        transform_only = "transform_only" in evidence.safe_context_signals

        return (
            jailbreak_count >= 2
            or bool(jailbreak_signals & {"fake_authority", "template_injection", "smuggling"})
            or ("hidden_prompt_request" in jailbreak_signals and not transform_only)
            or (jailbreak_count >= 1 and (is_injection or is_jailbreak))
            or (jailbreak_count >= 1 and malicious_payload and evidence.action_signals)
        )

    @staticmethod
    def _has_actionable_harm_payload(tier: str, evidence, actionable_override: bool) -> bool:
        transform_only = "transform_only" in evidence.safe_context_signals
        non_actionable_safe = bool(
            set(evidence.safe_context_signals)
            & {
                "education_overview",
                "defensive",
                "fictional_nonactionable",
                "training_or_synthetic",
            }
        )
        harmful_capability = bool(
            set(evidence.risk_families)
            & {"Weapons", "Violence", "Malware", "Fraud", "Privacy"}
        )
        action = bool(evidence.action_signals)
        objective = bool(evidence.objective_signals)
        target = bool(evidence.target_signals)

        if transform_only and not actionable_override:
            return False
        if tier == "HIGH" and not (non_actionable_safe and not actionable_override):
            return True
        return bool(
            harmful_capability
            and action
            and (
                objective
                or target
                or actionable_override
                or evidence.malicious_signal_count >= 2
            )
        )

    def score(self, raw_query: str) -> Dict[str, Any]:
        """
        Processes a raw query and assigns a final RiskLevel through confidence
        fusion instead of raw keyword blocking.
        """
        normalized = self.normalizer.normalize(raw_query)

        is_injection, inj_conf = self.injection_detector.detect(normalized)
        is_jailbreak, jb_conf = self.jailbreak_detector.detect(normalized)

        # Run intent classification (returns tier + category)
        tier, intent = self.intent_classifier.classify(normalized)
        evidence = self.semantic_signals.extract(normalized)
        context_safe = self.context.reduces_risk(normalized)
        semantic_safe = bool(evidence.safe_context_signals)
        context_override = self.context.has_actionable_override(normalized)
        semantic_evasion = bool(evidence.evasion_signals)
        actionable_override = context_override or semantic_evasion
        strong_semantic_safe = bool(
            set(evidence.safe_context_signals)
            & {"defensive", "fictional_nonactionable", "transform_only", "training_or_synthetic"}
        )
        safe_context = (
            (context_safe and not actionable_override)
            or (strong_semantic_safe and not semantic_evasion)
            or (semantic_safe and not actionable_override)
        )

        semantic = self.thai_safety.classify_prompt(raw_query)
        semantic_score = semantic.unsafe_probability if semantic.available else None

        # --- Decision ---
        structural_attack = (is_injection and inj_conf >= 0.6) or (is_jailbreak and jb_conf >= 0.6)
        semantic_malicious = self._has_malicious_composition(evidence)
        semantic_jailbreak = self._has_jailbreak_composition(evidence, is_injection, is_jailbreak)
        attack_framing = structural_attack or semantic_jailbreak
        actionable_harm_payload = self._has_actionable_harm_payload(
            tier,
            evidence,
            actionable_override,
        )
        semantic_high = (
            semantic_score is not None
            and semantic_score >= config.THAI_SAFETY_HIGH_THRESHOLD
        )
        semantic_medium = (
            semantic_score is not None
            and semantic_score >= config.THAI_SAFETY_MEDIUM_THRESHOLD
        )
        if intent == "General Benign" and evidence.primary_family:
            intent = evidence.primary_family

        if attack_framing and (actionable_harm_payload or semantic_malicious):
            risk_level = RiskLevel.HIGH_RISK
        elif attack_framing:
            risk_level = RiskLevel.MEDIUM_RISK
        elif actionable_harm_payload and (tier == "HIGH" or semantic_malicious):
            risk_level = RiskLevel.HIGH_RISK
        elif safe_context and not actionable_override:
            if semantic_high and (tier == "HIGH" or semantic_malicious):
                risk_level = RiskLevel.MEDIUM_RISK
            elif tier in {"HIGH", "MEDIUM"} or semantic_medium or evidence.risk_families:
                risk_level = RiskLevel.MEDIUM_RISK
            else:
                risk_level = RiskLevel.SAFE
        elif tier == "HIGH" or semantic_malicious:
            risk_level = RiskLevel.HIGH_RISK
        elif semantic_high and tier in {"HIGH", "MEDIUM"}:
            risk_level = RiskLevel.HIGH_RISK
        elif semantic_high and actionable_override:
            risk_level = RiskLevel.HIGH_RISK
        elif tier == "MEDIUM" or semantic_medium or (evidence.risk_families and evidence.malicious_signal_count >= 2):
            risk_level = RiskLevel.MEDIUM_RISK
        else:
            risk_level = RiskLevel.SAFE

        decision = "refuse" if risk_level == RiskLevel.HIGH_RISK else "allow"
        logger.debug(
            "[SAFETY] "
            f"rule_tier={tier} intent={intent} injection={is_injection}:{inj_conf:.2f} "
            f"jailbreak={is_jailbreak}:{jb_conf:.2f} "
            f"thai_classifier_score={semantic_score if semantic_score is not None else 'unavailable'} "
            f"semantic_families={evidence.risk_families} "
            f"semantic_actions={evidence.action_signals} "
            f"semantic_objectives={evidence.objective_signals} "
            f"semantic_targets={evidence.target_signals} "
            f"semantic_evasion={evidence.evasion_signals} "
            f"safe_context={safe_context} attack_framing={attack_framing} "
            f"actionable_harm_payload={actionable_harm_payload} "
            f"risk_level={risk_level} decision={decision}"
        )

        return {
            "risk_level": risk_level,
            "intent": intent,
            "is_injection": is_injection,
            "is_jailbreak": is_jailbreak,
            "rule_tier": tier,
            "safe_context": safe_context,
            "actionable_override": actionable_override,
            "thai_classifier_available": semantic.available,
            "thai_classifier_score": semantic_score,
            "thai_classifier_label": semantic.predicted_label,
            "thai_classifier_confidence": semantic.confidence,
            "matched_action_signals": evidence.action_signals,
            "matched_objective_signals": evidence.objective_signals,
            "matched_target_signals": evidence.target_signals,
            "matched_evasion_signals": evidence.evasion_signals,
            "matched_safe_context_signals": evidence.safe_context_signals,
            "matched_jailbreak_signals": evidence.jailbreak_signals,
            "semantic_risk_families": evidence.risk_families,
            "semantic_malicious_signal_count": evidence.malicious_signal_count,
            "attack_framing": attack_framing,
            "actionable_harm_payload": actionable_harm_payload,
        }
