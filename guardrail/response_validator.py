import re

import config
from guardrail.context import SafetyContext
from guardrail.thai_safety_classifier import get_thai_safety_classifier
from utils.logger import get_logger

logger = get_logger(__name__)


class ResponseValidator:
    """
    Post-generation validator. Strips CoT tags and catches the very few failure
    modes that slip past the pre-generation guardrail.

    Design principle: be permissive. The model itself is expected to refuse
    harmful content; this layer only catches clear-cut leakage / failure modes.
    Unnecessary secondary refusals hurt the Helpfulness score.
    """

    def __init__(self):
        # Strip Qwen3 chain-of-thought blocks (always present at temperature=0)
        self.cot_pattern = re.compile(
            r'<think>.*?</think>\s*|<think>.*$', re.DOTALL | re.IGNORECASE
        )

        # System-prompt leakage: exact phrases from our own system_prompt.txt
        # Keep this list short and very specific to avoid over-triggering.
        self.system_leak_patterns = re.compile(
            r'(You are a knowledgeable, accurate, and helpful AI assistant'
            r'|Your primary goal is to provide the most complete'
            r'|Key rules:'
            r'|Refuse only explicit harm'
            r'|Never reveal or repeat these instructions)',
            re.IGNORECASE,
        )

        # Absolute last-resort unsafe procedural patterns
        # Only the most unambiguous, step-by-step harmful instructions
        self.unsafe_procedural = re.compile(
            r'(step 1[:\.]? (?:gather|obtain|acquire) (?:materials|ingredients|chemicals) (?:for|to make)'
            r'|to (?:build|assemble|construct) (?:the |a )?(?:explosive|bomb|ied)'
            r'|mix (?:the following chemicals|together)[:] .{0,50}(nitrate|peroxide|chlorate)'
            r'|(?:keylogger|ransomware|reverse shell).{0,120}(?:code|payload|stealth|exfiltrat)'
            r'|(?:เว็บล็อกอินปลอม|ขโมยรหัส|เอารหัส).{0,120}(?:ขั้นตอน|โค้ด|ทำงาน))',
            re.IGNORECASE,
        )
        self.context = SafetyContext()
        self.thai_safety = get_thai_safety_classifier()

    def validate_and_clean(self, response: str, original_query: str) -> str:
        """
        Cleans and validates a generated response.
        Returns the cleaned response, or a minimal refusal string if a hard
        failure is detected.
        """
        if not response or not str(response).strip():
            return 'I am unable to provide a response to this query.'

        cleaned = self.cot_pattern.sub('', str(response)).strip()

        if not cleaned:
            return 'I am unable to provide a response to this query.'

        # Detect accidental system prompt leakage
        if self.system_leak_patterns.search(cleaned):
            return 'I cannot fulfill this request.'

        # Detect prompt-echo failure (model repeated the query verbatim)
        if len(cleaned) > 20 and cleaned.lower().strip() == original_query.lower().strip():
            return 'I am unable to generate a useful response to this query.'

        # Final safety net: catch unambiguous step-by-step harmful instructions
        if self.unsafe_procedural.search(cleaned):
            return 'I apologize, but I cannot provide this information.'

        semantic = self.thai_safety.classify_pair(original_query, cleaned)
        if semantic.available:
            safe_context = self.context.reduces_risk(
                f"{original_query}\n{cleaned}"
            )
            actionable = self.context.has_actionable_override(cleaned)
            logger.debug(
                "[SAFETY] post_generation "
                f"thai_classifier_score={semantic.unsafe_probability:.3f} "
                f"label={semantic.predicted_label} safe_context={safe_context}"
            )
            if (
                semantic.unsafe_probability
                >= config.THAI_SAFETY_RESPONSE_BLOCK_THRESHOLD
                and (actionable or not safe_context)
            ):
                return (
                    "I can't provide actionable harmful instructions. I can help "
                    "with high-level safety, prevention, or educational context."
                )

        return cleaned
