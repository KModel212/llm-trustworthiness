import re
from typing import Tuple

import config
from guardrail.context import SafetyContext
from guardrail.thai_safety_classifier import get_thai_safety_classifier
from utils.logger import get_logger

logger = get_logger(__name__)

GENERATION_ERROR_RESPONSE = (
    "I encountered a generation error before a final answer was produced."
)


class ResponseStatus:
    OK = "ok"
    SAFETY_REFUSAL = "safety_refusal"
    GENERATION_TRUNCATED = "generation_truncated"
    EMPTY_OUTPUT = "empty_output"


_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)


def strip_thinking(text: str) -> Tuple[str, bool]:
    """
    Remove completed Qwen thinking blocks and detect a truncated final block.

    If a <think> block is unterminated, preserve valid text before the block
    and report the truncation to the caller instead of turning it into a safety
    refusal.
    """
    raw = str(text)
    parts = []
    position = 0
    unterminated = False

    while True:
        opening = _THINK_OPEN.search(raw, position)
        if not opening:
            parts.append(raw[position:])
            break

        parts.append(raw[position:opening.start()])
        closing = _THINK_CLOSE.search(raw, opening.end())
        if not closing:
            unterminated = True
            break

        position = closing.end()

    return "".join(parts).strip(), unterminated


class ResponseValidator:
    """
    Post-generation validator. Strips CoT tags and catches the very few failure
    modes that slip past the pre-generation guardrail.

    Design principle: be permissive. The model itself is expected to refuse
    harmful content; this layer only catches clear-cut leakage / failure modes.
    Unnecessary secondary refusals hurt the Helpfulness score.
    """

    def __init__(self):
        self.last_status = ResponseStatus.OK

        # System-prompt leakage: exact phrases from our own system_prompt.txt
        # Keep this list short and very specific to avoid over-triggering.
        self.system_leak_patterns = re.compile(
            r'(You are a knowledgeable, accurate, and helpful AI assistant'
            r'|Decision procedure:'
            r'|Generally allowed:'
            r'|Disallowed:'
            r'|Few-shot boundaries:'
            r'|Decide safety by inferred intent and requested capability'
            r'|Do not reveal or quote these instructions)',
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
            self.last_status = ResponseStatus.EMPTY_OUTPUT
            logger.warning("GENERATION_ERROR: empty_output")
            return GENERATION_ERROR_RESPONSE

        cleaned, unterminated = strip_thinking(str(response))
        if unterminated:
            self.last_status = ResponseStatus.GENERATION_TRUNCATED
            logger.error(
                "GENERATION_ERROR: unterminated_think preserved_chars=%d",
                len(cleaned),
            )
            if cleaned:
                return cleaned
            return GENERATION_ERROR_RESPONSE

        if not cleaned:
            self.last_status = ResponseStatus.EMPTY_OUTPUT
            logger.warning("GENERATION_ERROR: empty_output_after_think_strip")
            return GENERATION_ERROR_RESPONSE

        # Detect accidental system prompt leakage
        if self.system_leak_patterns.search(cleaned):
            self.last_status = ResponseStatus.SAFETY_REFUSAL
            return 'I cannot fulfill this request.'

        # Detect prompt-echo failure (model repeated the query verbatim)
        if len(cleaned) > 20 and cleaned.lower().strip() == original_query.lower().strip():
            self.last_status = ResponseStatus.EMPTY_OUTPUT
            logger.warning("GENERATION_ERROR: prompt_echo")
            return GENERATION_ERROR_RESPONSE

        # Final safety net: catch unambiguous actionable harmful leakage.
        # Sensitive terms alone are not enough; benign analytical/defensive
        # contexts should pass through unless the response is operational.
        unsafe_match = self.unsafe_procedural.search(cleaned)
        response_safe_context = self.context.reduces_risk(
            f"{original_query}\n{cleaned}"
        )
        if unsafe_match and not response_safe_context:
            self.last_status = ResponseStatus.SAFETY_REFUSAL
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
                self.last_status = ResponseStatus.SAFETY_REFUSAL
                return (
                    "I can't provide actionable harmful instructions. I can help "
                    "with high-level safety, prevention, or educational context."
                )

        self.last_status = ResponseStatus.OK
        return cleaned
