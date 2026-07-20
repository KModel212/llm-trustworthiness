from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import config
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SafetyResult:
    unsafe_probability: float
    predicted_label: str
    confidence: float
    available: bool
    raw_logits: Optional[List[float]] = None
    error: Optional[str] = None


_SHARED_CLASSIFIER = None


def get_thai_safety_classifier():
    global _SHARED_CLASSIFIER
    if _SHARED_CLASSIFIER is None:
        _SHARED_CLASSIFIER = ThaiSafetyClassifier()
    return _SHARED_CLASSIFIER


class ThaiSafetyClassifier:
    """
    Offline wrapper for typhoon-ai/ThaiSafetyClassifier.

    Model-card format:
      input: <prompt> output: <llm_response>

    The model is a response classifier, not a pure prompt classifier. For
    pre-generation risk fusion, classify_prompt() uses a short proxy compliance
    response and treats the score as semantic evidence, never as the only reason
    to refuse contextually safe prompts.
    """

    def __init__(self):
        self.enabled = config.THAI_SAFETY_ENABLED
        self.available = False
        self.error: Optional[str] = None
        self.device = "cpu"
        self.tokenizer = None
        self.model = None

        if not self.enabled:
            self.error = "ThaiSafetyClassifier disabled by THAI_SAFETY_ENABLED=0"
            logger.warning(f"[SAFETY] {self.error}")
            return

        try:
            self._load()
        except Exception as e:
            self.error = str(e)
            logger.error(
                "[SAFETY] ThaiSafetyClassifier unavailable; semantic safety "
                f"classification disabled. error={self.error}"
            )

    def _load(self) -> None:
        import os

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_path = config.THAI_SAFETY_MODEL_PATH
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"THAI_SAFETY_MODEL_PATH not found: {model_path}"
            )

        if config.THAI_SAFETY_DEVICE == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.THAI_SAFETY_DEVICE

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.available = True
        logger.info(
            "[SAFETY] ThaiSafetyClassifier loaded offline "
            f"path={model_path} device={self.device}"
        )

    def _unavailable_result(self) -> SafetyResult:
        return SafetyResult(
            unsafe_probability=0.0,
            predicted_label="unavailable",
            confidence=0.0,
            available=False,
            error=self.error or "ThaiSafetyClassifier unavailable",
        )

    def format_pair(self, prompt: str, response: str) -> str:
        return f"input: {prompt or ''} output: {response or ''}"

    def classify(self, text: str) -> SafetyResult:
        return self.classify_pair("", text)

    def classify_prompt(self, prompt: str) -> SafetyResult:
        proxy_response = (
            "I will comply with the user's request and provide the concrete "
            "actionable details they asked for."
        )
        return self.classify_pair(prompt, proxy_response)

    def classify_pair(self, prompt: str, response: str) -> SafetyResult:
        results = self.classify_pairs([(prompt, response)])
        return results[0] if results else self._unavailable_result()

    def classify_pairs(self, pairs: List[tuple[str, str]]) -> List[SafetyResult]:
        if not self.available:
            return [self._unavailable_result() for _ in pairs]

        import torch

        texts = [self.format_pair(prompt, response) for prompt, response in pairs]
        outputs: List[SafetyResult] = []

        for start in range(0, len(texts), config.THAI_SAFETY_BATCH_SIZE):
            batch_texts = texts[start:start + config.THAI_SAFETY_BATCH_SIZE]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.THAI_SAFETY_MAX_LENGTH,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)

            for logit_row, prob_row in zip(logits.detach().cpu(), probs.detach().cpu()):
                safe_prob = float(prob_row[0])
                unsafe_prob = float(prob_row[1])
                predicted = "harmful" if unsafe_prob >= safe_prob else "safe"
                confidence = max(safe_prob, unsafe_prob)
                outputs.append(
                    SafetyResult(
                        unsafe_probability=unsafe_prob,
                        predicted_label=predicted,
                        confidence=confidence,
                        available=True,
                        raw_logits=[float(x) for x in logit_row.tolist()],
                    )
                )

        return outputs
