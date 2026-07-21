import importlib
import os
import unittest

import config
from guardrail.response_validator import (
    GENERATION_ERROR_RESPONSE,
    ResponseStatus,
    ResponseValidator,
    strip_thinking,
)
from model import InferenceEngine


class StripThinkingTests(unittest.TestCase):
    def test_completed_think_block_is_removed(self):
        cleaned, unterminated = strip_thinking(
            "<think>secret reasoning</think>Final answer"
        )

        self.assertEqual(cleaned, "Final answer")
        self.assertFalse(unterminated)

    def test_unterminated_think_block_preserves_prefix(self):
        cleaned, unterminated = strip_thinking(
            "Useful prefix<think>unfinished reasoning"
        )

        self.assertEqual(cleaned, "Useful prefix")
        self.assertTrue(unterminated)

    def test_normal_response_is_unchanged(self):
        cleaned, unterminated = strip_thinking("A normal final answer.")

        self.assertEqual(cleaned, "A normal final answer.")
        self.assertFalse(unterminated)

    def test_unterminated_think_does_not_become_safety_refusal(self):
        validator = ResponseValidator()

        cleaned = validator.validate_and_clean(
            "Useful prefix<think>unfinished reasoning",
            "What is useful?",
        )

        self.assertEqual(cleaned, "Useful prefix")
        self.assertEqual(validator.last_status, ResponseStatus.GENERATION_TRUNCATED)
        self.assertNotEqual(cleaned, "I am unable to provide a response to this query.")
        self.assertNotEqual(cleaned, GENERATION_ERROR_RESPONSE)


class ConfigDefaultsTests(unittest.TestCase):
    def test_generation_defaults(self):
        saved = {
            key: os.environ.get(key)
            for key in (
                "TEMPERATURE",
                "TOP_P",
                "TOP_K",
                "MAX_TOKENS",
                "SEED",
                "ENABLE_THINKING",
            )
        }
        try:
            for key in saved:
                os.environ.pop(key, None)
            reloaded_config = importlib.reload(config)

            self.assertEqual(reloaded_config.TEMPERATURE, 0.7)
            self.assertEqual(reloaded_config.TOP_P, 0.8)
            self.assertEqual(reloaded_config.TOP_K, 20)
            self.assertEqual(reloaded_config.MAX_TOKENS, 1536)
            self.assertEqual(reloaded_config.SEED, 42)
            self.assertFalse(reloaded_config.ENABLE_THINKING)
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            importlib.reload(config)


class ModelFallbackTests(unittest.TestCase):
    def test_chat_template_fallback_keeps_thinking_disabled(self):
        class FakeTokenizer:
            def __init__(self):
                self.calls = []

            def apply_chat_template(self, conversation, **kwargs):
                self.calls.append((conversation, kwargs))
                return "rendered prompt"

        class FakeLLM:
            def __init__(self):
                self.chat_calls = 0
                self.generate_calls = []

            def chat(self, **kwargs):
                self.chat_calls += 1
                raise TypeError(
                    "chat() got an unexpected keyword argument "
                    "'chat_template_kwargs'"
                )

            def generate(self, **kwargs):
                self.generate_calls.append(kwargs)
                return []

        tokenizer = FakeTokenizer()
        llm = FakeLLM()
        engine = InferenceEngine.__new__(InferenceEngine)
        engine.llm = llm
        engine.tokenizer = tokenizer
        engine.sampling_params = object()

        original_enable_thinking = config.ENABLE_THINKING
        try:
            config.ENABLE_THINKING = False
            messages = [[{"role": "user", "content": "hello"}]]
            engine._chat(messages)

            self.assertEqual(llm.chat_calls, 1)
            self.assertEqual(len(llm.generate_calls), 1)
            self.assertEqual(
                tokenizer.calls[0][1],
                {
                    "tokenize": False,
                    "add_generation_prompt": True,
                    "enable_thinking": False,
                },
            )
        finally:
            config.ENABLE_THINKING = original_enable_thinking


if __name__ == "__main__":
    unittest.main()
