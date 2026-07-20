"""
InferenceEngine - vLLM wrapper for the AI Thailand Benchmark 2026 pipeline.

Compatibility:
  - Qwen3-Instruct      (thinking disabled via chat_template_kwargs when supported)
  - Qwen2.5-Instruct
  - Llama-3.x-Instruct
  - Gemma-3-Instruct
  - Mistral-Instruct-v0.x

API note: llm.chat() was introduced in vLLM 0.4.0 and is the correct high-level
API for instruction-tuned models. It automatically applies the tokenizer's
chat template (tokenizer_config.json -> chat_template), so no manual prompt
formatting is needed. This is compatible with all models listed above.

For vLLM < 0.4.0, replace llm.chat() with llm.generate() after manually
applying the tokenizer.apply_chat_template() step.
"""

import os
from typing import Dict, List

import config
from utils.logger import get_logger

logger = get_logger(__name__)


class InferenceEngine:
    """
    Wraps vLLM for deterministic, offline, high-throughput batch inference.
    Model is loaded strictly from a local path specified by MODEL_PATH.
    """

    def __init__(self):
        # Lazy import: vLLM is Linux/GPU-only; importing at class level would
        # break any environment that runs the guardrail-only unit tests on CPU.
        from vllm import LLM, SamplingParams

        logger.info(f"Initializing vLLM engine - model: {config.MODEL_PATH}")

        if not os.path.exists(config.MODEL_PATH):
            raise FileNotFoundError(
                f"MODEL_PATH not found: {config.MODEL_PATH}. "
                "Ensure the model is mounted before starting the container."
            )

        try:
            self.llm = LLM(
                model=config.MODEL_PATH,
                trust_remote_code=config.TRUST_REMOTE_CODE,
                max_num_seqs=config.MAX_NUM_SEQS,
                max_model_len=config.MAX_MODEL_LEN,
                tensor_parallel_size=config.TENSOR_PARALLEL_SIZE,
                dtype=config.DTYPE,
                enable_prefix_caching=config.ENABLE_PREFIX_CACHING,
                enforce_eager=config.ENFORCE_EAGER,
                gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
            )
        except Exception as e:
            logger.exception(f"vLLM initialisation failed: {e}")
            raise

        # Deterministic generation as required by competition rules
        self.sampling_params = SamplingParams(
            temperature=config.TEMPERATURE,   # 0.0 -> greedy
            top_p=config.TOP_P,               # 1.0 -> no nucleus truncation
            max_tokens=config.MAX_TOKENS,
            detokenize=True,
        )

        # Load system prompt
        prompt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "prompts", "system_prompt.txt"
        )
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                self.system_prompt = f.read().strip()
            logger.info("System prompt loaded successfully.")
        except FileNotFoundError:
            logger.warning("System prompt file not found - using built-in default.")
            self.system_prompt = "You are a helpful and safe AI assistant."

    # ------------------------------------------------------------------
    def _chat(self, messages: List[List[Dict[str, str]]]):
        """
        Calls vLLM chat with Qwen3 thinking disabled when the tokenizer template
        supports it. Older vLLM builds may not accept chat_template_kwargs.
        """
        try:
            return self.llm.chat(
                messages=messages,
                sampling_params=self.sampling_params,
                use_tqdm=False,
                chat_template_kwargs={"enable_thinking": False},
            )
        except TypeError as e:
            logger.warning(
                "vLLM chat_template_kwargs unsupported; retrying without "
                f"explicit Qwen3 thinking control: {e}"
            )
            return self.llm.chat(
                messages=messages,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )

    # ------------------------------------------------------------------
    def _build_messages(
        self,
        queries: List[str],
        user_wrapped_system: bool = False,
        safer: bool = False,
    ) -> List[List[Dict[str, str]]]:
        """
        Wraps each query into a chat-formatted message list.
        llm.chat() passes this directly to the tokenizer's chat_template.
        """
        system_prompt = self.system_prompt
        if safer:
            system_prompt = (
                f"{self.system_prompt}\n\n"
                "Additional safety context: The user query was classified as "
                "potentially sensitive or ambiguous. Provide helpful high-level, "
                "educational, defensive, or safety-oriented information when "
                "appropriate, but do not provide actionable instructions for "
                "harm, crime, credential theft, malware, evasion, or violence."
            )

        if user_wrapped_system:
            return [
                [
                    {
                        "role": "user",
                        "content": (
                            f"{system_prompt}\n\n"
                            f"User query:\n{query}"
                        ),
                    }
                ]
                for query in queries
            ]

        return [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ]
            for query in queries
        ]

    # ------------------------------------------------------------------
    def generate(self, queries: List[str], safer: bool = False) -> List[str]:
        """
        Runs batched inference via vLLM's chat API.

        Args:
            queries: Raw user queries (already pre-filtered to SAFE/MEDIUM).

        Returns:
            List of generated text strings, one per input query.
        """
        if not queries:
            return []

        mode = "safer" if safer else "normal"
        logger.info(f"vLLM generating {len(queries)} responses ({mode} prompt)...")

        try:
            messages = self._build_messages(queries, safer=safer)
            outputs = self._chat(messages)
            return [out.outputs[0].text for out in outputs]
        except Exception as e:
            logger.exception(
                "Generation with system-role messages failed; retrying with a "
                f"user-wrapped system prompt for model compatibility: {e}"
            )

        try:
            fallback_messages = self._build_messages(
                queries, user_wrapped_system=True, safer=safer
            )
            outputs = self._chat(fallback_messages)
            return [out.outputs[0].text for out in outputs]
        except Exception as e:
            logger.exception(f"Generation error after fallback: {e}")
            return ["I encountered an internal error. Please try again."] * len(queries)
