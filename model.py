"""
InferenceEngine - vLLM wrapper for the AI Thailand Benchmark 2026 pipeline.
"""

import os
from typing import Dict, List

import config
from utils.logger import get_logger

logger = get_logger(__name__)


def _is_unsupported_chat_template_kwargs(error: TypeError) -> bool:
    message = str(error).lower()
    return (
        "chat_template_kwargs" in message
        and (
            "unexpected" in message
            or "unsupported" in message
            or "got an unexpected keyword" in message
            or "unexpected keyword argument" in message
        )
    )


class InferenceEngine:
    """
    Wraps vLLM for offline, high-throughput batch inference.
    """

    def __init__(self):
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

        # Qwen3 non-thinking recommended sampling preset.
        self.sampling_params = SamplingParams(
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            top_k=config.TOP_K,
            max_tokens=config.MAX_TOKENS,
            seed=config.SEED,
            detokenize=True,
        )
        logger.info(
            "Generation settings: temperature=%s top_p=%s top_k=%s "
            "max_tokens=%s seed=%s enable_thinking=%s",
            config.TEMPERATURE,
            config.TOP_P,
            config.TOP_K,
            config.MAX_TOKENS,
            config.SEED,
            config.ENABLE_THINKING,
        )

        # Get tokenizer so we can explicitly apply the chat template ourselves
        # if llm.chat() cannot forward enable_thinking=False.
        self.tokenizer = self.llm.get_tokenizer()

        prompt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "prompts",
            "system_prompt.txt",
        )

        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                self.system_prompt = f.read().strip()
            logger.info("System prompt loaded successfully.")
        except FileNotFoundError:
            logger.warning(
                "System prompt file not found - using built-in default."
            )
            self.system_prompt = "You are a helpful and safe AI assistant."

    # ------------------------------------------------------------------
    def _chat(self, messages: List[List[Dict[str, str]]]):
        """
        Run chat inference with Qwen3 thinking explicitly disabled.

        Preferred path:
            llm.chat(..., chat_template_kwargs={"enable_thinking": False})

        Compatibility fallback:
            manually apply tokenizer.apply_chat_template(
                enable_thinking=False
            )
            then call llm.generate().
        """
        try:
            return self.llm.chat(
                messages=messages,
                sampling_params=self.sampling_params,
                use_tqdm=False,
                chat_template_kwargs={
                    "enable_thinking": config.ENABLE_THINKING
                },
            )

        except TypeError as e:
            if not _is_unsupported_chat_template_kwargs(e):
                raise
            logger.warning(
                "Current vLLM does not support chat_template_kwargs in "
                "llm.chat(); applying chat template manually instead: %s",
                e,
            )

        # IMPORTANT:
        # Do not retry llm.chat() without enable_thinking=False.
        # That could silently re-enable Qwen3 thinking mode.
        prompts = []

        for conversation in messages:
            prompt = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=config.ENABLE_THINKING,
            )
            prompts.append(prompt)

        return self.llm.generate(
            prompts=prompts,
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )

    # ------------------------------------------------------------------
    def _extract_responses(self, outputs) -> List[str]:
        responses = [
            out.outputs[0].text
            for out in outputs
        ]
        self._log_unexpected_thinking(responses)
        return responses

    # ------------------------------------------------------------------
    def _log_unexpected_thinking(self, responses: List[str]) -> None:
        if config.ENABLE_THINKING:
            return

        unexpected_thinking = sum(
            "<think>" in text or "</think>" in text
            for text in responses
        )

        if unexpected_thinking:
            logger.error(
                "GENERATION_ERROR: unexpected_think_markers count=%d total=%d "
                "enable_thinking=False",
                unexpected_thinking,
                len(responses),
            )

    # ------------------------------------------------------------------
    def _build_messages(
        self,
        queries: List[str],
        user_wrapped_system: bool = False,
        safer: bool = False,
    ) -> List[List[Dict[str, str]]]:

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
    def generate(
        self,
        queries: List[str],
        safer: bool = False,
    ) -> List[str]:

        if not queries:
            return []

        mode = "safer" if safer else "normal"

        logger.info(
            f"vLLM generating {len(queries)} responses "
            f"({mode} prompt)..."
        )

        try:
            messages = self._build_messages(
                queries,
                safer=safer,
            )

            outputs = self._chat(messages)

            return self._extract_responses(outputs)

        except Exception as e:
            logger.exception(
                "Generation with system-role messages failed; "
                "retrying with a user-wrapped system prompt: %s",
                e,
            )

        try:
            fallback_messages = self._build_messages(
                queries,
                user_wrapped_system=True,
                safer=safer,
            )

            outputs = self._chat(fallback_messages)

            return self._extract_responses(outputs)

        except Exception as e:
            logger.exception(
                "Generation error after fallback: %s",
                e,
            )

            return [
                "I encountered an internal error. Please try again."
            ] * len(queries)
