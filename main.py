"""
AI Thailand Benchmark 2026 - Main Pipeline Orchestrator

Batch strategy:
  - Pre-generation guardrail runs on the entire dataset first (CPU-only, fast).
  - Only HIGH_RISK queries are refused without hitting the model.
  - SAFE and LOW_RISK queries proceed with the normal system prompt.
  - MEDIUM_RISK queries proceed with a stricter safety system prompt.
  - vLLM handles internal micro-batching; we feed the full generation list at
    once to maximise GPU utilisation with prefix caching.
  - Post-generation validation strips CoT and checks for leakage/failures.
  - Output rows are written in the exact original input order.
"""

import time
from typing import List

import config
from config import RiskLevel
from utils.logger import get_logger
from utils.io import load_dataset, save_submission
from utils.progress import report_progress
from guardrail.risk_scorer import RiskScorer
from guardrail.policy import RefusalPolicy
from guardrail.response_validator import ResponseValidator
from model import InferenceEngine

logger = get_logger(__name__)

INTERNAL_ERROR_RESPONSE = "I encountered an internal error. Please try again."

FALLBACK_RESPONSE = (
    "ขออภัย ระบบไม่สามารถสร้างคำตอบได้ในขณะนี้ "
    "โปรดลองใหม่อีกครั้งในภายหลัง"
)


def _submission_rows(ids: List[str], responses: List[str]) -> List[dict]:
    return [{"id": ids[i], "response": responses[i]} for i in range(len(ids))]


def _fallback_count(responses: List[str]) -> int:
    return sum(1 for response in responses if response == FALLBACK_RESPONSE)


def _exception_reason(exc: BaseException) -> str:
    message = str(exc).strip().replace("\n", " ")
    if not message:
        message = exc.__class__.__name__
    return f"{exc.__class__.__name__}:{message}"


def main() -> int:
    start = time.time()
    logger.info("Starting AI Thailand Benchmark 2026 inference pipeline.")

    fallback_on_disk = False
    n = 0
    ids: List[str] = []
    final_responses: List[str] = []
    generated_count = 0
    refused_count = 0
    internal_error_count = 0

    try:
        # -- 1. Load dataset before any model initialisation ------------------
        df = load_dataset(config.INPUT_PATH)
        n = len(df)
        ids = df["id"].astype(str).tolist()
        queries: List[str] = df["query"].fillna("").astype(str).tolist()

        # Create a complete valid fallback submission before loading large
        # runtime dependencies. If model startup fails, the platform still has
        # id,response output with the correct row count/order.
        final_responses = [FALLBACK_RESPONSE] * n
        save_submission(_submission_rows(ids, final_responses), config.OUTPUT_PATH)
        fallback_on_disk = True
        logger.info(f"FALLBACK_WRITTEN rows={n}")
        logger.info(
            f"Wrote fallback submission with {n} rows before model startup."
        )

        # -- 2. Initialise components ----------------------------------------
        risk_scorer = RiskScorer()
        refusal_policy = RefusalPolicy()
        validator = ResponseValidator()

        # vLLM is initialised here - allocates GPU memory once up-front.
        logger.info("REAL_MODEL_INITIALIZATION_STARTED")
        engine = InferenceEngine()
        logger.info("REAL_MODEL_INITIALIZED")

        # -- 3. Pre-generation guardrail pass (CPU) ---------------------------
        logger.info(f"Running guardrail analysis on {n} queries...")

        normal_indices: List[int] = []
        normal_queries: List[str] = []
        safer_indices: List[int] = []
        safer_queries: List[str] = []
        completed = 0

        for i, raw_query in enumerate(queries):
            # Truncate for guardrail analysis only (not for the model input)
            truncated = str(raw_query)[: config.GUARDRAIL_MAX_CHARS]
            risk_data = risk_scorer.score(truncated)

            if risk_data["risk_level"] == RiskLevel.HIGH_RISK:
                logger.debug(f"[{ids[i]}] HIGH_RISK -> refused ({risk_data['intent']})")
                final_responses[i] = refusal_policy.get_refusal(risk_data)
                completed += 1
                refused_count += 1
            elif risk_data["risk_level"] == RiskLevel.MEDIUM_RISK:
                safer_indices.append(i)
                safer_queries.append(raw_query)
            else:
                # SAFE / LOW_RISK -> normal generation
                normal_indices.append(i)
                normal_queries.append(raw_query)

        generation_total = len(normal_indices) + len(safer_indices)
        refused = n - generation_total
        logger.info(
            "Guardrail complete: "
            f"{len(normal_indices)} -> normal generation, "
            f"{len(safer_indices)} -> safer generation, {refused} -> refused."
        )

        if completed:
            save_submission(_submission_rows(ids, final_responses), config.OUTPUT_PATH)
            report_progress(completed)

        # -- 4. Batch inference (GPU via vLLM) --------------------------------
        generation_batches = [
            (normal_indices, normal_queries, False),
            (safer_indices, safer_queries, True),
        ]

        if generation_total:
            logger.info(f"GENERATION_STARTED rows={generation_total}")

        for batch_indices, batch_queries, safer in generation_batches:
            if not batch_queries:
                continue

            logger.info(f"Sending {len(batch_queries)} queries to vLLM...")

            generated = engine.generate(batch_queries, safer=safer)
            if len(generated) != len(batch_queries):
                logger.error(
                    "Generation count mismatch: "
                    f"{len(generated)} outputs for {len(batch_queries)} inputs."
                )
                generated = (
                    generated
                    + [INTERNAL_ERROR_RESPONSE] * len(batch_queries)
                )[: len(batch_queries)]

            # -- 5. Post-generation validation --------------------------------
            for idx, raw_output, original_query in zip(
                batch_indices, generated, batch_queries
            ):
                if raw_output == INTERNAL_ERROR_RESPONSE:
                    internal_error_count += 1
                else:
                    generated_count += 1
                final_responses[idx] = validator.validate_and_clean(
                    raw_output, original_query
                )
                completed += 1

            save_submission(_submission_rows(ids, final_responses), config.OUTPUT_PATH)
            report_progress(min(completed, n))

        # -- 6. Final durable save and required progress n --------------------
        save_submission(_submission_rows(ids, final_responses), config.OUTPUT_PATH)
        fallback_count = _fallback_count(final_responses)
        if generation_total:
            logger.info(
                f"GENERATION_COMPLETED rows={generated_count + internal_error_count}"
            )
        if generated_count and fallback_count:
            logger.warning(
                f"PARTIAL_FALLBACK_USED generated={generated_count} "
                f"fallback={fallback_count}"
            )
        logger.info(
            f"FINAL_SUBMISSION_WRITTEN generated={generated_count} "
            f"refused={refused_count} fallback={fallback_count} "
            f"internal_error={internal_error_count}"
        )
        report_progress(n)
        logger.info(f"PROGRESS_FINAL rows={n}")

        elapsed = time.time() - start
        logger.info(f"Pipeline finished in {elapsed:.1f}s. Output: {config.OUTPUT_PATH}")
        return 0
    except Exception as exc:
        logger.exception("Pipeline failed.")
        if fallback_on_disk and n:
            fallback_count = _fallback_count(final_responses)
            logger.warning(
                f"FALLBACK_USED reason={_exception_reason(exc)} rows={n}"
            )
            if generated_count and fallback_count:
                logger.warning(
                    f"PARTIAL_FALLBACK_USED generated={generated_count} "
                    f"fallback={fallback_count}"
                )
            logger.info(
                f"FINAL_SUBMISSION_WRITTEN generated={generated_count} "
                f"refused={refused_count} fallback={fallback_count} "
                f"internal_error={internal_error_count}"
            )
            logger.warning(
                "Keeping complete fallback submission and reporting final progress."
            )
            report_progress(n)
            logger.info(f"PROGRESS_FINAL rows={n}")
            return 0
        raise


if __name__ == "__main__":
    raise SystemExit(main())
