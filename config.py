import os


def _get_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _get_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))

# ---------------------------------------------------------
# I/O Paths (all configurable via environment variables)
# ---------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/weights/Qwen3-8B")
INPUT_PATH = os.environ.get("INPUT_PATH", "/model/test/dataset.csv")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/result/submission.csv")

# ---------------------------------------------------------
# Inference Configuration (vLLM + H100 40GB optimised)
# ---------------------------------------------------------
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.0"))
TOP_P = float(os.environ.get("TOP_P", "1.0"))
MAX_TOKENS = _get_int("MAX_TOKENS", 768)

# --- H100 40GB tuning ---
# Defaults are conservative for common local instruct models on H100 40GB.
# The baked fallback model is Qwen3-8B unless MODEL_PATH is overridden.
# MAX_NUM_SEQS=128: conservative default for Qwen3-8B on 40GB.
# MAX_MODEL_LEN=8192: enough for any realistic benchmark query + response.
#   Use 4096 if the model weights are larger (e.g. 14B) and KV cache is tight.
# GPU_MEMORY_UTILIZATION=0.88: leaves headroom for CUDA/vLLM overhead.
# TENSOR_PARALLEL_SIZE=1: single H100 - no tensor parallelism needed for 8B.
MAX_NUM_SEQS = _get_int("MAX_NUM_SEQS", 128)
MAX_MODEL_LEN = _get_int("MAX_MODEL_LEN", 8192)
GPU_MEMORY_UTILIZATION = _get_float("GPU_MEMORY_UTILIZATION", 0.88)
TENSOR_PARALLEL_SIZE = _get_int("TENSOR_PARALLEL_SIZE", 1)
DTYPE = os.environ.get("DTYPE", "auto")
TRUST_REMOTE_CODE = _get_bool("TRUST_REMOTE_CODE", True)
ENABLE_PREFIX_CACHING = _get_bool("ENABLE_PREFIX_CACHING", True)
ENFORCE_EAGER = _get_bool("ENFORCE_EAGER", False)

# ---------------------------------------------------------
# Guardrail Configuration
# ---------------------------------------------------------
# Truncate input for the heuristic safety pipeline only (not for the model).
# 8 000 chars approximately 2 000 tokens - well above any real benchmark query.
GUARDRAIL_MAX_CHARS = _get_int("GUARDRAIL_MAX_CHARS", 8000)

# ThaiSafetyClassifier is a Thai response-safety classifier. It is used as
# semantic evidence in risk fusion and as a high-confidence post-generation
# safety check. The model is downloaded at Docker build time and loaded
# offline at runtime.
THAI_SAFETY_ENABLED = _get_bool("THAI_SAFETY_ENABLED", True)
THAI_SAFETY_MODEL_ID = os.environ.get(
    "THAI_SAFETY_MODEL_ID", "typhoon-ai/ThaiSafetyClassifier"
)
THAI_SAFETY_MODEL_PATH = os.environ.get(
    "THAI_SAFETY_MODEL_PATH", "/models/thai-safety-classifier"
)
THAI_SAFETY_DEVICE = os.environ.get("THAI_SAFETY_DEVICE", "auto")
THAI_SAFETY_MAX_LENGTH = _get_int("THAI_SAFETY_MAX_LENGTH", 256)
THAI_SAFETY_BATCH_SIZE = _get_int("THAI_SAFETY_BATCH_SIZE", 16)

# Defaults favor false-refusal reduction. The classifier was trained as a
# prompt-response pair classifier, so pre-generation use is advisory. It only
# forces refusal when semantic confidence and rule/context evidence agree.
THAI_SAFETY_HIGH_THRESHOLD = _get_float("THAI_SAFETY_HIGH_THRESHOLD", 0.88)
THAI_SAFETY_MEDIUM_THRESHOLD = _get_float("THAI_SAFETY_MEDIUM_THRESHOLD", 0.65)
THAI_SAFETY_RESPONSE_BLOCK_THRESHOLD = _get_float(
    "THAI_SAFETY_RESPONSE_BLOCK_THRESHOLD", 0.92
)
THAI_SAFETY_RESPONSE_MEDIUM_THRESHOLD = _get_float(
    "THAI_SAFETY_RESPONSE_MEDIUM_THRESHOLD", 0.75
)


class RiskLevel:
    SAFE = "SAFE"
    LOW_RISK = "LOW_RISK"
    MEDIUM_RISK = "MEDIUM_RISK"
    HIGH_RISK = "HIGH_RISK"

