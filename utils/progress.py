import os
import subprocess

from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_PROGRESS_PATH = "/benchmark_lib/progress"


def report_progress(count: int) -> bool:
    """
    Report benchmark progress when the platform helper exists.

    Local runs usually do not mount /benchmark_lib/progress, so missing helper
    is treated as a no-op. The path is overrideable for tests.
    """
    progress_path = os.environ.get("PROGRESS_PATH", DEFAULT_PROGRESS_PATH)
    if not os.path.exists(progress_path):
        logger.debug(f"Progress helper not found at {progress_path}; skipping.")
        return False

    value = str(max(0, int(count)))
    try:
        completed = subprocess.run(
            [progress_path, value],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
    except Exception as exc:
        logger.warning(f"Failed to report progress {value}: {exc}")
        return False

    if completed.returncode != 0:
        logger.warning(
            "Progress helper returned non-zero status "
            f"{completed.returncode}: {completed.stderr.strip()}"
        )
        return False

    logger.info(f"Reported benchmark progress: {value}")
    return True
