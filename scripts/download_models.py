import argparse
import json
import multiprocessing as mp
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty

from huggingface_hub import HfApi, hf_hub_download


THAI_SAFETY_FILES = [
    "config.json",
    "model.safetensors",
    "spm.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "README.md",
]

QWEN3_8B_FILES = [
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors.index.json",
    "model-00001-of-00005.safetensors",
    "model-00002-of-00005.safetensors",
    "model-00003-of-00005.safetensors",
    "model-00004-of-00005.safetensors",
    "model-00005-of-00005.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]


@dataclass(frozen=True)
class RepoFile:
    name: str
    size: int | None


def default_cache_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown size"
    units = ["B", "KiB", "MiB", "GiB"]
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{value} B"


def cache_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for child in path.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def fetch_repo_file_metadata(model_id: str, revision: str, files: list[str]) -> dict[str, RepoFile]:
    info = HfApi().model_info(repo_id=model_id, revision=revision, files_metadata=True)
    siblings = {sibling.rfilename: sibling for sibling in info.siblings}
    missing = [filename for filename in files if filename not in siblings]
    if missing:
        raise RuntimeError(f"{model_id}@{revision} is missing required files: {missing}")

    return {
        filename: RepoFile(
            name=filename,
            size=getattr(siblings[filename], "size", None),
        )
        for filename in files
    }


def valid_local_file(path: Path, expected_size: int | None) -> bool:
    if not path.is_file():
        return False
    if expected_size is None:
        return path.stat().st_size > 0
    return path.stat().st_size == expected_size


def copy_if_needed(source: Path, destination: Path, expected_size: int | None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if valid_local_file(destination, expected_size):
        print(f"  staged file already valid: {destination}", flush=True)
        return

    temp_destination = destination.with_name(f".{destination.name}.tmp")
    if temp_destination.exists():
        temp_destination.unlink()
    shutil.copy2(source, temp_destination)
    actual_size = temp_destination.stat().st_size
    if expected_size is not None and actual_size != expected_size:
        temp_destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"Size mismatch after copying {destination.name}: "
            f"expected {expected_size}, got {actual_size}"
        )
    temp_destination.replace(destination)
    print(f"  staged {destination.name} ({format_bytes(actual_size)})", flush=True)


def hf_download_worker(
    queue: mp.Queue,
    model_id: str,
    revision: str,
    filename: str,
    cache_dir: str,
) -> None:
    try:
        path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=False,
        )
    except BaseException as exc:
        queue.put(("error", repr(exc)))
    else:
        queue.put(("ok", path))


def download_with_retry(
    model_id: str,
    revision: str,
    repo_file: RepoFile,
    cache_dir: Path,
    attempts: int,
    initial_backoff: float,
    stall_timeout: float,
    stall_min_bytes: int,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    last_error = "download did not start"

    for attempt in range(1, attempts + 1):
        print(
            f"[{model_id}] {repo_file.name}: attempt {attempt}/{attempts} "
            f"({format_bytes(repo_file.size)})",
            flush=True,
        )

        queue: mp.Queue = context.Queue()
        before_size = cache_size(cache_dir)
        last_size = before_size
        last_progress_at = time.monotonic()
        stalled = False
        process = context.Process(
            target=hf_download_worker,
            args=(queue, model_id, revision, repo_file.name, str(cache_dir)),
        )
        process.start()

        while process.is_alive():
            time.sleep(5)
            current_size = cache_size(cache_dir)
            delta = current_size - last_size
            if delta >= stall_min_bytes:
                print(
                    f"  cache received +{format_bytes(delta)} "
                    f"for {repo_file.name}",
                    flush=True,
                )
                last_size = current_size
                last_progress_at = time.monotonic()
            elif time.monotonic() - last_progress_at >= stall_timeout:
                process.terminate()
                process.join(timeout=20)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=10)
                received = max(0, current_size - before_size)
                last_error = (
                    f"stalled after receiving {format_bytes(received)} "
                    f"in {int(stall_timeout)}s"
                )
                print(f"  {repo_file.name}: {last_error}", flush=True)
                stalled = True
                break

        if stalled:
            pass
        elif process.exitcode == 0:
            try:
                status, payload = queue.get_nowait()
            except Empty:
                status, payload = "error", "download worker exited without a result"
            if status == "ok":
                cached_path = Path(payload)
                actual_size = cached_path.stat().st_size
                if repo_file.size is not None and actual_size != repo_file.size:
                    last_error = (
                        f"cached size mismatch for {repo_file.name}: "
                        f"expected {repo_file.size}, got {actual_size}"
                    )
                else:
                    print(
                        f"  cached {repo_file.name} at {cached_path} "
                        f"({format_bytes(actual_size)})",
                        flush=True,
                    )
                    return cached_path
            else:
                last_error = str(payload)
                print(f"  {repo_file.name}: {last_error}", flush=True)
        elif process.exitcode is not None and process.exitcode != 0:
            last_error = f"download worker exited with code {process.exitcode}"
            print(f"  {repo_file.name}: {last_error}", flush=True)

        if attempt < attempts:
            sleep_seconds = initial_backoff * (2 ** (attempt - 1))
            print(f"  retrying {repo_file.name} in {sleep_seconds:.0f}s", flush=True)
            time.sleep(sleep_seconds)

    raise RuntimeError(f"Failed to download {model_id}/{repo_file.name}: {last_error}")


def remove_local_hf_metadata(path: Path) -> None:
    local_cache = path / ".cache"
    if local_cache.exists():
        shutil.rmtree(local_cache)


def verify_staged_files(
    model_id: str,
    output_dir: Path,
    required_files: list[str],
    metadata: dict[str, RepoFile],
) -> None:
    missing = []
    bad_sizes = []
    for filename in required_files:
        path = output_dir / filename
        expected_size = metadata[filename].size
        if not path.is_file():
            missing.append(filename)
        elif expected_size is not None and path.stat().st_size != expected_size:
            bad_sizes.append((filename, expected_size, path.stat().st_size))

    if missing:
        raise RuntimeError(f"Missing staged files for {model_id}: {missing}")
    if bad_sizes:
        raise RuntimeError(f"Bad staged file sizes for {model_id}: {bad_sizes}")

    symlinks = [str(path) for path in output_dir.rglob("*") if path.is_symlink()]
    if symlinks:
        raise RuntimeError(f"Unexpected symlinks in staged model: {symlinks[:10]}")


def verify_qwen_shards(output_dir: Path) -> None:
    index = output_dir / "model.safetensors.index.json"
    if not index.is_file():
        raise RuntimeError(f"Missing Qwen safetensors index: {index}")

    data = json.loads(index.read_text(encoding="utf-8"))
    shards = sorted(set(data.get("weight_map", {}).values()))
    expected_shards = sorted(
        filename for filename in QWEN3_8B_FILES if filename.endswith(".safetensors")
    )
    if shards != expected_shards:
        raise RuntimeError(
            "Qwen safetensors shard list does not match expected files: "
            f"index={shards}, expected={expected_shards}"
        )

    missing = [name for name in shards if not (output_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing Qwen model shards referenced by index: {missing}")

    print(f"Verified Qwen safetensors shards: {', '.join(shards)}", flush=True)


def download_files(
    model_id: str,
    revision: str,
    output_dir: Path,
    files: list[str],
    cache_dir: Path,
    attempts: int,
    initial_backoff: float,
    stall_timeout: float,
    stall_min_bytes: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_local_hf_metadata(output_dir)
    metadata = fetch_repo_file_metadata(model_id, revision, files)

    print(
        f"Staging {model_id}@{revision} into {output_dir} "
        f"using cache {cache_dir}",
        flush=True,
    )

    for filename in files:
        repo_file = metadata[filename]
        destination = output_dir / filename
        if valid_local_file(destination, repo_file.size):
            print(
                f"[{model_id}] {filename}: already staged "
                f"({format_bytes(repo_file.size)})",
                flush=True,
            )
            continue

        cached_path = download_with_retry(
            model_id=model_id,
            revision=revision,
            repo_file=repo_file,
            cache_dir=cache_dir,
            attempts=attempts,
            initial_backoff=initial_backoff,
            stall_timeout=stall_timeout,
            stall_min_bytes=stall_min_bytes,
        )
        copy_if_needed(cached_path, destination, repo_file.size)

    verify_staged_files(model_id, output_dir, files, metadata)
    remove_local_hf_metadata(output_dir)


def download_thai_safety(
    model_id: str,
    revision: str,
    output_dir: Path,
    cache_dir: Path,
    attempts: int,
    initial_backoff: float,
    stall_timeout: float,
    stall_min_bytes: int,
) -> None:
    download_files(
        model_id=model_id,
        revision=revision,
        output_dir=output_dir,
        files=THAI_SAFETY_FILES,
        cache_dir=cache_dir,
        attempts=attempts,
        initial_backoff=initial_backoff,
        stall_timeout=stall_timeout,
        stall_min_bytes=stall_min_bytes,
    )


def download_qwen(
    model_id: str,
    revision: str,
    output_dir: Path,
    cache_dir: Path,
    attempts: int,
    initial_backoff: float,
    stall_timeout: float,
    stall_min_bytes: int,
) -> None:
    download_files(
        model_id=model_id,
        revision=revision,
        output_dir=output_dir,
        files=QWEN3_8B_FILES,
        cache_dir=cache_dir,
        attempts=attempts,
        initial_backoff=initial_backoff,
        stall_timeout=stall_timeout,
        stall_min_bytes=stall_min_bytes,
    )
    verify_qwen_shards(output_dir)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-model-id", default="Qwen/Qwen3-8B")
    parser.add_argument("--qwen-revision", default="main")
    parser.add_argument("--qwen-output-dir", default="/model/weights/Qwen3-8B")
    parser.add_argument("--thai-safety-model-id", default="typhoon-ai/ThaiSafetyClassifier")
    parser.add_argument("--thai-safety-revision", default="main")
    parser.add_argument("--thai-safety-output-dir", default="/models/thai-safety-classifier")
    parser.add_argument("--cache-dir", default=str(default_cache_dir()))
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--initial-backoff", type=float, default=10.0)
    parser.add_argument("--stall-timeout", type=float, default=300.0)
    parser.add_argument("--stall-min-bytes", type=int, default=1024 * 1024)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)

    download_qwen(
        model_id=args.qwen_model_id,
        revision=args.qwen_revision,
        output_dir=Path(args.qwen_output_dir),
        cache_dir=cache_dir,
        attempts=args.attempts,
        initial_backoff=args.initial_backoff,
        stall_timeout=args.stall_timeout,
        stall_min_bytes=args.stall_min_bytes,
    )
    print(f"Qwen model staged at {args.qwen_output_dir}", flush=True)

    download_thai_safety(
        model_id=args.thai_safety_model_id,
        revision=args.thai_safety_revision,
        output_dir=Path(args.thai_safety_output_dir),
        cache_dir=cache_dir,
        attempts=args.attempts,
        initial_backoff=args.initial_backoff,
        stall_timeout=args.stall_timeout,
        stall_min_bytes=args.stall_min_bytes,
    )
    print(f"ThaiSafetyClassifier staged at {args.thai_safety_output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
