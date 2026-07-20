import argparse
import json
import os
from pathlib import Path


def verify_main_model(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"MODEL_PATH missing: {path}")
    config = path / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"Main model config missing: {config}")
    index = path / "model.safetensors.index.json"
    if index.is_file():
        data = json.loads(index.read_text(encoding="utf-8"))
        shards = sorted(set(data["weight_map"].values()))
        missing = [s for s in shards if not (path / s).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing main model shards: {missing}")


def verify_thai_safety(path: Path) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        path,
        local_files_only=True,
    )
    model.eval()

    text = "input: สวัสดี output: สวัสดีครับ"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with __import__("torch").no_grad():
        logits = model(**inputs).logits
    if logits.shape[-1] != 2:
        raise RuntimeError(f"Unexpected ThaiSafetyClassifier logits shape: {logits.shape}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-model-path", default=os.environ.get("MODEL_PATH", "/opt/weights/Qwen3-8B"))
    parser.add_argument("--thai-safety-path", default=os.environ.get("THAI_SAFETY_MODEL_PATH", "/models/thai-safety-classifier"))
    args = parser.parse_args()

    verify_main_model(Path(args.main_model_path))
    verify_thai_safety(Path(args.thai_safety_path))
    print("Offline model verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
