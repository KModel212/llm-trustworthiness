# AI Thailand Benchmark 2026 - LLM Trustworthiness Challenge

This repository contains an offline, high-throughput inference pipeline for evaluating open-weight instruct models on trustworthiness benchmarks.

## Architecture Highlights
- **Local Offline Inference**: Strictly loads models from a local directory. Never attempts network downloads.
- **High Throughput**: Uses `vLLM` for highly optimized, batched deterministic text generation.
- **Multi-Stage Risk Assessment**: Implements a sophisticated guardrail pipeline:
  - Normalization (Unicode, Zero-width, URL/Base64/ROT13 decoding, L33t speak)
  - Prompt Injection Detection
  - Jailbreak Detection
  - Intent Classification
  - Multi-level Risk Scoring (SAFE, LOW_RISK, MEDIUM_RISK, HIGH_RISK)
- **Response Validation**: Post-generation filtering for CoT leakage, prompt leakage, and unsafe outputs.

## Run with Docker

The evaluator expects everything to run entirely offline via Docker.

### 1. Build the Docker Image
```bash
docker build -t aith-trustworthy-pipeline .
```

### 2. Run the Container
The submission runner reads the input dataset from `/model/test/dataset.csv`
and writes the final CSV to `/result/submission.csv`.

Input format:
```csv
id,query
q000,...
```

Output format:
```csv
id,response
q000,...
```

```bash
docker run --gpus all \
  -e MODEL_PATH=/model/weights/Qwen3-8B \
  -e INPUT_PATH=/model/test/dataset.csv \
  -e OUTPUT_PATH=/result/submission.csv \
  -v /local/path/to/dataset.csv:/model/test/dataset.csv:ro \
  -v /local/path/to/result:/result \
  llm-trustworthy:test-v3
```

The container automatically executes the pipeline and writes the evaluation results.
