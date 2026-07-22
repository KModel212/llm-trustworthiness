# syntax=docker/dockerfile:1.7
FROM vllm/vllm-openai:v0.23.0

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface
ENV HF_HUB_DOWNLOAD_TIMEOUT=60
ENV HF_HUB_ETAG_TIMEOUT=30
ARG MODEL_NAME=Qwen3-8B
ARG QWEN_MODEL_ID=Qwen/Qwen3-8B
ARG QWEN_REVISION=main
ARG THAI_SAFETY_MODEL_ID=typhoon-ai/ThaiSafetyClassifier
ARG THAI_SAFETY_REVISION=main
ENV MODEL_PATH=/opt/weights/${MODEL_NAME}
ENV INPUT_PATH=/model/test/dataset.csv
ENV OUTPUT_PATH=/result/submission.csv
ENV THAI_SAFETY_MODEL_PATH=/models/thai-safety-classifier

WORKDIR /app

COPY scripts/download_models.py ./scripts/download_models.py
RUN python3 -m pip install --no-cache-dir "huggingface_hub>=1.5.0,<2.0"
RUN --mount=type=cache,target=/root/.cache/huggingface,sharing=locked \
    HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python3 scripts/download_models.py \
    --qwen-model-id ${QWEN_MODEL_ID} \
    --qwen-revision ${QWEN_REVISION} \
    --qwen-output-dir ${MODEL_PATH} \
    --thai-safety-model-id ${THAI_SAFETY_MODEL_ID} \
    --thai-safety-revision ${THAI_SAFETY_REVISION} \
    --thai-safety-output-dir ${THAI_SAFETY_MODEL_PATH}

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

COPY config.py main.py model.py README.md ./
COPY guardrail/ ./guardrail/
COPY prompts/ ./prompts/
COPY utils/ ./utils/
COPY scripts/ ./scripts/

ENTRYPOINT ["python3", "main.py"]
