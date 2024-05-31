#!/bin/bash
set -x

HOME_DIR=""
GPU=${1}
PORT=${2}
MODEL=${3}
GPU_NUMS=1

echo ${PORT}

CUDA_VISIBLE_DEVICES="${GPU}" python -O -u -m vllm.entrypoints.api_server \
  --host=0.0.0.0 \
  --port=${PORT} \
  --model="${MODEL}" \
  --tokenizer="hf-internal-testing/llama-tokenizer" \
  --dtype="bfloat16" \
  --tensor-parallel-size=${GPU_NUMS}