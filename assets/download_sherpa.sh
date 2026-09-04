#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="${ROOT}/models"
mkdir -p "${MODELS_DIR}"
cd "${MODELS_DIR}"

MODEL_NAME="sherpa-onnx-streaming-zipformer-en-2023-06-26"
URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${MODEL_NAME}.tar.bz2"

if [[ -d "${MODEL_NAME}" ]]; then
  echo "Model exists: ${MODELS_DIR}/${MODEL_NAME}"
  exit 0
fi

echo "Downloading ${MODEL_NAME} …"
wget -q --show-progress -O "${MODEL_NAME}.tar.bz2" "${URL}"
tar xvf "${MODEL_NAME}.tar.bz2"
rm -f "${MODEL_NAME}.tar.bz2"

echo
echo "Model ready: ${MODELS_DIR}/${MODEL_NAME}"
echo
echo "  ./build/stt_mic \\"
echo "      --tokens  models/${MODEL_NAME}/tokens.txt \\"
echo "      --encoder models/${MODEL_NAME}/encoder-epoch-99-avg-1-chunk-16-left-128.onnx \\"
echo "      --decoder models/${MODEL_NAME}/decoder-epoch-99-avg-1-chunk-16-left-128.onnx \\"
echo "      --joiner  models/${MODEL_NAME}/joiner-epoch-99-avg-1-chunk-16-left-128.onnx"