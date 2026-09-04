#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$ROOT/models"
mkdir -p "$MODELS"

URL="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.onnx"

echo "Downloading $MODELS/yolo11n.onnx"
curl -L --fail -o "$MODELS/yolo11n.onnx" "$URL"
echo "Done. Size: $(du -h "$MODELS/yolo11n.onnx" | cut -f1)"
