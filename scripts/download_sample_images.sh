#!/usr/bin/env bash
# Download a few sample images for benchmarking / testing.
set -euo pipefail

OUT_DIR="benchmarks/sample_payloads"
mkdir -p "$OUT_DIR"

echo "Downloading sample images..."

# ImageNet-like public domain images
curl -sL "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg" \
  -o "$OUT_DIR/dog.jpg"
curl -sL "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg" \
  -o "$OUT_DIR/cat.jpg"
curl -sL "https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Dog_Breeds.jpg/320px-Dog_Breeds.jpg" \
  -o "$OUT_DIR/dog2.jpg" 2>/dev/null || true

echo "Sample images downloaded to $OUT_DIR/"
ls -lh "$OUT_DIR/"
