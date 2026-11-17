#!/usr/bin/env sh
# Downloads the two ONNX models from the OpenCV Zoo into static/models/.
# Used as part of the Render build command so the models are never committed to your repo.
# Licenses (verify yourself before use): YuNet = MIT, SFace = Apache-2.0 (see the opencv_zoo repository).
set -eu
DEST="$(dirname "$0")/../static/models"
BASE="https://github.com/opencv/opencv_zoo/raw/main/models"
mkdir -p "$DEST"

fetch() {  # url file min_bytes
  if [ -f "$DEST/$2" ] && [ "$(wc -c < "$DEST/$2")" -ge "$3" ]; then echo "$2 already present"; return; fi
  echo "Downloading $2 ..."
  curl -fL --retry 3 -o "$DEST/$2" "$1"
  size=$(wc -c < "$DEST/$2")
  if [ "$size" -lt "$3" ]; then echo "ERROR: $2 is only $size bytes (expected >= $3). Is it a Git LFS pointer or an error page?" >&2; exit 1; fi
}

fetch "$BASE/face_detection_yunet/face_detection_yunet_2023mar.onnx" face_detection_yunet_2023mar.onnx 100000
fetch "$BASE/face_recognition_sface/face_recognition_sface_2021dec.onnx" face_recognition_sface_2021dec.onnx 20000000
echo "Models ready in $DEST"
