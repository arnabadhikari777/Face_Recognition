# Model files (NOT bundled)

This folder must contain exactly these two files at runtime. They are downloaded by
`scripts/download_models.sh` (run automatically in the Render build command).

| File | Purpose | Source | Licence (verify) |
|---|---|---|---|
| `face_detection_yunet_2023mar.onnx` | face detection + 5 landmarks (about 0.2 MB) | OpenCV Zoo `models/face_detection_yunet` | MIT |
| `face_recognition_sface_2021dec.onnx` | 128-d face embedding (about 37 MB) | OpenCV Zoo `models/face_recognition_sface` | Apache-2.0 |

Do NOT substitute the `_int8` / `_int8bq` variants or any other model: embeddings from different
models or quantisations are not comparable. If you change a model, change `MODEL_NAME` /
`MODEL_VERSION` (config.py and static/js/face-engine.js) so old records are ignored, and
re-register everyone.

Manual download: `sh scripts/download_models.sh`. Check each LICENSE file in the OpenCV Zoo
repository yourself; the training data licence of any face model is a separate question.
