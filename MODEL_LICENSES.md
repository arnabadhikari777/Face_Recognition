# Model licences (read before deploying)

| Model | File | Source | Licence | Commercial use? |
|---|---|---|---|---|
| YuNet 2023mar | `face_detection_yunet_2023mar.onnx` | OpenCV Zoo | MIT | Yes |
| SCRFD 500M | `det_500m.onnx` (buffalo_s.zip) | InsightFace release v0.7 | InsightFace pretrained models: **non-commercial research only** | **No** |
| MobileFaceNet w600k_mbf | `w600k_mbf.onnx` (buffalo_s.zip) | InsightFace release v0.7 | same | **No** |
| IResNet-50 w600k_r50 | `w600k_r50.onnx` (buffalo_l.zip) | InsightFace release v0.7 | same | **No** |

* The InsightFace **code** is MIT, but its README states that the trained models (and the training data) are for non-commercial research purposes only.
  I did not find a licence that grants commercial or public-service use of these weights. Check <https://github.com/deepinsight/insightface#license> yourself; for a commercial product ask the InsightFace authors for a licence or train/choose other weights.
* Because of that, `scripts/download_models.sh` **refuses** to download them unless `ACCEPT_NONCOMMERCIAL_MODEL_LICENSES=1` is set. `render.yaml` sets it to `1`, which is your statement that this deployment is a personal / non-commercial research project. Change it if that is not true.
* The training data (WebFace600K) has its own terms; face-recognition law (consent, biometric data) is separate again.
* Redistribution: the models are NOT stored in the GitHub repository; every deployment downloads them from the official release.

## Models evaluated but not integrated
* **OpenCV SFace** (Apache-2.0): was the previous model. Its file is hosted on Git LFS (`media.githubusercontent.com`); I could not download it in my build sandbox to verify its SHA-256, and I will not write an unverified checksum into the manifest. Embeddings are 128-d and in a different space from the new models. Can be added later with a verified hash.
* **RetinaFace, AdaFace, other ArcFace ONNX models**: no ONNX weights with a verified checksum and a clear redistribution licence were reachable from the build environment. Not downloaded, not bundled.
* **SCRFD 10G (`det_10g.onnx`)**, **2d106det landmarks**: present in the same InsightFace archives (same licence class) but not wired in; they were not benchmarked, so they are not enabled.
