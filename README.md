# Face Recognition — multi-model, on-device, hybrid templates

Mobile-first web app. **All neural-network inference runs in the visitor's browser** (ONNX Runtime Web, in a Web Worker). Render only runs
Flask + template matching; MongoDB Atlas stores encrypted numeric templates. Raw images and video never leave the device.

* Detectors: SCRFD 500M (primary), YuNet (fallback / light profile). Recognizers: MobileFaceNet `w600k_mbf` (primary), IResNet-50 `w600k_r50` (second opinion). Different models are never mixed: separate templates, separate masters, separate calibration, score-level fusion.
* Registration: 6–9 quality-selected templates (front / left / right / optional up / down), duplicate + outlier rejection, per-model master template and pose prototypes.
* Recognition: multi-frame, multi-template, calibrated z-score fusion, margin to runner-up, model-disagreement detection, lazy second-model verification → `MATCH` / `UNCERTAIN` / `UNKNOWN` (or "retry" for poor quality). Similarity is a cosine score, never a probability.
* Profiles: Accuracy (default), Balanced, Compatibility. Models are downloaded automatically at build time from `models/model_manifest.json` with SHA-256 verification.

## Honest status
* Real-world accuracy is **not measured**: no consented evaluation dataset was available. Thresholds in `models/calibration.json` are provisional (`calibrated:false`). `tools/benchmark.py` + `tools/calibrate.py` are provided to do it on your data.
* The browser code (Web Worker, WebGPU fallback, Cache API, camera, UI) was **not run in a real browser or phone**. The identical engine core was run in Node with the real ONNX models and compared with an independent cv2/onnxruntime reference.
* Some downloaded model weights may be restricted to non-commercial research use. Review each model's license before deployment.

## Render
Build: `pip install -r requirements.txt && sh scripts/download_models.sh`
Start: `gunicorn app:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT`

## Layout
`app.py`, `config.py`, `backend/` (manifest, crypto, templates, face_matching, database, registration, routes, validation), `models/` (manifest + calibration), `scripts/` (download_models, migrate_db), `static/js/engine/` (browser engine), `static/js/*.js` (UI flows), `tools/` (benchmark, calibrate, metrics), `tests/`.
