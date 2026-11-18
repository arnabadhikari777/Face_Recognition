# Face Recognition (public, browser-side AI)

A single-page website with two functions: **Register Face** and **Recognize Face**.
Face detection and embedding run **in the visitor's browser** (ONNX Runtime Web). The Flask
server only validates requests and compares 128-number embeddings stored in MongoDB Atlas.
No login, no accounts, no images stored.

> **Read "Privacy and legal" and "Known limitations" before putting this on the public internet.**

## About this merged version
Two earlier drafts of this project were merged. The result is based on the draft with the real
pipeline (YuNet detector + SFace recogniser with landmark alignment, 128-d embeddings, strict
validation, tests). From the other draft I kept the oval face-guide overlay idea; its
512-d model contract, centre-crop fallback and unverified detector parsing were dropped because they
could silently produce bad embeddings. Embeddings are **not** compatible between the two drafts.

## Architecture

```
Browser (phone / desktop)                      Render Free (Flask + Gunicorn)        MongoDB Atlas M0
 camera -> YuNet detector -> 5 landmarks        POST /api/register-face               registered_faces
 -> align to 112x112 -> SFace -> 128-d unit     POST /api/recognize-face   <------->   {name, embeddings[3],
 vector (ONNX Runtime Web, WASM; WebGPU         GET  /api/health, /api/config          model_name, model_version,
 only if it works)  --- embeddings only --->    cosine matching in numpy               created_at}
```

* Images never leave the device; only the name (registration) and embeddings are sent.
* The server loads no AI model (numpy only), so it fits in Render Free RAM.
* The database is reachable only from Flask; the browser never sees credentials.

## Folder structure

```
app.py config.py requirements.txt Procfile render.yaml .python-version .env.example .gitignore
README.md DEPLOYMENT.md DEPLOYMENT_BN.md (Bengali guide)
backend/   __init__.py (app factory, headers, errors) database.py registration.py
           face_matching.py validation.py routes.py
templates/index.html
static/css/style.css
static/js/ app.js camera.js api.js face-engine.js registration.js recognition.js
static/models/README.md        (model files are downloaded here, see below)
scripts/download_models.sh
tests/     test_api.py test_validation.py test_matching.py test_config.py fakes.py test_face_engine.mjs
```
Additions to the suggested layout: `app.js` (page controller), `api.js` (fetch helpers),
`scripts/download_models.sh`, `tests/fakes.py`, `tests/test_face_engine.mjs`. `runtime.txt` is
replaced by `.python-version` because Render reads that file (Heroku uses runtime.txt).

## Models (chosen, not bundled)

| | Detector | Recognizer |
|---|---|---|
| Model | YuNet `face_detection_yunet_2023mar.onnx` | SFace `face_recognition_sface_2021dec.onnx` |
| Size | about 0.2 MB | about 37 MB |
| Output | boxes + 5 landmarks | 128-d embedding |
| Licence | MIT | Apache-2.0 |
| Source | github.com/opencv/opencv_zoo | github.com/opencv/opencv_zoo |

Why: both are small, run on ONNX Runtime Web's WASM backend, and carry permissive code/weights
licences. Larger models (ArcFace/InsightFace weights) were rejected because their pretrained
weights are restricted to non-commercial use. **I could not open the licence files from my
sandbox: check the LICENSE in each OpenCV Zoo model folder yourself, and note that the licence on
the weights does not settle questions about the training data.** Files are fetched at build time
by `scripts/download_models.sh`, so you do not redistribute them.

### Preprocessing pipeline (identical for register and recognize)

1. Video frame, letterboxed into a square (640, falling back to 320) for YuNet.
2. YuNet input: **BGR**, raw 0..255 floats, NCHW, no mean/std. Outputs `cls_/obj_/bbox_/kps_` at strides 8/16/32; score = sqrt(cls*obj) >= 0.7; NMS IoU 0.3.
3. Exactly one face required. Quality gates: size, position, pose, brightness, contrast, sharpness (values in `ENGINE` in `face-engine.js`; engineering defaults, not calibrated).
4. Alignment: least-squares similarity transform of the 5 landmarks onto the standard ArcFace 112x112 template, applied with a canvas transform.
5. SFace input: **RGB**, raw 0..255 floats, NCHW, 112x112. Output 128 floats, then L2-normalised.
6. Similarity = cosine = dot product of unit vectors.

Steps 2 and 5 follow OpenCV's reference implementation as I remember it and **have not been
checked against real model files**. If matches look random on first run, check channel order
(RGB vs BGR) and the detector output names first.

## Matching, thresholds and aggregation

| Setting | Default | Meaning |
|---|---|---|
| `MATCH_THRESHOLD` | 0.363 | Cosine threshold published by OpenCV for SFace. A starting point only. |
| `UNCERTAIN_LOW` | 0.30 | Scores in [0.30, 0.363) are "uncertain", below is "unknown". |
| `MATCH_MARGIN` | 0.05 | Best candidate must beat the runner-up by this much, otherwise "uncertain". |
| `MIN_SUPPORTING_SAMPLES` | 2 | Duplicate check: at least 2 of the 3 new samples must individually reach the threshold. |
| `SELF_CONSISTENCY_MIN` | 0.15 | The 3 registration samples must resemble each other (rejects garbage input). |

Aggregation: each query sample is compared with all stored samples of a person and the best
similarity is kept; the person's score is the mean over query samples. The score is a cosine
similarity, **not** a probability that the identification is correct. **These numbers are
uncalibrated for your cameras and users.** Before trusting results, collect consented test
pairs (same person / different people) on your target devices, plot the score distributions and
set the thresholds for the false-match rate you can accept.

## Environment variables

| Variable | Default | Notes |
|---|---|---|
| `MONGODB_URI` | (none, required) | Atlas connection string. Secret. |
| `MONGODB_DB_NAME` | `face_recognition_db` | |
| `MONGODB_COLLECTION` | `registered_faces` | |
| `MATCH_THRESHOLD`, `UNCERTAIN_LOW`, `MATCH_MARGIN`, `MIN_SUPPORTING_SAMPLES`, `SELF_CONSISTENCY_MIN` | see above | |
| `RETENTION_DAYS` | `90` | Records auto-deleted by a MongoDB TTL index (runs about every 60 s). |
| `MAX_RECORDS` | `5000` | Registration closes at this count (protects free tier). |
| `TRUST_PROXY` | `false` | Set `true` on Render so rate limits use the real client IP. |
| `ALLOWED_ORIGINS` | empty | Comma list for CORS. Empty = same-origin only. |
| `MAX_CONTENT_LENGTH` | `65536` | Request size limit in bytes. |
| `RATE_REGISTER_PER_HOUR` / `RATE_RECOGNIZE_PER_MINUTE` / `RATE_DEFAULT_PER_MINUTE` | 10 / 30 / 60 | Per client IP. |
| `PYTHON_VERSION` | (Render) | e.g. `3.12.3`; `.python-version` also provided. |

## Local run

```
python -m venv .venv && . .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
sh scripts/download_models.sh
cp .env.example .env     # then put your real MONGODB_URI into .env (app.py reads it for local runs; real environment variables win)
python app.py            # http://127.0.0.1:5000 ; localhost counts as a secure context for the camera
```
Phones need HTTPS for the camera, so test on Android via the deployed Render URL (see DEPLOYMENT.md).

Run the tests (Python 3.10+, Node 20+; no pip packages beyond Flask and numpy needed):
```
python -m unittest discover -s tests -t . -v
node --test tests/test_face_engine.mjs
```

## Test results (what was actually executed, 17 Nov 2026, in a sandbox without internet)

| Area | Item | Status |
|---|---|---|
| Backend | Flask app startup, `/`, static files, `/api/config` (dev server + curl) | PASSED |
| Backend | Health endpoint, ok and database-down (fake repository) | PASSED |
| Backend | Valid registration; invalid name/consent/model/dimension/count | PASSED |
| Backend | Duplicate registration; recognition match; unknown face | PASSED |
| Backend | NaN / Infinity / 1e999 / non-number / non-normalised embeddings rejected | PASSED |
| Backend | Rate limiting (429 + Retry-After), size limit (413), CORS allow-list, hidden 500 details | PASSED |
| Backend | Matching thresholds, ambiguity, duplicate aggregation (synthetic unit vectors) | PASSED |
| Backend | 34 Python unit tests total (incl. .env loader) | PASSED (34/34) |
| Frontend | Alignment math, YuNet output decoding (synthetic), NMS, blur/brightness/pose logic | PASSED (7/7 node tests) |
| Frontend | JavaScript syntax of all modules (`node --check`) | PASSED |
| Backend | Real MongoDB Atlas connection, indexes, TTL, `MongoRepository` class | NOT TESTED (pymongo and network unavailable) |
| Backend | Gunicorn start, Render deployment, HTTPS headers on Render | NOT TESTED |
| Model | Model download script, model loading, WASM inference, WebGPU and its fallback | NOT TESTED |
| Model | Matching consistency and threshold behaviour on real faces | NOT TESTED |
| Frontend | Camera permission, denial, no camera, camera switch, camera cleanup | NOT TESTED |
| Frontend | Three-angle registration, recognition workflow in a browser | NOT TESTED |
| Frontend | Mobile layout and Android Chrome behaviour, API failure UI | NOT TESTED |

Treat the project as **unverified end to end** until the NOT TESTED rows are done on a real device.

## Privacy and legal

* Face embeddings are biometric data. GDPR (EU/UK), BIPA (Illinois) and similar laws can require
  explicit written consent, a published retention policy and a way to delete data. A consent
  checkbox does **not** prove consent. I am not a lawyer: get legal advice before public launch.
* By the specification, a duplicate registration returns the stored name and recognition returns
  the name of a match. **Anyone on the internet can therefore test whether a face is registered
  and learn the name.** Consider whether that is acceptable; it is the biggest privacy risk here.
* No covert use: do not point the camera at people who do not know.
* Retention: automatic deletion after `RETENTION_DAYS`; owner-run deletion is in DEPLOYMENT.md (section L).

## Known limitations

* No liveness / anti-spoofing: a photo or screen of a registered person can match.
* Accuracy varies across lighting, cameras, ages and demographic groups; the thresholds are uncalibrated.
* Anyone can register, limited only by per-IP rate limits and `MAX_RECORDS`; there is no abuse moderation.
* The rate limiter is in process memory, so it resets on restart and only works correctly with one Gunicorn worker (as configured).
* Recognition scans all records (fine for thousands, not millions). Concurrent duplicate registrations can race.
* ONNX Runtime Web is loaded from jsDelivr (pinned 1.17.3, no SRI hash because I could not compute one offline). Self-host it for stricter security.
* WebGPU is switched off by default (`ENGINE.useWebGPU` in `face-engine.js`) because it can fail at run time on some devices; the WASM backend is used.
* iOS Safari and Firefox behaviour is untested.
* Render Free: sleeps after about 15 minutes idle (first request can take about a minute), limited RAM/CPU, no persistent disk, no static outbound IP.
* The first visit downloads about 37 MB of model data; browsers cache it afterwards.
