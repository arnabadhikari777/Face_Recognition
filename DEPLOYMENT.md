# Deployment Guide — Face Recognition Web App

This document covers MongoDB Atlas, model files, GitHub, and Render Free deployment.

---

## A. Create a MongoDB Atlas Free Cluster

1. Go to https://www.mongodb.com/cloud/atlas and sign up / log in.
2. Create a **Free (M0)** cluster. Choose a region close to your users.
3. Wait until the cluster status is **Available**.

---

## B. Create a Database User

1. In Atlas → **Database Access** → **Add New Database User**.
2. Authentication: Password.
3. Username + strong password (save both).
4. Privileges: **Read and write to any database** (or restrict to `face_recognition_db`).
5. Create user.

---

## C. Configure Secure Network Access

1. Atlas → **Network Access** → **Add IP Address**.
2. For Render (dynamic IPs) you typically need **Allow Access from Anywhere** (`0.0.0.0/0`).
   - This is common for free serverless/PaaS hosts. Prefer restricting later if you add a static outbound IP.
3. Confirm.

**Note:** Do not leave the cluster without authentication. Always use a strong DB user password.

---

## D. Obtain the MongoDB Connection String

1. Atlas → **Database** → **Connect** → **Drivers**.
2. Copy the SRV string, e.g.:
   ```
   mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
   ```
3. Replace `<username>` and `<password>` (URL-encode special characters in the password).
4. Optional: append database name: `...mongodb.net/face_recognition_db?retryWrites=true&w=majority`

---

## E. Model Files (Required)

Face detection and embedding run **in the browser** with ONNX Runtime Web.

Place at least a **recognizer** ONNX under `static/models/`:

| File (any of these names works) | Role | Typical size |
|---------------------------------|------|--------------|
| `w600k_mbf.onnx` / `mobilefacenet.onnx` | 512-d embedding (preferred) | ~4–14 MB |
| `det_10g.onnx` / `ultraface-rfb-320.onnx` | Optional detector | ~1–5 MB |

### Recommended sources (verify license before redistribution)

- InsightFace model zoo (buffalo_s / MobileFaceNet `w600k_mbf`):  
  https://github.com/deepinsight/insightface/tree/master/model_zoo  
  Official packs are released on GitHub Releases / Google Drive links listed in the InsightFace wiki.
- EdgeFace-XXS / XS (Apache-friendly small models):  
  https://github.com/yakhyo/edgeface-onnx

**License:** Confirm redistribution rights for the exact file you download. If the license does not allow bundling, keep models out of the public repo and document the download step for deployers.

After download:

```bash
mkdir -p static/models
# copy your .onnx files into static/models/
```

Preprocessing used by the app (must match training of the model):

- Input: 112×112 RGB face crop  
- Normalize: `(pixel - 127.5) / 127.5`  
- Output: L2-normalized embedding (512-d for MobileFaceNet ArcFace)

Without a recognizer model the UI will show a clear error asking for the ONNX files.

---

## F. Upload the Project to GitHub

1. Create a new **public** repository on GitHub (no README if you already have one locally).
2. Locally:

```bash
cd face-recognition
git init
git add .
git commit -m "Initial face recognition web app"
git branch -M main
git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
git push -u origin main
```

**Never commit** `.env`, real credentials, or real biometric data.

---

## G. Connect GitHub to Render

1. Sign up at https://render.com.
2. **New** → **Web Service**.
3. Connect your GitHub account and select the repository.
4. Configure:
   - **Name:** e.g. `face-recognition`
   - **Runtime:** Python
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** (leave empty if `Procfile` is present; Render uses it)
   - **Instance type:** Free

---

## H. Deploy the Flask Application

1. In Render → **Environment** add:

| Key | Value |
|-----|--------|
| `MONGODB_URI` | your Atlas connection string |
| `MONGODB_DB` | `face_recognition_db` |
| `SECRET_KEY` | long random string |
| `CORS_ORIGINS` | your Render URL, or `*` for testing |
| `DUPLICATE_THRESHOLD` | `0.55` (optional) |
| `RECOGNITION_THRESHOLD` | `0.48` (optional) |

2. Deploy. Wait for the first build to finish.
3. Open the service URL.

**Free tier notes:**

- Service sleeps after inactivity → first request after sleep has a cold start (30–60+ seconds).
- RAM is limited (~512 MB). This app does **not** load AI models on the server.
- No persistent local disk — all face records live in MongoDB Atlas.

---

## I. Test the Deployed Website

1. Open `https://YOUR-SERVICE.onrender.com`.
2. Call health: `https://YOUR-SERVICE.onrender.com/api/health`  
   Expect `"database": "connected"`.
3. Register a face (after placing models) and then recognize it.
4. Try duplicate registration — should return a 409-style message.

---

## J. Camera Permission on Android Chrome

1. Use **HTTPS** (Render provides HTTPS).
2. Tap the lock / site settings → allow Camera.
3. If permission was denied earlier, reset site permissions in Chrome settings.
4. Prefer the front camera for selfies; use **Switch Camera** for rear.

---

## K. Troubleshooting

| Issue | What to check |
|-------|----------------|
| Health shows database unreachable | `MONGODB_URI`, Atlas user password, Network Access `0.0.0.0/0` |
| Models not loading | Files under `static/models/`, correct `.onnx` names, browser console |
| CORS errors | `CORS_ORIGINS` includes the frontend origin |
| 429 Too many requests | Rate limit settings; wait 60 seconds |
| Cold start | Free tier sleep; wait and retry |
| Camera fails on mobile | HTTPS required; permission granted; no other app locking camera |

---

## L. Owner-Authorized Data Deletion

There is **no public Delete button** (by design).

To remove a registration:

1. Log in to MongoDB Atlas → Browse Collections → `face_recognition_db` → `registered_faces`.
2. Find the document by `name` or `_id`.
3. Delete the document.

Or from a secure shell with the same URI:

```python
from pymongo import MongoClient
from bson import ObjectId
client = MongoClient("YOUR_URI")
col = client["face_recognition_db"]["registered_faces"]
col.delete_one({"_id": ObjectId("...")})
# or
col.delete_many({"name": "Person Name"})
```

Implement retention policies according to your local privacy rules (biometric data).

---

## Local Development

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env with a local or Atlas URI
python app.py
```

Open http://127.0.0.1:5000

For production-like run:

```bash
gunicorn app:app --bind 0.0.0.0:5000
```

---

## Environment Variables Summary

| Variable | Required | Description |
|----------|----------|-------------|
| `MONGODB_URI` | Yes | Atlas connection string |
| `MONGODB_DB` | No | Default `face_recognition_db` |
| `SECRET_KEY` | Yes (prod) | Flask secret |
| `CORS_ORIGINS` | No | `*` or comma-separated origins |
| `DUPLICATE_THRESHOLD` | No | Cosine threshold for duplicate (default 0.55) |
| `RECOGNITION_THRESHOLD` | No | Cosine threshold for match (default 0.48) |
| `UNCERTAIN_THRESHOLD` | No | Below this → unknown (default 0.40) |
| `RATE_LIMIT_REQUESTS` | No | Default 30 |
| `RATE_LIMIT_WINDOW_SECONDS` | No | Default 60 |
| `FLASK_DEBUG` | No | `1` only for local debug |
| `PORT` | Auto | Set by Render |

---

## Architecture (short)

- **Browser:** Camera → ONNX Runtime Web (WASM / optional WebGPU) → face embedding.
- **Server (Render):** Flask + Gunicorn — validate, store/compare embeddings in MongoDB only.
- **Database:** MongoDB Atlas free cluster — stores name + embeddings + model metadata only (no images).
