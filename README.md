# Face Recognition Web Application

Public, single-page face registration and recognition site.

- **Client-side AI:** face detection & embedding run in the visitor’s browser (ONNX Runtime Web, WASM primary, WebGPU optional).
- **Backend:** Flask API on Render Free — stores and compares embeddings only.
- **Database:** MongoDB Atlas Free.
- **No accounts, no passwords, no image gallery.**

## Features

1. **Register Face** — name + informed consent + 3 pose samples → embeddings saved.
2. **Recognize Face** — consent + one capture → best match or unknown/uncertain.
3. Duplicate detection via configurable cosine similarity thresholds.
4. Mobile-friendly (Android Chrome + desktop).

## Quick start (local)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# set MONGODB_URI
# download ONNX models into static/models/  (see DEPLOYMENT.md)
python app.py
```

## Project structure

```
face-recognition/
├── app.py
├── config.py
├── requirements.txt
├── Procfile
├── runtime.txt
├── .env.example
├── .gitignore
├── README.md
├── DEPLOYMENT.md
├── backend/
│   ├── database.py
│   ├── face_matching.py
│   ├── registration.py
│   └── routes.py
├── templates/
│   └── index.html
└── static/
    ├── css/style.css
    ├── js/
    │   ├── camera.js
    │   ├── face-engine.js
    │   ├── registration.js
    │   └── recognition.js
    └── models/          # place .onnx files here
```

## Important

- Similarity scores are **not** identity probabilities.
- Results are never 100% accurate.
- Informed consent is required before capture.
- Only embeddings (not camera images) are stored.
- See **DEPLOYMENT.md** for Atlas, model download, Render, and deletion procedures.

## License / models

Application code is provided as-is for this project. Face model weights have their own licenses — verify before redistribution.
