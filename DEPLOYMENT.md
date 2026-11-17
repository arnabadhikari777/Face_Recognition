# Deployment guide (MongoDB Atlas Free + GitHub + Render Free)

Dashboard wording changes over time. If a button or menu is named differently, look for the
closest equivalent. Nothing here was executed against live Atlas/Render accounts (see README).

## A. Create a MongoDB Atlas free cluster
1. Sign up at mongodb.com/atlas and create an organisation and project.
2. Click **Create** (or **Build a Database**), choose the **M0 Free** tier, pick a nearby region, name it e.g. `face-cluster`.
3. Wait until the cluster is ready.

## B. Create a database user
1. **Database Access** -> **Add New Database User** -> authentication: Password.
2. Username e.g. `face_app`; click **Autogenerate Secure Password** and copy it somewhere safe (use a password manager, never GitHub).
3. Privileges: **Specific Privileges** -> role `readWrite` on database `face_recognition_db` only. Do not give admin roles.
   (Creating indexes needs `readWrite` plus index rights; `readWrite` includes `createIndex`/`collMod` on its own collections.)

## C. Network access
Render Free has no fixed outbound IP, so the only way to let it reach Atlas is to allow all addresses.
1. **Network Access** -> **Add IP Address** -> **Allow access from anywhere** (`0.0.0.0/0`).
2. This is an explicit, justified exception: access is still protected by TLS and the long random password of a least-privilege user.
   If you move to a host with a fixed IP (or a paid Render plan with static outbound IPs), replace this with that IP.

## D. Get the connection string
1. **Database** -> **Connect** -> **Drivers** (Python) -> copy the `mongodb+srv://...` string.
2. Replace `<db_user>` and `<db_password>` (URL-encode special characters in the password, e.g. `@` becomes `%40`).
3. The database and collection are created automatically on first write.

## E. Render environment variables (set in step H)
| Key | Value |
|---|---|
| `MONGODB_URI` | the string from D (mark as secret) |
| `TRUST_PROXY` | `true` |
| `PYTHON_VERSION` | `3.12.3` |
| `RETENTION_DAYS` | e.g. `90` (optional) |
Others are optional; see README.

## F. Upload the project to GitHub
```
cd face-recognition
git init
git add .
git status            # confirm .env is NOT listed and no *.onnx files are listed
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```
Create the empty public repository on github.com first (no README/licence/.gitignore). Never commit `.env`,
the connection string, or any real face images or embeddings. Add a LICENSE file of your choice for your own code.

## G. Connect GitHub to Render
1. render.com -> sign up with GitHub and authorise access to the repository.
2. **New +** -> **Web Service** -> select your repository.

## H. Deploy
Settings:
* Runtime: **Python 3**
* Build command: `pip install -r requirements.txt && sh scripts/download_models.sh`
* Start command: `gunicorn app:app --workers 1 --threads 4 --timeout 60 --bind 0.0.0.0:$PORT`
  (Render ignores `Procfile`; the file is only for hosts that read it.)
* Instance type: **Free**
* Health check path: `/api/health`
* Environment: add the variables from E.
* Click **Create Web Service**. The first build downloads the models (about 37 MB) from GitHub; if the build fails there, see K.

**Shortcut:** the repo contains `render.yaml`. In Render choose **New + -> Blueprint**, select the repo, and paste `MONGODB_URI` when asked.

Keep `--workers 1`: the rate limiter lives in process memory, and one worker keeps RAM low.

## I. Test the deployed site
1. `https://<your-service>.onrender.com/api/health` should return `{"database":"ok","status":"ok"}`. If the service was asleep, wait up to about a minute.
2. `.../api/config` should show `opencv-zoo-sface`, `2021dec`, `128`.
3. Open the home page, press **Register Face**, tick consent, enter a name, **Start Camera**; wait for the model download.
4. Capture three samples (front, left, right). Expect "Face registered successfully."
5. Register the same face again: expect "Face already registered" with the stored name and no new record.
6. **Recognize Face** with the same person: expect "Match Found" with a similarity score; with an unregistered person: "Unknown Face".
7. In Atlas **Browse Collections** confirm a document with name, 3 embeddings of 128 numbers, model fields and `created_at`, and no images.

## J. Test camera permission on Android Chrome
1. Open the HTTPS Render URL in Chrome (the camera does not work on plain `http://`).
2. Tap **Start Camera**; Chrome shows a permission prompt. Allow it; the preview must appear.
3. Denial test: tap Block, then reload and retry: the app must show "Camera permission was denied...". Re-enable in Chrome menu -> Settings -> Site settings -> Camera.
4. Rear camera: tap **Switch Camera** (only shown when more than one camera exists).
5. Close the workflow or finish: the camera indicator in the status bar must disappear (camera released).
6. Rotate the phone and confirm the layout still fits.
Record your results; the author has not run these.

## K. Troubleshooting
| Symptom | Likely cause / fix |
|---|---|
| Build fails in `download_models.sh` ("only N bytes") | GitHub returned an error page or LFS pointer. Run the script locally, check the URLs in the OpenCV Zoo repository, or commit the model files after verifying their licences and remove `static/models/*.onnx` from `.gitignore`. |
| Page says "face model files are missing" | `static/models/*.onnx` absent in the deployed build; check the build log. |
| "The face engine library could not be loaded" | jsDelivr blocked or offline; or CSP edited. Self-host `ort.min.js` and its `.wasm` files and update the script tag and CSP. |
| Detector error "Unexpected face detector outputs" | The detector file is not YuNet 2023mar; use the exact file. |
| `/api/health` returns 503 `database: unavailable` | Wrong `MONGODB_URI`, password not URL-encoded, user lacks permissions, or `0.0.0.0/0` not added. Read the Render logs (they never include the password). |
| Everything matches / nothing matches | Check preprocessing (README: RGB vs BGR), then calibrate thresholds. |
| 429 errors | Per-IP rate limit; wait or raise the `RATE_*` variables. |
| First request very slow | Render Free spin-down; wait up to about a minute. |
| Out of memory on Render | Keep one worker; lower `MAX_RECORDS`. |
| Camera does not start on a phone | Not HTTPS, permission blocked, or camera used by another app. |
| Registration says "storage limit reached" | `MAX_RECORDS` reached; delete records (L) or raise the limit (Atlas free tier holds 512 MB). |

## L. Owner-authorised data deletion
There is deliberately no public Delete button. Only the database owner may delete, after verifying
that the requester is the data subject (or has authority). Keep a log of requests outside the repo.

Atlas web UI: **Browse Collections** -> `face_recognition_db.registered_faces` -> filter `{ "name": "Jane Doe" }` ->
inspect the documents (several people can share a name; confirm the right one by `created_at`) -> trash icon -> confirm.

mongosh (connect with your own admin credentials, not the app user's secret from Render):
```
use face_recognition_db
db.registered_faces.find({ name: "Jane Doe" }, { name: 1, created_at: 1 })
db.registered_faces.deleteOne({ _id: ObjectId("<id from the find above>") })
// remove everything (e.g. decommissioning the site):
db.registered_faces.deleteMany({})
```
Automatic deletion: a TTL index on `created_at` removes records `RETENTION_DAYS` days after registration
(default 90; the background task runs roughly every minute). Changing `RETENTION_DAYS` updates the index
at the next app start. Remember to also delete Atlas backups/snapshots if you enable them (M0 has none).
