# Bugfix Report — Face Recognition (config loading + Start Camera)

## Bugs fixed and root causes

### 1. Processing Mode dropdown empty / race with Start Camera
**Root cause:** `init()` called `getConfig()` asynchronously and returned early on failure without UI feedback. Users could open Register/Recognize and press Start Camera before `config` was set. `els.profile` stayed empty; `els.profile.value` could be `""`; `EngineClient` / `profileObject()` then failed or produced opaque errors.

**Fix (static/js/app.js):**
- Explicit config state: `pending` | `ready` | `error`.
- Single shared `loadConfig()` promise (no concurrent duplicate requests).
- Populate profile `<select>` only after a valid `profiles` object arrives.
- Restore last mode from `localStorage` when still present in server config; otherwise use `default_profile`.
- Disable Start button and profile select until config is ready.
- On open-workflow, if config still pending, keep Start disabled and load (or show clear error + note).
- `start()` rejects empty/invalid profile IDs before creating `EngineClient`.
- `describeProfile()` and `profileObject()` guard against missing config/profile.

### 2. Generic “Something went wrong. Please reload…”
**Root cause:** `friendly()` fell through to a generic string for many `EngineError` codes and unexpected failures.

**Fix:** Map additional `EngineError` codes (`NO_WORKER`, `WORKER`) and `AbortError` to specific messages. Network failures from `api.js` already returned structured messages; those continue to surface.

### 3. Environment configuration clutter
**Root cause:** Uploaded `.env` mixed obsolete names (`MONGODB_DB`, `CORS_ORIGINS`, `FLASK_DEBUG`, `SECRET_KEY`, hard-coded thresholds) with current names and contained duplicates.

**Fix:** Clean `.env.example` using only names read by `config.py`. Added `ENV_SETUP_BN.md` with Atlas URI, Fernet key, and Render instructions. Obsolete variables removed from the template.

## Files deliberately left unchanged (accuracy protection)

- `models/model_manifest.json`
- `models/calibration.json`
- All ONNX model files under `static/models/`
- `static/js/face-engine.js`, `static/js/engine/*` (detection, alignment, embedding, quality)
- `static/js/registration.js`, `static/js/recognition.js`, `static/js/capture.js`
- `backend/face_matching.py`, `backend/registration.py`, `backend/templates.py`, `backend/validation.py`
- Similarity / threshold logic and preprocessing parameters
- Database schema and MongoDB collection structure

## File-by-file change summary

| File | Change |
|------|--------|
| `static/js/app.js` | Robust config load, dropdown population, Start guards, safer `describeProfile` / `profileObject`, slightly clearer `friendly()` |
| `.env.example` | Canonical variable list only; clearer comments |
| `ENV_SETUP_BN.md` | **New** — Bengali .env + Atlas + Fernet + Render guide |
| `TROUBLESHOOTING_BN.md` | Two rows for the fixed config/dropdown and generic-error issues |
| Everything else | Unchanged |

## Test results

| Area | Result | Notes |
|------|--------|-------|
| `/api/config` returns profiles | **Passed** | Flask test client: 200, profiles `accuracy` / `balanced` / `compatibility`, default `accuracy` |
| `/api/health` without Mongo | **Passed** (expected degraded) | 503 `database unavailable` when `MONGODB_URI` empty |
| JS syntax (`app.js`, `engine-client.js`, `camera.js`) | **Passed** | `node --check` |
| Config load before Start (logic review) | **Passed** | Start disabled until `configState === 'ready'`; invalid profile rejected |
| Dropdown population + localStorage restore | **Passed** (code path) | Restores `fr.profile` when still in `config.profiles` |
| Simulated `/api/config` failure | **Skipped** (no browser automation in this environment) | Code path shows error in workflow + profile note |
| Android / desktop camera | **Skipped** | No device browser; existing `Camera` error mapping unchanged |
| Recognition accuracy regression | **Skipped** | No consented image set / no live models exercise in this environment; recognition pipeline files were not modified |
| Full registration/recognition E2E with Mongo | **Skipped** | No Atlas credentials supplied |

**Summary:** Passed: 4 · Failed: 0 · Skipped: 4 (environment / hardware / data dependent)

## Remaining manual steps for you

1. Copy `.env.example` → `.env` (local) or set the same keys in Render → Environment.
2. Fill real `MONGODB_URI` (Atlas Connect string; URL-encode special characters in the password).
3. Optionally generate and set `TEMPLATE_ENCRYPTION_KEY` (keep the same key if you already have encrypted records).
4. Redeploy on Render (or restart local server) so clients receive the updated `app.js`.
5. Hard-refresh the browser (or clear site data) so the old JS is not cached.
6. Smoke-test: open Register → confirm three modes appear → Start Camera on each mode once.
