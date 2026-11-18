#!/usr/bin/env python3
"""Download + verify every AI model listed in models/model_manifest.json (stdlib only).

 * archives (InsightFace zips) are downloaded once, verified (SHA-256 + size), the needed .onnx members are extracted, the zip is deleted
 * single-file models are downloaded directly
 * every final .onnx is checked: SHA-256, size, valid ONNX protobuf header
 * already verified files are not downloaded again; corrupted files are deleted and fetched again
 * any required model that cannot be prepared => exit code 1 (the Render build fails; nothing is skipped silently)
 * models whose licence class is "noncommercial" need ACCEPT_NONCOMMERCIAL_MODEL_LICENSES=1 (see MODEL_LICENSES.md)

Environment (all optional): MODELS_DEST (default static/models), MODEL_SEED_DIR (local folder with already downloaded files,
used before the network - handy offline), DOWNLOAD_RETRIES (default 4), ACCEPT_NONCOMMERCIAL_MODEL_LICENSES.
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "models", "model_manifest.json")
DEST = os.environ.get("MODELS_DEST") or os.path.join(ROOT, "static", "models")
SEED = os.environ.get("MODEL_SEED_DIR", "")
RETRIES = int(os.environ.get("DOWNLOAD_RETRIES", "4"))


def log(msg):
    print(f"[download_models] {msg}", flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def looks_like_onnx(path):
    """Cheap structural check: an ONNX file is a protobuf ModelProto; the first field tag is 0x08 (ir_version).
    Also rejects Git-LFS pointer files and HTML error pages."""
    with open(path, "rb") as fh:
        head = fh.read(64)
    return len(head) > 8 and head[0] == 0x08 and not head.startswith(b"version https://git-lfs") and b"<html" not in head.lower()


def verify(path, sha, size, what):
    if not os.path.isfile(path):
        return False
    if size and os.path.getsize(path) != size:
        log(f"{what}: size mismatch ({os.path.getsize(path)} != {size})")
        return False
    if sha256_file(path) != sha:
        log(f"{what}: SHA-256 mismatch")
        return False
    if path.endswith(".onnx") and not looks_like_onnx(path):
        log(f"{what}: not a valid ONNX file")
        return False
    return True


def fetch(url, target, what):
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            log(f"{what}: downloading (attempt {attempt}/{RETRIES}) {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "face-recognition-model-downloader/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(target, "wb") as out:
                shutil.copyfileobj(resp, out, 1 << 20)
            return
        except Exception as exc:  # noqa: BLE001 - any network failure is retried
            last = exc
            log(f"{what}: failed: {exc}")
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"{what}: download failed after {RETRIES} attempts: {last}")


def seeded(name, sha, size):
    if SEED:
        p = os.path.join(SEED, name)
        if verify(p, sha, size, f"seed {name}"):
            return p
    return None


def main():
    with open(MANIFEST, encoding="utf-8") as fh:
        manifest = json.load(fh)
    os.makedirs(DEST, exist_ok=True)
    accept_nc = os.environ.get("ACCEPT_NONCOMMERCIAL_MODEL_LICENSES", "") in ("1", "true", "yes")
    archives = {a["id"]: a for a in manifest["archives"]}
    failures, ready = [], []
    todo_by_archive, todo_direct = {}, []

    for m in manifest["models"]:
        target = os.path.join(DEST, m["filename"])
        if m["license"]["class"] == "noncommercial" and not accept_nc:
            failures.append(f"{m['id']}: licence '{m['license']['name']}' is non-commercial. Set ACCEPT_NONCOMMERCIAL_MODEL_LICENSES=1 "
                            "only if your use is personal/non-commercial research (see MODEL_LICENSES.md).")
            continue
        if verify(target, m["sha256"], m["size_bytes"], m["id"]):
            log(f"{m['id']}: already verified, skipping")
            ready.append(m["id"])
            continue
        if os.path.exists(target):
            log(f"{m['id']}: existing file is corrupted, removing it")
            os.remove(target)
        (todo_by_archive.setdefault(m["source"]["archive_id"], []) if m["source"]["type"] == "archive" else todo_direct).append(m)

    for m in todo_direct:
        target = os.path.join(DEST, m["filename"])
        try:
            p = seeded(m["filename"], m["sha256"], m["size_bytes"])
            if p:
                shutil.copyfile(p, target)
            else:
                tmp = target + ".part"
                fetch(m["source"]["url"], tmp, m["id"])
                os.replace(tmp, target)
            if not verify(target, m["sha256"], m["size_bytes"], m["id"]):
                os.remove(target)
                raise RuntimeError("verification failed after download")
            ready.append(m["id"])
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{m['id']}: {exc}")

    for aid, models in todo_by_archive.items():
        a = archives[aid]
        with tempfile.TemporaryDirectory() as tmpdir:
            zpath = os.path.join(tmpdir, a["filename"])
            try:
                p = seeded(a["filename"], a["sha256"], a["size_bytes"])
                if p:
                    zpath = p
                else:
                    fetch(a["url"], zpath, a["filename"])
                if not verify(zpath, a["sha256"], a["size_bytes"], a["filename"]):
                    raise RuntimeError("archive verification failed")
                with zipfile.ZipFile(zpath) as zf:
                    for m in models:
                        target = os.path.join(DEST, m["filename"])
                        with zf.open(m["source"]["member"]) as src, open(target + ".part", "wb") as out:
                            shutil.copyfileobj(src, out, 1 << 20)
                        os.replace(target + ".part", target)
                        if verify(target, m["sha256"], m["size_bytes"], m["id"]):
                            ready.append(m["id"])
                        else:
                            os.remove(target)
                            failures.append(f"{m['id']}: extracted file failed verification")
            except Exception as exc:  # noqa: BLE001
                failures.extend(f"{m['id']}: {exc}" for m in models)

    required = {m["id"] for m in manifest["models"] if m["required"]}
    missing = sorted(required - set(ready))
    log(f"verified models: {', '.join(sorted(ready)) or '(none)'}")
    if failures or missing:
        for f in failures:
            log("ERROR " + f)
        log(f"BUILD FAILED: required models not ready: {', '.join(missing)}")
        return 1
    log(f"All {len(ready)} models are ready in {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
