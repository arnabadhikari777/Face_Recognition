/**
 * Browser-side face engine (ONNX Runtime Web).
 *
 *  Detector   : YuNet  face_detection_yunet_2023mar.onnx   (MIT)
 *  Recognizer : SFace  face_recognition_sface_2021dec.onnx (Apache-2.0)
 *  Both from https://github.com/opencv/opencv_zoo
 *
 * Pipeline (identical for registration and recognition):
 *   frame -> letterbox to SxS -> YuNet (BGR, raw 0..255, CHW) -> boxes + 5 landmarks
 *   -> similarity-transform the 5 landmarks onto the ArcFace 112x112 template
 *   -> SFace (RGB, raw 0..255, CHW, 112x112) -> 128-d -> L2 normalise.
 * Similarity = cosine (dot product of unit vectors).
 *
 * NOT TESTED in a real browser by the author: see README "Test results".
 */
export const ENGINE = {
  modelName: 'opencv-zoo-sface',
  modelVersion: '2021dec',
  embeddingDim: 128,
  detectorUrl: '/static/models/face_detection_yunet_2023mar.onnx',
  recognizerUrl: '/static/models/face_recognition_sface_2021dec.onnx',
  detSizes: [640, 320],     // tried in order; first size the detector accepts is used
  useWebGPU: false,         // WASM only by default: WebGPU can fail at run time on some devices
  ortBase: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/',  // must match the <script> tag version
  detScore: 0.7,
  nmsIou: 0.3,
  // Quality gates. These are engineering defaults, not calibrated values: tune on real devices.
  minFaceFrac: 0.18,        // face width / min(video w,h)
  maxFaceFrac: 0.85,
  minBrightness: 60, maxBrightness: 200, minContrast: 25,
  minSharpness: 40,         // variance of the Laplacian on the 112x112 aligned crop
  frontMaxYaw: 0.15,        // |nose offset| / eye distance
  sideMinYaw: 0.2, sideMaxYaw: 0.7,
};

export const SFACE_TEMPLATE = [
  [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]];

export class Cancelled extends Error {}
export class ModelError extends Error {
  constructor(code, message) { super(message); this.code = code; }
}

/* ---------- pure helpers (unit-testable without a browser) ---------- */

/** Least-squares similarity transform src->dst; returns canvas setTransform(a,b,c,d,e,f) values. */
export function similarityTransform(src, dst) {
  const n = src.length;
  let msx = 0, msy = 0, mdx = 0, mdy = 0;
  for (let i = 0; i < n; i++) { msx += src[i][0]; msy += src[i][1]; mdx += dst[i][0]; mdy += dst[i][1]; }
  msx /= n; msy /= n; mdx /= n; mdy /= n;
  let nr = 0, ni = 0, den = 0;
  for (let i = 0; i < n; i++) {
    const sx = src[i][0] - msx, sy = src[i][1] - msy, dx = dst[i][0] - mdx, dy = dst[i][1] - mdy;
    nr += dx * sx + dy * sy; ni += dy * sx - dx * sy; den += sx * sx + sy * sy;
  }
  const ar = nr / den, ai = ni / den;
  return { a: ar, b: ai, c: -ai, d: ar, e: mdx - (ar * msx - ai * msy), f: mdy - (ai * msx + ar * msy) };
}

export function applyTransform(t, [x, y]) { return [t.a * x + t.c * y + t.e, t.b * x + t.d * y + t.f]; }

const clamp01 = (v) => Math.min(1, Math.max(0, v));

/** Decode YuNet outputs (cls_/obj_/bbox_/kps_ at strides 8,16,32) into detections in letterboxed pixels. */
export function decodeYuNet(get, size, threshold) {
  const dets = [];
  for (const stride of [8, 16, 32]) {
    const cls = get(`cls_${stride}`), obj = get(`obj_${stride}`), bbox = get(`bbox_${stride}`), kps = get(`kps_${stride}`);
    if (!cls || !obj || !bbox || !kps) throw new ModelError('BAD_OUTPUTS', 'Unexpected face detector outputs.');
    const cols = Math.floor(size / stride), rows = Math.floor(size / stride);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const i = r * cols + c;
        const score = Math.sqrt(clamp01(cls[i]) * clamp01(obj[i]));
        if (!(score >= threshold)) continue;
        const cx = (c + bbox[i * 4]) * stride, cy = (r + bbox[i * 4 + 1]) * stride;
        const w = Math.exp(bbox[i * 4 + 2]) * stride, h = Math.exp(bbox[i * 4 + 3]) * stride;
        const lm = [];
        for (let n = 0; n < 5; n++) lm.push([(kps[i * 10 + 2 * n] + c) * stride, (kps[i * 10 + 2 * n + 1] + r) * stride]);
        dets.push({ x: cx - w / 2, y: cy - h / 2, w, h, score, lm });
      }
    }
  }
  return dets;
}

export function nms(dets, iouThr) {
  const sorted = [...dets].sort((p, q) => q.score - p.score), keep = [];
  for (const d of sorted) {
    let ok = true;
    for (const k of keep) {
      const ix = Math.max(0, Math.min(d.x + d.w, k.x + k.w) - Math.max(d.x, k.x));
      const iy = Math.max(0, Math.min(d.y + d.h, k.y + k.h) - Math.max(d.y, k.y));
      const inter = ix * iy, union = d.w * d.h + k.w * k.h - inter;
      if (union > 0 && inter / union > iouThr) { ok = false; break; }
    }
    if (ok) keep.push(d);
  }
  return keep;
}

export function grayStats(rgba, w, h) {
  const gray = new Float32Array(w * h);
  let sum = 0;
  for (let i = 0; i < w * h; i++) {
    const g = 0.299 * rgba[4 * i] + 0.587 * rgba[4 * i + 1] + 0.114 * rgba[4 * i + 2];
    gray[i] = g; sum += g;
  }
  const mean = sum / (w * h);
  let v = 0;
  for (let i = 0; i < gray.length; i++) v += (gray[i] - mean) ** 2;
  const contrast = Math.sqrt(v / gray.length);
  let lsum = 0, lsq = 0, cnt = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const l = gray[i - 1] + gray[i + 1] + gray[i - w] + gray[i + w] - 4 * gray[i];
      lsum += l; lsq += l * l; cnt++;
    }
  }
  const sharpness = lsq / cnt - (lsum / cnt) ** 2;
  return { brightness: mean, contrast, sharpness };
}

/** >0: nose displaced toward image-right (subject turned to THEIR left); <0: toward image-left. */
export function yawRatio(lm) {
  const mid = (lm[0][0] + lm[1][0]) / 2, eye = Math.abs(lm[1][0] - lm[0][0]) || 1;
  return (lm[2][0] - mid) / eye;
}

export function l2normalize(vec) {
  let s = 0;
  for (let i = 0; i < vec.length; i++) {
    if (!Number.isFinite(vec[i])) throw new ModelError('BAD_EMBEDDING', 'The face model returned an invalid result.');
    s += vec[i] * vec[i];
  }
  const n = Math.sqrt(s);
  if (!(n > 0)) throw new ModelError('BAD_EMBEDDING', 'The face model returned an invalid result.');
  return Array.from(vec, (v) => v / n);
}

/** Returns a user-facing message, or null when the face passes all checks for the requested pose. */
export function judgeFace(face, vw, vh, stats, pose, cfg = ENGINE) {
  const minDim = Math.min(vw, vh), margin = 0.01 * minDim;
  if (face.w < cfg.minFaceFrac * minDim) return 'Move closer to the camera.';
  if (face.w > cfg.maxFaceFrac * minDim) return 'Move back a little.';
  if (face.x < -margin || face.y < -margin || face.x + face.w > vw + margin || face.y + face.h > vh + margin) {
    return 'Keep your whole face inside the frame.';
  }
  const cx = face.x + face.w / 2, cy = face.y + face.h / 2;
  if (cx < 0.2 * vw || cx > 0.8 * vw || cy < 0.15 * vh || cy > 0.85 * vh) return 'Center your face in the frame.';
  const yaw = yawRatio(face.lm);
  if (pose === 'front' && Math.abs(yaw) > cfg.frontMaxYaw) return 'Look straight at the camera.';
  if (pose === 'any' && Math.abs(yaw) > 0.5) return 'Please face the camera.';
  if (pose === 'left' || pose === 'right') {
    const want = pose === 'left' ? 1 : -1;
    const toward = yaw * want;
    if (toward < cfg.sideMinYaw) return `Turn your face slightly to your ${pose}.`;
    if (toward > cfg.sideMaxYaw) return 'Not so far: turn only slightly.';
  }
  if (stats.brightness < cfg.minBrightness) return 'Too dark. Move to a brighter place.';
  if (stats.brightness > cfg.maxBrightness) return 'Too bright. Avoid strong light on your face.';
  if (stats.contrast < cfg.minContrast) return 'Low contrast. Face the light source.';
  if (stats.sharpness < cfg.minSharpness) return 'Image is blurry. Hold the camera steady.';
  return null;
}

/* ---------- engine ---------- */

async function fetchModel(url, onPct) {
  let res;
  try { res = await fetch(url); } catch { throw new ModelError('MODEL_FETCH', 'Could not download the face model.'); }
  if (!res.ok) throw new ModelError('MODEL_MISSING', 'Face model file is missing on the server.');
  const total = Number(res.headers.get('Content-Length')) || 0;
  if (!res.body || !res.body.getReader) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader(), chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length;
    if (total && onPct) onPct(Math.round((got / total) * 100));
  }
  const out = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

async function makeSession(bytes) {
  const attempts = ENGINE.useWebGPU && navigator.gpu ? [['webgpu', 'wasm'], ['wasm']] : [['wasm']];
  let last;
  for (const eps of attempts) {
    try {
      const session = await ort.InferenceSession.create(bytes, { executionProviders: eps, graphOptimizationLevel: 'all' });
      return { session, providers: eps.join('+') };
    } catch (e) { last = e; }
  }
  throw new ModelError('MODEL_LOAD', `The face model could not be loaded (${last && last.message ? last.message : 'unknown error'}).`);
}

export class FaceEngine {
  constructor() {
    this.det = null; this.rec = null; this.detSize = 0; this.backend = '';
    this._queue = Promise.resolve();
    this._detCanvas = document.createElement('canvas');
    this._detCtx = this._detCanvas.getContext('2d', { willReadFrequently: true });
    this._alignCanvas = document.createElement('canvas');
    this._alignCanvas.width = this._alignCanvas.height = 112;
    this._alignCtx = this._alignCanvas.getContext('2d', { willReadFrequently: true });
  }

  _enqueue(fn) {
    const run = this._queue.then(fn, fn);
    this._queue = run.catch(() => {});
    return run;
  }

  async init(onProgress = () => {}) {
    if (typeof ort === 'undefined') throw new ModelError('ORT_MISSING', 'The face engine library could not be loaded. Check your internet connection.');
    ort.env.wasm.numThreads = 1;   // avoids needing cross-origin isolation (SharedArrayBuffer)
    ort.env.wasm.proxy = false;
    ort.env.wasm.wasmPaths = ENGINE.ortBase;   // explicit, so the .wasm files always come from the pinned version
    onProgress(0, 'Loading face detector…');
    const detBytes = await fetchModel(ENGINE.detectorUrl, (p) => onProgress(p, 'Loading face detector…'));
    const d = await makeSession(detBytes);
    this.det = d.session;
    onProgress(0, 'Loading face recognition model (first load may take a minute)…');
    const recBytes = await fetchModel(ENGINE.recognizerUrl, (p) => onProgress(p, 'Loading face recognition model (first load may take a minute)…'));
    if (recBytes.length < 100000) throw new ModelError('MODEL_LOAD', 'The recognition model file is invalid (is it a Git LFS pointer?).');
    const r = await makeSession(recBytes);
    this.rec = r.session;
    this.backend = `detector:${d.providers}, recognizer:${r.providers}`;
    onProgress(100, 'Checking models…');
    await this._pickDetSize();
    const probe = await this._runRecognizer(new Float32Array(3 * 112 * 112));
    if (probe.length !== ENGINE.embeddingDim) {
      throw new ModelError('MODEL_LOAD', `Unexpected embedding size ${probe.length}; expected ${ENGINE.embeddingDim}.`);
    }
    console.info('Face engine ready:', this.backend, 'detector input', this.detSize);
  }

  async _pickDetSize() {
    let last;
    for (const s of ENGINE.detSizes) {
      try { await this._runDetector(new Float32Array(3 * s * s), s); this.detSize = s; return; } catch (e) { last = e; }
    }
    throw new ModelError('MODEL_LOAD', `The face detector rejected all input sizes (${last && last.message}).`);
  }

  async _runDetector(chw, size) {
    const input = new ort.Tensor('float32', chw, [1, 3, size, size]);
    const out = await this.det.run({ [this.det.inputNames[0]]: input });
    return decodeYuNet((n) => (out[n] ? out[n].data : null), size, ENGINE.detScore);
  }

  async _runRecognizer(chw) {
    const input = new ort.Tensor('float32', chw, [1, 3, 112, 112]);
    const out = await this.rec.run({ [this.rec.inputNames[0]]: input });
    return out[this.rec.outputNames[0]].data;
  }

  /** Detect faces in a <video>; coordinates are in video pixels. */
  detect(video) {
    return this._enqueue(async () => {
      const vw = video.videoWidth, vh = video.videoHeight, S = this.detSize;
      if (!vw || !vh) return [];
      const scale = Math.min(S / vw, S / vh);
      this._detCanvas.width = this._detCanvas.height = S;
      const ctx = this._detCtx;
      ctx.fillStyle = '#000'; ctx.fillRect(0, 0, S, S);
      ctx.drawImage(video, 0, 0, vw * scale, vh * scale);
      const { data } = ctx.getImageData(0, 0, S, S);
      const plane = S * S, chw = new Float32Array(3 * plane);
      for (let i = 0; i < plane; i++) { chw[i] = data[4 * i + 2]; chw[plane + i] = data[4 * i + 1]; chw[2 * plane + i] = data[4 * i]; } // BGR
      const dets = nms(await this._runDetector(chw, S), ENGINE.nmsIou);
      return dets.map((d) => ({ x: d.x / scale, y: d.y / scale, w: d.w / scale, h: d.h / scale, score: d.score,
        lm: d.lm.map(([x, y]) => [x / scale, y / scale]) }));
    });
  }

  _align(video, lm) {
    const t = similarityTransform(lm, SFACE_TEMPLATE), ctx = this._alignCtx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = '#000'; ctx.fillRect(0, 0, 112, 112);
    ctx.setTransform(t.a, t.b, t.c, t.d, t.e, t.f);
    ctx.drawImage(video, 0, 0);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    return ctx.getImageData(0, 0, 112, 112).data;
  }

  /** -> {ok, count, message, face?}. pose: 'front' | 'left' | 'right' | 'any'. */
  async analyze(video, pose = 'any') {
    const faces = await this.detect(video);
    if (faces.length === 0) return { ok: false, count: 0, message: 'No face detected. Please try again.' };
    if (faces.length > 1) return { ok: false, count: faces.length, message: 'Please make sure only one person is in the camera.' };
    const face = faces[0];
    const stats = grayStats(this._align(video, face.lm), 112, 112);
    const message = judgeFace(face, video.videoWidth, video.videoHeight, stats, pose);
    return message ? { ok: false, count: 1, message } : { ok: true, count: 1, face };
  }

  /** Embedding (unit-length array of 128 numbers) for a face found by analyze(). */
  embed(video, face) {
    return this._enqueue(async () => {
      const px = this._align(video, face.lm), plane = 112 * 112, chw = new Float32Array(3 * plane);
      for (let i = 0; i < plane; i++) { chw[i] = px[4 * i]; chw[plane + i] = px[4 * i + 1]; chw[2 * plane + i] = px[4 * i + 2]; } // RGB
      const raw = await this._runRecognizer(chw);
      if (raw.length !== ENGINE.embeddingDim) throw new ModelError('BAD_EMBEDDING', 'The face model returned an unexpected result.');
      return l2normalize(raw);
    });
  }
}

/* ---------- shared capture-flow plumbing ---------- */

export function startLiveCheck(video, engine, getPose, onResult) {
  let stopped = false;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  (async () => {
    while (!stopped) {
      try {
        const r = await engine.analyze(video, getPose());
        if (!stopped) onResult(r);
      } catch (e) {
        console.error(e);
        if (!stopped) onResult({ ok: false, message: 'Face check failed. Please reload the page.' });
        await sleep(1000);
      }
      await sleep(150);
    }
  })();
  return () => { stopped = true; };
}

export class CaptureFlow {
  constructor({ els, camera, engine, ui, modelInfo }) {
    Object.assign(this, { els, camera, engine, ui, modelInfo });
    this._cancelled = false; this._onCapture = null; this._onCancel = null;
  }
  waitForCapture() {
    if (this._cancelled) return Promise.reject(new Cancelled());
    return new Promise((resolve, reject) => { this._onCapture = resolve; this._onCancel = reject; });
  }
  capture() { const f = this._onCapture; this._onCapture = null; if (f) f(); }
  cancel() { this._cancelled = true; if (this._onCancel) this._onCancel(new Cancelled()); }
  assertActive() { if (this._cancelled) throw new Cancelled(); }
  liveHint(r) { this.ui.hint(r.ok ? 'Looks good. Press Capture.' : r.message, r.ok); }
}
