/** Main-thread handle to the inference Web Worker. Keeps the UI responsive; supports cancellation and memory cleanup. */
export class Cancelled extends Error {}
export class EngineError extends Error { constructor(code, message) { super(message); this.code = code; } }

const ORT_BASE = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/';   // pinned; must match the CSP in backend/__init__.py

export class EngineClient {
  constructor(manifest, profileId) {
    this.manifest = manifest; this.profileId = profileId; this.profile = manifest.profiles[profileId];
    this.worker = null; this.pending = new Map(); this.nextId = 1; this.epoch = 0; this.onProgress = () => {};
    this.canvas = document.createElement('canvas'); this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    this.backends = {};
  }

  /** Models loaded eagerly. Balanced mode loads the secondary recognizer lazily; accuracy/compatibility load everything. */
  initialModels() {
    const p = this.profile;
    return p.always_run_all_recognizers || p.recognizers.length === 1 ? [...p.detectors, ...p.recognizers] : [...p.detectors, p.recognizers[0]];
  }

  _send(msg, transfer) {
    return new Promise((resolve, reject) => {
      const id = this.nextId++;
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ ...msg, id, epoch: this.epoch }, transfer || []);
    });
  }

  async init(onProgress = () => {}) {
    this.onProgress = onProgress;
    if (typeof Worker === 'undefined') throw new EngineError('NO_WORKER', 'This browser does not support Web Workers.');
    this.worker = new Worker('/static/js/engine/worker.js');
    this.worker.onmessage = (ev) => {
      const m = ev.data;
      if (m.type === 'progress') return this.onProgress(m.modelId, m.pct);
      const p = this.pending.get(m.id); if (!p) return;
      this.pending.delete(m.id);
      if (m.ok) p.resolve(m.result);
      else p.reject(m.error.code === 'CANCELLED' ? new Cancelled() : new EngineError(m.error.code, m.error.message));
    };
    this.worker.onerror = (ev) => this._failAll(new EngineError('WORKER', 'The face engine crashed: ' + (ev.message || 'unknown error')));
    const webgpu = !!navigator.gpu && this.profile.backends.includes('webgpu');
    const res = await this._send({ type: 'init', manifest: this.manifest, profileId: this.profileId, models: this.initialModels(),
      ortBase: ORT_BASE, ortFile: webgpu ? 'ort.webgpu.min.js' : 'ort.wasm.min.js' });
    this.backends = res.backends;
  }

  _failAll(err) { for (const p of this.pending.values()) p.reject(err); this.pending.clear(); }

  /** Cancel every queued/running request (results are dropped). Safe to call at any time. */
  cancel() {
    this.epoch++;
    if (this.worker) this.worker.postMessage({ type: 'cancel', epoch: this.epoch });
    for (const p of this.pending.values()) p.reject(new Cancelled());
    this.pending.clear();
  }

  async ensureModels(ids, onProgress) {
    if (onProgress) this.onProgress = onProgress;
    const r = await this._send({ type: 'ensure', models: ids });
    Object.assign(this.backends, r.backends);
  }

  /** Grab the current video frame (downscaled so the longest side <= maxSide) as transferable RGBA. */
  grabFrame(video, maxSide) {
    const vw = video.videoWidth, vh = video.videoHeight; if (!vw || !vh) return null;
    const k = Math.min(1, maxSide / Math.max(vw, vh)), w = Math.round(vw * k), h = Math.round(vh * k);
    this.canvas.width = w; this.canvas.height = h;
    this.ctx.drawImage(video, 0, 0, w, h);
    return { buffer: this.ctx.getImageData(0, 0, w, h).data.buffer, width: w, height: h };
  }

  async analyze(frame, opts = {}) {
    const r = await this._send({ type: 'analyze', buffer: frame.buffer, width: frame.width, height: frame.height, opts }, [frame.buffer]);
    if (r.crop) r.crop = new Uint8Array(r.crop.buffer ? r.crop.buffer : r.crop);
    return r;
  }

  embed(crop, models) { return this._send({ type: 'embed', crop: crop.buffer.slice(0), models }); }
  release(models) { return this._send({ type: 'release', models }); }

  async dispose() {
    if (!this.worker) return;
    this.cancel();
    try { await Promise.race([this._send({ type: 'dispose' }), new Promise((r) => setTimeout(r, 500))]); } catch (_) { /* ignore */ }
    this.worker.terminate(); this.worker = null;
  }
}
