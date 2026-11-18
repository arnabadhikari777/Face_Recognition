/**
 * Runtime-agnostic face engine core. `ort` (onnxruntime-web in the browser worker / onnxruntime-node in tests) and the
 * model byte loader are injected, so exactly the same code path is exercised in the browser and in the Node tests.
 * Pipeline: detect (primary, adaptive fallback) -> 5-point alignment -> quality assessment -> per-model embeddings.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory(require('./preprocess.js'), require('./detectors.js'), require('./recognizers.js'), require('./quality.js'));
  } else { const F = root.FR = root.FR || {}; F.core = factory(F.preprocess, F.detectors, F.recognizers, F.quality); }
})(typeof self !== 'undefined' ? self : this, function (P, D, R, Q) {
  'use strict';

  class EngineError extends Error { constructor(code, message) { super(message); this.code = code; } }

  const iou = (a, b) => {
    const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x)), iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
    const i = ix * iy; return i / (a.w * a.h + b.w * b.h - i || 1);
  };

  class FaceEngineCore {
    /** @param {{ort:any, manifest:object, profileId:string, loadModelBytes:(spec,onPct)=>Promise<Uint8Array>}} o */
    constructor({ ort, manifest, profileId, loadModelBytes, detectorBackend = 'wasm' }) {
      this.detectorBackend = detectorBackend;
      this.ort = ort; this.manifest = manifest; this.profileId = profileId;
      this.profile = manifest.profiles[profileId];
      if (!this.profile) throw new EngineError('BAD_PROFILE', `Unknown profile ${profileId}`);
      this.loadModelBytes = loadModelBytes;
      this.sessions = new Map();      // model id -> {session, provider}
      this.specs = new Map(manifest.models.map((m) => [m.id, m]));
    }

    has(id) { return this.sessions.has(id); }
    backendOf(id) { const s = this.sessions.get(id); return s ? s.provider : null; }

    async load(id, onProgress = () => {}) {
      if (this.sessions.has(id)) return;
      const spec = this.specs.get(id);
      if (!spec) throw new EngineError('UNKNOWN_MODEL', `Model ${id} is not in the manifest.`);
      const bytes = await this.loadModelBytes(spec, (p) => onProgress(id, p));
      // Detectors stay on WASM (dynamic shapes / small); recognizers may use WebGPU first, then WASM.
      const wanted = spec.category === 'recognition' ? this.profile.backends : [this.detectorBackend];
      let last = null;
      for (const provider of wanted) {
        try {
          const session = await this.ort.InferenceSession.create(bytes, { executionProviders: [provider], graphOptimizationLevel: 'all' });
          if (spec.category === 'recognition') await this._probeRecognizer(session, id);
          this.sessions.set(id, { session, provider });
          return;
        } catch (e) { last = e; }
      }
      throw new EngineError('MODEL_LOAD', `The model ${id} could not be loaded (${last && last.message ? last.message : 'unknown error'}).`);
    }

    async _probeRecognizer(session, id) {      // catches broken WebGPU/WASM sessions early
      const reg = R.REGISTRY[id]; const crop = new Uint8Array(112 * 112 * 3).fill(128);
      for (let i = 0; i < crop.length; i += 7) crop[i] = (i * 31) & 255;
      const e = await R.embed(this.ort, session, reg, crop);
      if (e.length !== reg.dim) throw new Error('unexpected embedding size');
    }

    release(id) {
      const s = this.sessions.get(id);
      if (s) { try { if (s.session.release) s.session.release(); } catch (_) { /* ignore */ } this.sessions.delete(id); }
    }
    dispose() { for (const id of [...this.sessions.keys()]) this.release(id); }

    /** Run one detector on an RGBA frame; faces are in source pixels. */
    async detect(detectorId, rgba, w, h) {
      const det = D.REGISTRY[detectorId], s = this.sessions.get(detectorId);
      if (!det || !s) throw new EngineError('NOT_LOADED', `Detector ${detectorId} is not loaded.`);
      const thr = this.manifest.detection.score_threshold;
      const pre = det.preprocess(rgba, w, h);
      const out = await s.session.run({ [s.session.inputNames[0]]: new this.ort.Tensor('float32', pre.tensor, pre.shape) });
      let dets;
      if (detectorId === 'scrfd_500m') {
        dets = det.decode(s.session.outputNames.map((n) => ({ data: out[n].data, dims: out[n].dims })), det.inputSize, thr);
      } else {
        dets = det.decode((n) => (out[n] ? out[n].data : null), det.inputSize, thr);
      }
      return D.finalize(dets, det.nmsIou, pre.sx, pre.sy);
    }

    /** Adaptive detection: primary first, fallback only when the primary finds nothing or is unsure. */
    async detectAdaptive(rgba, w, h) {
      const [primary, fallback] = this.profile.detectors;
      let faces = await this.detect(primary, rgba, w, h), used = primary;
      const best = faces.reduce((m, f) => Math.max(m, f.score), 0);
      if (fallback && this.has(fallback) && (faces.length === 0 || best < this.manifest.detection.fallback_below_score)) {
        const alt = await this.detect(fallback, rgba, w, h);
        if (alt.length > faces.length || alt.reduce((m, f) => Math.max(m, f.score), 0) > best) { faces = alt; used = fallback; }
      }
      return { faces, used };
    }

    /**
     * Analyse one frame. frame={rgba,width,height}. opts={targetPose, deep, prevCenter}.
     * -> {ok, code, message, count, face?, quality?, crop?, detector}
     */
    async analyze(frame, opts = {}) {
      const { rgba, width: w, height: h } = frame;
      const { faces, used } = await this.detectAdaptive(rgba, w, h);
      if (faces.length === 0) return { ok: false, count: 0, code: 'no_face', message: 'No face detected. Please face the camera.', detector: used };
      faces.sort((a, b) => b.w * b.h - a.w * a.h);
      const big = faces[0];
      if (faces.some((f, i) => i > 0 && f.w >= 0.3 * big.w)) {
        return { ok: false, count: faces.length, code: 'multiple_faces', message: 'Please make sure only one person is in the camera.', detector: used };
      }
      const crop = P.warpToTemplate(rgba, w, h, big.lm);
      let agreement = null;
      if (opts.deep && this.profile.deep_landmark_check) {
        const other = this.profile.detectors.find((d) => d !== used && this.has(d));
        if (other) {
          const alt = (await this.detect(other, rgba, w, h)).filter((f) => iou(f, big) > 0.3).sort((a, b) => iou(b, big) - iou(a, big))[0];
          if (alt) agreement = Q.landmarkAgreement(big.lm, alt.lm);
        }
      }
      const quality = Q.assess(big, { frameW: w, frameH: h, crop, targetPose: opts.targetPose || 'any', agreement, prevCenter: opts.prevCenter || null }, this.manifest.quality);
      const first = quality.flags[0];
      return { ok: quality.ok, count: 1, code: first ? first.code : 'ok', message: first ? first.message : 'Looks good.', face: big, quality, crop, detector: used };
    }

    /** Embeddings of one aligned crop for the requested recognition models -> {modelId: unit Array}. */
    async embed(crop, modelIds) {
      const out = {};
      for (const id of modelIds) {
        const s = this.sessions.get(id);
        if (!s) throw new EngineError('NOT_LOADED', `Model ${id} is not loaded.`);
        out[id] = await R.embed(this.ort, s.session, R.REGISTRY[id], crop);
      }
      return out;
    }
  }

  return { FaceEngineCore, EngineError };
});
