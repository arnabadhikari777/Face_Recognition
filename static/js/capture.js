/** Shared capture plumbing: cancellable flow object, live quality hints, quality-ranked burst capture. */
import { Cancelled } from './engine-client.js';
export { Cancelled };   // one shared class: engine-client rejects with it, the flows and app.js test for it

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export class CaptureFlow {
  constructor({ els, camera, client, ui, config, profile }) {
    Object.assign(this, { els, camera, client, ui, config, profile });
    this._cancelled = false; this._onCapture = null; this._onSkip = null; this._onCancel = null;
  }
  waitForButton() {
    if (this._cancelled) return Promise.reject(new Cancelled());
    return new Promise((resolve, reject) => { this._onCapture = () => resolve('capture'); this._onSkip = () => resolve('skip'); this._onCancel = reject; });
  }
  capture() { const f = this._onCapture; this._onCapture = this._onSkip = null; if (f) f(); }
  skip() { const f = this._onSkip; this._onCapture = this._onSkip = null; if (f) f(); }
  cancel() { this._cancelled = true; this.client.cancel(); if (this._onCancel) this._onCancel(new Cancelled()); }
  assertActive() { if (this._cancelled) throw new Cancelled(); }
}

/** Live feedback while the user positions themselves (low-res frames, no embeddings). Returns a stop function. */
export function startLiveCheck(flow, getPose, onResult) {
  let stopped = false, prev = null;
  (async () => {
    while (!stopped) {
      try {
        const frame = flow.client.grabFrame(flow.els.video, flow.profile.preview_max_side);
        if (frame) {
          const r = await flow.client.analyze(frame, { targetPose: getPose(), deep: false, prevCenter: prev });
          prev = r.face ? [r.face.x + r.face.w / 2, r.face.y + r.face.h / 2] : null;
          if (!stopped) onResult(r);
        }
      } catch (e) { if (e && e.constructor && e.constructor.name === 'Cancelled') return; console.error(e); if (!stopped) onResult({ ok: false, message: 'Face check failed. Please reload the page.' }); await sleep(1000); }
      await sleep(120);
    }
  })();
  return () => { stopped = true; };
}

/**
 * Capture a burst of full-resolution frames, keep only those that pass every quality gate and return them best-first.
 * @returns {Promise<Array<{crop:Uint8Array, quality:object}>>}
 */
export async function captureBurst(flow, { pose, want, timeoutMs, onStatus }) {
  const { client, els, profile, config } = flow, good = [], t0 = performance.now();
  let prev = null, lastMsg = '';
  while (good.length < want && performance.now() - t0 < timeoutMs) {
    flow.assertActive();
    const frame = client.grabFrame(els.video, profile.capture_max_side);
    if (!frame) { await sleep(60); continue; }
    const r = await client.analyze(frame, { targetPose: pose, deep: true, prevCenter: prev });
    flow.assertActive();
    prev = r.face ? [r.face.x + r.face.w / 2, r.face.y + r.face.h / 2] : null;
    if (r.ok && r.quality.score >= config.quality.min_frame_quality) {
      good.push({ crop: r.crop, quality: r.quality });
      onStatus(`Good frame ${good.length} of ${want}…`, true);
    } else if (r.message !== lastMsg) { lastMsg = r.message; onStatus(r.message, false); }
    await sleep(90);                                // spacing so that frames differ slightly (motion / micro-expressions)
  }
  good.sort((a, b) => b.quality.score - a.quality.score);
  return good;
}

export function qualitySummary(q) {   // only non-identifying numbers are ever sent to the server
  return { score: Number(q.score.toFixed(4)), yaw: Number(q.pose.yaw.toFixed(3)), pitch: Number(q.pose.pitch.toFixed(3)),
    sharpness: Number(q.metrics.sharpness.toFixed(1)), brightness: Number(q.metrics.brightness.toFixed(1)),
    detection: Number(q.metrics.detScore.toFixed(3)), agreement: q.metrics.agreement == null ? null : Number(q.metrics.agreement.toFixed(3)) };
}
