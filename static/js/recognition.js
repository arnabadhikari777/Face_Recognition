import { CaptureFlow, startLiveCheck, captureBurst, qualitySummary } from './capture.js';
import { recognizeFace, packEmbeddings } from './api.js';

export class Recognition extends CaptureFlow {
  async run() {
    const { els, ui, client, config, profile } = this;
    const models = profile.recognizers, primary = models[0];
    els.progress.hidden = true; els.skip.hidden = true;
    els.instruction.textContent = 'Place one face in the frame, then press Scan.';
    ui.hint('Looking for a face…', false);
    const stop = startLiveCheck(this, () => 'any', (r) => ui.hint(r.ok ? 'Looks good. Press Scan.' : r.message, r.ok));
    try { await this.waitForButton(); } finally { stop(); }
    els.capture.disabled = true;
    let burst;
    try { burst = await captureBurst(this, { pose: 'any', want: config.recognition.burst_frames, timeoutMs: config.recognition.burst_timeout_ms, onStatus: (t, g) => ui.hint(t, g) }); }
    finally { els.capture.disabled = false; }
    ui.hideCamera();
    if (burst.length < config.recognition.min_frames) {
      return ui.showResult({ title: 'Image quality too low', kind: 'warn', lines: ['Not enough clear frames were captured.', 'Face the camera in good light, hold still and try again.'] });
    }
    const initial = profile.always_run_all_recognizers ? models : [primary];
    const frames = [];
    for (let i = 0; i < burst.length; i++) {
      ui.setProcessing(`Analysing frame ${i + 1} of ${burst.length}…`);
      frames.push({ crop: burst[i].crop, q: burst[i].quality, emb: await client.embed(burst[i].crop, initial) }); this.assertActive();
    }
    ui.setProcessing('Comparing with registered faces…');
    let res = await this._send(frames);
    let d = res.data || {};
    if (res.ok && d.status === 'need_secondary') {          // the server wants a second opinion from another model
      const need = d.need_models.filter((id) => models.includes(id));
      if (need.length) {
        ui.setProcessing('Double-checking with a second model…');
        await client.ensureModels(need, (id, pct) => ui.setProcessing(`Loading ${id} (${pct}%)…`));
        for (const f of frames) Object.assign(f.emb, await client.embed(f.crop, need));
        this.assertActive();
        res = await this._send(frames, true); d = res.data || {};
      }
    }
    ui.setProcessing(null);
    this._show(res, d);
  }

  _send(frames, final = false) {
    const { config } = this;
    return recognizeFace({ consent: true, final, profile: this.profile.id,
      frames: frames.map((f) => ({ pose: f.q.pose.bucket, quality: qualitySummary(f.q), embeddings: packEmbeddings(f.emb, config) })) });
  }

  _show(res, d) {
    const ev = d.evidence || {}, why = { model_disagreement: 'The recognition models disagree.', ambiguous: 'Two registered faces look alike.', low_support: 'Too few frames agreed.' };
    if (res.ok && d.status === 'match') ui_lines(this.ui, 'Match Found', [`Name: ${d.name}`, `Models agreeing: ${ev.models_agreeing} of ${ev.models_used}`, `Frames agreeing: ${ev.supporting_frames} of ${ev.frames}`], 'ok');
    else if (res.ok && d.status === 'unknown') ui_lines(this.ui, 'Unknown Face', ['No matching registered face found.'], 'warn');
    else if (res.ok && (d.status === 'uncertain' || d.status === 'need_secondary')) ui_lines(this.ui, 'Unable to identify reliably', [why[d.reason] || 'Please try again in better light.', 'Please try again.'], 'warn');
    else if (res.ok && d.status === 'retry') ui_lines(this.ui, 'Image quality too low', [d.message || 'Please try again in good light.'], 'warn');
    else this.ui.showError(d.message || 'Recognition failed. Please try again.');
  }
}
const ui_lines = (ui, title, lines, kind) => ui.showResult({ title, lines, kind });
