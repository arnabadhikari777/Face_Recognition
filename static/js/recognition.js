import { CaptureFlow, startLiveCheck } from './face-engine.js';
import { recognizeFace } from './api.js';

export class Recognition extends CaptureFlow {
  async run() {
    const { els, ui, engine } = this;
    els.progress.hidden = true;
    els.instruction.textContent = 'Place one face in the frame, then press Scan.';
    ui.hint('Looking for a face…', false);
    const stopLive = startLiveCheck(els.video, engine, () => 'any',
      (r) => ui.hint(r.ok ? 'Looks good. Press Scan.' : r.message, r.ok));
    let embedding = null;
    try {
      while (!embedding) {
        await this.waitForCapture();
        els.capture.disabled = true;
        try {
          const r = await engine.analyze(els.video, 'any');
          this.assertActive();
          if (!r.ok) { ui.hint(r.message, false); continue; }
          embedding = await engine.embed(els.video, r.face);
          this.assertActive();
        } finally { els.capture.disabled = false; }
      }
    } finally { stopLive(); }
    ui.hideCamera();
    ui.setProcessing('Comparing with registered faces…');
    const res = await recognizeFace({
      consent: true, model_name: this.modelInfo.model_name, model_version: this.modelInfo.model_version, embedding });
    ui.setProcessing(null);
    const d = res.data || {};
    if (res.ok && d.status === 'match') {
      ui.showResult({ title: 'Match Found', lines: [`Name: ${d.name}`, `Similarity Score: ${Number(d.similarity_score).toFixed(3)}`,
        'The score is a cosine similarity, not the probability that the identification is correct.'] });
    } else if (res.ok && d.status === 'unknown') {
      ui.showResult({ title: 'Unknown Face', kind: 'warn', lines: ['No matching registered face found.'] });
    } else if (res.ok && d.status === 'uncertain') {
      ui.showResult({ title: 'Unable to identify reliably. Please try again.', kind: 'warn' });
    } else {
      ui.showError(d.message || 'Recognition failed. Please try again.');
    }
  }
}
