import { CaptureFlow, startLiveCheck } from './face-engine.js';
import { registerFace } from './api.js';

const STEPS = [
  { pose: 'front', text: 'Look straight at the camera.' },
  { pose: 'left', text: "Slowly turn your face slightly to the person's left." },
  { pose: 'right', text: "Now turn your face slightly to the person's right." },
];

export class Registration extends CaptureFlow {
  async run(name) {
    const { els, ui, engine, camera } = this;
    const samples = [];
    els.progress.hidden = false;
    els.progress.max = STEPS.length;
    for (let i = 0; i < STEPS.length; i++) {
      const step = STEPS[i];
      els.progress.value = i;
      els.instruction.textContent = `Sample ${i + 1} of ${STEPS.length}: ${step.text}`;
      ui.hint('Looking for a face…', false);
      const stopLive = startLiveCheck(els.video, engine, () => step.pose, (r) => this.liveHint(r));
      try {
        let sample = null;
        while (!sample) {
          await this.waitForCapture();
          els.capture.disabled = true;
          try {
            const r = await engine.analyze(els.video, step.pose);
            this.assertActive();
            if (!r.ok) { ui.hint(r.message, false); continue; }
            sample = await engine.embed(els.video, r.face);
            this.assertActive();
          } finally { els.capture.disabled = false; }
        }
        samples.push(sample);
        els.progress.value = i + 1;
      } finally { stopLive(); }
    }
    ui.hideCamera();
    ui.setProcessing('Checking for duplicates and saving…');
    const res = await registerFace({
      name, consent: true, model_name: this.modelInfo.model_name, model_version: this.modelInfo.model_version,
      embeddings: samples });
    ui.setProcessing(null);
    const d = res.data || {};
    if (res.status === 201 && d.status === 'registered') {
      ui.showResult({ title: 'Face registered successfully.' });
      els.name.value = '';
    } else if (d.status === 'duplicate') {
      ui.showResult({ title: 'Face already registered', kind: 'warn',
        lines: [`Registered name: ${d.existing_name}`, 'No new record was created.'] });
    } else if (d.status === 'uncertain') {
      ui.showResult({ title: 'Unable to confirm', kind: 'warn', lines: [d.message] });
    } else {
      ui.showError(d.message || 'Registration failed. Please try again.');
    }
  }
}
