import { CaptureFlow, startLiveCheck, captureBurst, qualitySummary } from './capture.js';
import { registerFace, packEmbeddings } from './api.js';

const T = () => window.FR.templates;

export class Registration extends CaptureFlow {
  async run(name) {
    const { els, ui, client, config, profile } = this;
    const plan = config.registration.plan, models = profile.recognizers, primary = models[0];
    const total = plan.reduce((s, p) => s + p.count, 0);
    const chosen = [];           // {pose, crop, quality}
    els.progress.hidden = false; els.progress.max = total; els.progress.value = 0;

    for (const step of plan) {
      let tries = 0;
      for (;;) {
        els.instruction.textContent = `${step.text}${step.count > 1 ? ` (${step.count} samples)` : ''}${step.optional ? ' Optional.' : ''}`;
        els.skip.hidden = !step.optional; ui.hint('Looking for a face…', false);
        const stop = startLiveCheck(this, () => step.pose, (r) => ui.hint(r.ok ? 'Looks good. Press Capture.' : r.message, r.ok));
        let action;
        try { action = await this.waitForButton(); } finally { stop(); }
        if (action === 'skip') break;
        els.capture.disabled = true; els.skip.hidden = true;
        let burst;
        try { burst = await captureBurst(this, { pose: step.pose, want: step.count + 2, timeoutMs: config.registration.burst_timeout_ms, onStatus: (t, g) => ui.hint(t, g) }); }
        finally { els.capture.disabled = false; }
        const picked = [];
        for (const b of burst) { if (picked.length < step.count) picked.push(b); }
        if (picked.length < step.count && !step.optional) {
          if (++tries >= 3) throw new Error('Could not capture enough clear frames. Improve the lighting and try again.');
          ui.hint('Not enough clear frames. Try again.', false); continue;
        }
        picked.forEach((b) => chosen.push({ pose: step.pose, crop: b.crop, quality: b.quality }));
        els.progress.value = chosen.length;
        break;
      }
    }
    els.skip.hidden = true;
    if (chosen.length < config.registration.min_templates) throw new Error(`Only ${chosen.length} good samples were captured; at least ${config.registration.min_templates} are needed.`);

    // Embeddings for every enrolment model of the profile. Lazy models are loaded now.
    ui.hideCamera(); ui.setProcessing('Preparing face models…');
    const missing = models.filter((id) => !client.backends[id]);
    if (missing.length) await client.ensureModels(missing, (id, pct) => ui.setProcessing(`Loading ${id} (${pct}%)…`));
    const samples = [];
    for (let i = 0; i < chosen.length; i++) {
      ui.setProcessing(`Processing sample ${i + 1} of ${chosen.length}…`);
      const emb = await client.embed(chosen[i].crop, models); this.assertActive();
      samples.push({ pose: chosen[i].pose, q: chosen[i].quality, emb });
    }
    // Client-side hygiene on the primary model (the server repeats it): duplicates and outliers.
    const sel = T().selectTemplates(samples.map((s) => ({ pose: s.pose, quality: s.q.score, vec: s.emb[primary] })),
      { ...config.registration, consistency_min: config.consistency_min });
    if (sel.keep.length < config.registration.min_templates) {
      ui.setProcessing(null);
      return ui.showResult({ title: 'Samples were not consistent', kind: 'warn',
        lines: ['Some frames looked too different or too similar to each other.', 'Please try again in steady, even lighting.'] });
    }
    ui.setProcessing('Checking for duplicates and saving…');
    const res = await registerFace({ name, consent: true, profile: this.profile.id,
      samples: sel.keep.map((i) => ({ pose: samples[i].pose, quality: qualitySummary(samples[i].q), embeddings: packEmbeddings(samples[i].emb, config) })) });
    ui.setProcessing(null);
    const d = res.data || {};
    if (res.status === 201 && d.status === 'registered') { ui.showResult({ title: 'Face registered successfully.', lines: [`Saved ${d.templates} face templates for ${d.models.length} model(s).`] }); els.name.value = ''; }
    else if (d.status === 'upgraded') ui.showResult({ title: 'Registration updated', lines: [d.message] });
    else if (d.status === 'duplicate') ui.showResult({ title: 'Face already registered', kind: 'warn', lines: [`Registered name: ${d.existing_name}`, 'No new record was created.'] });
    else if (d.status === 'uncertain') ui.showResult({ title: 'Unable to confirm', kind: 'warn', lines: [d.message] });
    else ui.showError(d.message || 'Registration failed. Please try again.');
  }
}
