/**
 * Client-side sample hygiene before anything is sent: duplicate rejection and outlier detection on the PRIMARY model's
 * embeddings. The server repeats the same checks (backend/templates.py) - the client only saves a round trip.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./preprocess.js'));
  else { const F = root.FR = root.FR || {}; F.templates = factory(F.preprocess); }
})(typeof self !== 'undefined' ? self : this, function (P) {
  'use strict';

  const median = (a) => { const s = [...a].sort((x, y) => x - y), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };

  /**
   * samples: [{pose, quality (0..1), vec:unit Array}] -> {keep:[indices], dropped:[{index, reason:'duplicate'|'outlier'|'limit'}]}
   * cfg: {duplicate_similarity, outlier_mad_k, max_templates, consistency_min}
   */
  function selectTemplates(samples, cfg) {
    const order = samples.map((_, i) => i).sort((a, b) => samples[b].quality - samples[a].quality);
    const dropped = [], kept = [];
    for (const i of order) {                                   // best quality first; later near-identical same-pose samples are duplicates
      const dup = kept.some((k) => samples[k].pose === samples[i].pose && P.cosine(samples[k].vec, samples[i].vec) >= cfg.duplicate_similarity);
      if (dup) dropped.push({ index: i, reason: 'duplicate' }); else kept.push(i);
    }
    let pool = kept;
    if (pool.length >= 4) {                                    // robust outlier rejection on mean similarity to the others
      const means = pool.map((i) => pool.reduce((s, j) => (j === i ? s : s + P.cosine(samples[i].vec, samples[j].vec)), 0) / (pool.length - 1));
      const med = median(means), mad = median(means.map((m) => Math.abs(m - med))) * 1.4826;
      const floor = Math.max(cfg.consistency_min, med - cfg.outlier_mad_k * Math.max(mad, 0.02));
      pool = pool.filter((i, k) => { if (means[k] < floor) { dropped.push({ index: i, reason: 'outlier' }); return false; } return true; });
    }
    pool.sort((a, b) => samples[b].quality - samples[a].quality);
    for (const i of pool.slice(cfg.max_templates)) dropped.push({ index: i, reason: 'limit' });
    return { keep: pool.slice(0, cfg.max_templates), dropped };
  }

  return { selectTemplates, median };
});
