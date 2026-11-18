/**
 * Model byte loader used inside the Web Worker: download with progress -> verify size + SHA-256 against the manifest ->
 * keep a verified copy in the Cache API so the next visit needs no download. A corrupted/partial download is never used.
 */
(function (root, factory) {
  const F = root.FR = root.FR || {}; F.modelStore = factory();
  if (typeof module === 'object' && module.exports) module.exports = F.modelStore;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';
  const CACHE = 'fr-models-v1';

  class ModelFetchError extends Error { constructor(code, message) { super(message); this.code = code; } }

  async function sha256Hex(bytes) {
    const d = await crypto.subtle.digest('SHA-256', bytes);
    return Array.from(new Uint8Array(d), (b) => b.toString(16).padStart(2, '0')).join('');
  }

  async function verified(bytes, spec) {
    if (bytes.length !== spec.size_bytes) return false;
    return (await sha256Hex(bytes)) === spec.sha256;
  }

  /** @returns {Promise<Uint8Array>} verified model bytes */
  async function loadModelBytes(spec, onPct = () => {}) {
    const url = '/static/models/' + spec.filename, key = url + '?sha256=' + spec.sha256;
    let cache = null;
    try { cache = await caches.open(CACHE); const hit = await cache.match(key); if (hit) { const b = new Uint8Array(await hit.arrayBuffer()); if (await verified(b, spec)) { onPct(100); return b; } await cache.delete(key); } } catch (_) { /* Cache API unavailable: just download */ }
    let res;
    try { res = await fetch(url); } catch (_) { throw new ModelFetchError('MODEL_FETCH', `Could not download the face model (${spec.id}). Check your connection.`); }
    if (!res.ok) throw new ModelFetchError('MODEL_MISSING', `The face model ${spec.id} is missing on the server.`);
    const total = Number(res.headers.get('Content-Length')) || spec.size_bytes;
    const reader = res.body && res.body.getReader ? res.body.getReader() : null;
    let bytes;
    if (reader) {
      bytes = new Uint8Array(spec.size_bytes); let got = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (got + value.length > bytes.length) throw new ModelFetchError('MODEL_CORRUPT', `The model ${spec.id} is larger than expected.`);
        bytes.set(value, got); got += value.length; onPct(Math.min(99, Math.round((got / total) * 100)));
      }
      if (got !== spec.size_bytes) throw new ModelFetchError('MODEL_CORRUPT', `The model ${spec.id} download was incomplete.`);
    } else { bytes = new Uint8Array(await res.arrayBuffer()); }
    if (!(await verified(bytes, spec))) throw new ModelFetchError('MODEL_CORRUPT', `The model ${spec.id} failed its integrity check. Reload the page; if it persists the server files are damaged.`);
    if (cache) { try { await cache.put(key, new Response(bytes.slice().buffer, { headers: { 'Content-Type': 'application/octet-stream' } })); } catch (_) { /* quota: ignore */ } }
    onPct(100);
    return bytes;
  }

  return { loadModelBytes, ModelFetchError, sha256Hex };
});
