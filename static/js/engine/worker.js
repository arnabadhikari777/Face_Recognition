/* Classic Web Worker: ALL neural-network inference happens here, on the user's device, off the UI thread.
 * Messages in : {id,type:'init'|'ensure'|'analyze'|'embed'|'release'|'dispose'|'cancel', ...}
 * Messages out: {id,ok,result|error} and {type:'progress',modelId,pct}
 * A monotonically increasing epoch lets the page cancel unused inference: jobs of an older epoch are dropped. */
'use strict';
const BASE = self.location.pathname.replace(/[^/]*$/, '');
importScripts(BASE + 'preprocess.js', BASE + 'detectors.js', BASE + 'recognizers.js', BASE + 'quality.js', BASE + 'engine-core.js', BASE + 'model-store.js');

let core = null, epoch = 0, chain = Promise.resolve();

function reply(id, ok, payload, transfer) { self.postMessage(ok ? { id, ok, result: payload } : { id, ok, error: payload }, transfer || []); }
const errInfo = (e) => ({ code: e && e.code ? e.code : 'ERROR', message: e && e.message ? e.message : String(e) });

async function loadList(ids) {
  for (const id of ids) {
    if (core.has(id)) continue;
    await core.load(id, (mid, pct) => self.postMessage({ type: 'progress', modelId: mid, pct }));
  }
}

async function handle(m) {
  switch (m.type) {
    case 'init': {
      importScripts(m.ortBase + m.ortFile);
      self.ort.env.wasm.numThreads = 1;            // no SharedArrayBuffer / cross-origin isolation required
      self.ort.env.wasm.proxy = false;
      self.ort.env.wasm.wasmPaths = m.ortBase;     // .wasm files from the same pinned ONNX Runtime Web version
      core = new self.FR.core.FaceEngineCore({ ort: self.ort, manifest: m.manifest, profileId: m.profileId, loadModelBytes: self.FR.modelStore.loadModelBytes });
      await loadList(m.models);
      return { backends: Object.fromEntries(m.models.map((id) => [id, core.backendOf(id)])) };
    }
    case 'ensure': await loadList(m.models); return { backends: Object.fromEntries(m.models.map((id) => [id, core.backendOf(id)])) };
    case 'analyze': {
      const r = await core.analyze({ rgba: new Uint8Array(m.buffer), width: m.width, height: m.height }, m.opts || {});
      const crop = r.crop ? r.crop : null;
      return { r: { ok: r.ok, code: r.code, message: r.message, count: r.count, detector: r.detector, quality: r.quality || null, face: r.face || null, crop }, transfer: crop ? [crop.buffer] : [] };
    }
    case 'embed': return { r: await core.embed(new Uint8Array(m.crop), m.models) };
    case 'release': m.models.forEach((id) => core.release(id)); return { r: true };
    case 'dispose': if (core) core.dispose(); core = null; return { r: true };
    default: throw new Error('unknown message ' + m.type);
  }
}

self.onmessage = (ev) => {
  const m = ev.data;
  if (m.type === 'cancel') { epoch = m.epoch; return; }
  chain = chain.then(async () => {
    if (m.epoch !== undefined && m.epoch < epoch) return reply(m.id, false, { code: 'CANCELLED', message: 'cancelled' });
    try { const out = await handle(m); reply(m.id, true, out && out.r !== undefined ? out.r : out, out && out.transfer); }
    catch (e) { reply(m.id, false, errInfo(e)); }
  });
};
