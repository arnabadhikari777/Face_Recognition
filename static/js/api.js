/** Backend communication. Never throws for HTTP errors; returns {ok,status,data}. */
async function request(path, options = {}) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 70000);     // Render Free can need ~50 s to wake up
  try {
    const res = await fetch(path, { ...options, signal: ctrl.signal });
    let data = null;
    try { data = await res.json(); } catch { /* non-JSON body */ }
    return { ok: res.ok, status: res.status, data };
  } catch (e) {
    const message = e && e.name === 'AbortError'
      ? 'The server took too long to respond. It may be waking up; please try again in a minute.'
      : 'Cannot reach the server. Check your internet connection and try again.';
    return { ok: false, status: 0, data: { status: 'error', code: 'network', message } };
  } finally { clearTimeout(timer); }
}

const post = (path, body) => request(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

export const getConfig = () => request('/api/config');
export const registerFace = (payload) => post('/api/register-face', payload);
export const recognizeFace = (payload) => post('/api/recognize-face', payload);

/** {model_id: {version, vector}} for the models in `vectors` (model_id -> unit array). */
export function packEmbeddings(vectors, config) {
  const out = {};
  for (const [id, vec] of Object.entries(vectors)) out[id] = { version: config.models.find((m) => m.id === id).version, vector: vec };
  return out;
}
