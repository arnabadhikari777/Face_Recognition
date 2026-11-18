import { Camera, CameraError } from './camera.js';
import { EngineClient, EngineError } from './engine-client.js';
import { Cancelled } from './capture.js';
import { getConfig } from './api.js';
import { Registration } from './registration.js';
import { Recognition } from './recognition.js';

const $ = (id) => document.getElementById(id);
const els = {
  home: $('home'), workflow: $('workflow'), title: $('workflow-title'), setup: $('setup'),
  nameField: $('name-field'), name: $('name-input'), consent: $('consent'), consentText: $('consent-text'),
  profile: $('profile-select'), profileNote: $('profile-note'),
  start: $('start-btn'), loading: $('loading'), loadProgress: $('load-progress'), loadingText: $('loading-text'),
  cameraArea: $('camera-area'), video: $('preview'), instruction: $('instruction'), progress: $('capture-progress'),
  hintEl: $('hint'), capture: $('capture-btn'), skip: $('skip-btn'), switchBtn: $('switch-btn'), processing: $('processing'),
  processingText: $('processing-text'), result: $('result'), error: $('error'), close: $('close-btn'), engineInfo: $('engine-info'),
};

const MODES = {
  register: { title: 'Register Face', Flow: Registration, capture: 'Capture', needsName: true,
    consent: 'I confirm that the person whose face is being registered has given informed consent.' },
  recognize: { title: 'Recognize Face', Flow: Recognition, capture: 'Scan', needsName: false,
    consent: "I confirm that I have permission to capture and check this person's face." },
};

const camera = new Camera(els.video);
let config = null, client = null, clientProfile = null, flow = null, mode = null, session = 0;

const ui = {
  hint(text, good) { els.hintEl.textContent = text; els.hintEl.classList.toggle('good', !!good); },
  hideCamera() { camera.stop(); els.cameraArea.hidden = true; },
  setProcessing(text) { els.processing.hidden = text == null; if (text != null) els.processingText.textContent = text; },
  showError(text) { els.error.textContent = text; els.error.hidden = false; },
  showResult({ title, lines = [], kind = 'ok' }) {
    els.result.replaceChildren();
    const strong = document.createElement('strong'); strong.textContent = title; els.result.append(strong);
    for (const l of lines) { const p = document.createElement('div'); p.textContent = l; els.result.append(p); }
    els.result.classList.toggle('warn', kind === 'warn');
    els.result.hidden = false;
  },
  clear() { els.error.hidden = true; els.result.hidden = true; },
};

function showLoading(text, pct) {
  els.loading.hidden = false; els.loadingText.textContent = text;
  if (typeof pct === 'number' && pct > 0) els.loadProgress.value = pct; else els.loadProgress.removeAttribute('value');
}

function friendly(e) {
  if (e instanceof CameraError) return e.message;
  if (e instanceof EngineError) {
    if (e.code === 'MODEL_MISSING') return 'The face model files are missing on the server. The site owner must check the deployment (see TROUBLESHOOTING_BN.md).';
    if (e.code === 'MODEL_CORRUPT' || e.code === 'MODEL_FETCH') return e.message;
    return `${e.message} If this keeps happening your device may be too limited for this mode: choose Balanced or Compatibility mode and try again.`;
  }
  if (e && e.message && !(e instanceof TypeError)) return e.message;
  console.error(e);
  return 'Something went wrong. Please reload the page and try again.';
}

function profileObject(id) { return { id, ...config.profiles[id] }; }

function describeProfile() {
  const p = config.profiles[els.profile.value];
  const names = p.recognizers.map((id) => config.models.find((m) => m.id === id).name.split(' (')[0]);
  const mb = Math.round(config.models.filter((m) => [...p.detectors, ...p.recognizers].includes(m.id)).reduce((s, m) => s + m.size_bytes, 0) / 1048576);
  let note = `${p.label}: ${names.join(' + ')}. First load downloads about ${mb} MB (then cached).`;
  if (els.profile.value === 'accuracy' && ((navigator.deviceMemory && navigator.deviceMemory <= 2) || (navigator.hardwareConcurrency && navigator.hardwareConcurrency <= 2))) {
    note += ' This device looks low on memory or CPU: if loading fails, choose Balanced or Compatibility mode.';
  }
  els.profileNote.textContent = note;
}

function open(kind) {
  close();
  mode = MODES[kind];
  els.title.textContent = mode.title;
  els.consentText.textContent = mode.consent;
  els.nameField.hidden = !mode.needsName;
  els.capture.textContent = mode.capture;
  els.consent.checked = false;
  ui.clear();
  els.setup.hidden = false; els.workflow.hidden = false; els.home.hidden = true;
}

function close() {
  session++;
  if (flow) flow.cancel();
  flow = null;
  ui.hideCamera(); ui.setProcessing(null); els.loading.hidden = true; ui.clear();
  els.workflow.hidden = true; els.home.hidden = false;
}

async function ensureClient(profileId, mine) {
  if (client && clientProfile === profileId) return;
  if (client) { await client.dispose(); client = null; }       // profile changed: free the old models
  const c = new EngineClient(config, profileId);
  await c.init((id, pct) => { if (mine === session) showLoading(`Loading ${config.models.find((m) => m.id === id).name.split(' (')[0]}…`, pct); });
  client = c; clientProfile = profileId;
  els.engineInfo.textContent = 'Engine: ' + Object.entries(c.backends).map(([id, b]) => `${id}:${b}`).join(', ');
}

async function start() {
  ui.clear();
  const mine = ++session;
  let name = '';
  if (mode.needsName) { name = els.name.value.trim(); if (!name) return ui.showError('Please enter a name.'); }
  if (!els.consent.checked) return ui.showError('Please confirm consent to continue.');
  els.setup.hidden = true;
  const profileId = els.profile.value;
  try {
    if (!config) {
      showLoading('Connecting to the server…');
      const r = await getConfig();
      if (!r.ok) throw new Error((r.data && r.data.message) || 'Cannot reach the server.');
      config = r.data;
    }
    localStorage.setItem('fr.profile', profileId);
    await ensureClient(profileId, mine);
    if (mine !== session) return;
    els.loading.hidden = true;
    await camera.start();
    if (mine !== session) { camera.stop(); return; }
    els.cameraArea.hidden = false;
    els.switchBtn.hidden = !(await camera.hasMultipleCameras());
    flow = new mode.Flow({ els, camera, client, ui, config, profile: profileObject(profileId) });
    await flow.run(name);
  } catch (e) {
    if (!(e instanceof Cancelled) && mine === session) ui.showError(friendly(e));
  } finally {
    if (mine === session) {
      flow = null;
      ui.hideCamera(); ui.setProcessing(null); els.loading.hidden = true;
      els.setup.hidden = false; els.consent.checked = false;
      // Memory cleanup: lazily loaded secondary models are released after use in Balanced mode.
      if (client && !config.profiles[profileId].always_run_all_recognizers) {
        const extra = config.profiles[profileId].recognizers.slice(1);
        if (extra.length) client.release(extra).then(() => extra.forEach((id) => delete client.backends[id])).catch(() => {});
      }
    }
  }
}

async function init() {
  const r = await getConfig();
  if (!r.ok) { return; }                 // handled again on Start; the buttons still work
  config = r.data;
  for (const [id, p] of Object.entries(config.profiles)) { const o = document.createElement('option'); o.value = id; o.textContent = p.label; els.profile.append(o); }
  let saved = null; try { saved = localStorage.getItem('fr.profile'); } catch (_) { /* storage blocked */ }
  els.profile.value = config.profiles[saved] ? saved : config.default_profile;
  els.profile.addEventListener('change', describeProfile);
  describeProfile();
}

els.start.addEventListener('click', start);
$('btn-register').addEventListener('click', () => open('register'));
$('btn-recognize').addEventListener('click', () => open('recognize'));
els.close.addEventListener('click', close);
els.capture.addEventListener('click', () => { if (flow) flow.capture(); });
els.skip.addEventListener('click', () => { if (flow) flow.skip(); });
els.switchBtn.addEventListener('click', async () => { try { await camera.switch(); } catch (e) { ui.showError(friendly(e)); } });
window.addEventListener('pagehide', () => { camera.stop(); if (client) client.cancel(); });
init();
