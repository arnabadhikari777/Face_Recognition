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
/** Config load state: 'pending' | 'ready' | 'error' */
let configState = 'pending';
let configLoadPromise = null;
let configRetryCount = 0;

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
    if (e.code === 'NO_WORKER') return e.message;
    if (e.code === 'WORKER') return e.message;
    return `${e.message} If this keeps happening your device may be too limited for this mode: choose Balanced or Compatibility mode and try again.`;
  }
  if (e && e.name === 'AbortError') return 'The request was cancelled or timed out. Please try again.';
  if (e && e.message && !(e instanceof TypeError)) return e.message;
  console.error(e);
  return 'Something went wrong. Please reload the page and try again.';
}

function profileObject(id) {
  if (!config || !config.profiles || !config.profiles[id]) {
    throw new Error('Invalid or missing processing mode. Please reload and try again.');
  }
  return { id, ...config.profiles[id] };
}

function describeProfile() {
  if (!config || !config.profiles) {
    els.profileNote.textContent = 'Loading configuration…';
    return;
  }
  const id = els.profile.value;
  const p = config.profiles[id];
  if (!p) {
    els.profileNote.textContent = 'Select a processing mode.';
    return;
  }
  const names = (p.recognizers || []).map((rid) => {
    const m = (config.models || []).find((mm) => mm.id === rid);
    return m ? m.name.split(' (')[0] : rid;
  });
  const ids = [...(p.detectors || []), ...(p.recognizers || [])];
  const mb = Math.round(
    (config.models || []).filter((m) => ids.includes(m.id)).reduce((s, m) => s + (m.size_bytes || 0), 0) / 1048576
  );
  let note = `${p.label}: ${names.join(' + ')}. First load downloads about ${mb} MB (then cached).`;
  if (id === 'accuracy' && ((navigator.deviceMemory && navigator.deviceMemory <= 2) || (navigator.hardwareConcurrency && navigator.hardwareConcurrency <= 2))) {
    note += ' This device looks low on memory or CPU: if loading fails, choose Balanced or Compatibility mode.';
  }
  els.profileNote.textContent = note;
}

function setStartEnabled(enabled) {
  els.start.disabled = !enabled;
  els.profile.disabled = !enabled;
}

function populateProfiles() {
  while (els.profile.firstChild) els.profile.removeChild(els.profile.firstChild);
  if (!config || !config.profiles) return;
  for (const [id, p] of Object.entries(config.profiles)) {
    const o = document.createElement('option');
    o.value = id;
    o.textContent = p.label || id;
    els.profile.append(o);
  }
  let saved = null;
  try { saved = localStorage.getItem('fr.profile'); } catch (_) { /* storage blocked */ }
  const preferred = (saved && config.profiles[saved]) ? saved : (config.default_profile || Object.keys(config.profiles)[0]);
  if (preferred && config.profiles[preferred]) els.profile.value = preferred;
  describeProfile();
}

/**
 * Load /api/config once (or retry). Concurrent callers share the same promise.
 * Resolves with config object or throws a user-facing Error.
 */
function loadConfig(force = false) {
  if (!force && configState === 'ready' && config) return Promise.resolve(config);
  if (!force && configLoadPromise) return configLoadPromise;

  configState = 'pending';
  setStartEnabled(false);
  if (els.profileNote) els.profileNote.textContent = 'Loading configuration…';

  configLoadPromise = (async () => {
    const r = await getConfig();
    if (!r.ok) {
      const msg = (r.data && r.data.message) || (r.status === 0
        ? 'Cannot reach the server. Check your internet connection and try again.'
        : `Server error (${r.status}). Please try again.`);
      configState = 'error';
      config = null;
      throw new Error(msg);
    }
    if (!r.data || !r.data.profiles || typeof r.data.profiles !== 'object') {
      configState = 'error';
      config = null;
      throw new Error('Server returned an invalid configuration. Please try again later.');
    }
    config = r.data;
    configState = 'ready';
    configRetryCount = 0;
    populateProfiles();
    setStartEnabled(true);
    return config;
  })();

  // Allow a later force-retry to create a new promise after this one settles.
  configLoadPromise.finally(() => { configLoadPromise = null; });
  return configLoadPromise;
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

  // If config is not ready yet, keep Start disabled and attempt load (with retry UI on failure).
  if (configState !== 'ready') {
    setStartEnabled(false);
    els.profileNote.textContent = 'Loading configuration…';
    loadConfig().catch((e) => {
      if (mode) {  // still in this workflow
        ui.showError(e.message + ' You can close and try again, or reload the page.');
        els.profileNote.textContent = 'Configuration unavailable. Close and retry, or reload.';
      }
    });
  } else {
    setStartEnabled(true);
    describeProfile();
  }
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
  if (!config || !config.profiles || !config.profiles[profileId]) {
    throw new Error('Invalid processing mode selected. Please choose another mode or reload.');
  }
  const c = new EngineClient(config, profileId);
  await c.init((id, pct) => {
    if (mine === session) {
      const m = (config.models || []).find((mm) => mm.id === id);
      const label = m ? m.name.split(' (')[0] : id;
      showLoading(`Loading ${label}…`, pct);
    }
  });
  client = c; clientProfile = profileId;
  els.engineInfo.textContent = 'Engine: ' + Object.entries(c.backends).map(([id, b]) => `${id}:${b}`).join(', ');
}

async function start() {
  ui.clear();
  const mine = ++session;
  let name = '';
  if (mode.needsName) { name = els.name.value.trim(); if (!name) return ui.showError('Please enter a name.'); }
  if (!els.consent.checked) return ui.showError('Please confirm consent to continue.');
  if (configState !== 'ready' || !config) {
    return ui.showError('Configuration is still loading or failed. Please wait a moment or reload the page.');
  }
  const profileId = els.profile.value;
  if (!profileId || !config.profiles[profileId]) {
    return ui.showError('Please select a valid processing mode.');
  }
  els.setup.hidden = true;
  try {
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
      if (client && config && config.profiles[profileId] && !config.profiles[profileId].always_run_all_recognizers) {
        const extra = config.profiles[profileId].recognizers.slice(1);
        if (extra.length) client.release(extra).then(() => extra.forEach((id) => delete client.backends[id])).catch(() => {});
      }
    }
  }
}

async function init() {
  setStartEnabled(false);
  els.profileNote.textContent = 'Loading configuration…';
  try {
    await loadConfig();
  } catch (e) {
    // Non-fatal at boot: user can still open a workflow and retry, or reload.
    console.warn('Initial /api/config failed:', e.message);
    // Leave a note on the home screen via profile note if they open a flow.
  }
  // Wire change handler once.
  els.profile.addEventListener('change', describeProfile);
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
