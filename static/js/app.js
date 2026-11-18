import { Camera, CameraError } from './camera.js';
import { FaceEngine, ENGINE, Cancelled, ModelError } from './face-engine.js';
import { getConfig } from './api.js';
import { Registration } from './registration.js';
import { Recognition } from './recognition.js';

const $ = (id) => document.getElementById(id);
const els = {
  home: $('home'), workflow: $('workflow'), title: $('workflow-title'), setup: $('setup'),
  nameField: $('name-field'), name: $('name-input'), consent: $('consent'), consentText: $('consent-text'),
  start: $('start-btn'), loading: $('loading'), loadProgress: $('load-progress'), loadingText: $('loading-text'),
  cameraArea: $('camera-area'), video: $('preview'), instruction: $('instruction'), progress: $('capture-progress'),
  hintEl: $('hint'), capture: $('capture-btn'), switchBtn: $('switch-btn'), processing: $('processing'),
  processingText: $('processing-text'), result: $('result'), error: $('error'), close: $('close-btn'),
};

const MODES = {
  register: { title: 'Register Face', Flow: Registration, capture: 'Capture', needsName: true,
    consent: 'I confirm that the person whose face is being registered has given informed consent.' },
  recognize: { title: 'Recognize Face', Flow: Recognition, capture: 'Scan', needsName: false,
    consent: "I confirm that I have permission to capture and check this person's face." },
};

const camera = new Camera(els.video);
let engine = null, modelInfo = null, flow = null, mode = null, session = 0;

const ui = {
  hint(text, good) { els.hintEl.textContent = text; els.hintEl.classList.toggle('good', !!good); },
  hideCamera() { camera.stop(); els.cameraArea.hidden = true; },
  setProcessing(text) { els.processing.hidden = text == null; if (text != null) els.processingText.textContent = text; },
  showError(text) { els.error.textContent = text; els.error.hidden = false; },
  showResult({ title, lines = [], kind = 'ok' }) {
    els.result.replaceChildren();
    const strong = document.createElement('strong');
    strong.textContent = title;
    els.result.append(strong);
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
  if (e instanceof CameraError || e instanceof ModelError) {
    if (e.code === 'MODEL_MISSING') return 'The face model files are missing on the server. The site owner must add them (see README).';
    return e.message;
  }
  console.error(e);
  return 'Something went wrong. Please reload the page and try again.';
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
  els.setup.hidden = false;
  els.workflow.hidden = false;
  els.home.hidden = true;
}

function close() {
  session++;                      // invalidates any running flow's cleanup
  if (flow) flow.cancel();
  flow = null;
  ui.hideCamera(); ui.setProcessing(null); els.loading.hidden = true; ui.clear();
  els.workflow.hidden = true; els.home.hidden = false;
}

async function start() {
  ui.clear();
  const mine = ++session;
  let name = '';
  if (mode.needsName) {
    name = els.name.value.trim();
    if (!name) return ui.showError('Please enter a name.');
  }
  if (!els.consent.checked) return ui.showError('Please confirm consent to continue.');
  els.setup.hidden = true;
  try {
    if (!modelInfo) {
      showLoading('Connecting to the server…');
      const r = await getConfig();
      if (!r.ok) throw new Error((r.data && r.data.message) || 'Cannot reach the server.');
      modelInfo = r.data;
    }
    if (modelInfo.model_name !== ENGINE.modelName || modelInfo.model_version !== ENGINE.modelVersion
        || modelInfo.embedding_dim !== ENGINE.embeddingDim) {
      modelInfo = null;
      throw new Error('This page is out of date. Please reload it.');
    }
    if (!engine) {
      const e = new FaceEngine();
      await e.init((pct, text) => { if (mine === session) showLoading(text, pct); });
      engine = e;
    }
    if (mine !== session) return;
    els.loading.hidden = true;
    await camera.start();
    if (mine !== session) { camera.stop(); return; }
    els.cameraArea.hidden = false;
    els.switchBtn.hidden = !(await camera.hasMultipleCameras());
    flow = new mode.Flow({ els, camera, engine, ui, modelInfo });
    await flow.run(name);
  } catch (e) {
    if (!(e instanceof Cancelled) && mine === session) ui.showError(friendly(e));
  } finally {
    if (mine === session) {
      flow = null;
      ui.hideCamera(); ui.setProcessing(null); els.loading.hidden = true;
      els.setup.hidden = false; els.consent.checked = false;
    }
  }
}

els.start.addEventListener('click', start);
$('btn-register').addEventListener('click', () => open('register'));
$('btn-recognize').addEventListener('click', () => open('recognize'));
els.close.addEventListener('click', close);
els.capture.addEventListener('click', () => { if (flow) flow.capture(); });
els.switchBtn.addEventListener('click', async () => {
  try { await camera.switch(); } catch (e) { ui.showError(friendly(e)); }
});
window.addEventListener('pagehide', () => camera.stop());
