// INTEGRATION test with REAL ONNX models (onnxruntime-node) and real photos. Skipped automatically if models/fixtures/reference are absent.
//   MODELS_DIR   folder produced by scripts/download_models.sh   (default static/models)
//   FIXTURES_DIR folder with *.jpg                              (not shipped: faces of real people need consent)
//   REF_JSON     output of tests/reference/make_reference.py
// Verifies that the JS pipeline (letterbox, SCRFD decode, alignment warp, per-model tensor, ORT, L2 norm) matches an
// independent cv2 + onnxruntime-python reference, and that the two detectors agree on landmarks.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const MODELS = process.env.MODELS_DIR || 'static/models', FIX = process.env.FIXTURES_DIR || '', REF = process.env.REF_JSON || '';
let ort = null, jpeg = null;
try { ort = require('onnxruntime-node'); jpeg = require('jpeg-js'); } catch { /* optional dev dependency */ }
const ready = ort && jpeg && FIX && REF && fs.existsSync(REF) && fs.existsSync(path.join(MODELS, 'w600k_r50.onnx'));
const opts = { skip: ready ? false : 'needs onnxruntime-node, jpeg-js, MODELS_DIR, FIXTURES_DIR, REF_JSON' };

const { FaceEngineCore } = require('../../static/js/engine/engine-core.js');
const P = require('../../static/js/engine/preprocess.js');
const Q = require('../../static/js/engine/quality.js');
const manifest = require('../../models/model_manifest.json');
// Documented tolerances.
//  warpCosine   : alignment + tensor + ONNX + L2 using the REFERENCE landmarks (pure preprocessing/model parity).
//  landmarkRel  : end-to-end landmark error relative to inter-ocular distance (cv2.resize vs the JS resize differ by sub-pixel
//                 amounts, which moves detector landmarks by ~1 px on big photos).
//  e2eCosine    : end-to-end embedding cosine; lower than warpCosine because embeddings are sensitive to ~1 px landmark jitter.
const TOL = { warpCosine: 0.999, landmarkRel: 0.02, e2eCosine: 0.97 };

async function makeEngine() {
  const eng = new FaceEngineCore({ ort, manifest, profileId: 'accuracy', detectorBackend: 'cpu',
    loadModelBytes: async (spec) => new Uint8Array(fs.readFileSync(path.join(MODELS, spec.filename))) });
  // onnxruntime-node uses the 'cpu' provider; the profile's browser backends (webgpu/wasm) do not exist in Node.
  eng.profile = { ...eng.profile, backends: ['cpu'] };
  for (const id of ['scrfd_500m', 'yunet_2023mar', 'w600k_mbf', 'w600k_r50']) await eng.load(id);
  return eng;
}
const load = (f) => { const d = jpeg.decode(fs.readFileSync(path.join(FIX, f)), { useTArray: true }); return { rgba: d.data, width: d.width, height: d.height }; };

test('JS pipeline matches the independent python reference (landmarks + embeddings, both models)', opts, async () => {
  const ref = JSON.parse(fs.readFileSync(REF, 'utf8')), eng = await makeEngine(), embs = {};
  for (const [file, r] of Object.entries(ref)) {
    if (!r) continue;
    const fr = load(file), { faces } = await eng.detectAdaptive(fr.rgba, fr.width, fr.height);
    faces.sort((a, b) => b.w * b.h - a.w * a.h);
    assert.equal(faces.length, r.faces, `${file}: face count`);
    const lmErr = faces[0].lm.reduce((s, p, i) => s + Math.hypot(p[0] - r.landmarks[i][0], p[1] - r.landmarks[i][1]), 0) / 5;
    const eye = Math.hypot(r.landmarks[1][0] - r.landmarks[0][0], r.landmarks[1][1] - r.landmarks[0][1]);
    assert.ok(lmErr / eye < TOL.landmarkRel, `${file}: mean landmark error ${lmErr.toFixed(3)} px = ${(lmErr / eye).toFixed(4)} of eye distance`);
    const crop = P.warpToTemplate(fr.rgba, fr.width, fr.height, faces[0].lm);
    const e = await eng.embed(crop, ['w600k_mbf', 'w600k_r50']);
    embs[file] = e;
    for (const m of ['w600k_mbf', 'w600k_r50']) {
      const c = P.cosine(e[m], r.embeddings[m]);
      console.log(`  ${file} ${m}: cosine(js, reference) = ${c.toFixed(5)}, landmark error ${lmErr.toFixed(3)} px (${(lmErr / eye).toFixed(4)} of eye distance)`);
      assert.ok(c >= TOL.e2eCosine, `${file} ${m}: cosine to reference ${c}`);
      const wc = P.cosine((await eng.embed(P.warpToTemplate(fr.rgba, fr.width, fr.height, r.landmarks), [m]))[m], r.embeddings[m]);
      console.log(`  ${file} ${m}: warp-only parity (reference landmarks) = ${wc.toFixed(5)}`);
      assert.ok(wc >= TOL.warpCosine, `${file} ${m}: warp-only cosine ${wc}`);
    }
  }
  for (const m of ['w600k_mbf', 'w600k_r50']) {   // sanity: same person scores clearly above different people
    const g = P.cosine(embs['obama.jpg'][m], embs['obama2.jpg'][m]), i = P.cosine(embs['obama.jpg'][m], embs['biden.jpg'][m]);
    console.log(`  ${m}: genuine ${g.toFixed(3)}  impostor ${i.toFixed(3)}`);
    assert.ok(g > i + 0.4, `${m}: genuine ${g} vs impostor ${i}`);
  }
  eng.dispose();
});

test('SCRFD and YuNet agree on the five landmarks (also proves landmark ordering is consistent)', opts, async () => {
  const eng = await makeEngine();
  for (const file of ['obama.jpg', 'obama2.jpg', 'biden.jpg']) {
    const fr = load(file);
    const a = (await eng.detect('scrfd_500m', fr.rgba, fr.width, fr.height))[0], b = (await eng.detect('yunet_2023mar', fr.rgba, fr.width, fr.height))[0];
    assert.ok(a && b, `${file}: both detectors find the face`);
    const ag = Q.landmarkAgreement(a.lm, b.lm);
    console.log(`  ${file}: landmark agreement ${ag.toFixed(3)}`);
    assert.ok(ag > 0.85, `${file}: agreement ${ag}`);
  }
  eng.dispose();
});

test('analyze(): returns crop + quality for a single face, rejects group photo, handles no face', opts, async () => {
  const eng = await makeEngine();
  const one = await eng.analyze(load('obama.jpg'), { targetPose: 'any', deep: true });
  assert.equal(one.count, 1); assert.equal(one.crop.length, 112 * 112 * 3); assert.ok(one.quality.score > 0 && one.quality.score <= 1);
  console.log('  obama.jpg quality', one.ok, one.code, JSON.stringify({ score: +one.quality.score.toFixed(3), agreement: one.quality.metrics.agreement, pose: one.quality.pose.bucket }));
  const grp = await eng.analyze(load('t1.jpg'), { targetPose: 'any' });
  assert.equal(grp.code, 'multiple_faces');
  const blank = await eng.analyze({ rgba: new Uint8Array(640 * 480 * 4).fill(120), width: 640, height: 480 }, {});
  assert.equal(blank.code, 'no_face');
  eng.dispose();
});
