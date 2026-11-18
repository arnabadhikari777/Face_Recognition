// UNIT tests for the pure engine modules (no ONNX models needed). Synthetic inputs only - these prove the maths,
// NOT real-world recognition accuracy.
import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const P = require('../../static/js/engine/preprocess.js');
const D = require('../../static/js/engine/detectors.js');
const Q = require('../../static/js/engine/quality.js');
const R = require('../../static/js/engine/recognizers.js');
const manifest = require('../../models/model_manifest.json');

test('estimateSimilarity recovers scale/rotation/translation exactly', () => {
  const ang = 0.3, s = 1.7, tx = 12, ty = -5;
  const src = P.ARCFACE_112.map(([x, y]) => [x, y]);
  const dst = src.map(([x, y]) => [s * (Math.cos(ang) * x - Math.sin(ang) * y) + tx, s * (Math.sin(ang) * x + Math.cos(ang) * y) + ty]);
  const t = P.estimateSimilarity(src, dst);
  assert.ok(Math.abs(t.a - s * Math.cos(ang)) < 1e-9 && Math.abs(t.b - s * Math.sin(ang)) < 1e-9);
  assert.ok(Math.abs(t.tx - tx) < 1e-9 && Math.abs(t.ty - ty) < 1e-9);
});

test('warpToTemplate: identity placement reproduces the source pixels', () => {
  const w = 160, h = 160, rgba = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const o = (y * w + x) * 4; rgba[o] = x; rgba[o + 1] = y; rgba[o + 2] = 77; rgba[o + 3] = 255; }
  const lm = P.ARCFACE_112.map(([x, y]) => [x + 20, y + 10]);
  const crop = P.warpToTemplate(rgba, w, h, lm);
  assert.equal(crop.length, 112 * 112 * 3);
  const i = (50 * 112 + 60) * 3;
  assert.equal(crop[i], 80); assert.equal(crop[i + 1], 60); assert.equal(crop[i + 2], 77);
});

test('letterboxTopLeft keeps aspect ratio and zero-pads', () => {
  const rgba = new Uint8Array(200 * 100 * 4).fill(255);
  const { rgba: out, scale } = P.letterboxTopLeft(rgba, 200, 100, 64);
  assert.equal(scale, 0.32);
  assert.equal(out[(10 * 64 + 10) * 4], 255);
  assert.equal(out[(40 * 64 + 10) * 4], 0);
});

test('l2normalize / cosine', () => {
  const v = P.l2normalize([3, 4]);
  assert.ok(Math.abs(P.cosine(v, v) - 1) < 1e-12);
  assert.throws(() => P.l2normalize([0, 0]));
  assert.throws(() => P.l2normalize([NaN, 1]));
});

test('nms removes overlapping boxes, keeps best', () => {
  const a = { x: 0, y: 0, w: 10, h: 10, score: 0.9, lm: [] }, b = { x: 1, y: 1, w: 10, h: 10, score: 0.8, lm: [] }, c = { x: 50, y: 50, w: 10, h: 10, score: 0.7, lm: [] };
  const k = D.nms([b, a, c], 0.3);
  assert.deepEqual(k.map((d) => d.score), [0.9, 0.7]);
});

test('YuNet decode places a synthetic detection correctly', () => {
  const S = 640, out = {};
  for (const st of [8, 16, 32]) {
    const n = S / st; out['cls_' + st] = new Float32Array(n * n); out['obj_' + st] = new Float32Array(n * n);
    out['bbox_' + st] = new Float32Array(n * n * 4); out['kps_' + st] = new Float32Array(n * n * 10);
  }
  const n = 80, r = 20, c = 30, i = r * n + c;           // stride 8 cell
  out.cls_8[i] = 0.81; out.obj_8[i] = 1;
  out.bbox_8.set([0.5, 0.5, Math.log(5), Math.log(6)], i * 4);
  out.kps_8.set([1, 1, 2, 1, 1.5, 2, 1, 3, 2, 3], i * 10);
  const d = D.yunet.decode((nme) => out[nme], S, 0.5);
  assert.equal(d.length, 1);
  assert.ok(Math.abs(d[0].score - 0.9) < 1e-6);
  assert.ok(Math.abs(d[0].x + d[0].w / 2 - (c + 0.5) * 8) < 1e-4 && Math.abs(d[0].w - 40) < 1e-4);
  assert.deepEqual(d[0].lm[0].map((v) => +v.toFixed(3)), [(1 + c) * 8, (1 + r) * 8]);
});

test('SCRFD decode places a synthetic detection correctly', () => {
  const S = 640, outs = [];
  for (const dim of [1, 4, 10]) for (const st of [8, 16, 32]) { const n = S / st; outs.push({ data: new Float32Array(n * n * 2 * dim), dims: [n * n * 2, dim] }); }
  const n = 80, r = 10, c = 12, a = 1, i = (r * n + c) * 2 + a;
  outs[0].data[i] = 0.77;
  outs[3].data.set([1, 2, 3, 4], i * 4);
  outs[6].data.set([0.5, 0.5, -0.5, 0.5, 0, 1, 0, 2, 1, 2], i * 10);
  const d = D.scrfd.decode(outs, S, 0.5);
  assert.equal(d.length, 1);
  assert.deepEqual([d[0].x, d[0].y, d[0].w, d[0].h], [c * 8 - 8, r * 8 - 16, 32, 48]);
  assert.deepEqual(d[0].lm[2], [c * 8, r * 8 + 8]);
});

test('model preprocessing: RGB, (x-127.5)/127.5, CHW', () => {
  const rgb = new Uint8Array(112 * 112 * 3); rgb[0] = 255; rgb[1] = 0; rgb[2] = 127.5 | 0;
  for (const id of ['w600k_mbf', 'w600k_r50']) {
    const t = R.REGISTRY[id].toTensor(rgb);
    assert.equal(t.length, 3 * 112 * 112);
    assert.ok(Math.abs(t[0] - 1) < 1e-6); assert.ok(Math.abs(t[112 * 112] + 1) < 1e-6);
  }
});

test('quality: blur lowers sharpness; dark/bright flagged; pose buckets', () => {
  const sharp = new Uint8Array(112 * 112 * 3), blur = new Uint8Array(112 * 112 * 3);
  for (let y = 0; y < 112; y++) for (let x = 0; x < 112; x++) for (let c = 0; c < 3; c++) {
    sharp[(y * 112 + x) * 3 + c] = ((x >> 2) + (y >> 2)) % 2 ? 200 : 50; blur[(y * 112 + x) * 3 + c] = 125 + 10 * Math.sin(x / 9);
  }
  assert.ok(Q.cropMetrics(sharp).sharpness > 10 * Q.cropMetrics(blur).sharpness);
  const T = manifest.quality, base = { x: 400, y: 200, w: 300, h: 380, score: 0.95, lm: P.ARCFACE_112.map(([x, y]) => [x * 3 + 400, y * 3 + 200]) };
  const ok = Q.assess(base, { frameW: 1280, frameH: 720, crop: sharp, targetPose: 'front' }, T);
  assert.equal(ok.pose.bucket, 'front');
  const dark = Q.assess(base, { frameW: 1280, frameH: 720, crop: new Uint8Array(112 * 112 * 3).fill(10), targetPose: 'front' }, T);
  assert.ok(dark.flags.some((f) => f.code === 'dark'));
  const turned = { ...base, lm: base.lm.map((p, i) => (i === 2 ? [p[0] + 25, p[1]] : p)) };
  assert.equal(Q.assess(turned, { frameW: 1280, frameH: 720, crop: sharp, targetPose: 'front' }, T).pose.bucket, 'left');
  const up = { ...base, lm: base.lm.map((p, i) => (i === 2 ? [p[0], p[1] - 14] : p)) };
  assert.equal(Q.bucketPose(Q.poseFromLandmarks(up.lm), T), 'up');
});

test('manifest is self-consistent', () => {
  const ids = new Set(manifest.models.map((m) => m.id));
  for (const p of Object.values(manifest.profiles)) for (const id of [...p.detectors, ...p.recognizers]) assert.ok(ids.has(id), id);
  for (const m of manifest.models) { assert.match(m.sha256, /^[0-9a-f]{64}$/); assert.ok(m.size_bytes > 0); }
  for (const id of Object.keys(R.REGISTRY)) assert.ok(ids.has(id));
});
