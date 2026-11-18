// Pure-logic tests for static/js/face-engine.js. Run: node --test tests/test_face_engine.mjs
// These do NOT run the ONNX models (no browser / no model files in this environment).
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  similarityTransform, applyTransform, SFACE_TEMPLATE, decodeYuNet, nms, grayStats, yawRatio,
  l2normalize, judgeFace, ENGINE,
} from '../static/js/face-engine.js';

test('similarity transform recovers a known scale/rotation/translation', () => {
  const th = 0.17, k = 2.3, tx = 140, ty = 90;
  const fwd = ([x, y]) => [k * (Math.cos(th) * x - Math.sin(th) * y) + tx, k * (Math.sin(th) * x + Math.cos(th) * y) + ty];
  const src = SFACE_TEMPLATE.map(fwd);
  const t = similarityTransform(src, SFACE_TEMPLATE);
  src.forEach((p, i) => {
    const q = applyTransform(t, p);
    assert.ok(Math.abs(q[0] - SFACE_TEMPLATE[i][0]) < 1e-6 && Math.abs(q[1] - SFACE_TEMPLATE[i][1]) < 1e-6);
  });
});

test('decodeYuNet decodes one synthetic detection', () => {
  const S = 64, o = {};
  for (const st of [8, 16, 32]) {
    const n = (S / st) ** 2;
    o[`cls_${st}`] = new Float32Array(n); o[`obj_${st}`] = new Float32Array(n);
    o[`bbox_${st}`] = new Float32Array(n * 4); o[`kps_${st}`] = new Float32Array(n * 10);
  }
  const i = 2 * 8 + 3; // r=2,c=3 at stride 8
  o.cls_8[i] = 0.9; o.obj_8[i] = 0.9;
  o.bbox_8.set([0.5, 0.5, Math.log(2), Math.log(2.5)], i * 4);
  const dets = decodeYuNet((n) => o[n], S, 0.7);
  assert.equal(dets.length, 1);
  const d = dets[0];
  assert.ok(Math.abs(d.w - 16) < 1e-4 && Math.abs(d.h - 20) < 1e-4);
  assert.ok(Math.abs(d.x + d.w / 2 - 28) < 1e-4 && Math.abs(d.y + d.h / 2 - 20) < 1e-4);
  assert.ok(Math.abs(d.score - 0.9) < 1e-6);
  assert.equal(d.lm.length, 5);
  assert.throws(() => decodeYuNet(() => null, S, 0.7));
});

test('nms removes overlapping boxes, keeps distant ones', () => {
  const a = { x: 0, y: 0, w: 10, h: 10, score: 0.9 }, b = { x: 1, y: 1, w: 10, h: 10, score: 0.8 };
  const c = { x: 50, y: 50, w: 10, h: 10, score: 0.7 };
  assert.deepEqual(nms([b, c, a], 0.3).map((d) => d.score), [0.9, 0.7]);
});

test('grayStats: flat image has zero sharpness; noisy image is sharp', () => {
  const w = 112, flat = new Uint8ClampedArray(w * w * 4).fill(128);
  const f = grayStats(flat, w, w);
  assert.ok(Math.abs(f.brightness - 128) < 1e-3 && f.sharpness < 1e-6 && f.contrast < 1e-6);
  const noisy = new Uint8ClampedArray(w * w * 4);
  for (let i = 0; i < w * w; i++) { const v = (i * 7919) % 2 ? 40 : 200; noisy.set([v, v, v, 255], 4 * i); }
  assert.ok(grayStats(noisy, w, w).sharpness > ENGINE.minSharpness);
});

test('yawRatio sign: nose toward image-right is positive', () => {
  const lm = (nx) => [[40, 50], [80, 50], [nx, 70], [45, 90], [75, 90]];
  assert.ok(yawRatio(lm(75)) > 0.2);
  assert.ok(yawRatio(lm(45)) < -0.2);
  assert.ok(Math.abs(yawRatio(lm(60))) < 1e-9);
});

test('l2normalize', () => {
  const v = l2normalize([3, 4]);
  assert.ok(Math.abs(v[0] - 0.6) < 1e-12 && Math.abs(v[1] - 0.8) < 1e-12);
  assert.throws(() => l2normalize([NaN, 1]));
  assert.throws(() => l2normalize([0, 0]));
});

test('judgeFace messages', () => {
  const good = { brightness: 120, contrast: 50, sharpness: 200 };
  const mk = (nx, w = 300) => ({ x: 170, y: 90, w, h: w, lm: [[200, 200], [300, 200], [nx, 260], [215, 320], [285, 320]] });
  const vw = 640, vh = 480;
  assert.equal(judgeFace(mk(250), vw, vh, good, 'front'), null);
  assert.match(judgeFace(mk(250, 40), vw, vh, good, 'front'), /closer/);
  assert.match(judgeFace(mk(250, 470), vw, vh, good, 'front'), /back/);
  assert.match(judgeFace(mk(280), vw, vh, good, 'front'), /straight/);
  assert.equal(judgeFace(mk(280), vw, vh, good, 'left'), null);      // nose toward image-right = subject's left
  assert.match(judgeFace(mk(280), vw, vh, good, 'right'), /right/);
  assert.equal(judgeFace(mk(220), vw, vh, good, 'right'), null);
  assert.match(judgeFace(mk(250), vw, vh, { ...good, brightness: 20 }, 'front'), /dark/);
  assert.match(judgeFace(mk(250), vw, vh, { ...good, brightness: 250 }, 'front'), /bright/);
  assert.match(judgeFace(mk(250), vw, vh, { ...good, sharpness: 1 }, 'front'), /blurry/);
  assert.equal(judgeFace(mk(280), vw, vh, good, 'any'), null);
});
