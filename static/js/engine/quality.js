/**
 * Face Quality Assessment Engine. Pure functions.
 * Every threshold lives in the model manifest (models/model_manifest.json -> "quality") and is PROVISIONAL until
 * calibrated with tools/benchmark.py on real, consented images from your own cameras.
 * "Occlusion" is a heuristic proxy (detector score + two-detector landmark agreement + box inside frame), NOT a dedicated
 * occlusion model.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./preprocess.js'));
  else { root.FR = root.FR || {}; root.FR.quality = factory(root.FR.preprocess); }
})(typeof self !== 'undefined' ? self : this, function (P) {
  'use strict';

  const FRONT_PITCH_RATIO = (P.ARCFACE_112[2][1] - P.ARCFACE_112[0][1]) / (P.ARCFACE_112[3][1] - P.ARCFACE_112[0][1]); // ~0.494

  /** Head pose proxies from the five landmarks. yaw>0: nose toward image-right (subject turned to THEIR left). */
  function poseFromLandmarks(lm) {
    const eyeMidX = (lm[0][0] + lm[1][0]) / 2, eyeMidY = (lm[0][1] + lm[1][1]) / 2;
    const mouthMidY = (lm[3][1] + lm[4][1]) / 2;
    const eyeDist = Math.hypot(lm[1][0] - lm[0][0], lm[1][1] - lm[0][1]) || 1;
    const yaw = (lm[2][0] - eyeMidX) / eyeDist;
    const pitch = (lm[2][1] - eyeMidY) / ((mouthMidY - eyeMidY) || 1) - FRONT_PITCH_RATIO;
    const roll = Math.atan2(lm[1][1] - lm[0][1], lm[1][0] - lm[0][0]) * 180 / Math.PI;
    return { yaw, pitch, roll, eyeDist };
  }

  function bucketPose(p, t) {
    if (Math.abs(p.yaw) <= t.front_max_yaw && Math.abs(p.pitch) <= t.front_max_pitch) return 'front';
    if (Math.abs(p.yaw) >= t.side_min_yaw && Math.abs(p.yaw) >= Math.abs(p.pitch) * 3) return p.yaw > 0 ? 'left' : 'right';
    if (Math.abs(p.yaw) < t.side_min_yaw && Math.abs(p.pitch) >= t.vertical_min_pitch) return p.pitch < 0 ? 'up' : 'down';
    return 'other';
  }

  /** Metrics on an aligned 112x112 RGB crop. */
  function cropMetrics(rgb, size = 112) {
    const n = size * size, gray = new Float32Array(n);
    let sum = 0, dark = 0, bright = 0;
    for (let i = 0; i < n; i++) {
      const g = 0.299 * rgb[3 * i] + 0.587 * rgb[3 * i + 1] + 0.114 * rgb[3 * i + 2];
      gray[i] = g; sum += g; if (g < 10) dark++; else if (g > 245) bright++;
    }
    const mean = sum / n;
    let v = 0; for (let i = 0; i < n; i++) v += (gray[i] - mean) ** 2;
    const contrast = Math.sqrt(v / n);
    let ls = 0, lq = 0, c = 0, gx2 = 0, gy2 = 0;
    for (let y = 1; y < size - 1; y++) for (let x = 1; x < size - 1; x++) {
      const i = y * size + x;
      const l = gray[i - 1] + gray[i + 1] + gray[i - size] + gray[i + size] - 4 * gray[i];
      ls += l; lq += l * l; c++;
      const gx = gray[i + 1] - gray[i - 1], gy = gray[i + size] - gray[i - size];
      gx2 += gx * gx; gy2 += gy * gy;
    }
    return {
      brightness: mean, contrast,
      sharpness: lq / c - (ls / c) ** 2,                  // variance of the Laplacian
      motionBlur: Math.abs(Math.log2((gx2 + 1) / (gy2 + 1))), // strong directional imbalance of gradients
      clipFrac: (dark + bright) / n,
    };
  }

  const clamp01 = (x) => Math.max(0, Math.min(1, x));
  const ramp = (x, lo, hi) => clamp01((x - lo) / (hi - lo));   // 0 at lo, 1 at hi

  /** Landmark agreement of two detectors: 1 - mean landmark distance / inter-ocular distance (clamped). */
  function landmarkAgreement(a, b) {
    const eye = Math.hypot(a[1][0] - a[0][0], a[1][1] - a[0][1]) || 1;
    let s = 0; for (let i = 0; i < 5; i++) s += Math.hypot(a[i][0] - b[i][0], a[i][1] - b[i][1]);
    return clamp01(1 - (s / 5) / eye);
  }

  /**
   * Full assessment. ctx = {frameW, frameH, crop, targetPose ('any'|'front'|'left'|'right'|'up'|'down'), agreement (0..1|null), prevCenter ([x,y]|null)}
   * -> {score 0..1, ok, flags:[{code,message}], metrics, pose:{yaw,pitch,roll,bucket}}
   */
  function assess(face, ctx, T) {
    const m = cropMetrics(ctx.crop), p = poseFromLandmarks(face.lm), bucket = bucketPose(p, T);
    const minDim = Math.min(ctx.frameW, ctx.frameH), flags = [];
    const add = (code, message) => flags.push({ code, message });
    const margin = 0.01 * minDim;
    const cx = face.x + face.w / 2, cy = face.y + face.h / 2;
    const inside = !(face.x < -margin || face.y < -margin || face.x + face.w > ctx.frameW + margin || face.y + face.h > ctx.frameH + margin);
    let motion = 0;
    if (ctx.prevCenter) motion = Math.hypot(cx - ctx.prevCenter[0], cy - ctx.prevCenter[1]) / minDim;

    if (face.score < T.min_detection_score) add('low_confidence', 'Face not clear. Face the camera in good light.');
    if (p.eyeDist < T.min_eye_distance_px || face.w < T.min_face_frac * minDim) add('too_small', 'Move closer to the camera.');
    else if (face.w > T.max_face_frac * minDim) add('too_large', 'Move back a little.');
    if (!inside) add('outside', 'Keep your whole face inside the frame.');
    else if (cx < 0.2 * ctx.frameW || cx > 0.8 * ctx.frameW || cy < 0.15 * ctx.frameH || cy > 0.85 * ctx.frameH) add('off_center', 'Center your face in the frame.');
    if (Math.abs(p.roll) > T.max_roll_deg) add('tilted', 'Keep your head upright.');
    const tp = ctx.targetPose || 'any';
    if (tp === 'any') { if (Math.abs(p.yaw) > T.any_max_yaw) add('pose', 'Please face the camera.'); }
    else if (bucket !== tp) {
      const msg = { front: 'Look straight at the camera.', left: 'Turn your face slightly to your left.', right: 'Turn your face slightly to your right.',
        up: 'Tilt your chin slightly up.', down: 'Tilt your chin slightly down.' }[tp];
      add('pose', msg);
    } else if (tp === 'left' || tp === 'right') { if (Math.abs(p.yaw) > T.side_max_yaw) add('pose_extreme', 'Not so far: turn only slightly.'); }
    if (m.brightness < T.min_brightness) add('dark', 'Too dark. Move to a brighter place.');
    else if (m.brightness > T.max_brightness) add('bright', 'Too bright. Avoid strong light on your face.');
    else if (m.contrast < T.min_contrast) add('low_contrast', 'Low contrast. Face the light source.');
    if (m.clipFrac > T.max_clip_frac) add('exposure', 'Lighting is uneven. Avoid strong backlight or shadows.');
    if (m.sharpness < T.min_sharpness) add('blurry', 'Image is blurry. Hold the camera steady.');
    else if (m.motionBlur > T.max_motion_blur || motion > T.max_motion_frac) add('motion', 'Hold still for a moment.');
    if (ctx.agreement != null && ctx.agreement < T.min_landmark_agreement) add('landmarks', 'Face position unclear. Remove hair, hands or masks from your face.');

    const comps = [
      ramp(face.score, T.min_detection_score, 0.95),
      ramp(p.eyeDist, T.min_eye_distance_px, 2.2 * T.min_eye_distance_px),
      ramp(m.sharpness, T.min_sharpness, 4 * T.min_sharpness),
      1 - clamp01(Math.abs(m.brightness - 128) / 128),
      1 - clamp01(Math.abs(p.roll) / (2 * T.max_roll_deg)),
    ];
    if (ctx.agreement != null) comps.push(ramp(ctx.agreement, T.min_landmark_agreement - 0.1, 0.98));
    const score = Math.exp(comps.reduce((s, c) => s + Math.log(Math.max(c, 0.02)), 0) / comps.length); // geometric mean
    return { score, ok: flags.length === 0, flags, metrics: { ...m, eyeDist: p.eyeDist, detScore: face.score, agreement: ctx.agreement, motion },
      pose: { yaw: p.yaw, pitch: p.pitch, roll: p.roll, bucket } };
  }

  return { poseFromLandmarks, bucketPose, cropMetrics, landmarkAgreement, assess, FRONT_PITCH_RATIO };
});
