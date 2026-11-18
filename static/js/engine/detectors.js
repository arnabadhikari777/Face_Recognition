/**
 * Face detectors. One dedicated preprocessing + decoding implementation per model.
 *
 *  YuNet (OpenCV Zoo, 2023mar): input NCHW float32, BGR, raw 0..255, FIXED 640x640.
 *      outputs cls_/obj_/bbox_/kps_ at strides 8/16/32. Landmarks: image-left eye, image-right eye, nose, left mouth, right mouth.
 *  SCRFD det_500m (InsightFace buffalo_s): input NCHW float32, RGB, (x-127.5)/128, dynamic size (multiple of 32).
 *      9 outputs: scores[3], bbox distances[3], keypoint offsets[3] for strides 8/16/32, 2 anchors per cell.
 * Faces are returned in SOURCE image pixels: {x,y,w,h,score,lm:[[x,y]*5]}.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./preprocess.js'));
  else { root.FR = root.FR || {}; root.FR.detectors = factory(root.FR.preprocess); }
})(typeof self !== 'undefined' ? self : this, function (P) {
  'use strict';

  function nms(dets, iouThr) {
    const sorted = [...dets].sort((p, q) => q.score - p.score), keep = [];
    for (const d of sorted) {
      let ok = true;
      for (const k of keep) {
        const ix = Math.max(0, Math.min(d.x + d.w, k.x + k.w) - Math.max(d.x, k.x));
        const iy = Math.max(0, Math.min(d.y + d.h, k.y + k.h) - Math.max(d.y, k.y));
        const inter = ix * iy, union = d.w * d.h + k.w * k.h - inter;
        if (union > 0 && inter / union > iouThr) { ok = false; break; }
      }
      if (ok) keep.push(d);
    }
    return keep;
  }

  const clamp01 = (v) => Math.min(1, Math.max(0, v));
  const STRIDES = [8, 16, 32];

  /* ---------------- YuNet ---------------- */
  const yunet = {
    id: 'yunet_2023mar',
    inputSize: 640,
    /** -> {tensor: Float32Array(3*S*S) BGR raw, shape, scale} */
    preprocess(rgba, w, h) {
      const S = this.inputSize, lb = P.letterboxTopLeft(rgba, w, h, S), plane = S * S, chw = new Float32Array(3 * plane);
      for (let i = 0; i < plane; i++) { chw[i] = lb.rgba[4 * i + 2]; chw[plane + i] = lb.rgba[4 * i + 1]; chw[2 * plane + i] = lb.rgba[4 * i]; }
      return { tensor: chw, shape: [1, 3, S, S], sx: lb.sx, sy: lb.sy };
    },
    /** get(name) -> Float32Array|null */
    decode(get, size, threshold) {
      const dets = [];
      for (const stride of STRIDES) {
        const cls = get('cls_' + stride), obj = get('obj_' + stride), bbox = get('bbox_' + stride), kps = get('kps_' + stride);
        if (!cls || !obj || !bbox || !kps) throw new Error('Unexpected face detector outputs.');
        const cols = Math.floor(size / stride), rows = cols;
        for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
          const i = r * cols + c, score = Math.sqrt(clamp01(cls[i]) * clamp01(obj[i]));
          if (!(score >= threshold)) continue;
          const cx = (c + bbox[i * 4]) * stride, cy = (r + bbox[i * 4 + 1]) * stride;
          const w = Math.exp(bbox[i * 4 + 2]) * stride, h = Math.exp(bbox[i * 4 + 3]) * stride;
          const lm = [];
          for (let n = 0; n < 5; n++) lm.push([(kps[i * 10 + 2 * n] + c) * stride, (kps[i * 10 + 2 * n + 1] + r) * stride]);
          dets.push({ x: cx - w / 2, y: cy - h / 2, w, h, score, lm });
        }
      }
      return dets;
    },
    nmsIou: 0.3,
  };

  /* ---------------- SCRFD ---------------- */
  const scrfd = {
    id: 'scrfd_500m',
    inputSize: 640,
    preprocess(rgba, w, h) {
      const S = this.inputSize, lb = P.letterboxTopLeft(rgba, w, h, S), plane = S * S, chw = new Float32Array(3 * plane);
      for (let i = 0; i < plane; i++) {
        chw[i] = (lb.rgba[4 * i] - 127.5) / 128; chw[plane + i] = (lb.rgba[4 * i + 1] - 127.5) / 128; chw[2 * plane + i] = (lb.rgba[4 * i + 2] - 127.5) / 128;
      } // RGB
      return { tensor: chw, shape: [1, 3, S, S], sx: lb.sx, sy: lb.sy };
    },
    /** outputs: array of {data, dims} in session output order. Groups by last dim (1=score, 4=bbox, 10=kps), larger first = stride 8. */
    decode(outputs, size, threshold) {
      const group = (dim) => outputs.filter((o) => o.dims[o.dims.length - 1] === dim).sort((a, b) => b.data.length / dim - a.data.length / dim);
      const sc = group(1), bb = group(4), kp = group(10);
      if (sc.length !== 3 || bb.length !== 3 || kp.length !== 3) throw new Error('Unexpected face detector outputs.');
      const dets = [];
      STRIDES.forEach((stride, k) => {
        const n = Math.floor(size / stride), scores = sc[k].data, bbox = bb[k].data, kps = kp[k].data;
        if (scores.length !== n * n * 2) throw new Error('Face detector output does not match the input size.');
        for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) for (let a = 0; a < 2; a++) {
          const i = (r * n + c) * 2 + a, score = scores[i];
          if (!(score >= threshold)) continue;
          const cx = c * stride, cy = r * stride;
          const x1 = cx - bbox[i * 4] * stride, y1 = cy - bbox[i * 4 + 1] * stride;
          const x2 = cx + bbox[i * 4 + 2] * stride, y2 = cy + bbox[i * 4 + 3] * stride;
          const lm = [];
          for (let q = 0; q < 5; q++) lm.push([cx + kps[i * 10 + 2 * q] * stride, cy + kps[i * 10 + 2 * q + 1] * stride]);
          dets.push({ x: x1, y: y1, w: x2 - x1, h: y2 - y1, score, lm });
        }
      });
      return dets;
    },
    nmsIou: 0.4,
  };

  /** Map letterboxed detections back to source pixels and run NMS. */
  function finalize(dets, nmsIou, sx, sy) {
    return nms(dets, nmsIou).map((d) => ({
      x: d.x / sx, y: d.y / sy, w: d.w / sx, h: d.h / sy, score: d.score,
      lm: d.lm.map(([x, y]) => [x / sx, y / sy]),
    }));
  }

  return { yunet, scrfd, nms, finalize, REGISTRY: { yunet_2023mar: yunet, scrfd_500m: scrfd } };
});
