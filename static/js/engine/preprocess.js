/**
 * Geometry + pixel helpers shared by every model (pure JS, no DOM, runs in a Web Worker and in Node tests).
 * Model-specific tensor layouts live in recognizers.js / detectors.js, NOT here.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else { root.FR = root.FR || {}; root.FR.preprocess = factory(); }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  /** InsightFace/ArcFace 112x112 five-point template: left eye, right eye, nose, left mouth, right mouth (image-left first). */
  const ARCFACE_112 = [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]];

  /** Least-squares similarity transform (Umeyama, no reflection) mapping src -> dst.
   *  x' = a*x - b*y + tx ; y' = b*x + a*y + ty */
  function estimateSimilarity(src, dst) {
    const n = src.length;
    let msx = 0, msy = 0, mdx = 0, mdy = 0;
    for (let i = 0; i < n; i++) { msx += src[i][0]; msy += src[i][1]; mdx += dst[i][0]; mdy += dst[i][1]; }
    msx /= n; msy /= n; mdx /= n; mdy /= n;
    let nr = 0, ni = 0, den = 0;
    for (let i = 0; i < n; i++) {
      const sx = src[i][0] - msx, sy = src[i][1] - msy, dx = dst[i][0] - mdx, dy = dst[i][1] - mdy;
      nr += dx * sx + dy * sy; ni += dy * sx - dx * sy; den += sx * sx + sy * sy;
    }
    if (!(den > 0)) throw new Error('degenerate landmarks');
    const a = nr / den, b = ni / den;
    return { a, b, tx: mdx - (a * msx - b * msy), ty: mdy - (b * msx + a * msy) };
  }

  function applySimilarity(t, p) { return [t.a * p[0] - t.b * p[1] + t.tx, t.b * p[0] + t.a * p[1] + t.ty]; }

  /** Warp an RGBA image into a size x size RGB crop (bilinear, constant-0 border = cv2.warpAffine defaults).
   *  Returns Uint8Array(size*size*3) in RGB order. */
  function warpToTemplate(rgba, w, h, landmarks, size = 112, template = ARCFACE_112) {
    const t = estimateSimilarity(landmarks, template);
    const s2 = t.a * t.a + t.b * t.b;
    const out = new Uint8Array(size * size * 3);
    const px = (x, y, c) => (x < 0 || y < 0 || x >= w || y >= h) ? 0 : rgba[(y * w + x) * 4 + c];
    for (let v = 0; v < size; v++) {
      for (let u = 0; u < size; u++) {
        const du = u - t.tx, dv = v - t.ty;
        const sx = (t.a * du + t.b * dv) / s2, sy = (-t.b * du + t.a * dv) / s2;
        const x0 = Math.floor(sx), y0 = Math.floor(sy), fx = sx - x0, fy = sy - y0;
        const o = (v * size + u) * 3;
        for (let c = 0; c < 3; c++) {
          const val = px(x0, y0, c) * (1 - fx) * (1 - fy) + px(x0 + 1, y0, c) * fx * (1 - fy)
            + px(x0, y0 + 1, c) * (1 - fx) * fy + px(x0 + 1, y0 + 1, c) * fx * fy;
          out[o + c] = Math.max(0, Math.min(255, Math.round(val)));
        }
      }
    }
    return out;
  }

  /** Bilinear resize of RGBA into a new RGBA buffer. */
  function resizeRGBA(src, w, h, nw, nh) {
    const out = new Uint8Array(nw * nh * 4);
    const rx = w / nw, ry = h / nh;
    for (let y = 0; y < nh; y++) {
      const sy = Math.max(0, Math.min(h - 1, (y + 0.5) * ry - 0.5)), y0 = Math.floor(sy), y1 = Math.min(h - 1, y0 + 1), fy = sy - y0;
      for (let x = 0; x < nw; x++) {
        const sx = Math.max(0, Math.min(w - 1, (x + 0.5) * rx - 0.5)), x0 = Math.floor(sx), x1 = Math.min(w - 1, x0 + 1), fx = sx - x0;
        for (let c = 0; c < 4; c++) {
          const v = src[(y0 * w + x0) * 4 + c] * (1 - fx) * (1 - fy) + src[(y0 * w + x1) * 4 + c] * fx * (1 - fy)
            + src[(y1 * w + x0) * 4 + c] * (1 - fx) * fy + src[(y1 * w + x1) * 4 + c] * fx * fy;
          out[(y * nw + x) * 4 + c] = v + 0.5;
        }
      }
    }
    return out;
  }

  /** Fit an image inside size x size keeping aspect ratio, placed top-left, zero padded (as InsightFace SCRFD does).
   *  Returns {rgba (size*size*4), scale, sx, sy}. */
  function letterboxTopLeft(rgba, w, h, size) {
    const scale = Math.min(size / w, size / h);
    const nw = Math.max(1, Math.min(size, Math.round(w * scale))), nh = Math.max(1, Math.min(size, Math.round(h * scale)));
    const resized = (nw === w && nh === h) ? rgba : resizeRGBA(rgba, w, h, nw, nh);
    const out = new Uint8Array(size * size * 4);
    for (let y = 0; y < nh; y++) out.set(resized.subarray(y * nw * 4, (y + 1) * nw * 4), y * size * 4);
    return { rgba: out, scale: Math.min(size / w, size / h), sx: nw / w, sy: nh / h };  // sx/sy: exact per-axis factors after integer rounding
  }

  function l2normalize(vec) {
    let s = 0;
    for (let i = 0; i < vec.length; i++) {
      if (!Number.isFinite(vec[i])) throw new Error('non-finite value in embedding');
      s += vec[i] * vec[i];
    }
    const n = Math.sqrt(s);
    if (!(n > 0)) throw new Error('zero-norm embedding');
    return Array.from(vec, (v) => v / n);
  }

  function cosine(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }

  return { ARCFACE_112, estimateSimilarity, applySimilarity, warpToTemplate, resizeRGBA, letterboxTopLeft, l2normalize, cosine };
});
