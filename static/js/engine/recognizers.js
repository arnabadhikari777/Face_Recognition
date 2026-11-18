/**
 * Recognition models. Each model has its OWN preprocessing function on purpose (never shared), plus its own
 * output handling. Verified against the InsightFace reference (ArcFaceONNX.get_feat):
 *   cv2.dnn.blobFromImages(imgs, scalefactor=1/127.5, size=(112,112), mean=(127.5,127.5,127.5), swapRB=True)
 *   => RGB, (x - 127.5) / 127.5, NCHW, 112x112. Output: 512-d, must be L2-normalised before cosine comparison.
 * Two models can have numerically identical preprocessing; they are still kept separate so that a future model
 * change cannot silently alter the other one.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./preprocess.js'));
  else { root.FR = root.FR || {}; root.FR.recognizers = factory(root.FR.preprocess); }
})(typeof self !== 'undefined' ? self : this, function (P) {
  'use strict';

  function mbfTensor(rgb112) {            // w600k_mbf (MobileFaceNet, WebFace600K)
    const plane = 112 * 112, t = new Float32Array(3 * plane);
    for (let i = 0; i < plane; i++) for (let c = 0; c < 3; c++) t[c * plane + i] = (rgb112[3 * i + c] - 127.5) / 127.5;
    return t;
  }
  function r50Tensor(rgb112) {            // w600k_r50 (IResNet-50, WebFace600K)
    const plane = 112 * 112, t = new Float32Array(3 * plane);
    for (let i = 0; i < plane; i++) for (let c = 0; c < 3; c++) t[c * plane + i] = (rgb112[3 * i + c] - 127.5) / 127.5;
    return t;
  }

  const REGISTRY = {
    w600k_mbf: { id: 'w600k_mbf', dim: 512, inputSize: 112, toTensor: mbfTensor },
    w600k_r50: { id: 'w600k_r50', dim: 512, inputSize: 112, toTensor: r50Tensor },
  };

  /** Run one model on an aligned RGB crop -> unit-length Array(dim). `session` is an ort.InferenceSession. */
  async function embed(ort, session, spec, rgb112) {
    const input = new ort.Tensor('float32', spec.toTensor(rgb112), [1, 3, spec.inputSize, spec.inputSize]);
    const out = await session.run({ [session.inputNames[0]]: input });
    const raw = out[session.outputNames[0]].data;
    if (raw.length !== spec.dim) throw new Error(`Model ${spec.id} returned ${raw.length} values, expected ${spec.dim}.`);
    return P.l2normalize(raw);
  }

  return { REGISTRY, embed };
});
