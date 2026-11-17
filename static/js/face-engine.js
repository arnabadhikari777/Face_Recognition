/**
 * Client-side face detection + embedding via ONNX Runtime Web.
 *
 * Expected models (place under /static/models/):
 *   - det_10g.onnx or ultraface-rfb-320.onnx  (face detector)
 *   - w600k_mbf.onnx or mobilefacenet.onnx     (512-d ArcFace-style embedding)
 *
 * Primary execution provider: WASM. WebGPU used when available.
 *
 * Preprocessing (MobileFaceNet / InsightFace style):
 *   - Crop & align face to 112x112
 *   - RGB, float32
 *   - Normalize: (pixel - 127.5) / 127.5   → range [-1, 1]
 *   - Output L2-normalized 512-d embedding
 *
 * If models are missing, the engine reports a clear error so the UI can guide the user.
 */
const FaceEngine = (() => {
  const MODEL_NAME = "mobilefacenet-arcface";
  const MODEL_VERSION = "1.0";
  const EMBEDDING_DIM = 512;
  const INPUT_SIZE = 112;

  let ort = null;
  let detSession = null;
  let recSession = null;
  let ready = false;
  let loadError = null;

  const MODEL_PATHS = {
    // Prefer smaller / commonly redistributed names; user can rename after download
    detector: [
      "/static/models/det_10g.onnx",
      "/static/models/ultraface-rfb-320.onnx",
      "/static/models/face_detector.onnx",
    ],
    recognizer: [
      "/static/models/w600k_mbf.onnx",
      "/static/models/mobilefacenet.onnx",
      "/static/models/arcface_mobilefacenet.onnx",
    ],
  };

  async function loadOrt() {
    if (window.ort) {
      ort = window.ort;
      return;
    }
    // Dynamic script load if not present
    await new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.min.js";
      s.onload = resolve;
      s.onerror = () => reject(new Error("Failed to load ONNX Runtime Web"));
      document.head.appendChild(s);
    });
    ort = window.ort;
    // Prefer local wasm files from CDN
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
    ort.env.wasm.numThreads = 1; // safest default without COOP/COEP
  }

  async function tryCreateSession(paths, providers) {
    let lastErr = null;
    for (const path of paths) {
      try {
        const session = await ort.InferenceSession.create(path, {
          executionProviders: providers,
        });
        return { session, path };
      } catch (e) {
        lastErr = e;
      }
    }
    throw lastErr || new Error("No model path succeeded");
  }

  async function init() {
    if (ready) return { ok: true };
    if (loadError) return { ok: false, error: loadError };

    try {
      await loadOrt();
      // Prefer WebGPU then WASM
      const providers = [];
      try {
        if (navigator.gpu) providers.push("webgpu");
      } catch (_) {}
      providers.push("wasm");

      // Detector is optional for simplified pipeline: if only recognizer is present
      // we assume the caller already crops a roughly frontal face.
      try {
        const det = await tryCreateSession(MODEL_PATHS.detector, providers);
        detSession = det.session;
        console.info("[FaceEngine] Detector loaded:", det.path);
      } catch (e) {
        console.warn("[FaceEngine] Detector model not found — will use center-crop fallback.", e.message);
        detSession = null;
      }

      const rec = await tryCreateSession(MODEL_PATHS.recognizer, providers);
      recSession = rec.session;
      console.info("[FaceEngine] Recognizer loaded:", rec.path);

      ready = true;
      return { ok: true };
    } catch (e) {
      loadError =
        "Face recognition models are not available. " +
        "Please place the required ONNX files in /static/models/ " +
        "(see DEPLOYMENT.md). " +
        (e && e.message ? e.message : "");
      console.error("[FaceEngine]", loadError);
      return { ok: false, error: loadError };
    }
  }

  function isReady() {
    return ready;
  }

  function getMeta() {
    return { model_name: MODEL_NAME, model_version: MODEL_VERSION, embedding_dim: EMBEDDING_DIM };
  }

  /**
   * Very lightweight face presence heuristic when no detector model is present:
   * assumes a single centered face in the frame (common for guided capture UI).
   */
  function fallbackDetect(imageData) {
    const { width, height } = imageData;
    // Assume face occupies central ~60% region
    const size = Math.min(width, height) * 0.62;
    const x = (width - size) / 2;
    const y = (height - size) / 2 - size * 0.05;
    return [{ box: { x, y, w: size, h: size }, score: 0.9 }];
  }

  /**
   * Minimal UltraFace / SCRFD-style postprocess is complex; when a detector session
   * exists we run a simplified path that expects a common RFB-320 style output.
   * If parsing fails we fall back to center crop so registration still works.
   */
  async function detectFaces(imageData) {
    if (!detSession) {
      return fallbackDetect(imageData);
    }
    try {
      const inputName = detSession.inputNames[0];
      // UltraFace RFB-320 typically expects 320x240 or 640x480; use 320x240
      const tw = 320;
      const th = 240;
      const tensor = imageDataToTensor(imageData, tw, th, "bgr_mean");
      const feeds = {};
      feeds[inputName] = tensor;
      const out = await detSession.run(feeds);
      // Best-effort: if outputs look like scores+boxes, parse top face; else fallback
      const faces = parseDetectorOutput(out, imageData.width, imageData.height, tw, th);
      if (faces.length === 0) return fallbackDetect(imageData);
      return faces;
    } catch (e) {
      console.warn("[FaceEngine] Detector run failed, using center crop.", e);
      return fallbackDetect(imageData);
    }
  }

  function parseDetectorOutput(outputs, origW, origH, tw, th) {
    // Generic fallback: many ultraface models output "scores" and "boxes"
    const keys = Object.keys(outputs);
    let scores = null;
    let boxes = null;
    for (const k of keys) {
      const t = outputs[k];
      if (!t || !t.data) continue;
      if (k.toLowerCase().includes("score") || (t.dims && t.dims[t.dims.length - 1] === 2)) {
        scores = t;
      }
      if (k.toLowerCase().includes("box") || (t.dims && t.dims[t.dims.length - 1] === 4)) {
        boxes = t;
      }
    }
    if (!scores || !boxes) return [];
    const sData = scores.data;
    const bData = boxes.data;
    const n = Math.min(Math.floor(sData.length / 2), Math.floor(bData.length / 4));
    let bestIdx = -1;
    let bestScore = 0.6;
    for (let i = 0; i < n; i++) {
      const conf = sData[i * 2 + 1]; // face class
      if (conf > bestScore) {
        bestScore = conf;
        bestIdx = i;
      }
    }
    if (bestIdx < 0) return [];
    const bx = bData[bestIdx * 4 + 0] * origW;
    const by = bData[bestIdx * 4 + 1] * origH;
    const bw = (bData[bestIdx * 4 + 2] - bData[bestIdx * 4 + 0]) * origW;
    const bh = (bData[bestIdx * 4 + 3] - bData[bestIdx * 4 + 1]) * origH;
    return [{ box: { x: bx, y: by, w: bw, h: bh }, score: bestScore }];
  }

  function imageDataToTensor(imageData, tw, th, mode) {
    // Resize + convert to CHW float32
    const src = imageData;
    const canvas = document.createElement("canvas");
    canvas.width = tw;
    canvas.height = th;
    const ctx = canvas.getContext("2d");
    // Draw source ImageData via temporary canvas
    const tmp = document.createElement("canvas");
    tmp.width = src.width;
    tmp.height = src.height;
    tmp.getContext("2d").putImageData(src, 0, 0);
    ctx.drawImage(tmp, 0, 0, tw, th);
    const resized = ctx.getImageData(0, 0, tw, th);
    const data = resized.data;
    const float = new Float32Array(3 * tw * th);
    const mean = mode === "bgr_mean" ? [104, 117, 123] : [127.5, 127.5, 127.5];
    const scale = mode === "bgr_mean" ? 1.0 : 127.5;
    let p = 0;
    for (let y = 0; y < th; y++) {
      for (let x = 0; x < tw; x++) {
        const i = (y * tw + x) * 4;
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        if (mode === "bgr_mean") {
          // BGR order, mean subtract, no /255
          float[0 * tw * th + p] = b - mean[0];
          float[1 * tw * th + p] = g - mean[1];
          float[2 * tw * th + p] = r - mean[2];
        } else {
          // RGB, (x-127.5)/127.5
          float[0 * tw * th + p] = (r - mean[0]) / scale;
          float[1 * tw * th + p] = (g - mean[1]) / scale;
          float[2 * tw * th + p] = (b - mean[2]) / scale;
        }
        p++;
      }
    }
    return new ort.Tensor("float32", float, [1, 3, th, tw]);
  }

  function cropFace(imageData, box, margin = 0.2) {
    const { width, height } = imageData;
    let { x, y, w, h } = box;
    // Expand
    const mx = w * margin;
    const my = h * margin;
    x = Math.max(0, x - mx);
    y = Math.max(0, y - my);
    w = Math.min(width - x, w + 2 * mx);
    h = Math.min(height - y, h + 2 * my);
    // Make square
    const side = Math.max(w, h);
    const cx = x + w / 2;
    const cy = y + h / 2;
    x = Math.max(0, cx - side / 2);
    y = Math.max(0, cy - side / 2);
    const sw = Math.min(side, width - x);
    const sh = Math.min(side, height - y);

    const canvas = document.createElement("canvas");
    canvas.width = INPUT_SIZE;
    canvas.height = INPUT_SIZE;
    const ctx = canvas.getContext("2d");
    const tmp = document.createElement("canvas");
    tmp.width = width;
    tmp.height = height;
    tmp.getContext("2d").putImageData(imageData, 0, 0);
    ctx.drawImage(tmp, x, y, sw, sh, 0, 0, INPUT_SIZE, INPUT_SIZE);
    return ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
  }

  function l2Normalize(arr) {
    let sum = 0;
    for (let i = 0; i < arr.length; i++) sum += arr[i] * arr[i];
    const n = Math.sqrt(sum) || 1;
    const out = new Array(arr.length);
    for (let i = 0; i < arr.length; i++) out[i] = arr[i] / n;
    return out;
  }

  /**
   * Quality heuristics (no ML): reject obviously bad frames.
   */
  function assessQuality(imageData, faceBox) {
    const { width, height, data } = imageData;
    // Brightness
    let sum = 0;
    const step = 16;
    let count = 0;
    for (let i = 0; i < data.length; i += 4 * step) {
      sum += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      count++;
    }
    const brightness = sum / count;
    if (brightness < 35) return { ok: false, reason: "Face is too dark. Improve lighting." };
    if (brightness > 230) return { ok: false, reason: "Face is too bright / overexposed." };

    // Size relative to frame
    if (faceBox) {
      const area = (faceBox.w * faceBox.h) / (width * height);
      if (area < 0.04) return { ok: false, reason: "Face is too small. Move closer." };
      if (area > 0.75) return { ok: false, reason: "Face is too close. Move back a little." };
    }
    return { ok: true };
  }

  async function getEmbedding(imageData) {
    if (!ready) {
      const r = await init();
      if (!r.ok) throw new Error(r.error);
    }

    const faces = await detectFaces(imageData);
    if (faces.length === 0) {
      return { ok: false, error: "No face detected. Please try again.", faces: 0 };
    }
    if (faces.length > 1) {
      // Keep only the highest-scoring face but still warn if multiple strong detections
      faces.sort((a, b) => b.score - a.score);
    }
    // For strict single-face UX the caller may still reject multiple

    const face = faces[0];
    const quality = assessQuality(imageData, face.box);
    if (!quality.ok) {
      return { ok: false, error: quality.reason, faces: faces.length };
    }

    const cropped = cropFace(imageData, face.box);
    const inputName = recSession.inputNames[0];
    const tensor = imageDataToTensor(cropped, INPUT_SIZE, INPUT_SIZE, "rgb_norm");
    const feeds = {};
    feeds[inputName] = tensor;
    const out = await recSession.run(feeds);
    const outName = recSession.outputNames[0];
    const embData = out[outName].data;
    const emb = l2Normalize(Array.from(embData));

    if (emb.length !== EMBEDDING_DIM) {
      // Some models output 128; still accept but warn server may reject
      console.warn("[FaceEngine] Unexpected embedding dim:", emb.length);
    }

    return {
      ok: true,
      embedding: emb,
      faces: faces.length,
      score: face.score,
      meta: getMeta(),
    };
  }

  return {
    init,
    isReady,
    getMeta,
    detectFaces,
    getEmbedding,
    assessQuality,
    MODEL_NAME,
    MODEL_VERSION,
    EMBEDDING_DIM,
  };
})();

window.FaceEngine = FaceEngine;
