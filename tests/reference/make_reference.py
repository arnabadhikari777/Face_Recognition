"""Independent PYTHON reference for the browser pipeline (uses cv2 + onnxruntime, NOT the JS code).

For each fixture image it runs SCRFD det_500m with the InsightFace decoding, aligns with skimage-style similarity
transform + cv2.warpAffine, builds the blob with cv2.dnn.blobFromImage exactly as insightface's ArcFaceONNX.get_feat does
and writes landmarks + embeddings of every recognition model to a JSON file. tests/js/parity.test.mjs compares the JS
engine against it. Usage: python tests/reference/make_reference.py <models_dir> <images_dir> <out.json>
"""
import glob
import json
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

ARC = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)


def estimate(src, dst):  # same math as skimage SimilarityTransform.estimate (umeyama)
    src_m, dst_m = src.mean(0), dst.mean(0)
    s, d = src - src_m, dst - dst_m
    cov = d.T @ s / len(src)
    U, S, Vt = np.linalg.svd(cov)
    det = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    D = np.diag([1, det])
    R = U @ D @ Vt
    scale = (S * np.diag(D)).sum() / s.var(0).sum()
    t = dst_m - scale * R @ src_m
    return np.hstack([scale * R, t[:, None]]).astype(np.float32)


def scrfd_detect(sess, img_bgr, size=640, thr=0.5, nms_thr=0.4):
    h, w = img_bgr.shape[:2]
    r = min(size / w, size / h)
    nw, nh = int(w * r), int(h * r)
    det = np.zeros((size, size, 3), np.uint8)
    det[:nh, :nw] = cv2.resize(img_bgr, (nw, nh))
    # NOTE: InsightFace divides by the single ratio r, which is off by up to ~0.3% on the non-limiting axis because of the integer
    # rounding of nw/nh. We deliberately map back with the exact per-axis factors (nw/w, nh/h), same as the JS engine.
    sx, sy = nw / w, nh / h
    blob = cv2.dnn.blobFromImage(det, 1.0 / 128, (size, size), (127.5, 127.5, 127.5), swapRB=True)
    outs = sess.run(None, {sess.get_inputs()[0].name: blob})
    boxes, kpss, scores_all = [], [], []
    for idx, stride in enumerate((8, 16, 32)):
        sc, bb, kp = outs[idx], outs[idx + 3] * stride, outs[idx + 6] * stride
        n = size // stride
        ys, xs = np.mgrid[:n, :n]
        centers = np.stack([xs, ys], -1).reshape(-1, 2).astype(np.float32) * stride
        centers = np.repeat(centers, 2, axis=0)
        keep = np.where(sc[:, 0] >= thr)[0]
        b = np.stack([centers[:, 0] - bb[:, 0], centers[:, 1] - bb[:, 1], centers[:, 0] + bb[:, 2], centers[:, 1] + bb[:, 3]], 1)
        k = kp.reshape(-1, 5, 2) * 1.0 + centers[:, None, :]
        boxes.append(b[keep] / np.array([sx, sy, sx, sy], np.float32)); kpss.append(k[keep] / np.array([sx, sy], np.float32)); scores_all.append(sc[keep, 0])
    boxes, kpss, scores = np.concatenate(boxes), np.concatenate(kpss), np.concatenate(scores_all)
    order = scores.argsort()[::-1]
    boxes, kpss, scores = boxes[order], kpss[order], scores[order]
    keep = []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    idx = np.arange(len(scores))
    while idx.size:
        i = idx[0]; keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[idx[1:]]), np.maximum(y1[i], y1[idx[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[idx[1:]]), np.minimum(y2[i], y2[idx[1:]])
        inter = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1)
        ovr = inter / (areas[i] + areas[idx[1:]] - inter)
        idx = idx[1:][ovr <= nms_thr]
    return boxes[keep], kpss[keep], scores[keep]


def main(models, images, out):
    so = ort.SessionOptions(); so.log_severity_level = 3
    det = ort.InferenceSession(os.path.join(models, "det_500m.onnx"), so, providers=["CPUExecutionProvider"])
    recs = {m: ort.InferenceSession(os.path.join(models, f), so, providers=["CPUExecutionProvider"])
            for m, f in (("w600k_mbf", "w600k_mbf.onnx"), ("w600k_r50", "w600k_r50.onnx"))}
    result = {}
    for path in sorted(glob.glob(os.path.join(images, "*.jpg"))):
        img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)  # camera frames carry no EXIF rotation
        b, k, s = scrfd_detect(det, img)
        if len(s) == 0:
            result[os.path.basename(path)] = None; continue
        i = int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))
        M = estimate(k[i], ARC)
        aligned = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)
        blob = cv2.dnn.blobFromImages([aligned], 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        emb = {}
        for name, sess in recs.items():
            e = sess.run(None, {sess.get_inputs()[0].name: blob})[0][0]
            emb[name] = (e / np.linalg.norm(e)).tolist()
        result[os.path.basename(path)] = {"faces": int(len(s)), "score": float(s[i]), "box": b[i].tolist(), "landmarks": k[i].tolist(), "embeddings": emb}
    json.dump(result, open(out, "w"))
    print("wrote", out, {k: (v and v["faces"]) for k, v in result.items()})


if __name__ == "__main__":
    main(*sys.argv[1:4])
