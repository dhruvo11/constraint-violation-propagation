"""CVP inference (Algorithm 1) and the per-document features shared by every method."""
import os
import pickle

import numpy as np
import torch

from . import ocr, visual
from .schema import (CLASSES, CLASS_INDEX, FIELD_CLASS, FIELDS, candidate_fields, evaluate,
                     signature_class)

HEADS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "heads.pkl")


def pool(H, boxes, pad=3):
    """Per-field pooled heatmap statistics: max, mean, 90th percentile."""
    out = np.zeros((len(FIELDS), 3), np.float32)
    h, w = H.shape
    for i, f in enumerate(FIELDS):
        x0, y0, x1, y1 = boxes[f]
        r = H[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)]
        if r.size:
            out[i] = [r.max(), r.mean(), np.percentile(r, 90)]
    return out


def constraint_features(o):
    w, r = evaluate(o["values"])
    fail = np.array([o["values"][f] is None for f in FIELDS], np.float32)
    conf = np.array([o["conf"][f] for f in FIELDS], np.float32)
    return np.concatenate([w, r, fail, conf]), w, r


def field_prior(p_cls, w):
    """Constraint-branch posterior over fields: each class's mass spread over its
    candidate fields; the coherent class spreads uniformly over all fields."""
    pi = np.full(len(FIELDS), p_cls[CLASS_INDEX["coherent"]] / len(FIELDS))
    for k in CLASSES[1:]:
        cand = candidate_fields(k, w)
        for f in cand:
            pi[FIELDS.index(f)] += p_cls[CLASS_INDEX[k]] / len(cand)
    return pi / pi.sum()


def attribution_features(p_cls, w, pooled):
    pi = field_prior(p_cls, w)
    return np.concatenate([np.log(pi + 1e-6)[:, None], pooled], 1)  # 10 x 4


def visual_detection_features(pooled):
    m = np.sort(pooled[:, 0])[::-1]
    return np.array([m[0], m[1], pooled[:, 0].mean(), pooled[:, 1].max()], np.float32)


def noisy_or(p_violation, p_visual):
    """Detection as in Algorithm 1: forged if the constraints are violated OR the pixels
    look edited; a soft OR keeps each branch's evidence undiluted by the other."""
    return 1 - (1 - p_violation) * (1 - p_visual)


def cvp_rule(p_cls, w, pooled):
    """Algorithm 1 as stated: arg-max class, candidate set, max-pooled heatmap inside it."""
    k = CLASSES[int(np.argmax(p_cls))]
    cand = candidate_fields(k, w)
    return max(cand, key=lambda f: pooled[FIELDS.index(f), 0])


class CVP:
    """End-to-end system used by the demo: image -> decision, class, field, evidence."""

    def __init__(self, device=None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.reader = ocr.Reader(device="cpu")
        self.visual = visual.load(self.device)
        with open(HEADS_PATH, "rb") as fh:
            self.heads = pickle.load(fh)

    def __call__(self, arr):
        o = self.reader.read(arr)
        feats, w, r = constraint_features(o)
        H, _ = visual.heatmap(self.visual, [arr], self.device)
        H = H[0]
        pooled = pool(H, o["boxes"])
        p_cls = self.heads["g_phi"].predict_proba(self.heads["g_scaler"].transform(feats[None]))[0]
        p_vis = float(self.heads["det"].predict_proba(visual_detection_features(pooled)[None])[0, 1])
        p_tamper = float(noisy_or(1 - p_cls[0], p_vis))
        scores = attribution_features(p_cls, w, pooled) @ self.heads["attr_w"]
        k = CLASSES[int(np.argmax(p_cls))]
        return dict(ocr=o, w=w, r=r, p_cls=p_cls, cls=k, candidates=candidate_fields(k, w),
                    signature_class=signature_class(w), heatmap=H, pooled=pooled,
                    p_tamper=p_tamper, tampered=p_tamper >= self.heads["det_threshold"],
                    field=FIELDS[int(np.argmax(scores))], field_scores=scores,
                    field_prior=field_prior(p_cls, w))
