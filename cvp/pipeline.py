"""CVP inference (Algorithm 1) and the per-document features shared by every method."""
import os
import pickle

import numpy as np
import torch

from . import doctypes, ocr, visual

MODELS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def heads_path(doctype_name):
    return os.path.join(MODELS, f"heads_{doctype_name}.pkl")


def pool(H, boxes, schema, pad=3):
    """Per-field pooled heatmap statistics: max, mean, 90th percentile."""
    out = np.zeros((len(schema.field_names), 3), np.float32)
    h, w = H.shape
    for i, f in enumerate(schema.field_names):
        x0, y0, x1, y1 = boxes[f]
        r = H[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)]
        if r.size:
            out[i] = [r.max(), r.mean(), np.percentile(r, 90)]
    return out


def class_posterior(model, X, n_classes):
    """g_phi's class posterior as a full row per class name. A class no training document
    was labelled with (a field no rule reads is never the target of a single-field edit)
    gets probability 0 instead of shifting every later column."""
    P = np.zeros((len(X), n_classes))
    P[:, model.classes_] = model.predict_proba(X)
    return P


def constraint_features(o, schema):
    """[w; r; e; q]: violations, residuals, per-field parse failures and OCR confidences."""
    w, r = schema.evaluate(o["values"])
    fail = np.array([o["values"][f] is None for f in schema.field_names], np.float32)
    conf = np.array([o["conf"][f] for f in schema.field_names], np.float32)
    return np.concatenate([w, r, fail, conf]), w, r


def field_prior(p_cls, w, schema, shared=False):
    """Constraint-branch posterior over fields: each class's mass spread over its
    candidate fields; the coherent class spreads uniformly over all fields.

    With shared=True, the classes whose signature fits the observed violations
    (necessary <= V <= possible) pool their mass and spread it evenly over all their
    candidates. The violations alone cannot tell those fields apart (a shared signature,
    e.g. cash and change), so the constraint branch does not prefer one of them and the
    visual branch decides."""
    F = schema.field_names
    pi = np.full(len(F), p_cls[schema.class_index["coherent"]] / len(F))
    V = schema.violated(w) if shared else set()
    fits = [k for k in schema.classes if schema.necessary[k] <= V <= schema.possible[k]] if V else []
    if fits:
        C = schema.consistent_fields(w)
        mass = sum(p_cls[schema.class_index[k]] for k in fits)
        for f in C:
            pi[schema.field_index[f]] += mass / len(C)
    for k in schema.classes:
        if k in fits:
            continue
        cand = schema.candidate_fields(k, w)
        for f in cand:
            pi[schema.field_index[f]] += p_cls[schema.class_index[k]] / len(cand)
    return pi / pi.sum()


HEAT = ("prob", "log", "logmax")


def attribution_features(p_cls, w, pooled, schema, shared=False, heat="prob"):
    """Per-field evidence for the attribution head, n_fields x (1 + k): the log field prior,
    then the pooled heatmap statistics -- max, mean and 90th percentile as probabilities
    ("prob"), the same in log space ("log"), or only the log of the max ("logmax"). In log
    space the score is a weighted product of experts, prior^a * heat^b, so a saturated
    heatmap (1e-7 against 2e-7) still ranks the fields instead of vanishing beside the prior."""
    pi = field_prior(p_cls, w, schema, shared)
    if heat == "prob":
        h = pooled
    elif heat == "log":
        h = np.log(np.clip(pooled, 1e-12, None))
    elif heat == "logmax":
        h = np.log(np.clip(pooled[:, :1], 1e-12, None))
    else:
        raise ValueError(f"heat must be one of {HEAT}")
    return np.concatenate([np.log(pi + 1e-6)[:, None], h], 1)


def visual_detection_features(pooled):
    m = np.sort(pooled[:, 0])[::-1]
    return np.array([m[0], m[1], pooled[:, 0].mean(), pooled[:, 1].max()], np.float32)


def noisy_or(p_violation, p_visual):
    """Detection as in Algorithm 1: forged if the constraints are violated OR the pixels
    look edited; a soft OR keeps each branch's evidence undiluted by the other."""
    return 1 - (1 - p_violation) * (1 - p_visual)


def cvp_rule(p_cls, w, pooled, schema):
    """Algorithm 1 as stated: arg-max class, candidate set, max-pooled heatmap inside it."""
    k = schema.class_names[int(np.argmax(p_cls))]
    cand = schema.candidate_fields(k, w)
    return max(cand, key=lambda f: pooled[schema.field_index[f], 0])


_SHARED = {}


def shared_models(device):
    """The OCR and the visual branch serve every document type: load them once."""
    if device not in _SHARED:
        _SHARED[device] = (ocr.Reader(device="cpu"), visual.load(device))
    return _SHARED[device]


class CVP:
    """End-to-end system: image -> decision, class, attributed field, and the evidence."""

    def __init__(self, doctype="transcript", device=None):
        self.doctype = doctypes.get(doctype) if isinstance(doctype, str) else doctype
        self.schema = self.doctype.schema
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        with open(heads_path(self.doctype.name), "rb") as fh:
            self.heads = pickle.load(fh)
        self.reader, self.visual = shared_models(self.device)

    def __call__(self, arr):
        s = self.schema
        o = self.reader.read(arr, self.doctype)
        feats, w, r = constraint_features(o, s)
        H, _ = visual.heatmap(self.visual, [arr], self.device)
        H = H[0]
        pooled = pool(H, o["boxes"], s)
        p_cls = class_posterior(self.heads["g_phi"], self.heads["g_scaler"].transform(feats[None]),
                                len(s.class_names))[0]
        p_vis = float(self.heads["det"].predict_proba(visual_detection_features(pooled)[None])[0, 1])
        p_tamper = float(noisy_or(1 - p_cls[0], p_vis))
        shared = self.heads.get("shared_prior", False)  # how the heads were fitted
        heat = self.heads.get("heat", "prob")
        scores = attribution_features(p_cls, w, pooled, s, shared, heat) @ self.heads["attr_w"]
        k = s.class_names[int(np.argmax(p_cls))]
        return dict(ocr=o, w=w, r=r, p_cls=p_cls, cls=k, candidates=s.candidate_fields(k, w),
                    signature_class=s.signature_class(w), consistent=s.consistent_fields(w), heatmap=H, pooled=pooled,
                    p_tamper=p_tamper, tampered=p_tamper >= self.heads["det_threshold"],
                    field=s.field_names[int(np.argmax(scores))], field_scores=scores,
                    field_prior=field_prior(p_cls, w, s, shared))
