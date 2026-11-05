"""CVP end to end on one real receipt photo, as the real-data evaluation runs it.

The constraint branch reads every value where the receipt's layout labels put it
(cvp/real/read.py) and scores the reading with the receipt's own schema; the violated
rules nominate the consistent fields. The visual branch's heatmap (on the photo's visual
view) picks among them, or among all fields when no rule is violated (Algorithm 1 with
the signature lookup). Decision: a violated rule, or else enough heat.
"""
import json
import os

import numpy as np
import torch

from .. import paths, visual
from . import cord, forge, read, view
from . import receipt_schema as RS

# The real-photo visual branch, if it has been fine-tuned (scripts/reproduce_real.sh),
# else the synthetic one; the heat threshold comes from validation documents.
VISUAL_CHOICES = ["visual_cord_crude.pt", "visual_resunet.pt"]
THRESHOLD_FILE = os.path.join(paths.BASE, "results", "real", "cord_threshold.json")


def default_visual():
    for name in VISUAL_CHOICES:
        p = os.path.join(paths.MODELS, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("no trained visual branch in models/")


class RealCVP:
    def __init__(self, visual_path=None, device=None):
        from doctr.models import recognition_predictor
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.visual_path = visual_path or default_visual()
        self.reco = recognition_predictor("crnn_vgg16_bn", pretrained=True)
        m = visual.ResUNet(pretrained=False)
        m.load_state_dict(torch.load(self.visual_path, map_location="cpu"))
        self.model = m.eval().to(self.device)
        if self.device == "mps":
            self.model = self.model.half()
        self.threshold = 0.5
        if os.path.exists(THRESHOLD_FILE):
            t = json.load(open(THRESHOLD_FILE))
            self.threshold = t.get(os.path.basename(self.visual_path), self.threshold)

    def __call__(self, r, img):
        """Analyse photo `img` (RGB uint8) of receipt `r` (its layout labels locate the fields)."""
        rg = cord.from_lines(r.lines, rid=r.id)
        S = RS.schema(rg)
        words = forge.field_words(r)
        text = read.read_values(img, r.lines, self.reco)
        values = RS.values(cord.from_lines(r.lines, read=lambda w: text[id(w)]))
        w, res = S.evaluate(values)
        violated = sorted(S.violated(w), key=S.c_index.get)
        consistent = S.consistent_fields(w) if violated else list(S.field_names)
        candidates = S.candidate_fields(S.signature_class(w), w) if violated else list(S.field_names)
        small, s = view.visual_view(img)
        H = visual.heatmap(self.model, [small], self.device)[0][0]
        size = (img.shape[1], img.shape[0])
        boxes = {f: forge.bbox(words[f], size) for f in S.field_names if f in words}
        heat = {}
        for f, b in boxes.items():
            x0, y0, x1, y1 = view.scale_box(b, s)
            reg = H[max(0, y0 - 3):y1 + 3, max(0, x0 - 3):x1 + 3]
            heat[f] = float(reg.max()) if reg.size else 0.0
        pool = [f for f in consistent if f in heat] or list(heat)
        field = max(pool, key=lambda f: heat[f]) if pool else None
        p_vis = float(H.max())
        return dict(schema=S, truth=RS.values(rg), values=values, w=w, violated=violated,
                    consistent=consistent, candidates=candidates, heatmap=H, scale=s, boxes=boxes,
                    heat=heat, field=field, p_visual=p_vis, tampered=bool(violated) or p_vis >= self.threshold,
                    texts={wd["text"]: text[id(wd)] for line in r.lines for wd in cord.value_words(line)})
