"""The analysis service behind the web app. It runs CVP on a document and returns everything the
page shows as plain data: the verdict and its reasons, every rule with its numbers, every field
with its box and heat, the image, the heatmap, and the answer key when the forgery is ours.

Four kinds of document:
- synthetic: the six template types (forged on request with the evaluation's forgers, or uploaded);
- cord: a real CORD v2 test receipt photo, forged on request with the evaluation's seeds;
- aiforge: a CORD receipt edited by a commercial AI editor (AIForge-Doc; local data only);
- receipt: any receipt photo, read end to end without labels (experimental, cvp/real/upload.py).
"""
import base64
import copy
import io
import json
import os
import threading
import time
import traceback
import uuid
from collections import OrderedDict

import numpy as np
from PIL import Image

from cvp import doctypes, paths, pipeline, visual
from cvp.core import render, tamper

from . import explain

CATEGORIES = ("genuine", "T1", "T2", "T3", "T4")
REAL_VISUAL = "visual_cord_crude.pt"     # real photos: the branch fine-tuned on CORD photos (crude edits)
ZERO_SHOT = "visual_resunet.pt"          # AIForge-Doc: the synthetic branch, as in the paper's Section 8.5
MAX_UPLOAD_SIDE = 2400


def _jpeg_uri(arr, max_side=1600, quality=88):
    im = Image.fromarray(arr)
    s = min(1.0, max_side / max(im.size))
    if s < 1.0:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _png_uri(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _heat_uri(H, max_width=900):
    """The heatmap as a transparent red layer (RGBA PNG), laid over the image by the page."""
    H = np.clip(np.nan_to_num(np.asarray(H, np.float32)), 0, 1)
    h, w = H.shape
    if w > max_width:
        H = np.asarray(Image.fromarray(H).resize((max_width, round(h * max_width / w)), Image.BILINEAR))
    a = (H ** 1.4 * 215).astype(np.uint8)
    rgba = np.zeros(H.shape + (4,), np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2], rgba[..., 3] = 235, 40, 40, a
    return _png_uri(rgba)


def _box(b, s=1.0):
    return [round(float(c) * s, 1) for c in b] if b is not None else None


def _rule_status(scope, values):
    return "unread" if any(values.get(f) is None for f in scope) else None


class Engine:
    def __init__(self):
        self.lock = threading.Lock()
        self._syn, self._real, self._reader = {}, {}, None
        self.state = dict(ready=False, step="starting", error=None)
        self.cache = {}
        self._kept = OrderedDict()  # recent experimental readings, so a corrected value can be re-checked

    # ------------------------------------------------------------------ what this machine can do
    def availability(self):
        from cvp.real import aiforge, cord
        out = dict(synthetic={n: os.path.exists(pipeline.heads_path(n)) for n in doctypes.names()})
        try:
            import doctr  # noqa: F401
            has_doctr = True
        except ImportError:
            has_doctr = False
        has_real_model = os.path.exists(os.path.join(paths.MODELS, REAL_VISUAL))
        out["cord"] = bool(has_doctr and has_real_model and cord.files("test"))
        out["aiforge"] = {v: bool(out["cord"] and aiforge.available(v)) for v in ("v1", "v2")}
        out["receipt"] = bool(has_doctr and has_real_model)
        out["cord_receipts"] = cord.count_receipts("test") if out["cord"] else 0
        return out

    # ------------------------------------------------------------------ models, loaded once
    def synthetic(self, name):
        if name not in self._syn:
            self._syn[name] = pipeline.CVP(name)
        return self._syn[name]

    def real(self, model=REAL_VISUAL):
        """RealCVP with the given visual branch; the OCR is shared between branches."""
        if model not in self._real:
            from cvp.real.system import RealCVP, THRESHOLD_FILE
            if not self._real:
                self._real[model] = RealCVP(visual_path=os.path.join(paths.MODELS, model))
            else:
                base = next(iter(self._real.values()))
                other = copy.copy(base)
                m = visual.ResUNet(pretrained=False)
                import torch
                m.load_state_dict(torch.load(os.path.join(paths.MODELS, model), map_location="cpu"))
                m = m.eval().to(base.device)
                other.model = m.half() if base.device == "mps" else m
                other.visual_path = os.path.join(paths.MODELS, model)
                thresholds = json.load(open(THRESHOLD_FILE)) if os.path.exists(THRESHOLD_FILE) else {}
                other.threshold = thresholds.get(model, 0.5)
                self._real[model] = other
        return self._real[model]

    def reader(self):
        if self._reader is None:
            from doctr.models import ocr_predictor
            self._reader = ocr_predictor(pretrained=True)
        return self._reader

    def warm_up(self, then=None):
        """Load the models in the background so the first click is quick; then run `then`."""
        def go():
            try:
                av = self.availability()
                self.state["step"] = "loading the OCR and the visual branch"
                with self.lock:
                    self.synthetic("transcript")
                if av["cord"]:
                    self.state["step"] = "loading the real-photo models"
                    with self.lock:
                        self.real(REAL_VISUAL)
                        if any(av["aiforge"].values()):
                            self.real(ZERO_SHOT)
                if av["receipt"]:
                    self.state["step"] = "loading the receipt reader"
                    with self.lock:
                        self.reader()
                self.state.update(ready=True, step="ready")
                if then:
                    then()
            except Exception as e:  # the page shows it; the rest of the app keeps working
                self.state.update(ready=True, step="ready", error=f"{type(e).__name__}: {e}")
                traceback.print_exc()
        threading.Thread(target=go, daemon=True).start()

    # ------------------------------------------------------------------ the four kinds of document
    def forge_synthetic(self, name, seed, category, field=None):
        DT = doctypes.get(name)
        S = DT.schema
        if field and (field not in S.field_names or not S.constrained(field) or category not in ("T1", "T2")):
            field = None
        rng = np.random.default_rng(int(seed))
        doc = render.generate(DT, rng)
        img, info = tamper.apply(doc, DT, category, rng, field=field)
        truth = dict(category=category, forged=bool(info["edited"]) or category != "genuine",
                     edited=list(info["edited"]), rects=[_box(r) for r in info["rects"]],
                     changes=[dict(field=f, label=S.by_name[f].label, old=explain.fmt(doc.values[f]),
                                   new=explain.fmt(info["values"][f])) for f in info["edited"]])
        title = f"{DT.title} #{int(seed)}"
        return self._synthetic_result(name, img, title, truth, source=f"Made by our generator, {explain.CATEGORY[category]}")

    def analyse_synthetic(self, name, arr):
        DT = doctypes.get(name)
        arr = np.asarray(Image.fromarray(arr).convert("RGB").resize(DT.page))
        return self._synthetic_result(name, arr, f"Uploaded {DT.title.lower()}", None, source="Uploaded by you")

    def forge_cord(self, k, category, field=None):
        from cvp.real import cord, forge
        k = max(0, min(cord.count_receipts("test") - 1, int(k)))
        r = cord.receipt("test", k)
        if category not in ("T1", "T2"):
            field = None
        rng = np.random.default_rng(1000 * k + forge.CATEGORIES.index(category))
        img, info = forge.forge(r, category, rng, field=field)
        sysm = self.real(REAL_VISUAL)
        res = sysm(r, img)
        labels = {f.name: f.label for f in res["schema"].fields}
        truth = dict(category=info["category"], forged=bool(info["edited"]), edited=list(info["edited"]),
                     rects=info["rects"], changes=[dict(field=f, label=labels.get(f, f), old=explain.fmt(info["old"][f]),
                                                        new=explain.fmt(info["new"][f])) for f in info["edited"]])
        return self._real_result(res, img, f"CORD receipt #{k}", truth, "cord",
                                 source=f"A real shop receipt (CORD v2 test set), {explain.CATEGORY[info['category']]} "
                                        "made by our forger", model=REAL_VISUAL, threshold=sysm.threshold)

    def aiforge(self, version, spec):
        from cvp.real import aiforge, cord, forge
        row = next(r for r in aiforge.metadata(version) if r.get("spec_id") == spec)
        split, k = aiforge.cord_index(row)
        r = cord.receipt(split, k)
        fake = aiforge.rgb(os.path.join(aiforge.local(version), row["image"]))
        mask = aiforge.rgb(os.path.join(aiforge.local(version), row["mask"]))[..., 0] > 127
        sysm = self.real(ZERO_SHOT)
        res = sysm(r, fake)
        S = res["schema"]
        labels = {f.name: f.label for f in S.fields}
        size = (fake.shape[1], fake.shape[0])
        text_edit = row.get("field_name") in aiforge.TEXT_FIELDS
        field = None if text_edit else aiforge.edited_field(forge.field_words(r), mask, size, S.field_names)
        ys, xs = np.nonzero(mask)
        rect = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None
        tool = aiforge.TOOLS.get(row.get("assigned_tool"), row.get("assigned_tool"))
        what = labels[field].lower() if field else ("an item name" if text_edit else "a value no rule reads")
        truth = dict(category="ai", forged=True, edited=[field] if field else [], rects=[rect] if rect else [],
                     changes=[dict(field=field, label=labels[field] if field else what,
                                   old=row.get("original_value", ""), new=row.get("forged_value", ""))],
                     what=what, tool=tool)
        return self._real_result(res, fake, f"AI-edited receipt ({tool})", truth, "aiforge",
                                 source=f"AIForge-Doc {version}: a CORD receipt whose {what} {tool} rewrote "
                                        "(CC BY-NC-SA 4.0)", model=ZERO_SHOT, threshold=sysm.threshold)

    def analyse_receipt(self, arr, title="Uploaded receipt photo", source="Uploaded by you", truth=None):
        from cvp.real import upload, view
        h, w = arr.shape[:2]
        s0 = min(1.0, MAX_UPLOAD_SIDE / max(h, w))
        if s0 < 1.0:
            arr = np.asarray(Image.fromarray(arr).resize((round(w * s0), round(h * s0)), Image.LANCZOS))
        t0 = time.time()
        reading = upload.read(arr, self.reader())
        sysm = self.real(REAL_VISUAL)
        small, s = view.visual_view(arr)
        if truth and truth.get("rects"):  # given in the original image's pixels
            truth = dict(truth, rects=[_box(r, s0 * s) for r in truth["rects"]])
        H = visual.heatmap(sysm.model, [small], sysm.device)[0][0]
        res = upload.analyse(reading, H, s)
        out = self._receipt_result(reading, res, small, H, s, title, source, truth, time.time() - t0)
        out["id"] = uuid.uuid4().hex[:12]
        self._keep(out["id"], (reading, H, small, s, title, source, truth))
        return out

    def _keep(self, rid, payload):
        self._kept[rid] = payload
        self._kept.move_to_end(rid)
        while len(self._kept) > 30:
            self._kept.popitem(last=False)

    def recheck(self, rid, values):
        """Re-check an experimental reading after the user corrected some of its values."""
        from cvp.real import upload
        if rid not in self._kept:
            raise KeyError("This reading has expired; analyse the photo again.")
        reading, H, small, s, title, source, truth = self._kept[rid]
        rd = copy.deepcopy(reading)
        rec = rd.receipt
        for f, raw in values.items():
            v = None if raw in (None, "") else float(str(raw).replace(",", ""))
            num = "".join(ch for ch in f if ch.isdigit())
            base = f[:len(f) - len(num)] if num else f
            if base in ("price", "cnt", "unit") and num and 0 < int(num) <= len(rec.items):
                setattr(rec.items[int(num) - 1], {"price": "price", "cnt": "cnt", "unit": "unitprice"}[base], v)
            elif f in ("subtotal", "tax", "service", "discount", "total", "cash", "change", "card", "qty"):
                setattr(rec, f, v)
            else:
                raise ValueError(f"unknown field {f}")
        res = upload.analyse(rd, H, s)
        # the answer key describes the photo as read, not the user's corrections: leave it out here
        out = self._receipt_result(rd, res, small, H, s, title, source, None, 0.0)
        out["id"] = rid
        out["corrected"] = [dict(field=f, label=rd.labels.get(f, f), value=values[f]) for f in sorted(values)]
        self._keep(rid, (rd, H, small, s, title, source, truth))
        return out

    # ------------------------------------------------------------------ results as plain data
    def _synthetic_result(self, name, img, title, truth, source):
        t0 = time.time()
        sysm = self.synthetic(name)
        res = sysm(img)
        S = sysm.schema
        labels = {f.name: f.label for f in S.fields}
        o = res["ocr"]
        values = o["values"]
        violated = set(S.violated(res["w"]))
        rules = []
        for c in S.constraints:
            status = _rule_status(c.scope, values) or ("broken" if c.name in violated else "ok")
            calc = explain.arithmetic(name, c.name, values, c.scope, labels)
            rules.append(dict(name=c.name, text=explain.rule_text(name, c.name, c.text), status=status,
                              detail=calc or explain.values_text(c.scope, values, labels),
                              fields=list(c.scope), kind=getattr(c, "kind", "")))
        coherent = res["cls"] == "coherent"
        suspects = [] if coherent or not violated else list(res["consistent"])
        strong = [] if coherent or not violated else list(res["candidates"])
        blamed = res["field"] if res["tampered"] else None
        heat = {f: float(res["pooled"][i, 0]) for i, f in enumerate(S.field_names)}
        fields = [dict(name=f, label=labels[f], read=o["text"][f], value=explain.fmt(values.get(f)),
                       heat=heat[f], box=_box(o["boxes"][f]),
                       role="blamed" if f == blamed else "suspect" if f in strong else
                       "suspect-weak" if f in suspects else "none") for f in S.field_names]
        broken = [r["text"] for r in rules if r["status"] == "broken"]
        status = "tampered" if res["tampered"] else "genuine"
        unread = [labels[f] for f in S.field_names if values.get(f) is None]
        story = explain.narrate(dict(status=status, mode="synthetic", labels=labels, blamed=blamed, suspects=suspects,
                                     broken=broken, heat=heat, unread=unread, n_fields=len(S.field_names),
                                     n_rules=len(S.constraints),
                                     prior={f: float(p) for f, p in zip(S.field_names, res["field_prior"])},
                                     score=res["p_tamper"], threshold=sysm.heads["det_threshold"]))
        post = sorted(zip(S.class_names, res["p_cls"]), key=lambda x: -x[1])[:3]
        tech = [("Probability of tampering", f"{res['p_tamper']:.3f} (alarm at {sysm.heads['det_threshold']:.3f})"),
                ("Signature class", res["cls"]),
                ("Class posterior", ", ".join(f"{k} {p:.2f}" for k, p in post)),
                ("Visual model", "ResNet-18 U-Net on RGB + ELA, trained on synthetic documents"),
                ("Time", f"{time.time() - t0:.1f} s")]
        return self._package("synthetic", doctypes.get(name).title, title, source, img, res["heatmap"], fields,
                             rules, status, story, truth, blamed, labels, tech, doctype=name)

    def _real_result(self, res, img, title, truth, mode, source, model, threshold):
        from cvp.real import view
        t0 = time.time()
        S = res["schema"]
        labels = {f.name: f.label for f in S.fields}
        values = res["values"]
        small, s = view.visual_view(img)
        violated = set(res["violated"])
        rules = []
        for c in S.constraints:
            status = _rule_status(c.scope, values) or ("broken" if c.name in violated else "ok")
            calc = explain.arithmetic("real", c.name, values, c.scope, labels)
            rules.append(dict(name=c.name, text=explain.rule_text("real", c.name, c.text), status=status,
                              detail=calc or explain.values_text(c.scope, values, labels), fields=list(c.scope)))
        suspects = list(res["consistent"]) if violated else []
        strong = list(res["candidates"]) if violated else []
        blamed = res["field"] if res["tampered"] else None
        heat = res["heat"]
        fields = []
        for f in S.field_names:
            fields.append(dict(name=f, label=labels[f], read=explain.fmt(values.get(f)), value=explain.fmt(values.get(f)),
                               printed=explain.fmt(res["truth"].get(f)), heat=float(heat.get(f, 0.0)),
                               box=_box(res["boxes"].get(f), s),
                               role="blamed" if f == blamed else "suspect" if f in strong else
                               "suspect-weak" if f in suspects else "none"))
        if truth:
            truth = dict(truth, rects=[_box(r, s) for r in truth.get("rects", []) if r is not None])
        broken = [r["text"] for r in rules if r["status"] == "broken"]
        status = "tampered" if res["tampered"] else "genuine"
        unread = [labels[f] for f in S.field_names if values.get(f) is None]
        story = explain.narrate(dict(status=status, mode=mode, labels=labels, blamed=blamed, suspects=suspects,
                                     broken=broken, heat=heat, unread=unread, n_fields=len(S.field_names),
                                     n_rules=len(S.constraints), score=res["p_visual"] if threshold <= 1 else None,
                                     threshold=threshold))
        tech = [("Highest heat", f"{res['p_visual']:.3f} (an alarm on heat alone needs {threshold:.3f})"),
                ("Visual model", "fine-tuned on CORD photos (crude edits)" if model == REAL_VISUAL else
                 "trained on synthetic documents only (zero-shot)"),
                ("Reading", "docTR at each value's labelled position (CORD layout labels)"),
                ("Time", f"{time.time() - t0:.1f} s")]
        return self._package(mode, "Real receipt photo" if mode == "cord" else "AI-edited receipt", title, source,
                             small, res["heatmap"], fields, rules, status, story, truth, blamed, labels, tech,
                             doctype="real")

    def _receipt_result(self, reading, res, small, H, s, title, source, truth, seconds):
        from cvp.real import upload
        S = res["schema"]
        labels = dict(reading.labels)
        values = res["values"]
        broken_hard, broken_soft = set(res["broken_hard"]), set(res["broken_soft"])
        rules = []
        for c in S.constraints:
            st = _rule_status(c.scope, values)
            if not st:
                st = "broken" if c.name in broken_hard else "warning" if c.name in broken_soft else "ok"
            calc = explain.arithmetic("real", c.name, values, c.scope, labels)
            rules.append(dict(name=c.name, text=explain.rule_text("real", c.name, c.text), status=st,
                              detail=calc or explain.values_text(c.scope, values, labels), fields=list(c.scope),
                              level="alarm" if c.name in upload.HARD else "warning"))
        suspects = list(res["consistent"])
        blamed = res["field"]
        heat = res["heat"]
        fields = []
        for f in S.field_names:
            bs = reading.boxes.get(f, [])
            fields.append(dict(name=f, label=labels.get(f, f), read=explain.fmt(values.get(f)),
                               value=explain.fmt(values.get(f)), heat=float(heat.get(f, 0.0)),
                               box=_box(bs[0], s) if bs else None, boxes=[_box(b, s) for b in bs],
                               role="blamed" if (f == blamed and res["tampered"]) else
                               "suspect" if f in suspects else "none"))
        if not S.constraints:
            status = "unknown"
        elif res["tampered"]:
            status = "tampered"
        elif res["warning"]:
            status = "warning"
        else:
            status = "genuine"
        broken = [r["text"] for r in rules if r["status"] in ("broken", "warning")]
        story = explain.narrate(dict(status=status, mode="receipt", labels=labels,
                                     blamed=blamed if status == "tampered" else None, suspects=suspects,
                                     broken=broken, heat=heat, unread=[], n_fields=len(S.field_names),
                                     n_rules=len(S.constraints)))
        tech = [("Mode", "experimental: docTR reads the whole photo; rows are matched to the bill's fields"),
                ("Alarm", "only the bill-level rules (total, change, payment) raise an alarm; item rules warn"),
                ("Visual model", "fine-tuned on CORD photos (crude edits); used only to rank suspects"),
                ("Time", f"{seconds:.1f} s")]
        rows_read = [dict(text=row["text"], kind=row["kind"], value=explain.fmt(row["value"]) if row["value"] is not None
                          else "", box=_box(row["box"], s)) for row in upload.row_table(reading)]
        out = self._package("receipt", "Receipt photo (experimental)", title, source, small, H, fields, rules, status,
                            story, truth, blamed if status == "tampered" else None, labels, tech, doctype="real")
        out["rows"] = rows_read
        out["notes"] = list(reading.notes)
        return out

    def _package(self, mode, kind, title, source, img, H, fields, rules, status, story, truth, blamed, labels, tech,
                 doctype):
        img = np.asarray(img)
        h, w = img.shape[:2]
        out = dict(mode=mode, kind=kind, title=title, source=source, doctype=doctype,
                   verdict=dict(status=status, headline=story["headline"], reason=story["reason"],
                                blamed=labels.get(blamed) if blamed else None),
                   image=dict(src=_png_uri(img) if mode == "synthetic" else _jpeg_uri(img, max_side=max(h, w)),
                              width=w, height=h),
                   heat=dict(src=_heat_uri(H)) if H is not None else None,
                   fields=fields, rules=rules, steps=story["steps"],
                   tech=[dict(label=a, value=b) for a, b in tech], notes=[])
        if truth is not None:
            suspects = [f["name"] for f in fields if f["role"] in ("suspect", "suspect-weak", "blamed")]
            verdict, text = explain.answer(truth, blamed, status, labels, suspects)
            out["truth"] = dict(truth, verdict=verdict, text=text)
        return out


def run_safely(fn, *a, **k):
    """(result, None) or (None, a message for the page)."""
    try:
        return fn(*a, **k), None
    except Exception as e:
        traceback.print_exc()
        return None, f"{type(e).__name__}: {e}"
