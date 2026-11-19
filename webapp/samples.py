"""The app's ready-made examples: a guided tour for a demonstration, then more of each kind.
Every example is computed live by the engine (with fixed seeds), not stored."""
import io

import numpy as np
from PIL import Image

T = "transcript"
TOUR = [
    dict(id="tour-1", kind="synthetic", doctype=T, seed=9, category="genuine",
         title="A genuine mark sheet",
         intro="Start with an honest document. CVP reads every field, checks the document's own rules "
               "(the marks add up to the total, the percentage matches the total, the grade matches the "
               "percentage) and looks for edit traces in the pixels.",
         notice="Every rule holds and the heatmap stays dark, so the verdict is genuine."),
    dict(id="tour-2", kind="synthetic", doctype=T, seed=3, category="T1", field="m2",
         title="A clumsy edit",
         intro="Someone pasted a new number over subject mark 2 in a different font. This is the kind of "
               "edit ordinary forensic tools were built for.",
         notice="Both kinds of evidence agree: the marks no longer add up to the total, and the heatmap "
                "lights up exactly on the pasted number."),
    dict(id="tour-3", kind="synthetic", doctype=T, seed=2, category="T2", field="m3",
         title="A smooth, AI-style edit",
         intro="Now subject mark 3 is redrawn smoothly, in the page's own ink and font, the way a generative "
               "editor would. Pixel-only detectors mostly miss edits like this.",
         notice="The sum rule still breaks, because the new mark doesn't add up. But it fits all five marks "
                "equally: no rule can say which one changed (Proposition 1). The faint heat decides."),
    dict(id="tour-4", kind="synthetic", doctype="receipt", seed=4, category="T2", field="change",
         title="When the rules can't decide",
         intro="On a shop receipt the change was edited. Cash − total = change now fails, and that fits "
               "two fields equally: the cash or the change.",
         notice="The rules shortlist two suspects; the image and what the model learned about receipts "
                "pick the change."),
    dict(id="tour-5", kind="cord", k=2, category="T1", field="change",
         title="A real receipt photo",
         intro="The same idea on a real photo of a real receipt from an Indonesian shop (the CORD dataset). "
               "We changed the change from 72,000 to 92,000.",
         notice="100,000 − 28,000 should be 72,000, so either the cash or the change was edited; the "
                "heatmap is much warmer on the change."),
    dict(id="tour-6", kind="aiforge", version="v2", spec="000000905",
         title="A real AI forgery (GPT-Image-2)",
         intro="This receipt's subtotal was rewritten by GPT-Image-2, OpenAI's image editor, for the "
               "AIForge-Doc benchmark. In its authors' study, people comparing it with the original were at chance.",
         notice="The image model sees nothing at all: the heatmap is empty. But 194,000 of items can't make a "
                "subtotal of 109,293, and only the subtotal fits both broken rules."),
    dict(id="tour-7", kind="synthetic", doctype=T, seed=8, category="T4",
         title="The limit: an edit that keeps every rule",
         intro="Finally, a careful forger who changes a date and keeps every rule intact. Now the rules have "
                "nothing to say, and only the pixels are left.",
         notice="The heatmap is warm on the edited date, but not quite enough to raise the alarm. This is "
                "the honest limit of CVP: an edit that breaks no rule is only as visible as its pixels."),
]

MORE = [
    ("synthetic", "Template documents", [
        dict(id="syn-invoice-qty", kind="synthetic", doctype="invoice", seed=3, category="T2", field="qty2",
             title="GST invoice: a quantity changed"),
        dict(id="syn-invoice-total", kind="synthetic", doctype="invoice", seed=4, category="T2", field="total",
             title="GST invoice: the total changed"),
        dict(id="syn-receipt-total", kind="synthetic", doctype="receipt", seed=5, category="T1", field="total",
             title="Shop receipt: the total pasted over"),
        dict(id="syn-cert-grade", kind="synthetic", doctype="certificate", seed=6, category="T2", field="grade",
             title="Certificate: the grade changed"),
        dict(id="syn-passport-dob", kind="synthetic", doctype="passport", seed=6, category="T2", field="viz_dob",
             title="Passport: the date of birth changed"),
        dict(id="syn-passport-mrz", kind="synthetic", doctype="passport", seed=7, category="T1", field="mrz2",
             title="Passport: the machine-readable line edited"),
        dict(id="syn-id-valid", kind="synthetic", doctype="idcard", seed=7, category="T2", field="valid",
             title="Student ID: the validity extended"),
        dict(id="syn-id-genuine", kind="synthetic", doctype="idcard", seed=9, category="genuine",
             title="Student ID: genuine"),
        dict(id="syn-transcript-total", kind="synthetic", doctype=T, seed=5, category="T2", field="total",
             title="Mark sheet: the total changed"),
    ]),
    ("cord", "Real receipt photos (CORD)", [
        dict(id="cord-7-T2", kind="cord", k=7, category="T2", title="Receipt #7: a smooth edit"),
        dict(id="cord-14-genuine", kind="cord", k=14, category="genuine", title="Receipt #14: genuine"),
        dict(id="cord-20-T1", kind="cord", k=20, category="T1", title="Receipt #20: an obvious edit"),
        dict(id="cord-11-T3", kind="cord", k=11, category="T3", title="Receipt #11: totals fixed too"),
    ]),
    ("aiforge", "Real AI forgeries (AIForge-Doc)", [
        dict(id="ai-905", kind="aiforge", version="v2", spec="000000905",
             title="GPT-Image-2 rewrote a subtotal"),
        dict(id="ai-820", kind="aiforge", version="v1", spec="000000820",
             title="Gemini rewrote an item price: three suspects"),
        dict(id="ai-864", kind="aiforge", version="v1", spec="000000864",
             title="Ideogram rewrote the cash: cash or change?"),
        dict(id="ai-022", kind="aiforge", version="v2", spec="000000022",
             title="GPT-Image-2 rewrote an item code: a miss"),
    ]),
    ("receipt", "Experimental: reading any receipt photo", [
        dict(id="rx-cord-14", kind="receipt-cord", k=14, title="A real photo, read without labels"),
        dict(id="rx-gst-genuine", kind="receipt-gst", seed=5, category="genuine",
             title="An Indian GST receipt, genuine"),
        dict(id="rx-gst-total", kind="receipt-gst", seed=5, category="T2", field="total",
             title="An Indian GST receipt, total changed"),
        dict(id="rx-gst-item", kind="receipt-gst", seed=5, category="T2", field="amt2",
             title="An Indian GST receipt, an item price changed"),
    ]),
]


def reader_field(f):
    """The experimental reader's name for a field of our GST shop receipt (cvp/doctypes/receipt.py)."""
    fixed = {"total": "total", "subtotal": "subtotal", "cash": "cash", "change": "change", "cgst": "tax",
             "sgst": "tax", "tot_qty": "qty"}
    if f in fixed:
        return fixed[f]
    for a, b in (("amt", "price"), ("rate", "unit"), ("qty", "cnt")):
        if f.startswith(a) and f[len(a):].isdigit():
            return b + f[len(a):]
    return None


def catalogue(av):
    """Groups of examples for the page, each marked available or not on this machine."""
    def ok(s):
        k = s["kind"]
        if k == "synthetic":
            return av["synthetic"].get(s["doctype"], False)
        if k in ("cord",):
            return av["cord"]
        if k == "aiforge":
            return av["aiforge"].get(s["version"], False)
        if k == "receipt-cord":
            return av["receipt"] and av["cord"]
        if k == "receipt-gst":
            return av["receipt"] and av["synthetic"].get("receipt", False)
        return False

    def card(s):
        return dict(id=s["id"], title=s["title"], kind=s["kind"], available=ok(s),
                    intro=s.get("intro"), notice=s.get("notice"))
    groups = [dict(id="tour", title="Guided tour", items=[card(s) for s in TOUR])]
    groups += [dict(id=g, title=t, items=[card(s) for s in items]) for g, t, items in MORE]
    return groups


def find(sid):
    for s in TOUR:
        if s["id"] == sid:
            return s
    for _, _, items in MORE:
        for s in items:
            if s["id"] == sid:
                return s
    raise KeyError(sid)


def run(engine, s):
    k = s["kind"]
    if k == "synthetic":
        return engine.forge_synthetic(s["doctype"], s["seed"], s["category"], s.get("field"))
    if k == "cord":
        return engine.forge_cord(s["k"], s["category"], s.get("field"))
    if k == "aiforge":
        return engine.aiforge(s["version"], s["spec"])
    if k == "receipt-cord":
        from cvp.real import cord
        r = cord.receipt("test", s["k"])
        img = np.array(Image.open(io.BytesIO(r.image)).convert("RGB"))
        return engine.analyse_receipt(img, title=f"CORD receipt #{s['k']}, read without labels",
                                      source="A real shop receipt (CORD v2), given to the reader without its labels",
                                      truth=dict(category="genuine", forged=False, edited=[], rects=[], changes=[]))
    if k == "receipt-gst":
        from cvp import doctypes
        from cvp.core import render, tamper
        DT = doctypes.get("receipt")
        rng = np.random.default_rng(s["seed"])
        doc = render.generate(DT, rng)
        img, info = tamper.apply(doc, DT, s["category"], rng, field=s.get("field"))
        big = np.asarray(Image.fromarray(img).resize((img.shape[1] * 2, img.shape[0] * 2), Image.LANCZOS))
        labels = {f.name: f.label for f in DT.schema.fields}
        truth = dict(category=s["category"], forged=bool(info["edited"]),
                     edited=sorted({reader_field(f) for f in info["edited"]} - {None}),
                     rects=[[2 * c for c in r] for r in info["rects"]],
                     changes=[dict(field=reader_field(f), label=labels[f], old=DT.schema.by_name[f].fmt(doc.values[f]),
                                   new=DT.schema.by_name[f].fmt(info["values"][f])) for f in info["edited"]],
                     what=", ".join(labels[f].lower() for f in info["edited"]) or None)
        return engine.analyse_receipt(big, title=f"GST shop receipt #{s['seed']} (as a photo)",
                                      source="Made by our generator and read as if it were a photo", truth=truth)
    raise ValueError(k)
