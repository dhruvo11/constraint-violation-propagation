"""Ready-to-upload documents for the app's "Analyse an image" tab: forged and genuine pages of
each document type, from the scenarios in DEMO.md, saved as PNG (lossless, so the upload is
exactly the page that was made). Each file is checked through the tab's own upload path, and
samples/README.md says what the app should find on each.

  make_samples.py
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvp import doctypes, paths  # noqa: E402
from cvp.core import render, tamper  # noqa: E402

OUT = os.path.join(paths.BASE, "samples")
# (file name, type, seed, forgery, field) -- the scenarios of DEMO.md
SAMPLES = [
    ("1_marksheet_forged_mark3.png", "transcript", 2, "T2", "m3"),
    ("2_marksheet_forged_obvious.png", "transcript", 3, "T1", "m2"),
    ("3_marksheet_genuine.png", "transcript", 9, "genuine", None),
    ("4_invoice_forged_quantity.png", "invoice", 3, "T2", "qty2"),
    ("5_receipt_forged_change.png", "receipt", 4, "T2", "change"),
    ("6_certificate_forged_grade.png", "certificate", 6, "T2", "grade"),
    ("7_passport_forged_birthdate.png", "passport", 6, "T2", "viz_dob"),
    ("8_idcard_forged_validity.png", "idcard", 7, "T2", "valid"),
    ("9_idcard_genuine.png", "idcard", 9, "genuine", None),
]


def main():
    import app  # the app's own upload path: app.analyse(type title, image as numpy)
    os.makedirs(OUT, exist_ok=True)
    notes = ["# Sample documents for the app's \"Analyse an image\" tab", "",
             "Open the tab, choose the **Document type** named below, drag the file onto the upload box (or click "
             "it and pick the file), then press **Analyse**.", "",
             "| File | Document type to choose | What was done to it | What the app finds |", "|---|---|---|---|"]
    for fname, name, seed, cat, field in SAMPLES:
        DT = doctypes.get(name)
        rng = np.random.default_rng(seed)
        doc = render.generate(DT, rng)
        img, info = tamper.apply(doc, DT, cat, rng, field=field)
        Image.fromarray(img).save(os.path.join(OUT, fname))
        labels = {f.name: f.label for f in DT.schema.fields}
        upload = np.asarray(Image.open(os.path.join(OUT, fname)).convert("RGB"))
        _, report, _ = app.analyse(DT.title, upload)
        verdict = "TAMPERED" if "**TAMPERED**" in report.split("\n")[0] else "GENUINE"
        blamed = next((ln.split("**Attributed field:** ")[1].split(" (")[0] for ln in report.split("\n\n")
                       if ln.startswith("**Attributed field:**")), None)
        done = "nothing (genuine)" if not info["edited"] else (
            f"{app.CAT_LABELS[cat]}: {', '.join(labels[f] for f in info['edited'])} changed")
        found = verdict + (f", blames {blamed}" if blamed else "")
        ok = (verdict == "GENUINE") if not info["edited"] else (verdict == "TAMPERED" and blamed in
                                                               [labels[f] for f in info["edited"]])
        notes.append(f"| `{fname}` | {DT.title} | {done} | {found} {'(correct)' if ok else '(MISSED)'} |")
        print(f"{fname:36s} {DT.title:28s} {done:60s} -> {found} {'OK' if ok else 'MISSED'}", flush=True)
    with open(os.path.join(OUT, "README.md"), "w") as fh:
        fh.write("\n".join(notes) + "\n")


if __name__ == "__main__":
    main()
