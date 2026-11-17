"""Page numbers for the final report's contents, list of tables and list of figures.

Renders report/final_report/CVP_Final_Report.docx to PDF with LibreOffice, finds the page on
which each heading and caption is printed, and writes them as JSON for build_report.py --pages.
Printed page numbers start at 1 on the ABSTRACT page (the cover, certificate and declaration
carry none), as in the template.

  report_pages.py --soffice /path/to/soffice [--out pages.json]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_report as B  # noqa: E402


def norm(text):
    return re.sub(r"\s+", " ", text).strip().upper()


def render(soffice, docx, outdir):
    profile = "file://" + os.path.join(outdir, "lo-profile")
    subprocess.run([soffice, f"-env:UserInstallation={profile}", "--headless", "--convert-to", "pdf",
                    "--outdir", outdir, docx], check=True, capture_output=True, timeout=600)
    return os.path.join(outdir, os.path.splitext(os.path.basename(docx))[0] + ".pdf")


def page_texts(pdf):
    out = subprocess.run(["pdftotext", "-layout", pdf, "-"], check=True, capture_output=True, text=True).stdout
    return [norm(p) for p in out.split("\f")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soffice", required=True)
    ap.add_argument("--docx", default=B.OUT)
    ap.add_argument("--out", default=os.path.join(B.ROOT, "pages.json"))
    ap.add_argument("--pdf-out", help="also keep the rendered PDF here")
    a = ap.parse_args()
    doc = B.Doc()
    for name in sorted(os.listdir(B.CONTENT)):
        if re.match(r"\d\d_.*\.txt$", name):
            B.parse_content(doc, open(os.path.join(B.CONTENT, name)).read())
    with tempfile.TemporaryDirectory() as tmp:
        pdf = render(a.soffice, a.docx, tmp)
        pages = page_texts(pdf)
        if a.pdf_out:
            with open(pdf, "rb") as src, open(a.pdf_out, "wb") as dst:
                dst.write(src.read())
    first = next(i for i, t in enumerate(pages) if "ABSTRACT" in t)  # printed page 1
    toc_page = next(i for i, t in enumerate(pages) if "TABLE OF CONTENTS" in t)

    def where(text, last=True, after=0):
        hits = [i for i, t in enumerate(pages) if i >= after and norm(text) in t]
        if not hits:
            return None
        return (hits[-1] if last else hits[0]) - first + 1

    result, missing = {}, []
    for name in ["ABSTRACT", "ACKNOWLEDGEMENT", "TABLE OF CONTENTS"]:
        result[name] = where(name, last=False)
    for name in ["LIST OF TABLES", "LIST OF FIGURES", "LIST OF SYMBOLS, ABBREVIATIONS AND NOMENCLATURE"]:
        result[name] = where(name, after=toc_page + 1, last=False)
    for _, text in doc.toc:
        result[text] = where(text)
    for cap in doc.tables + doc.figures:
        key = cap.split(":")[0]
        result[key] = where(key + ":")
    missing = [k for k, v in result.items() if v is None]
    json.dump({k: v for k, v in result.items() if v is not None}, open(a.out, "w"), indent=1)
    print(f"{len(pages)} PDF pages; printed page 1 is PDF page {first + 1}; {len(result) - len(missing)} of "
          f"{len(result)} entries located; written to {a.out}")
    if missing:
        print("not found:", missing)


if __name__ == "__main__":
    main()
