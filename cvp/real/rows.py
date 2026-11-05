"""Reading a receipt nobody has labelled: OCR the whole page (docTR detection + recognition),
group the words into printed rows, and pair each summary key ("TOTAL", "CASH", "CHANGE"...)
with the rightmost amount on its row: on a receipt, a key and its amount share a row.
Used for Find it again! scans, whose transcriptions carry no positions, and for receipts
uploaded to the app.
"""
import re

import numpy as np

_AMOUNT = re.compile(r"-?\(?(?:RM\s?)?-?\d{1,3}(?:,\d{3})*\.\d{2}\)?|-?\d+\.\d{2}")
# Summary keys, most specific first. A row counts for a key when its text (amounts removed)
# matches; the first such row from the top wins, except that rows below a "GST SUMMARY"
# table are not read (that table repeats totals and tax per tax code).
KEYS = {
    "total": [r"TOTAL SALES ?\(?INCL", r"TOTAL ?\(?INCL", r"GRAND TOTAL", r"NETT? ?TOTAL", r"TOTAL AMOUNT",
              r"TOTAL PAYABLE", r"AMOUNT DUE", r"ROUNDED TOTAL", r"^TOTAL\b(?!.*\b(QTY|ITEM|GST|TAX|DISC|SAVING|EXCL))"],
    "subtotal": [r"TOTAL SALES ?\(?EXCL", r"SUB ?-? ?TOTAL", r"TOTAL ?\(?EXCL", r"^TOTAL SALES\b(?!.*INCL)"],
    "tax": [r"TOTAL GST", r"^GST\b(?!.*(ID|NO|REG|SUMMARY))", r"\bGST ?@", r"SERVICE TAX", r"^TAX\b(?!.*INVOICE)"],
    "discount": [r"DISCOUNT"],
    "rounding": [r"ROUND"],
    "cash": [r"^CASH\b(?!.*(SALES|BILL|INVOICE|RECEIPT))", r"TENDERED", r"^TENDER", r"^PAYMENT\b"],
    "change": [r"CHANGE"],
}
STOP = re.compile(r"GST SUMMARY|SUMMARY OF GST|TAX CODE")


def words_of(page, size):
    """(text, x0, y0, x1, y1) in pixels for every word docTR found on a page."""
    W, H = size
    out = []
    for b in page.blocks:
        for line in b.lines:
            for w in line.words:
                (x0, y0), (x1, y1) = w.geometry
                out.append((w.value, x0 * W, y0 * H, x1 * W, y1 * H))
    return out


def group_rows(words):
    """Words grouped into printed rows (by vertical overlap), each sorted left to right."""
    if not words:
        return []
    words = sorted(words, key=lambda w: (w[2] + w[4]) / 2)
    h = np.median([w[4] - w[2] for w in words])
    rows, cur, centre = [], [words[0]], (words[0][2] + words[0][4]) / 2
    for w in words[1:]:
        c = (w[2] + w[4]) / 2
        if abs(c - centre) <= 0.5 * h:
            cur.append(w)
            centre = np.mean([(x[2] + x[4]) / 2 for x in cur])
        else:
            rows.append(sorted(cur, key=lambda x: x[1]))
            cur, centre = [w], c
    rows.append(sorted(cur, key=lambda x: x[1]))
    return rows


def amount(text):
    m = _AMOUNT.findall(text.replace(" ", ""))
    if not m:
        return None
    s = m[-1].replace("RM", "").replace(",", "").replace("(", "-").replace(")", "")
    try:
        return float(s)
    except ValueError:
        return None


def summary(rows):
    """{key: amount} from the rows of a receipt (see KEYS)."""
    texts = [" ".join(w[0] for w in r).upper() for r in rows]
    stop = next((i for i, t in enumerate(texts) if STOP.search(t)), len(texts))
    out = {}
    for key, pats in KEYS.items():
        for pat in pats:
            rx = re.compile(pat)
            for t in texts[:stop]:
                keytext = _AMOUNT.sub("", t.replace(" ", " ")).strip()
                if rx.search(keytext):
                    v = amount(t)
                    if v is not None:
                        out[key] = v
                        break
            if key in out:
                break
    return out
