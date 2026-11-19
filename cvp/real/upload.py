"""Reading a receipt photo that nobody has labelled: the app's experimental mode.

docTR reads the whole page (text detection and recognition), rows.group_rows puts the words
into printed rows, and every row is classified: a summary row (subtotal, tax, service charge,
discount, rounding, total, cash, card, change, item count) when its words name one, an item
row when it carries an amount above the summary, or neither. The values fill a cord.Receipt,
so the photo gets its own constraint graph from receipt_schema, exactly like a labelled CORD
receipt, and the theory (consistent sets, Proposition 1 classes) applies unchanged. Every value
keeps the box of the word it was read from, for the heatmap and for the page.

Only arithmetic that the reading supports is checked: the items must add up to within a factor
of two of the subtotal (else the item rows were probably misread and the item rule is left out),
and a line is checked only where its count, unit price and price were all found.
"""
import re
from dataclasses import dataclass, field

from . import cord, money, rows
from . import receipt_schema as RS

# Summary rows, most specific first: the first pattern that matches a row's words decides it.
SUMMARY = [
    ("qty", r"TOTAL\s*\(?\s*(QTY|QUANTITY|ITE[MN]\w*|PCS)\b(?!.*\d[.,]\d{3})|\bNO\.?\s*OF\s*ITEMS|\bQTY\s*(TOTAL)?\s*:|"
            r"\bTOTAL\s*ITE[MN]|^\s*ITE[MN]S?\s*:?\s*[\d.,]+\s*$"),
    ("coupon", r"\bCOUPON\b|\bVOUCHER\b|\bGIFT\s*CARD\b|\bPOINTS?\s*(REDEEM\w*)?\b|\bREDEEM\w*"),
    ("total", r"ROUNDED\s*TOTAL|TOTAL\s*\(?ROUNDED"),
    ("change", r"\bCHANGE\b|\bKEMBALI(AN)?\b|\bCHANGE\s*DUE\b|\bBALANCE\s*(RETURN(ED)?|TO\s*CUSTOMER)\b|^\s*BALANCE\b"),
    ("tax", r"\bPB\s?-?\s?[1IL]\b|\bP\.?\s?REST\w*|\bRESTO\w*\s*TAX\b|\bSERVICE\s*TAX\b|\b(C|S|I|UT)\s?\.?\s?GST\b(?!\s*(IN|NO|REG|ID))|\bGST\b(?!\s*(IN|NO|REG|ID|SUMMARY))"
            r"|\bVAT\b(?!\s*(NO|REG))|\bTAX\b(?!\s*INVOICE)|\bPPN\b|\bPB\s?1\b|\bPAJAK\b|\bCESS\b"),
    ("service", r"\bSERVICE(\s*CHARGE)?\b|\bSERV\w*\s*CH\w*|\bSVC\b|\bS\.?\s?CHG\b|\bSERVIS\b|\bLAYANAN\b"),
    ("rounding", r"\bROUND(ING)?(\s*OFF|\s*ADJ\w*)?\b|\bPEMBULATAN\b"),
    ("discount", r"\bDISC(OUNT)?\b|\bDISKON\b|\bPOTONGAN\b|\bSAVINGS?\b"),
    ("subtotal", r"\bSUB\s*-?\s*TOTAL\b|\bSUB\s*-?\s*TTL\b|\bSUBTTL\b|\bSUB\s*-?\s*TOT\b|\bTOTAL\s*BEFORE\s*TAX\b|\bITEM\s*TOTAL\b|\bTAXABLE\s*(VALUE|AMOUNT)\b"
                 r"|\bGROSS(\s*(AMOUNT|TOTAL))?\b|\bJUMLAH\b"),
    ("card", r"\bCARD\b|\bCREDIT\b|\bDEBIT\b|\bVISA\b|\bMASTER\s*CARD\b|\bKARTU\b|\bEDC\b|\bUPI\b|\bGPAY\b"
             r"|\bPAYTM\b|\bPHONE\s*PE\b|\bQRIS\b|\bE-?\s?MONEY\b|\bGO-?PAY\b|\bOVO\b|\bDANA\b"),
    ("cash", r"\bCASH\b|\bTUNAI\b|\bTENDER(ED)?\b|\bAMOUNT\s*(PAID|RECEIVED)\b|\bPAID\b|\bRECEIVED\b|\bBAYAR\b"),
    ("total", r"\bGRAND\s*TOTAL\b|\bNET\s*(AMOUNT|TOTAL|PAYABLE)\b|\bTOTAL\b|\bAMOUNT\s*(DUE|PAYABLE)\b"
              r"|\bBILL\s*AMOUNT\b|\bNET\s*AMT\b|\bTOTAL\s*BAYAR\b|\bJUMLAH\s*BAYAR\b"),
]
LABELS = {"subtotal": "Subtotal", "tax": "Tax", "service": "Service charge", "discount": "Discount",
          "rounding": "Rounding", "total": "Total", "cash": "Cash", "card": "Other payment", "change": "Change",
          "qty": "Item count"}
# Rows that are neither items nor summary: the shop's details, the bill's number, date and staff.
OTHER = re.compile(r"\b(DATE|TIME|TGL|JAM|BILL\s*NO|INVOICE|INV\b|NO\.|ORDER|TOKEN|TABLE|MEJA|COVERS?|PAX|GUEST"
                   r"|CASHIER|KASIR|WAITER|SERVER|STAFF|OPERATOR|GSTIN|FSSAI|NPWP|PHONE|TEL|MOBILE|PH\b|WWW|HTTP"
                   r"|EMAIL|THANK|TERIMA|KASIH|ROAD|STREET|JL\.?|JALAN|NAGAR|LAYOUT|BENGALURU|BANGALORE|PIN\b"
                   r"|REF|TRX|TRANS|APPROVAL|AUTH|BATCH|TID|MID|ACCOUNT|A/C|WIFI|PASSWORD|VISIT|WELCOME)\b")
_CURRENCY = r"(?:RP|RS|RM|INR|IDR|MYR|USD|SGD|EUR|\$|₹|€|£)"
_AMOUNT = re.compile(rf"^[(\-]?{_CURRENCY}?\.?\s?[-]?\d{{1,3}}(?:[.,\s]\d{{3}})*(?:[.,]\d{{1,2}})?[)\-]?$|"
                     rf"^[(\-]?{_CURRENCY}?\.?\s?[-]?\d{{1,2}}(?:,\d{{2}})+,\d{{3}}(?:\.\d{{1,2}})?[)\-]?$|"  # 1,23,456.00
                     rf"^[(\-]?{_CURRENCY}?\.?\s?[-]?\d+(?:[.,]\d{{1,2}})?[)\-]?$", re.I)
_COUNT = re.compile(r"^(?:[xX]\s?)?(\d{1,3})(?:\s?[xX@]|[.,]0+)?$")


def is_amount(token):
    """A word that is a money amount (not a date, a time, a percentage or a code)."""
    t = token.strip()
    if not t or any(ch in t for ch in ":/%#") or re.search(r"\d-\d", t):
        return False
    if re.search(r"[A-Za-z]", re.sub(_CURRENCY, "", t, flags=re.I)):
        return False
    return bool(_AMOUNT.match(t))


@dataclass
class Row:
    text: str
    words: list                         # (text, x0, y0, x1, y1)
    kind: str = "other"                 # "item", a summary field name, or "other"
    value: float = None
    box: tuple = None                   # the box of the word the value was read from
    cnt: float = None
    unit: float = None


@dataclass
class Reading:
    receipt: cord.Receipt
    rows: list
    boxes: dict = field(default_factory=dict)   # schema field -> list of word boxes
    labels: dict = field(default_factory=dict)  # schema field -> label
    notes: list = field(default_factory=list)   # what could not be read reliably


_LAKH = re.compile(r"^\D*\d{1,2}(?:,\d{2})+,\d{3}(?:\.\d{1,2})?\D*$")


def value_of(token):
    """The amount a word prints; Indian grouping (1,23,456.00) is read whole."""
    if _LAKH.match(token.strip()):
        neg = token.strip().startswith(("-", "("))
        v = float(re.sub(r"[^0-9.]", "", token))
        return -v if neg else v
    return money.amount(token)


def _amounts(row_words):
    """[(value, box)] for the words of a row that are amounts, left to right; a currency word
    or a lone minus sign before an amount is kept with it."""
    out = []
    for w in row_words:
        if is_amount(w[0]):
            v = value_of(w[0])
            if v is not None:
                out.append((v, w[1:]))
    return out


_COUNT_WORD = re.compile(r"^[xX]?(\d{1,3})[xX@]?$")


def _item_amounts(row_words):
    """The row's amounts that can be prices: not a bare year or a bare number of seven or more
    digits (a phone or account number)."""
    out = []
    for v, b in _amounts(row_words):
        t = next(w[0] for w in row_words if tuple(w[1:]) == tuple(b))
        bare = re.sub(r"[^0-9]", "", t)
        if re.fullmatch(r"\d+", t.strip("()-")) and (len(bare) >= 7 or (len(bare) == 4 and 1900 <= int(bare) <= 2100)):
            continue
        out.append((v, b))
    return out


def _letters(row_words):
    return re.sub(r"[^A-Z]", "", "".join(w[0].upper() for w in row_words if not is_amount(w[0])))


KEYWORDS = ("SUBTOTAL", "TOTAL", "CHANGE", "KEMBALI", "KEMBALIAN", "DISCOUNT", "DISKON", "SERVICE", "CHARGE",
            "TUNAI", "TENDERED", "ROUNDING", "BALANCE", "AMOUNT", "PAYABLE", "GRAND")


def _fuzzy(text):
    """The row's words with OCR near-misses of the summary keywords corrected ("SUB_TSTAL" ->
    "SUBTOTAL", "SERVI" -> "SERVICE"): a word of five or more letters at least 80% similar."""
    from difflib import SequenceMatcher
    out = []
    for tok in text.split():
        letters = re.sub(r"[^A-Z]", "", tok)
        best = max(KEYWORDS, key=lambda k: SequenceMatcher(None, letters, k).ratio()) if len(letters) >= 5 else None
        if best and letters != best and SequenceMatcher(None, letters, best).ratio() >= 0.8:
            tok = best
        out.append(tok)
    return " ".join(out)


def classify(r):
    """The row's kind and value: a summary field (its rightmost amount), else 'item' when it has
    a price and a name, else 'other' (an item's price row is joined to its name in parse)."""
    up = re.sub(r"\s+", " ", r.text.upper())
    fz = _fuzzy(up)
    for name, pat in SUMMARY:
        if re.search(pat, up) or (fz != up and re.search(pat, fz)):
            amts = _amounts(r.words)
            if amts:
                r.kind, (r.value, r.box) = name, amts[-1]
                if name == "qty":
                    r.value = float(int(round(r.value)))
            return r
    amts = _item_amounts(r.words)
    if amts and len(_letters(r.words)) >= 2 and not OTHER.search(up):
        _as_item(r, amts)
    return r


def _as_item(r, amts):
    r.kind, (r.value, r.box) = "item", amts[-1]
    counts = [w for w in r.words if _COUNT_WORD.match(w[0]) and tuple(w[1:]) != tuple(r.box)]
    if counts:
        c = float(_COUNT_WORD.match(counts[0][0]).group(1))
        r.cnt = c if 0 < c < 1000 else None
    if r.cnt and len(amts) >= 2:
        unit = amts[-2][0]
        if abs(r.cnt * unit - r.value) <= max(1.0, 0.005 * abs(r.value)) and unit != r.cnt:
            r.unit = unit


def read(img, predictor):
    """Read receipt photo `img` (RGB uint8) with docTR `predictor` (an ocr_predictor)."""
    H, W = img.shape[:2]
    page = predictor([img]).pages[0]
    return parse(rows.group_rows(rows.words_of(page, (W, H))))


def _join_two_row_items(rs):
    """An item printed on two rows, its name above and its count and price below
    ("NASI GORENG" / "1 x 25,000  25,000"), becomes one item row."""
    for i in range(1, len(rs)):
        r, above = rs[i], rs[i - 1]
        if r.kind != "other" or above.kind != "other" or _item_amounts(above.words):
            continue
        amts = _item_amounts(r.words)
        if amts and len(_letters(r.words)) < 2 and len(_letters(above.words)) >= 2 and \
                not OTHER.search(above.text.upper()):
            r.words = above.words + r.words
            r.text = above.text + " " + r.text
            above.words, above.text = [], ""
            _as_item(r, amts)


def parse(row_words):
    """A Reading from the words of a receipt grouped into rows (top to bottom)."""
    rs = [classify(Row(" ".join(w[0] for w in r), list(r))) for r in row_words]
    _join_two_row_items(rs)
    rs = [r for r in rs if r.words]
    first_summary = next((i for i, r in enumerate(rs) if r.kind in ("subtotal", "total", "tax")), len(rs))
    for i, r in enumerate(rs):
        if r.kind == "item" and i > first_summary:
            r.kind = "other"  # amounts below the bill (tax tables, card slips) are not items
    # Several total rows: the last one before the payment is the total. An earlier one followed by
    # tax, service or discount rows is the subtotal when none is printed ("Total" ... "Grand total").
    tot = [i for i, r in enumerate(rs) if r.kind == "total"]
    if len(tot) > 1:
        top = max(abs(rs[i].value) for i in tot)
        final = [i for i in tot if abs(rs[i].value) >= 0.5 * top][-1]  # not a misread count ("TOTAL ITENS 1")
        has_sub = any(r.kind == "subtotal" for r in rs)
        for i in tot:
            if i == final:
                continue
            between = {r.kind for r in rs[i + 1:final]} if i < final else set()
            rs[i].kind = "subtotal" if (not has_sub and between & {"tax", "service", "discount"}) else "other"
            has_sub = has_sub or rs[i].kind == "subtotal"
    final_total = next((i for i in range(len(rs) - 1, -1, -1) if rs[i].kind == "total"), None)
    sub = next((i for i, r in enumerate(rs) if r.kind == "subtotal"), None)
    if sub is not None and final_total is not None and sub < final_total:
        # an amount between the subtotal and the total whose name the OCR garbled ("PBI", "P.Rest 10 %")
        # is a charge, or a discount when negative
        for r in rs[sub + 1:final_total]:
            amts = _amounts(r.words)
            if r.kind == "other" and amts and abs(amts[-1][0]) < abs(rs[sub].value):
                r.value, r.box = amts[-1]
                r.kind = "discount" if r.value < 0 or r.text.strip().startswith("-") else "tax"
    for i, r in enumerate(rs):
        if r.kind == "coupon":  # a coupon or voucher: a discount above the total, a payment below it
            r.kind = "card" if final_total is not None and i > final_total else "discount"
    found, boxes = {}, {}
    for r in rs:
        if r.kind not in LABELS:
            continue
        v = abs(r.value) if r.kind == "discount" else r.value
        if r.kind in ("tax", "discount", "service", "card") and r.kind in found:  # CGST + SGST; card + coupon
            found[r.kind] += v
            boxes[r.kind].append(r.box)
        elif r.kind not in found:
            found[r.kind], boxes[r.kind] = v, [r.box]
    notes = []
    target = found.get("subtotal", found.get("total"))
    for r in rs:
        if r.kind == "item" and target and abs(r.value) > 1.01 * abs(target):
            r.kind = "other"  # a number larger than the whole bill (a postcode, a code) is no price
    items = [r for r in rs if r.kind == "item"]
    if items and target:
        s = sum(r.value for r in items)
        if not (0.5 * target <= s <= 2.0 * target):
            notes.append(f"The item lines add up to {s:,.2f}, far from the "
                         f"{'subtotal' if 'subtotal' in found else 'total'} ({target:,.2f}). They were probably "
                         "misread, so the rule that the items add up is not checked.")
            for r in items:
                r.kind = "other"
            items = []
    its = []
    for k, r in enumerate(items, 1):
        name = re.sub(r"\s+", " ", " ".join(w[0] for w in r.words if not is_amount(w[0]))).strip()
        its.append(cord.Item(name=name, cnt=r.cnt if r.unit is not None else None, unitprice=r.unit, price=r.value))
        boxes[f"price{k}"] = [r.box]
        if r.unit is not None:
            boxes[f"unit{k}"] = [_item_amounts(r.words)[-2][1]]
            cw = next((w for w in r.words if _COUNT_WORD.match(w[0]) and tuple(w[1:]) != tuple(r.box)), None)
            if cw:
                boxes[f"cnt{k}"] = [cw[1:]]
    qty_ok = bool(its) and all(it.cnt for it in its)
    if "qty" in found and not qty_ok:
        notes.append("The printed item count is not checked: not every item line shows its count.")
    rec = cord.Receipt(id="upload", items=its, subtotal=found.get("subtotal"), tax=found.get("tax"),
                       service=found.get("service"), discount=found.get("discount"), total=found.get("total"),
                       cash=found.get("cash"), change=found.get("change"), card=found.get("card"),
                       qty=found.get("qty") if qty_ok else None)
    S = RS.schema(rec)
    labels = {f.name: f.label for f in S.fields}
    for f in ("tax", "discount", "service", "card"):
        if f in labels and len(boxes.get(f, [])) > 1:
            labels[f] += f" ({len(boxes[f])} lines)"
    if "rounding" in found:
        notes.append(f"A rounding of {found['rounding']:+,.2f} is printed: totals within one currency unit "
                     "count as equal, so it is allowed for.")
    return Reading(rec, rs, {f: b for f, b in boxes.items() if f in labels}, labels, notes)


def heat_of(H, boxes, s, pad=3):
    """The maximum of heatmap H (on a view scaled by s) over each field's boxes."""
    out = {}
    h, w = H.shape
    for f, bs in boxes.items():
        vals = []
        for b in bs:
            x0, y0, x1, y1 = (int(round(c * s)) for c in b)
            reg = H[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)]
            if reg.size:
                vals.append(float(reg.max()))
        out[f] = max(vals) if vals else 0.0
    return out


HARD = ("total", "change", "tender")  # the bill's own arithmetic: reliable to read, so a break is an alarm


def analyse(reading, H, s):
    """CVP on a reading. The bill-level rules (subtotal + tax = total, cash - total = change, cash
    covers the total) raise the alarm; the item-level rules (items add up, count x unit price =
    price, item counts) only raise a warning, because item lines are often misread. The broken
    rules shortlist the fields they fit and the heat picks among them: it ranks, it never accuses."""
    S = RS.schema(reading.receipt)
    v = RS.values(reading.receipt)
    w, res = S.evaluate(v)
    hard = [c.name for c in S.constraints if c.name in HARD]
    w_hard = w.copy()
    for j, c in enumerate(S.constraints):
        if c.name not in HARD:
            w_hard[j] = 0.0
    violated = sorted(S.violated(w), key=S.c_index.get)
    broken_hard = [c for c in violated if c in HARD]
    basis = w_hard if broken_hard else w
    consistent = S.consistent_fields(basis) if violated else []
    heat = heat_of(H, reading.boxes, s)
    pool = [f for f in consistent if f in heat]
    field_ = max(pool, key=lambda f: heat[f]) if pool else None
    return dict(schema=S, values=v, w=w, residuals=res, violated=violated, broken_hard=broken_hard,
                broken_soft=[c for c in violated if c not in HARD], consistent=consistent,
                candidates=S.candidate_fields(S.signature_class(basis), basis) if violated else [], heat=heat,
                field=field_, tampered=bool(broken_hard), warning=bool(violated) and not broken_hard,
                checked=len(S.constraints), checked_hard=len(hard))


def words_box(words):
    xs0, ys0, xs1, ys1 = zip(*[w[1:] for w in words])
    return min(xs0), min(ys0), max(xs1), max(ys1)


def row_table(reading):
    """What was read, row by row, for the page: (text, kind, value, box)."""
    return [dict(text=r.text, kind=r.kind, value=r.value, box=words_box(r.words) if r.words else None)
            for r in reading.rows]


def demo_words(lines):
    """Words laid out as rows from plain text lines (for tests): one row per line."""
    out = []
    for i, line in enumerate(lines):
        x = 10.0
        row = []
        for t in line.split():
            row.append((t, x, 20.0 * i + 2, x + 9.0 * len(t), 20.0 * i + 16))
            x += 9.0 * len(t) + 8
        out.append(row)
    return out


__all__ = ["read", "parse", "analyse", "heat_of", "row_table", "is_amount", "LABELS", "Reading", "Row",
           "demo_words"]
