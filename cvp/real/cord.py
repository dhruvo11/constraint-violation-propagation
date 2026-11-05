"""CORD v2: 1,000 receipts photographed in Indonesian shops and restaurants, with key-value
labels for every printed field (Park et al., "CORD: A Consolidated Receipt Dataset for
Post-OCR Parsing", NeurIPS Document Intelligence Workshop 2019; CC BY 4.0).

The data lives in data/real/cord-v2 (Hugging Face: naver-clova-ix/cord-v2). Each receipt
has a structured parse (`gt_parse`: menu items, sub_total, total) and every labelled word
with its quadrilateral (`valid_line`), which locates each field on the photo.
"""
import glob
import json
import os
from dataclasses import dataclass, field

from .. import paths
from .money import amount, count

ROOT = paths.real_dir("cord-v2")
SPLITS = ("train", "validation", "test")


@dataclass
class Item:
    name: str
    cnt: float = None
    unitprice: float = None
    price: float = None
    addons: list = field(default_factory=list)  # prices of sub-items (toppings, sizes...)
    discount: float = None


@dataclass
class Receipt:
    id: str
    items: list
    subtotal: float = None
    tax: float = None
    service: float = None
    discount: float = None
    total: float = None
    cash: float = None
    change: float = None
    card: float = None
    qty: float = None  # the printed number of items
    lines: list = field(default_factory=list)  # labelled words: [{category, words: [{text, quad}]}]
    size: tuple = None  # (width, height) of the photo
    image: bytes = None  # JPEG/PNG bytes when loaded with images=True


def _as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def _section(p, key):
    s = p.get(key)
    return s if isinstance(s, dict) else {}


def parse(gt, image=None, rid=""):
    """A Receipt from one CORD ground-truth record (the JSON in its `ground_truth` column)."""
    p = gt["gt_parse"]
    items = []
    for it in _as_list(p.get("menu")):
        if not isinstance(it, dict):
            continue
        addons = [amount(s.get("price")) for s in _as_list(it.get("sub")) if isinstance(s, dict)]
        items.append(Item(name=str(it.get("nm", "")), cnt=count(it.get("cnt")), unitprice=amount(it.get("unitprice")),
                          price=amount(it.get("price")), addons=[a for a in addons if a is not None],
                          discount=amount(it.get("discountprice"))))
    st, tt = _section(p, "sub_total"), _section(p, "total")
    size = gt.get("meta", {}).get("image_size", {})
    return Receipt(id=rid, items=items, subtotal=amount(st.get("subtotal_price")), tax=amount(st.get("tax_price")),
                   service=amount(st.get("service_price")), discount=amount(st.get("discount_price")),
                   total=amount(tt.get("total_price")), cash=amount(tt.get("cashprice")),
                   change=amount(tt.get("changeprice")), card=amount(tt.get("creditcardprice")),
                   qty=count(tt.get("menuqty_cnt")), lines=gt.get("valid_line", []),
                   size=(size.get("width"), size.get("height")), image=image)


AMOUNTS = {"sub_total.subtotal_price": "subtotal", "sub_total.tax_price": "tax", "sub_total.service_price": "service",
           "sub_total.discount_price": "discount", "total.total_price": "total", "total.cashprice": "cash",
           "total.changeprice": "change", "total.creditcardprice": "card"}


def value_words(line):
    """The words of a labelled line that print its value (not its key, such as "TOTAL")."""
    return [w for w in line["words"] if not w.get("is_key")]


def number_text(texts):
    """The printed number among a line's value words: the last word with a digit (amounts are
    printed rightmost: "Rp. | 111,000", or a value repeated in two columns), except digit
    groups split by a space ("2 | 600"), which join."""
    nums = [t for t in texts if any(ch.isdigit() for ch in t)]
    if not nums:
        return " ".join(texts)
    tail = 0
    while tail + 1 < len(nums) and len([c for c in nums[-1 - tail] if c.isdigit()]) == 3 \
            and nums[-1 - tail] != nums[-2 - tail] and nums[-1 - tail].isdigit():
        tail += 1
    return "".join(nums[len(nums) - 1 - tail:])


def from_lines(lines, read=None, rid="", size=None):
    """A Receipt rebuilt from labelled words. `read(word)` gives each value word's text
    (default: the dataset's transcription), so the same code builds the receipt as an OCR
    reads it. A menu item is the set of lines sharing a group id; "menu.sub.*" lines are its
    add-ons."""
    read = read or (lambda w: w["text"])
    groups, fields = {}, {}
    for line in lines:
        texts = [read(w) for w in value_words(line)]
        cat = line["category"]
        text = " ".join(texts).strip() if cat in ("menu.nm", "menu.sub.nm") else number_text(texts)
        if cat.startswith("menu."):
            groups.setdefault(line["group_id"], []).append((cat, text))
        elif cat in AMOUNTS:
            fields[AMOUNTS[cat]] = amount(text)
        elif cat == "total.menuqty_cnt":
            fields["qty"] = count(text)
    items = []
    for _, entries in sorted(groups.items()):
        it = Item(name="")
        for cat, text in entries:
            if cat == "menu.nm":
                it.name = text
            elif cat == "menu.cnt":
                it.cnt = count(text)
            elif cat == "menu.unitprice":
                it.unitprice = amount(text)
            elif cat == "menu.price":
                it.price = amount(text)
            elif cat == "menu.discountprice":
                it.discount = amount(text)
            elif cat == "menu.sub.price" and amount(text) is not None:
                it.addons.append(amount(text))
        items.append(it)
    return Receipt(id=rid, items=items, lines=lines, size=size, **fields)


def files(split, root=ROOT):
    return sorted(glob.glob(os.path.join(root, "data", f"{split}-*.parquet")))


def receipts(split, root=ROOT, images=False):
    """Every receipt of a split, in dataset order; images are loaded only when asked for."""
    import pyarrow.parquet as pq
    k = 0
    for path in files(split, root):
        t = pq.read_table(path, columns=["ground_truth"] + (["image"] if images else []))
        gts = t.column("ground_truth").to_pylist()
        imgs = t.column("image").to_pylist() if images else [None] * len(gts)
        for gt, img in zip(gts, imgs):
            yield parse(json.loads(gt), img["bytes"] if img else None, rid=f"{split}_{k:04d}")
            k += 1
