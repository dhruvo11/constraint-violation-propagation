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
