"""Rendering and forgery generation: determinism, ground truth and the taxonomy."""
import numpy as np
import pytest

from cvp import doctypes
from cvp.core import render, tamper
from cvp.core.render import jpeg_bytes

T = doctypes.get("transcript")
S = T.schema


def doc(seed):
    return render.generate(T, np.random.default_rng(seed))


def test_rendering_is_deterministic():
    assert jpeg_bytes(doc(7).image, 90) == jpeg_bytes(doc(7).image, 90)


def test_field_boxes_lie_on_the_page_and_hold_ink():
    d = doc(3)
    h, w = d.image.shape[:2]
    gray = d.image.mean(2)
    for f, (x0, y0, x1, y1) in d.boxes.items():
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h, f
        assert gray[y0:y1, x0:x1].min() < 120, f"no ink in {f}"


@pytest.mark.parametrize("category", tamper.CATEGORIES)
def test_categories(category):
    rng = np.random.default_rng(11)
    d = render.generate(T, rng)
    img, info = tamper.apply(d, T, category, rng)
    assert img.shape == d.image.shape
    violations = S.evaluate(info["values"])[0].sum()
    if category == "genuine":
        assert info["edited"] == [] and info["values"] == d.values and violations == 0
    elif category in ("T1", "T2"):
        assert len(info["edited"]) == 1 and violations > 0
        assert all(info["values"][f] == d.values[f] for f in S.field_names if f not in info["edited"])
    else:
        assert info["edited"] and violations == 0  # coherent: the constraint branch is blind


@pytest.mark.parametrize("category", ["T1", "T2"])
def test_edit_changes_only_the_edited_region(category):
    rng = np.random.default_rng(5)
    d = render.generate(T, rng)
    img, info = tamper.apply(d, T, category, rng, field="m3")
    diff = np.abs(img.astype(int) - d.image.astype(int)).max(2) > 40
    x0, y0, x1, y1 = info["rects"][0]
    outside = diff.copy()
    outside[max(0, y0 - 8):y1 + 8, max(0, x0 - 8):x1 + 8] = False
    assert diff[y0:y1, x0:x1].any(), "the edit left no trace in its own region"
    assert outside.mean() < 0.002, "the edit leaked outside its region"
