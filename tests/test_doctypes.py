"""Properties every document type must have, checked for all of them."""
import numpy as np
import pytest

from cvp import doctypes, ocr
from cvp.core import render, tamper
from cvp.doctypes.base import BLANK

TYPES = doctypes.names()


def spec(name, seed):
    return doctypes.get(name).sample_spec(np.random.default_rng(seed))


@pytest.mark.parametrize("name", TYPES)
def test_sampled_documents_are_coherent(name):
    S = doctypes.get(name).schema
    for seed in range(40):
        v, _, _ = spec(name, seed)
        assert S.violated(S.evaluate(v)[0]) == set(), (seed, S.violated(S.evaluate(v)[0]))


@pytest.mark.parametrize("name", TYPES)
def test_every_constraint_reads_declared_fields(name):
    S = doctypes.get(name).schema
    for c in S.constraints:
        assert set(c.scope) <= set(S.field_names), c.name
        assert set(c.sensitive) <= set(c.scope), c.name
        for g in c.exchangeable:
            assert set(g) <= set(c.scope), c.name


@pytest.mark.parametrize("name", TYPES)
def test_single_field_edits_violate_and_stay_consistent(name):
    """A T1/T2 value always breaks a rule, and the edited field is among those the
    violation pattern is consistent with (the candidate set never excludes the truth)."""
    T = doctypes.get(name)
    S = T.schema
    for seed in range(120):
        rng = np.random.default_rng(seed)
        v, _, _ = T.sample_spec(rng)
        f = T.pick_field(v, rng)
        assert S.constrained(f), f"{f} has no rule but was picked for a single-field edit"
        edited = dict(v, **{f: T.violating_value(f, v, rng)})
        w = S.evaluate(edited)[0]
        assert w.sum() > 0, (seed, f, v[f], edited[f])
        assert f in S.consistent_fields(w), (seed, f, S.violated(w), S.consistent_fields(w))


@pytest.mark.parametrize("name", TYPES)
def test_coherent_edits_keep_every_constraint(name):
    T = doctypes.get(name)
    S = T.schema
    for seed in range(60):
        rng = np.random.default_rng(seed)
        v, _, _ = T.sample_spec(rng)
        new, edited, primary = T.coherent_edit(v, rng)
        assert edited and primary in edited, (seed, edited, primary)
        assert S.violated(S.evaluate(new)[0]) == set(), (seed, primary, S.violated(S.evaluate(new)[0]))
        assert all(S.by_name[f].fmt(new[f]) == S.by_name[f].fmt(v[f]) for f in S.field_names if f not in edited)


@pytest.mark.parametrize("name", TYPES)
def test_printed_values_read_back(name):
    """fmt and parse are inverse on sampled values, so the OCR reads what was printed."""
    T = doctypes.get(name)
    for seed in range(20):
        v, _, _ = spec(name, seed)
        for f in T.schema.fields:
            assert f.parse(f.fmt(v[f.name])) == v[f.name], (f.name, v[f.name], f.fmt(v[f.name]))


@pytest.mark.parametrize("name", TYPES)
def test_rendering_is_deterministic_and_boxes_hold_ink(name):
    T = doctypes.get(name)
    a = render.generate(T, np.random.default_rng(3))
    b = render.generate(T, np.random.default_rng(3))
    assert np.array_equal(a.image, b.image)
    h, w = a.image.shape[:2]
    assert (w, h) == T.page
    gray = a.image.mean(2)
    for f, (x0, y0, x1, y1) in a.boxes.items():
        if a.values[f] == BLANK:
            continue
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h, f
        assert gray[y0:y1, x0:x1].min() < 120, f"no ink in {f}"


@pytest.mark.parametrize("name", TYPES)
def test_values_fit_their_ocr_slots(name):
    """Every printed value, genuine or forged, lies inside the crop the OCR reads."""
    T = doctypes.get(name)
    for seed in range(30):
        rng = np.random.default_rng(seed)
        v, c, st = T.sample_spec(rng)
        if seed % 2:
            v = T.random_values(rng, v)
        d = render.render(T, v, c, st)
        _, info = tamper.apply(d, T, tamper.CATEGORIES[seed % 5], rng)
        for f in T.schema.field_names:
            x0, y0, x1, y1 = ocr.slot_rect(T, f, st.ox, st.oy, d.image.shape)
            b = info["boxes"][f]
            if b[2] - b[0] > 2:
                assert x0 <= b[0] and y0 <= b[1] and b[2] <= x1 and b[3] <= y1, (seed, f, b, (x0, y0, x1, y1))


@pytest.mark.parametrize("name", TYPES)
def test_page_registration_recovers_the_offset(name):
    T = doctypes.get(name)
    for seed in range(10):
        d = render.generate(T, np.random.default_rng(100 + seed))
        ox, oy = ocr.register(ocr.gray_of(d.image), T.header_rule, T.rule_min)
        assert abs(ox - d.style.ox) <= 1 and abs(oy - d.style.oy) <= 1, (seed, (ox, oy), (d.style.ox, d.style.oy))


@pytest.mark.parametrize("name", TYPES)
@pytest.mark.parametrize("category", tamper.CATEGORIES)
def test_forgery_categories(name, category):
    T = doctypes.get(name)
    S = T.schema
    rng = np.random.default_rng(11)
    d = render.generate(T, rng)
    img, info = tamper.apply(d, T, category, rng)
    assert img.shape == d.image.shape
    violations = S.evaluate(info["values"])[0].sum()
    if category == "genuine":
        assert info["edited"] == [] and violations == 0
    elif category in ("T1", "T2"):
        assert len(info["edited"]) == 1 and violations > 0
    else:
        assert info["edited"] and violations == 0


@pytest.mark.parametrize("name", TYPES)
def test_edits_stay_in_their_region(name):
    T = doctypes.get(name)
    for category in ("T1", "T2"):
        rng = np.random.default_rng(5)
        d = render.generate(T, rng)
        img, info = tamper.apply(d, T, category, rng)
        diff = np.abs(img.astype(int) - d.image.astype(int)).max(2) > 40
        x0, y0, x1, y1 = info["rects"][0]
        outside = diff.copy()
        outside[max(0, y0 - 8):y1 + 8, max(0, x0 - 8):x1 + 8] = False
        assert diff[y0:y1, x0:x1].any(), "the edit left no trace in its own region"
        assert outside.mean() < 0.002, "the edit leaked outside its region"
