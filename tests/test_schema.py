"""The constraint machinery and the theory it implements (Section III, Appendices A-B)."""
import datetime as dt

import numpy as np
import pytest

from cvp import doctypes
from cvp.doctypes.transcript import MARK_FIELDS, coherent_values

T = doctypes.get("transcript")
S = T.schema


def example():
    """The worked example of Appendix A."""
    return coherent_values([62, 58, 45, 72, 60], "1RUA24CSE0157", dt.date(2025, 7, 12))


def signature(values):
    return S.violated(S.evaluate(values)[0])


def test_worked_example_is_coherent():
    v = example()
    assert v["total"] == 297 and v["pct"] == 59.4 and v["grade"] == "C"
    assert signature(v) == set()


def test_worked_example_signatures():
    v = dict(example(), m3=95)
    assert signature(v) == {"c1"}
    v = dict(example(), total=397)
    assert signature(v) == {"c1", "c2"}  # c3 reads the printed percentage, which is unchanged
    v = dict(example(), m3=145)
    assert signature(v) == {"c1", "c4_m3"}


def test_equivalence_classes_follow_proposition_1():
    assert S.class_fields["mark"] == MARK_FIELDS
    assert [k for k in S.classes if len(S.class_fields[k]) == 1] == ["total", "pct", "grade", "date", "roll"]


def test_reduction_factor_appendix_b():
    rho, expected = S.reduction_factor()
    assert expected == pytest.approx(3.0) and rho == pytest.approx(0.70)


def test_candidate_sets():
    w = S.evaluate(dict(example(), m3=95))[0]
    assert S.signature_class(w) == "mark"
    assert S.candidate_fields("mark", w) == MARK_FIELDS  # indistinguishable to constraints
    w = S.evaluate(dict(example(), m3=145))[0]
    assert S.candidate_fields(S.signature_class(w), w) == ["m3"]  # the unary rule points at m3


@pytest.mark.parametrize("seed", range(20))
def test_proposition_1_exchangeable_marks(seed):
    """Equal in-range perturbations of two different marks give identical (w, r)."""
    rng = np.random.default_rng(seed)
    v, _, _ = T.sample_spec(rng)
    a, b = rng.choice(MARK_FIELDS, 2, replace=False)
    delta = int(rng.integers(-10, 11)) or 3
    if not all(0 <= v[f] + delta <= 100 for f in (a, b)):
        pytest.skip("perturbation leaves the valid range")
    wa, ra = S.evaluate(dict(v, **{a: v[a] + delta}))
    wb, rb = S.evaluate(dict(v, **{b: v[b] + delta}))
    assert np.array_equal(wa, wb) and np.array_equal(ra, rb)


@pytest.mark.parametrize("seed", range(30))
def test_sampled_documents_are_coherent(seed):
    v, _, _ = T.sample_spec(np.random.default_rng(seed))
    assert signature(v) == set()


@pytest.mark.parametrize("seed", range(60))
def test_single_field_edits_violate_and_are_classified(seed):
    rng = np.random.default_rng(seed)
    v, _, _ = T.sample_spec(rng)
    f = S.field_names[seed % len(S.field_names)]
    edited = dict(v, **{f: T.violating_value(f, v, rng)})
    w = S.evaluate(edited)[0]
    assert w.sum() > 0
    assert S.signature_class(w) == S.field_class[f]
    assert f in S.candidate_fields(S.field_class[f], w)


@pytest.mark.parametrize("seed", range(30))
def test_coherent_edits_keep_every_constraint(seed):
    rng = np.random.default_rng(seed)
    v, _, _ = T.sample_spec(rng)
    new, edited, primary = T.coherent_edit(v, rng)
    assert edited and primary in edited
    assert signature(new) == set()


def test_unreadable_fields_count_as_violations():
    v = dict(example(), total=None)
    w, r = S.evaluate(v)
    assert w[S.c_index["c1"]] == 1 and w[S.c_index["c2"]] == 1 and r[S.c_index["c1"]] == 0
