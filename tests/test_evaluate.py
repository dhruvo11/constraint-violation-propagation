"""Scoring rules of the evaluation (scripts/evaluate.py, and the real-data scripts that must agree with it)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import evaluate as E  # noqa: E402


def test_only_exact_ties_split_the_credit():
    """A uniform guess is scored by its expectation; a saturated heatmap that still ranks
    the edited field first is a correct arg-max, not a tie."""
    d = {"edited": np.array([[0, 1, 0, 0]] * 3, bool)}
    scores = np.array([[0.25, 0.25, 0.25, 0.25],
                       [1.2e-7, 2.1e-7, 0.9e-7, 0.0],
                       [0.9, 0.1, 0.0, 0.0]])
    assert np.allclose(E.attribution_correct(scores, d), [0.25, 1.0, 0.0])


def test_heat_forms_of_the_attribution_features():
    """prob: [log prior, max, mean, p90]; log: the same statistics in log space; logmax: the
    log of the max only. The prior column does not depend on the form."""
    from cvp import doctypes, pipeline
    S = doctypes.get("transcript").schema
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(len(S.class_names)))
    w = np.zeros(len(S.constraints), np.float32)
    w[S.c_index["c1"]] = 1
    pooled = rng.uniform(1e-8, 1, (len(S.field_names), 3)).astype(np.float32)
    f = {h: pipeline.attribution_features(p, w, pooled, S, heat=h) for h in pipeline.HEAT}
    assert f["prob"].shape == f["log"].shape == (len(S.field_names), 4) and f["logmax"].shape == (len(S.field_names), 2)
    assert np.allclose(f["prob"][:, 1:], pooled) and np.allclose(f["log"][:, 1:], np.log(pooled))
    assert np.allclose(f["logmax"][:, 1], np.log(pooled[:, 0]))
    assert np.allclose(f["prob"][:, 0], f["log"][:, 0]) and np.allclose(f["prob"][:, 0], f["logmax"][:, 0])


def test_any_edited_field_counts():
    """Coherent edits change several fields: naming any of them is a correct attribution."""
    d = {"edited": np.array([[1, 0, 1]], bool)}
    assert E.attribution_correct(np.array([[0.1, 0.2, 0.7]]), d)[0] == 1.0


# ------------------------------------------------------------------ AIForge-Doc (scripts/real_aiforge.py)
def _aiforge_rows(violated, hmax):
    """The same document as real_aiforge.py and real_evaluate_cord.py store it."""
    fields = ["cash", "change", "total"]
    pooled = np.array([[0.2, 0.1, 0.15], [0.7, 0.3, 0.5], [0.9, 0.4, 0.8]], np.float32)
    w = np.array([1.0 if violated else 0.0, 0.0])
    common = dict(fields=fields, w=w, consistent=["cash", "change"] if violated else [])
    ai = dict(common, heat={"zero-shot": dict(pooled=pooled, hmax=hmax)})
    cord = dict(common, candidates=["change"] if violated else [], pooled=pooled, hmax=hmax)
    return ai, cord


def test_aiforge_scores_match_the_cord_evaluation():
    """AIForge-Doc forgeries are scored exactly like our own CORD forgeries: same detection score
    and same field scores for every method, with and without a broken rule."""
    import real_aiforge as RA
    from real_evaluate_cord import score_row
    for violated in (True, False):
        ai, cord = _aiforge_rows(violated, hmax=0.9)
        det, attr = score_row(cord)
        for m in ("Visual-only", "Constraint-only", "CVP"):
            s_det, s_fields = RA.scores(ai, "zero-shot", m)
            assert np.isclose(s_det, det[m]) and np.allclose(s_fields, attr[m]), (violated, m)


def test_aiforge_alarm_is_a_broken_rule_or_heat():
    """An alarm is a broken rule, or else heat at the threshold, as in real_evaluate_cord.py; a
    threshold above 1 (the zero-shot branch's) leaves the alarms to the rules alone."""
    import real_aiforge as RA
    cases = [(True, 0.0, 1.01, True), (False, 0.99, 1.01, False), (True, 0.0, 0.98, True),
             (False, 0.99, 0.98, True), (False, 0.5, 0.98, False)]
    for violated, hmax, t, expected in cases:
        ai, _ = _aiforge_rows(violated, hmax)
        assert RA.alarm(ai, "zero-shot", t) is expected, (violated, hmax, t)


def test_aiforge_cord_index():
    """AIForge-Doc numbers the CORD v2 receipts train 0-799, validation 800-899, test 900-999."""
    import real_aiforge as RA
    assert RA.cord_index({"image_id": "cord_00000"}) == ("train", 0)
    assert RA.cord_index({"image_id": "cord_00799"}) == ("train", 799)
    assert RA.cord_index({"image_id": "cord_00800"}) == ("validation", 0)
    assert RA.cord_index({"image_id": "cord_00999"}) == ("test", 99)
