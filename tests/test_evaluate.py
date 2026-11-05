"""Scoring rules of the evaluation (scripts/evaluate.py)."""
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
