"""End-to-end checks with the trained models (skipped when the models are not present)."""
import os

import numpy as np
import pytest

from cvp import doctypes, ocr, paths, pipeline, visual
from cvp.core import render, tamper

T = doctypes.get("transcript")
S = T.schema
needs_models = pytest.mark.skipif(
    not all(os.path.exists(p) for p in (ocr.MODEL_PATH, visual.MODEL_PATH, pipeline.heads_path(T.name))),
    reason="trained models not found; see README to train them")


@pytest.fixture(scope="module")
def system():
    return pipeline.CVP(T, device="cpu")


@needs_models
def test_ocr_reads_generated_documents():
    reader = ocr.Reader()
    right = total = 0
    for seed in range(8):
        d = render.generate(T, np.random.default_rng(1000 + seed))
        o = reader.read(d.image, T)
        for f in S.fields:
            right += o["text"][f.name] == f.fmt(d.values[f.name])
            total += 1
    assert right / total >= 0.95


@needs_models
@pytest.mark.parametrize("seed,category,field", [(2, "T2", "m3"), (3, "T1", "m2"), (5, "T2", "total")])
def test_cvp_attributes_single_field_edits(system, seed, category, field):
    rng = np.random.default_rng(seed)
    d = render.generate(T, rng)
    img, info = tamper.apply(d, T, category, rng, field=field)
    res = system(img)
    assert res["tampered"]
    assert res["field"] in info["edited"]


@needs_models
def test_cvp_passes_a_genuine_document(system):
    rng = np.random.default_rng(9)
    d = render.generate(T, rng)
    img, _ = tamper.apply(d, T, "genuine", rng)
    res = system(img)
    assert not res["tampered"] and res["cls"] == "coherent"


def test_paths_are_inside_the_project():
    for p in (paths.MODELS, paths.data_dir("transcript"), paths.feature_dir("transcript")):
        assert os.path.commonpath([paths.BASE, p]) == paths.BASE
