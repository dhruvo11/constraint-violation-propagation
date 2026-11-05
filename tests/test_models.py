"""End-to-end checks with the trained models (skipped when the models are not present)."""
import os

import numpy as np
import pytest

from cvp import doctypes, ocr, paths, pipeline, visual
from cvp.core import render, tamper

TYPES = doctypes.names()


def have_models(name):
    return all(os.path.exists(p) for p in (ocr.MODEL_PATH, visual.MODEL_PATH, pipeline.heads_path(name)))


_systems = {}


def system(name):
    if name not in _systems:
        _systems[name] = pipeline.CVP(name, device="cpu")
    return _systems[name]


@pytest.mark.parametrize("name", TYPES)
def test_ocr_reads_generated_documents(name):
    if not os.path.exists(ocr.MODEL_PATH):
        pytest.skip("OCR model not trained")
    T = doctypes.get(name)
    reader = ocr.Reader()
    right = total = 0
    for seed in range(8):
        d = render.generate(T, np.random.default_rng(1000 + seed))
        o = reader.read(d.image, T)
        for f in T.schema.fields:
            right += o["text"][f.name] == f.fmt(d.values[f.name])
            total += 1
    assert right / total >= 0.95


@pytest.mark.parametrize("name", TYPES)
def test_cvp_attributes_crude_single_field_edits(name):
    """Crude edits leave both kinds of evidence: CVP should flag and attribute them."""
    if not have_models(name):
        pytest.skip("trained models not found")
    T = doctypes.get(name)
    hits = 0
    for seed in range(6):
        rng = np.random.default_rng(seed)
        d = render.generate(T, rng)
        img, info = tamper.apply(d, T, "T1", rng)
        res = system(name)(img)
        hits += res["tampered"] and res["field"] in info["edited"]
    assert hits >= 5


@pytest.mark.parametrize("name", TYPES)
def test_cvp_passes_genuine_documents(name):
    if not have_models(name):
        pytest.skip("trained models not found")
    T = doctypes.get(name)
    passed = 0
    for seed in range(6):
        rng = np.random.default_rng(100 + seed)
        d = render.generate(T, rng)
        img, _ = tamper.apply(d, T, "genuine", rng)
        passed += not system(name)(img)["tampered"]
    assert passed >= 5


def test_paths_are_inside_the_project():
    for name in TYPES:
        for p in (paths.MODELS, paths.data_dir(name), paths.feature_dir(name)):
            assert os.path.commonpath([paths.BASE, p]) == paths.BASE
