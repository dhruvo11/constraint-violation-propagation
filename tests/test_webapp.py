"""The web app: its plain-English explanations (webapp/explain.py) and its API (webapp/server.py)."""
import os

import pytest

from cvp import pipeline
from webapp import explain

LABELS = {"cash": "Cash", "change": "Change", "total": "Total", "m1": "Mark 1", "m2": "Mark 2"}


def test_the_reason_names_the_only_field_that_fits():
    s = explain.narrate(dict(status="tampered", mode="cord", labels=LABELS, blamed="total", suspects=["total"],
                             broken=["Subtotal + tax = total"], heat={"total": 0.0}, n_fields=3, n_rules=2))
    assert s["headline"] == "Tampered" and "only **Total** can explain that" in s["reason"]
    assert [x["title"] for x in s["steps"]] == ["Read the document", "Check the rules", "Shortlist the suspects",
                                                "Look at the pixels", "Decide"]
    assert "the only field that fits" in s["steps"][-1]["text"]


def test_the_reason_says_when_the_image_decides():
    s = explain.narrate(dict(status="tampered", mode="synthetic", labels=LABELS, blamed="change",
                             suspects=["cash", "change"], broken=["Cash − total = change"],
                             heat={"cash": 0.1, "change": 0.9}, n_fields=3, n_rules=1))
    assert "looks most edited at **Change**" in s["reason"] and "Proposition 1" in s["steps"][2]["text"]


def test_genuine_warm_and_experimental_wordings():
    warm = explain.narrate(dict(status="genuine", mode="synthetic", labels=LABELS, heat={"m1": 0.98},
                                n_fields=2, n_rules=3, score=0.94, threshold=0.95))
    assert "stays below the alarm level (0.95)" in warm["reason"]
    rx = explain.narrate(dict(status="warning", mode="receipt", labels=LABELS, suspects=["m1", "m2"],
                              broken=["The item prices add up to the subtotal"], heat={}, n_fields=2, n_rules=1))
    assert rx["headline"] == "Check this one" and "warning, not an alarm" in rx["reason"]
    none = explain.narrate(dict(status="unknown", mode="receipt", labels={}, heat={}, n_fields=0, n_rules=0))
    assert none["headline"] == "Can't verify"


def test_the_answer_key():
    edited = dict(forged=True, edited=["change"])
    assert explain.answer(edited, "change", "tampered", LABELS)[0] == "right"
    assert explain.answer(edited, "cash", "tampered", LABELS)[0] == "partial"
    assert explain.answer(edited, None, "genuine", LABELS)[0] == "wrong"
    assert explain.answer(edited, None, "warning", LABELS, ["cash", "change"])[1].endswith("among the suspects.")
    assert explain.answer(dict(forged=False, edited=[]), None, "genuine", LABELS)[0] == "right"
    assert explain.answer(dict(forged=False, edited=[]), "m1", "tampered", LABELS)[0] == "wrong"
    names = dict(forged=True, edited=[], what="an item name")
    assert explain.answer(names, None, "genuine", LABELS) == ("wrong", "Missed: the edit changed an item name, "
                                                                       "which no rule reads.")


def test_rules_in_words_with_their_numbers():
    v = dict(m1=60, m2=70, m3=80, m4=90, m5=50, total=355)
    assert explain.arithmetic("transcript", "c1", v, (), {}) == "60 + 70 + 80 + 90 + 50 = 350; the total says 355"
    assert explain.arithmetic("real", "change", dict(cash=100000.0, total=28000.0, change=92000.0), (), {}) == \
        "100,000 (cash) − 28,000 = 72,000; the change says 92,000"
    assert explain.rule_text("real", "line3", "") == "Count × unit price = price (item 3)"
    assert explain.rule_text("transcript", "c4_m1", "0 <= mark 1 <= 100") == "0 ≤ mark 1 ≤ 100"
    assert explain.fmt(1234.5) == "1,234.50" and explain.fmt(None) == "unreadable"


# ------------------------------------------------------------------ the API (without the startup warm-up)
@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from webapp.server import app
    return TestClient(app)  # not used as a context manager: the models load only when a test needs them


def test_the_page_and_the_catalogue(client):
    assert client.get("/").status_code == 200 and "Guided tour" in client.get("/").text
    cat = client.get("/api/catalog").json()
    tour = next(g for g in cat["groups"] if g["id"] == "tour")
    assert len(tour["items"]) == 7 and len(cat["doctypes"]) == 6
    assert set(cat["categories"]) == {"genuine", "T1", "T2", "T3", "T4"}


def test_bad_requests_are_refused(client):
    assert client.post("/api/upload", files={"file": ("x.png", b"not an image", "image/png")},
                       data={"kind": "receipt"}).status_code == 400
    assert client.post("/api/forge", json={"doctype": "nope", "seed": 1, "category": "T2"}).status_code == 400
    assert client.post("/api/forge", json={"doctype": "transcript", "seed": 1, "category": "T9"}).status_code == 400
    assert client.post("/api/sample/no-such-example").status_code == 404
    assert client.post("/api/recheck", json={"id": "expired", "values": {"total": "1"}}).status_code == 404


@pytest.mark.skipif(not os.path.exists(pipeline.heads_path("transcript")), reason="needs the trained models")
def test_a_forged_mark_sheet_end_to_end(client):
    r = client.post("/api/forge", json={"doctype": "transcript", "seed": 2, "category": "T2", "field": "m3"}).json()
    assert r["verdict"]["status"] == "tampered" and r["verdict"]["blamed"] == "Subject mark 3"
    assert r["truth"]["verdict"] == "right" and r["image"]["src"].startswith("data:image/png")
    broken = [x for x in r["rules"] if x["status"] == "broken"]
    assert broken and broken[0]["detail"].endswith("the total says 311")
    assert {f["name"] for f in r["fields"] if f["role"] in ("suspect", "suspect-weak", "blamed")} == \
        {"m1", "m2", "m3", "m4", "m5"}
