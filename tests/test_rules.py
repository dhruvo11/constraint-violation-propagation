"""The published algorithms behind the new rules, and each type's constraint structure."""
import datetime as dt

import numpy as np
import pytest

from cvp import doctypes, pipeline
from cvp.core import rules as R
from cvp.doctypes import certificate, invoice, passport, receipt


# ---------------------------------------------------------------- published algorithms
def test_icao_check_digits_match_the_9303_specimen():
    assert R.icao_check("L898902C3") == "6"
    assert R.icao_check("740812") == "2" and R.icao_check("120415") == "9"
    assert R.icao_check("ZE184226B<<<<<") == "1"
    assert R.icao_check("L898902C36" + "7408122" + "1204159" + "ZE184226B<<<<<1") == "0"


def test_mrz_reproduces_the_9303_specimen():
    v = dict(viz_num="L898902C3", viz_surname="ERIKSSON", viz_given="ANNA MARIA", viz_dob=dt.date(1974, 8, 12),
             viz_sex="F", viz_exp=dt.date(2012, 4, 15), viz_pno="ZE184226B")
    assert passport.build_mrz(v) == ("P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
                                     "L898902C36UTO7408122F1204159ZE184226B<<<<<10")


def test_gstin_check_character():
    assert R.gstin_check("27AAPFU0939F1Z") == "V"  # the sample GSTIN used across GST documentation
    rng = np.random.default_rng(0)
    for _ in range(200):
        g = R.make_gstin(rng)
        assert R.gstin_valid(g)
        assert not R.gstin_valid(invoice.typo(g, rng, range(2, 14)))  # Luhn mod 36 catches every typo


def test_indian_money_and_words():
    assert R.inr(123456789) == "12,34,567.89" and R.inr(99) == "0.99" and R.inr(-36) == "-0.36"
    assert R.parse_inr("12,34,567.89") == 123456789 and R.parse_inr("12.3") is None
    assert R.rupees_words(15880) == "RUPEES FIFTEEN THOUSAND EIGHT HUNDRED EIGHTY ONLY"
    assert R.rupees_words(123456) == "RUPEES ONE LAKH TWENTY THREE THOUSAND FOUR HUNDRED FIFTY SIX ONLY"
    for n in [0, 7, 19, 20, 99, 100, 101, 999, 1000, 12345, 99999, 100000, 9999999]:
        assert R.parse_rupees_words(R.rupees_words(n)) == n


def test_financial_year_and_dates():
    assert R.financial_year(dt.date(2026, 3, 31)) == "25-26" and R.financial_year(dt.date(2026, 4, 1)) == "26-27"
    assert R.parse_dmony(R.fmt_dmony(dt.date(1974, 8, 12))) == dt.date(1974, 8, 12)
    assert R.add_years(dt.date(2024, 2, 29), 1) == dt.date(2025, 2, 28)


def test_certificate_check_letter_catches_digit_changes():
    c = certificate.cert_no(2025, 41273)
    assert certificate.cert_ok(c)
    for i, ch in enumerate(c):
        if ch.isdigit():
            for d in "0123456789":
                if d != ch:
                    assert not certificate.cert_ok(c[:i] + d + c[i + 1:])


# ---------------------------------------------------------------- constraint structure
def test_invoice_lines_form_proposition_1_classes():
    S = invoice.SCHEMA
    for k in range(1, invoice.N + 1):
        assert S.class_fields[f"line{k}"] == [f"qty{k}", f"rate{k}"]
    assert all(len(S.class_fields[k]) == 1 for k in S.classes if not k.startswith("line"))


@pytest.mark.parametrize("seed", range(10))
def test_invoice_qty_and_rate_are_exchangeable(seed):
    """Scaling a line's quantity or its rate by the same factor gives identical (w, r)."""
    T = doctypes.get("invoice")
    v, _, _ = T.sample_spec(np.random.default_rng(seed))
    wa, ra = T.schema.evaluate(dict(v, qty1=v["qty1"] * 3))
    wb, rb = T.schema.evaluate(dict(v, rate1=v["rate1"] * 3))
    assert np.array_equal(wa, wb) and np.allclose(ra, rb)


def test_invoice_tax_regime():
    T = doctypes.get("invoice")
    S = T.schema
    for seed in range(30):
        v, _, _ = T.sample_spec(np.random.default_rng(seed))
        intra = v["seller_gstin"][:2] == v["pos"][:2]
        assert (v["igst"] == 0) == intra and (v["cgst"] > 0) == intra
        assert v["total"] % 100 == 0 and abs(v["round_off"]) <= 50 and v["words"] * 100 == v["total"]
    v, _, _ = T.sample_spec(np.random.default_rng(1))
    other = next(s for s in R.STATES if s != v["pos"][:2])
    moved = dict(v, pos=f"{other}-{R.STATES[other]}")  # place of supply no longer the buyer's state
    assert "c_sp" in S.violated(S.evaluate(moved)[0])


def test_receipt_has_no_proposition_1_ambiguity_but_shares_signatures():
    S = receipt.SCHEMA
    assert all(len(S.class_fields[k]) == 1 for k in S.classes)
    w = np.zeros(len(S.constraints), np.float32)
    w[S.c_index["c_chg"]] = 1
    assert set(S.consistent_fields(w)) == {"cash", "change"}


def test_shared_prior_does_not_choose_between_fields_with_one_signature():
    """g_phi may lean to one of two fields that break the same rules; the shared prior
    gives them equal mass (the visual branch decides) and changes nothing for a single class."""
    S = receipt.SCHEMA
    w = np.zeros(len(S.constraints), np.float32)
    w[S.c_index["c_chg"]] = 1
    p = np.zeros(len(S.class_names))
    p[S.class_index[S.field_class["change"]]], p[S.class_index[S.field_class["cash"]]] = 0.9, 0.1
    by_class, shared = pipeline.field_prior(p, w, S), pipeline.field_prior(p, w, S, shared=True)
    change, cash = S.field_index["change"], S.field_index["cash"]
    assert by_class[change] > 5 * by_class[cash]
    assert np.isclose(shared[change], shared[cash]) and np.isclose(shared[change] + shared[cash], 1.0)
    T = doctypes.get("transcript").schema  # five marks, one Proposition 1 class: nothing to pool
    w = np.zeros(len(T.constraints), np.float32)
    w[T.c_index["c1"]] = 1
    p = np.zeros(len(T.class_names))
    p[T.class_index["mark"]] = 1.0
    assert np.allclose(pipeline.field_prior(p, w, T), pipeline.field_prior(p, w, T, shared=True))


def test_passport_equalities_are_two_sided():
    """A careful forger who recomputes the check digits leaves the same signature whether
    the printed date of birth or its MRZ copy was changed."""
    T = doctypes.get("passport")
    S = T.schema
    v, _, _ = T.sample_spec(np.random.default_rng(2))
    born = v["viz_dob"]
    viz = dict(v, viz_dob=R.add_years(born, 1 if born.year % 10 < 9 else -1))  # one digit of the year
    m = v["mrz2"]
    dob = passport.yymmdd(viz["viz_dob"])
    mrz = dict(v, mrz2=passport.line2(m[0:9], dob, m[20], m[21:27], m[28:42].rstrip("<")))
    wa, wb = S.evaluate(viz)[0], S.evaluate(mrz)[0]
    assert S.violated(wa) == S.violated(wb) == {"c_dob"}
    assert set(S.consistent_fields(wa)) == {"viz_dob", "mrz2"}
    # A forger who keeps the old check digits is caught: weights 7, 3, 1 are coprime to 10,
    # so any single-digit substitution changes both the field's check digit and the composite.
    careless = dict(v, mrz2=passport.line2(m[0:9], dob, m[20], m[21:27], m[28:42].rstrip("<"), fix=False, old=m))
    assert {"c_dob", "c_cd2", "c_cmp"} <= S.violated(S.evaluate(careless)[0])


def test_unconstrained_names():
    for name in ("certificate", "idcard"):
        S = doctypes.get(name).schema
        assert not S.constrained("name") and all(S.constrained(f) for f in S.field_names if f != "name")
