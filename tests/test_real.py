"""Real-document readers and rules, on hand-made records (no downloaded data needed)."""
import pytest

from cvp.real import cord, money, receipt_rules


@pytest.mark.parametrize("text,value", [
    ("60.000", 60000), ("24,000", 24000), ("17500", 17500), ("48982.68", 48982.68), ("1,234.56", 1234.56),
    ("1.234.567", 1234567), ("Rp. 111,000", 111000), ("@11000", 11000), ("-60.000", -60000), ("(1,000)", -1000),
    ("-2 600", -2600), ("5.455", 5455), ("0", 0), ("0.00", 0), ("2.5", 2.5)])
def test_amounts_as_printed_on_receipts(text, value):
    assert money.amount(text) == pytest.approx(value)


@pytest.mark.parametrize("text", ["100%", "0%", "", "abc", None, ["1", "2"]])
def test_texts_that_are_not_amounts(text):
    assert money.amount(text) is None


@pytest.mark.parametrize("text,value", [
    ("2", 2), ("1X", 1), ("x2", 2), ("1 x", 1), ("2.00", 2), ("1.00xITEMs", 1), ("(Qty=4.00)", 4), ("(2", 2)])
def test_counts_are_never_negative(text, value):
    assert money.count(text) == value


def record(**over):
    """A consistent CORD-style ground truth: 2 x 11,000 + 17,500 + 10% tax, paid in cash."""
    gt = {"gt_parse": {
        "menu": [{"nm": "ES TEH", "cnt": "2x", "unitprice": "@11,000", "price": "22,000"},
                 {"nm": "NASI GORENG", "cnt": "1", "price": "17,500", "sub": {"nm": "+ TELUR", "price": "3,000"}}],
        "sub_total": {"subtotal_price": "42,500", "tax_price": "4,250"},
        "total": {"total_price": "46,750", "cashprice": "50,000", "changeprice": "3,250", "menuqty_cnt": "3"}},
        "meta": {"image_size": {"width": 400, "height": 800}}, "valid_line": []}
    for path, v in over.items():
        sec, key = path.split(".")
        gt["gt_parse"][sec][key] = v
    return gt


def test_a_consistent_receipt_satisfies_every_rule():
    r = cord.parse(record())
    assert r.items[0].cnt == 2 and r.items[0].unitprice == 11000 and r.items[1].addons == [3000]
    assert receipt_rules.check(r) == {"line": [True], "sum": True, "total": True, "change": True,
                                      "tender": True, "count": True}


@pytest.mark.parametrize("edit,broken", [
    ({"total.total_price": "48,750"}, {"total", "change"}),
    ({"total.total_price": "56,750"}, {"total", "change", "tender"}),  # now more than the cash paid
    ({"total.changeprice": "13,250"}, {"change"}),
    ({"sub_total.subtotal_price": "52,500"}, {"sum", "total"}),
    ({"total.cashprice": "40,000"}, {"change", "tender"}),
    ({"total.menuqty_cnt": "4"}, {"count"}),
])
def test_one_edited_value_breaks_the_rules_that_read_it(edit, broken):
    res = receipt_rules.check(cord.parse(record(**edit)))
    assert {k for k, v in res.items() if v is False or (isinstance(v, list) and not all(v))} == broken


def test_edits_within_rounding_tolerance_are_invisible_to_the_rules():
    """A limitation worth stating: changing the total by 100 rupiah (0.2%) looks like rounding."""
    res = receipt_rules.check(cord.parse(record(**{"total.total_price": "46,850"})))
    assert res["total"] is True


def test_tax_included_prices_and_card_payments_are_genuine_conventions():
    r = cord.parse(record(**{"sub_total.subtotal_price": "42,500", "total.total_price": "42,500",
                             "total.cashprice": "20,000", "total.changeprice": "0"}))
    r.card = 22500.0  # part paid by card
    res = receipt_rules.check(r)
    assert res["total"] and res["tender"] and res["change"]


def test_rules_that_read_missing_values_do_not_apply():
    gt = record()
    del gt["gt_parse"]["total"]["cashprice"], gt["gt_parse"]["total"]["menuqty_cnt"]
    res = receipt_rules.check(cord.parse(gt))
    assert res["change"] is None and res["tender"] is None and res["count"] is None and res["total"] is True
