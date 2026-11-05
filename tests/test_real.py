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


def test_receipts_rebuild_from_labelled_words_without_their_keys():
    """Key words ("TOTAL", "PB1 (10%)") are not part of a value; an item is a group of lines;
    `read` replaces each value word's text, as an OCR would."""
    def line(cat, group, *words):
        return {"category": cat, "group_id": group, "sub_group_id": 0,
                "words": [{"text": t, "is_key": k, "quad": {}} for t, k in words]}
    lines = [line("menu.nm", 3, ("ES", 0), ("TEH", 0)), line("menu.cnt", 3, ("2x", 0)),
             line("menu.price", 3, ("22,000", 0)), line("menu.nm", 4, ("ROTI", 0)), line("menu.price", 4, ("9,000", 0)),
             line("menu.sub.price", 4, ("1,000", 0)), line("sub_total.tax_price", 5, ("PB1", 1), ("(10%)", 1), ("3,200", 0)),
             line("total.total_price", 6, ("TOTAL", 1), ("35,200", 0))]
    r = cord.from_lines(lines)
    assert [(it.name, it.cnt, it.price, it.addons) for it in r.items] == [("ES TEH", 2, 22000, []), ("ROTI", None, 9000, [1000])]
    assert r.tax == 3200 and r.total == 35200
    misread = cord.from_lines(lines, read=lambda w: w["text"].replace("35,200", "36,200"))
    assert misread.total == 36200 and misread.tax == 3200


def test_rules_that_read_missing_values_do_not_apply():
    gt = record()
    del gt["gt_parse"]["total"]["cashprice"], gt["gt_parse"]["total"]["menuqty_cnt"]
    res = receipt_rules.check(cord.parse(gt))
    assert res["change"] is None and res["tender"] is None and res["count"] is None and res["total"] is True


def test_a_receipts_own_schema_follows_the_rules_and_proposition_1():
    """Each receipt gets its own constraint graph. Without unit prices, item prices are read
    only through their sum: one class, so an edit of either leaves the same signature."""
    from cvp.real import receipt_schema as RS
    r = cord.parse(record())
    S = RS.schema(r)
    assert set(S.constraint_names) == {"line1", "sum", "total", "change", "tender", "count"}
    w, _ = S.evaluate(RS.values(r))
    assert not S.violated(w)  # the genuine receipt
    gt = record()
    del gt["gt_parse"]["menu"][0]["unitprice"]
    r = cord.parse(gt)
    S = RS.schema(r)
    assert S.field_class["price1"] == S.field_class["price2"]
    v = RS.values(r)
    v["price2"] += 5000
    w, _ = S.evaluate(v)
    assert S.violated(w) == {"sum"}
    assert set(S.candidate_fields(S.signature_class(w), w)) == {"price1", "price2"}


def test_schema_and_validity_rules_agree_on_every_edit():
    from cvp.real import receipt_schema as RS
    for edit in ({"total.total_price": "48,750"}, {"total.changeprice": "13,250"}, {"total.menuqty_cnt": "4"},
                 {"sub_total.subtotal_price": "52,500"}, {"total.cashprice": "40,000"}):
        r = cord.parse(record(**edit))
        res = receipt_rules.check(r)
        broken = {k for k, v in res.items() if v is False or (isinstance(v, list) and not all(v))}
        S = RS.schema(cord.parse(record()))  # the genuine receipt's structure, evaluated on the edited reading
        w, _ = S.evaluate(RS.values(r))
        assert {c.rstrip("0123456789") for c in S.violated(w)} == broken


def test_without_a_subtotal_the_items_add_up_to_the_total():
    """Many receipts print no subtotal (e.g. 17,500 + 46,000 + 27,500 = TOTAL 91,000)."""
    from cvp.real import receipt_schema as RS
    gt = {"gt_parse": {"menu": [{"nm": "A", "price": "17500"}, {"nm": "B", "price": "46000"}, {"nm": "C", "price": "27500"}],
                       "total": {"total_price": "91000", "cashprice": "91000"}}, "meta": {}, "valid_line": []}
    r = cord.parse(gt)
    assert receipt_rules.check(r)["sum"] is True
    S = RS.schema(r)
    v = RS.values(r)
    v["price2"] = 48000
    w, _ = S.evaluate(v)
    assert S.violated(w) == {"sum"} and len(S.candidate_fields(S.signature_class(w), w)) == 3  # one class of prices


@pytest.mark.parametrize("words,text", [(["Rp.", "111,000"], "111,000"), (["46.636", "46.636"], "46.636"),
                                        (["-2", "600"], "-2600"), (["1", "x"], "1"), (["17500"], "17500")])
def test_the_number_among_a_lines_value_words(words, text):
    assert cord.number_text(words) == text


@pytest.mark.parametrize("value,like,out", [(36000, "46000", "36000"), (12500, "11,000", "12,500"),
                                            (65000, "60.000", "65.000"), (12500, "@11000", "@12500"),
                                            (121000, "Rp. 111,000", "Rp. 121,000"), (52341.2, "48982.68", "52341.20"),
                                            (3, "1X", "3X"), (1234567, "1.234.567", "1.234.567"), (21000, "-2 600", "-21 000")])
def test_forged_values_are_printed_in_the_receipts_own_style(value, like, out):
    from cvp.real import forge
    assert forge.printed(value, like) == out


def test_forged_amounts_change_for_real_and_keep_their_digits():
    import numpy as np
    from cvp.real import forge
    rng = np.random.default_rng(0)
    for old in (17500.0, 91000.0, 4664.0, 230000.0, 700.0):
        for _ in range(50):
            new = forge.other_value(old, rng)
            assert abs(new - old) > max(1.0, 0.02 * old) and len(str(int(new))) == len(str(int(old)))
            assert new % forge.granularity(old) == 0


def test_coherent_edits_keep_every_rule():
    import numpy as np
    from cvp.real import forge, receipt_schema as RS
    r = cord.parse(record())
    S = RS.schema(r)
    for seed in range(20):
        ch = forge.coherent(r, np.random.default_rng(seed))
        assert ch and any(f.startswith("price") for f in ch)
        v = dict(RS.values(r), **ch)
        w, _ = S.evaluate(v)
        assert not S.violated(w), (seed, ch)


def test_the_visual_view_scales_large_photos_and_pads_narrow_ones():
    import numpy as np
    from cvp.real import view
    big = np.full((4000, 3000, 3), 200, np.uint8)
    v, s = view.visual_view(big)
    assert max(v.shape[:2]) == view.MAX_SIDE and abs(s - view.MAX_SIDE / 4000) < 1e-9
    assert view.scale_box([400, 800, 1200, 1600], s) == [160, 320, 480, 640]
    narrow = np.full((500, 200, 3), 200, np.uint8)
    v, s = view.visual_view(narrow, min_side=256)
    assert s == 1.0 and v.shape[:2] == (500, 256)  # padded on the right: boxes keep their coordinates
    v, _ = view.visual_view(narrow)
    assert v.shape[:2] == (500, 200)  # no padding at test time
