"""The experimental reader of receipt photos (cvp/real/upload.py), on receipts typed as rows of words."""
import numpy as np

from cvp.real import upload as U

BILL = ["MY CAFE", "MG Road Bengaluru 560001", "Bill No 1234  Date 12/03/2024", "Item Qty Rate Amount",
        "Paneer Tikka 2 180.00 360.00", "Masala Chai 3 30.00 90.00", "Total 450.00", "CGST @2.5% 11.25",
        "SGST 2.5 % 11.25", "Grand Total 472.50", "Cash 500.00", "Change 27.50", "Thank you"]
NO_HEAT = np.zeros((900, 700), np.float32)


def read(lines):
    return U.parse(U.demo_words(lines))


def check(lines):
    rd = read(lines)
    return rd, U.analyse(rd, NO_HEAT, 1.0)


def test_amounts_and_not_amounts():
    for t in ("450", "1,234.50", "Rp.111,000", "(1,000)", "-2,600", "25.000", "₹1,23,456.00"):
        assert U.is_amount(t), t
    for t in ("12:30", "12/03/2024", "2.5%", "XXXX1234", "080-2345", "#12"):
        assert not U.is_amount(t), t
    assert U.value_of("₹1,23,456.00") == 123456.0 and U.value_of("1,23,456") == 123456.0
    assert U.value_of("25.000") == 25000.0 and U.value_of("12.50") == 12.5


def test_an_indian_bill_is_read_and_holds():
    rd, res = check(BILL)
    r = rd.receipt
    assert (r.subtotal, r.tax, r.total, r.cash, r.change) == (450.0, 22.5, 472.5, 500.0, 27.5)
    assert [(it.cnt, it.unitprice, it.price) for it in r.items] == [(2.0, 180.0, 360.0), (3.0, 30.0, 90.0)]
    assert rd.labels["tax"] == "Tax (2 lines)"            # CGST and SGST together
    assert res["violated"] == [] and not res["tampered"] and not res["warning"] and res["checked"] >= 4


def test_an_edited_total_is_an_alarm_on_the_total():
    lines = list(BILL)
    lines[9] = "Grand Total 482.50"
    _, res = check(lines)
    assert res["tampered"] and set(res["broken_hard"]) == {"total", "change"}
    assert res["consistent"] == ["total"]                # the only field both broken rules read


def test_an_edited_item_price_only_warns():
    lines = list(BILL)
    lines[5] = "Masala Chai 3 30.00 99.00"
    _, res = check(lines)
    assert not res["tampered"] and res["warning"]
    assert "sum" in res["broken_soft"] or any(c.startswith("line") for c in res["broken_soft"])


def test_a_charge_with_a_garbled_name_counts():
    rd = read(["Nasi Goreng 1 30,000", "SUBTOTAL 325,600", "Service chrg 17,908", "10% PBI 34,351", "TOTAL 377,859"])
    kinds = {r.text: r.kind for r in rd.rows}
    assert kinds["10% PBI 34,351"] == "tax" and kinds["Service chrg 17,908"] == "service"


def test_ocr_near_misses_of_keywords():
    rd = read(["ICE COFFEE 82,000", "SUB_TSTAL 82,000", "TOTAL 82,000", "CASH 100,000", "KEMBALL 18,000"])
    assert rd.receipt.subtotal == 82000.0 and rd.receipt.change == 18000.0


def test_a_coupon_after_the_total_is_a_payment():
    _, res = check(["Tea 1 50,000", "SUBTOTAL 50,000", "TOTAL 50,000", "Coupon 20,000", "CASH 40,000",
                    "Change 10,000"])
    assert res["violated"] == []                          # 40,000 + 20,000 - 50,000 = 10,000


def test_misread_items_are_dropped_with_a_note():
    rd = read(["Mysuru Bakery 570001", "Bread 40.00", "SUBTOTAL 40.00", "TOTAL 40.00"])
    assert all(r.kind != "item" or r.value < 1000 for r in rd.rows)   # the postcode is no price
    rd = read(["Bread 10.00", "Jam 5.00", "SUBTOTAL 40.00", "TOTAL 40.00"])
    assert not rd.receipt.items and rd.notes                          # far from the subtotal: not checked


def test_several_totals_the_count_and_the_payment():
    rd = read(["Soup 1 28,000", "SUB TOTAL 28,000", "TOTAL SALES 28,000", "TOTAL ITENS 1", "CASH 100,000",
               "CHANGE 72,000"])
    assert rd.receipt.total == 28000.0                    # not the misread item count


def test_heat_ranks_the_suspects():
    lines = list(BILL)
    lines[11] = "Change 37.50"                           # cash or change: the rule can't tell
    rd = read(lines)
    H = np.zeros((900, 700), np.float32)
    x0, y0, x1, y1 = (int(c) for c in rd.boxes["change"][0])
    H[y0:y1, x0:x1] = 0.9
    res = U.analyse(rd, H, 1.0)
    assert set(res["consistent"]) == {"cash", "change"} and res["field"] == "change"
