import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from classify_exceptions import classify_exception, make_reason


def row(**kwargs):
    base = {
        "invoice_id": "INV-1", "payment_id": "PMT-1",
        "invoice_amount": 1000.0, "payment_amount": 1000.0,
        "amount_difference": 0.0, "customer_similarity": 100.0,
        "date_difference": 2, "reference_match": True,
        "confidence_score": 80.0,
    }
    base.update(kwargs)
    return pd.Series(base)


def assert_type(r, expected):
    typ, severity, action = classify_exception(r)
    assert typ == expected
    assert severity in {"HIGH", "MEDIUM", "LOW"}
    assert action


def test_amount_mismatch():
    assert_type(row(payment_amount=1500, amount_difference=-500), "AMOUNT_MISMATCH")


def test_missing_payment():
    assert_type(row(payment_id=None), "MISSING_PAYMENT")


def test_duplicate_payment():
    assert classify_exception(row(invoice_id=None), duplicate_payment=True)[0] == "DUPLICATE_PAYMENT"


def test_customer_ambiguity():
    assert_type(row(customer_similarity=45), "CUSTOMER_AMBIGUITY")


def test_date_mismatch():
    assert_type(row(date_difference=30), "DATE_MISMATCH")


def test_missing_reference():
    assert_type(row(reference_match=False, confidence_score=85), "MISSING_REFERENCE")


def test_unmatched_transaction():
    assert_type(row(invoice_id=None, customer_similarity=0), "UNMATCHED_TRANSACTION")


def test_partial_payment():
    assert_type(row(payment_amount=600, amount_difference=400, customer_similarity=95), "PARTIAL_PAYMENT")


def test_other():
    assert_type(row(customer_similarity=80, date_difference=5, reference_match=True), "OTHER")


def test_reason_amount_mismatch():
    r = row(payment_amount=1150, amount_difference=-150)
    typ, _, _ = classify_exception(r)
    assert make_reason(r, typ) == "Payment is ₹150.00 higher than invoice amount."
