import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reconciler import reconcile


def make_inputs(invoice_amount=1000.0, payment_amount=1000.0, payer="Acme Pvt Ltd", payment_date="2026-08-30", reference="INV-1"):
    invoices = pd.DataFrame([{
        "invoice_id": "INV-1", "customer_name": "Acme Private Limited",
        "invoice_date": "2026-08-01", "due_date": "2026-08-31",
        "amount": invoice_amount, "currency": "USD", "description": "Consulting services"
    }])
    payments = pd.DataFrame([{
        "payment_id": "PMT-1", "payer_name": payer,
        "payment_date": payment_date, "amount": payment_amount,
        "currency": "USD", "method": "Wire", "reference_note": reference
    }])
    settlements = pd.DataFrame([{
        "settlement_id": "STL-1", "payment_id": "PMT-1",
        "settlement_date": "2026-08-06", "gross_amount": payment_amount,
        "fee": 0.0, "settled_amount": payment_amount, "status": "settled"
    }])
    return invoices, payments, settlements


def test_exact_match():
    result = reconcile(*make_inputs())
    row = result[result.invoice_id == "INV-1"].iloc[0]
    assert row.status == "MATCHED"
    assert row.payment_id == "PMT-1"
    assert row.confidence_score >= 90


def test_fuzzy_customer_match():
    inputs = make_inputs(payer="Acme Prvate Ltd", reference="payment")
    result = reconcile(*inputs)
    row = result[result.invoice_id == "INV-1"].iloc[0]
    assert row.customer_similarity >= 70
    assert row.payment_id == "PMT-1"
    assert row.status in {"MATCHED", "REVIEW"}


def test_amount_mismatch():
    result = reconcile(*make_inputs(payment_amount=500.0))
    row = result[result.invoice_id == "INV-1"].iloc[0]
    assert row.amount_difference == 500.0
    assert row.status != "MATCHED"


def test_missing_payment():
    invoices, payments, settlements = make_inputs()
    result = reconcile(invoices, payments.iloc[0:0], settlements.iloc[0:0])
    row = result[result.invoice_id == "INV-1"].iloc[0]
    assert row.status == "UNMATCHED"
    assert row.payment_id is None


def test_duplicate_payment():
    invoices, payments, settlements = make_inputs()
    duplicate = payments.copy()
    duplicate["payment_id"] = "PMT-2"
    duplicate["reference_note"] = "duplicate transfer"
    payments = pd.concat([payments, duplicate], ignore_index=True)
    settlement2 = settlements.copy()
    settlement2["payment_id"] = "PMT-2"
    settlement2["settlement_id"] = "STL-2"
    settlements = pd.concat([settlements, settlement2], ignore_index=True)

    result = reconcile(invoices, payments, settlements)
    invoice_rows = result[result.invoice_id == "INV-1"]
    unmatched_payment_rows = result[(result.invoice_id.isna()) & (result.payment_id == "PMT-2")]
    assert len(invoice_rows) == 1
    assert invoice_rows.iloc[0].status == "MATCHED"
    assert len(unmatched_payment_rows) == 1


def test_unmatched_payment():
    invoices, payments, settlements = make_inputs()
    stray = payments.copy()
    stray["payment_id"] = "PMT-99"
    stray["payer_name"] = "Completely Different Customer"
    stray["amount"] = 7777.77
    stray["reference_note"] = "unrelated transfer"
    payments = pd.concat([payments, stray], ignore_index=True)
    stray_settlement = settlements.copy()
    stray_settlement["payment_id"] = "PMT-99"
    stray_settlement["settlement_id"] = "STL-99"
    stray_settlement["gross_amount"] = 7777.77
    stray_settlement["settled_amount"] = 7777.77
    settlements = pd.concat([settlements, stray_settlement], ignore_index=True)

    result = reconcile(invoices, payments, settlements)
    row = result[(result.invoice_id.isna()) & (result.payment_id == "PMT-99")].iloc[0]
    assert row.status == "UNMATCHED"
