from pathlib import Path

import pandas as pd

from src.finance_tools import (
    FinanceDataStore,
    get_cash_position,
    get_exception_summary,
    get_exceptions,
    get_invoice_details,
    get_payment_details,
    get_reconciliation_summary,
    get_top_exceptions,
    search_transactions,
)


BASE = Path(__file__).resolve().parents[1]


def test_reconciliation_summary_uses_stored_metrics():
    result = get_reconciliation_summary(BASE)
    assert result["total_records"] == 100
    assert result["matched_count"] == 71
    assert result["review_count"] == 17
    assert result["unmatched_count"] == 12
    assert result["precision"] == 1.0


def test_invoice_details_returns_real_source_records():
    result = get_invoice_details("INV-2024-00000", BASE)
    assert result["found"] is True
    assert result["invoice"]["invoice_id"] == "INV-2024-00000"
    assert result["reconciliation"]["status"] == "MATCHED"
    assert result["matching_signals"]["confidence_score"] == 91.69


def test_payment_details_returns_associated_records():
    result = get_payment_details("PMT-100000", BASE)
    assert result["found"] is True
    assert result["payment"]["payment_id"] == "PMT-100000"
    assert result["associated_invoices"][0]["invoice_id"] == "INV-2024-00000"


def test_exception_filters_are_deterministic():
    result = get_exceptions("PARTIAL_PAYMENT", "HIGH", BASE)
    assert result["count"] == 6
    assert all(x["exception_type"] == "PARTIAL_PAYMENT" for x in result["exceptions"])


def test_top_exceptions_is_value_sorted():
    result = get_top_exceptions(5, BASE)
    values = [abs(float(x["invoice_amount"])) for x in result["exceptions"]]
    assert len(values) == 5
    assert values == sorted(values, reverse=True)


def test_exception_summary_has_expected_categories():
    result = get_exception_summary(BASE)
    categories = {x["exception_type"] for x in result["by_type"]}
    assert "DATE_MISMATCH" in categories
    assert "PARTIAL_PAYMENT" in categories
    assert result["total_exceptions"] == 66


def test_search_transactions_finds_invoice_and_payment_ids():
    invoice_result = search_transactions("INV-2024-00000", 10, BASE)
    assert invoice_result["count"] >= 1
    assert any(r.get("invoice_id") == "INV-2024-00000" for r in invoice_result["results"])

    payment_result = search_transactions("PMT-100000", 10, BASE)
    assert payment_result["count"] >= 1
    assert any(r.get("payment_id") == "PMT-100000" for r in payment_result["results"])


def test_cash_position_uses_settled_amounts():
    result = get_cash_position(BASE)
    settlements = pd.read_csv(BASE / "data" / "raw" / "settlements.csv")
    expected = settlements.loc[settlements["status"].str.casefold() == "settled", "settled_amount"].sum()
    assert result["settled_cash_total"] == round(float(expected), 2)
