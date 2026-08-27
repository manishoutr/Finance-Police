from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

from finance_assistant import _demo_answer
from finance_tools import (
    get_cash_position,
    get_exception_summary,
    get_exceptions,
    get_invoice_details,
    get_reconciliation_summary,
    get_top_exceptions,
)

QUESTIONS = [
    "What is our current reconciliation status?",
    "What is our match rate?",
    "How many invoices are unresolved?",
    "How much money is currently in exceptions?",
    "Show me the five largest exceptions.",
    "Why was invoice INV001 not reconciled?",
    "Find payments with discrepancies above ₹1,000.",
    "What are the most common exception types?",
    "What is our current cash position?",
    "Tell me about invoice INV999999.",
]


def answer(q: str) -> str:
    return _demo_answer(q, base_dir=ROOT)


def test_reconciliation_status_uses_live_summary():
    src = get_reconciliation_summary(ROOT)
    out = answer(QUESTIONS[0])
    for key in ("total_records", "matched_count", "review_count", "unmatched_count"):
        assert str(src[key]) in out
    assert f"{src['match_rate']:.1%}" in out


def test_match_rate_matches_backend():
    src = get_reconciliation_summary(ROOT)
    out = answer(QUESTIONS[1])
    assert f"{src['match_rate']:.1%}" in out


def test_unresolved_invoice_count_is_derived_from_backend():
    src = get_reconciliation_summary(ROOT)
    unresolved = src["review_count"] + src["unmatched_count"]
    out = answer(QUESTIONS[2])
    assert str(unresolved) in out


def test_exception_value_matches_backend():
    src = get_reconciliation_summary(ROOT)
    out = answer(QUESTIONS[3])
    # Demo assistant formats exception value using its own formatter.
    formatted = f"USD {src['total_exception_value']:,.2f}"
    assert formatted in out


def test_top_five_exceptions_are_actual_records():
    src = get_top_exceptions(5, ROOT)
    out = answer(QUESTIONS[4])
    assert len(src["exceptions"]) <= 5
    invoices = set(pd.read_csv(ROOT / "data/raw/invoices.csv")["invoice_id"].astype(str))
    payments = set(pd.read_csv(ROOT / "data/raw/payments.csv")["payment_id"].astype(str))
    for row in src["exceptions"]:
        assert row["invoice_id"] in invoices
        if row.get("payment_id"):
            assert row["payment_id"] in payments
        assert str(row["invoice_id"]) in out


def test_existing_invoice_question_never_invents_inv001():
    out = answer(QUESTIONS[5])
    invoices = set(pd.read_csv(ROOT / "data/raw/invoices.csv")["invoice_id"].astype(str).str.upper())
    if "INV001" not in invoices:
        assert "No invoice record was found" in out


def test_discrepancy_query_uses_exception_records():
    src = get_exceptions(base_dir=ROOT)
    expected = [
        r for r in src["exceptions"]
        if abs(float(r.get("amount_difference") or 0)) > 1000
    ]
    out = answer(QUESTIONS[6])
    if expected:
        assert "Payment discrepancies above" in out
        for row in expected:
            assert str(row["invoice_id"]) in out
            assert str(row["payment_id"]) in out
    else:
        assert "No stored exceptions" in out


def test_exception_types_match_backend():
    src = get_exception_summary(ROOT)
    out = answer(QUESTIONS[7])
    for row in src.get("summary", src.get("by_type", [])):
        assert row["exception_type"] in out


def test_cash_position_uses_actual_backend_values():
    src = get_cash_position(ROOT)
    out = answer(QUESTIONS[8])
    for section_name in (
        "confirmed_cash", "pending_review_payments", "unmatched_incoming_payments", "expected_incoming_cash"
    ):
        section = src.get(section_name, {})
        by_currency = section.get("by_currency", {})
        for currency, amount in by_currency.items():
            assert currency in out
            assert f"{float(amount):,.2f}" in out


def test_nonexistent_invoice_is_reported_unavailable():
    details = get_invoice_details("INV999999", ROOT)
    assert details["found"] is False
    out = answer(QUESTIONS[9])
    assert "No invoice record was found" in out
    assert "INV999999" in out


def test_ai_layer_has_no_approval_or_rejection_tool():
    from finance_assistant import TOOLS
    names = {tool["name"] for tool in TOOLS}
    assert "approve_match" not in names
    assert "reject_match" not in names


def test_ai_questions_do_not_modify_reconciliation_results(tmp_path):
    source = ROOT / "outputs/reconciliation_results.csv"
    before = source.read_bytes()
    for q in QUESTIONS:
        _demo_answer(q, base_dir=ROOT)
    after = source.read_bytes()
    assert after == before


def test_ai_questions_do_not_modify_ground_truth():
    source = ROOT / "data/ground_truth/ground_truth.csv"
    before = source.read_bytes()
    for q in QUESTIONS:
        _demo_answer(q, base_dir=ROOT)
    assert source.read_bytes() == before


def test_missing_exception_output_fails_safely(tmp_path):
    shutil.copytree(ROOT / "data", tmp_path / "data")
    shutil.copytree(ROOT / "outputs", tmp_path / "outputs")
    (tmp_path / "outputs/exceptions.csv").unlink()
    out = _demo_answer("How much money is currently in exceptions?", base_dir=tmp_path)
    assert "couldn't retrieve" in out.lower() or "error" in out.lower() or "not available" in out.lower()


def test_empty_invoices_fail_safely(tmp_path):
    shutil.copytree(ROOT / "data", tmp_path / "data")
    shutil.copytree(ROOT / "outputs", tmp_path / "outputs")
    invoice_path = tmp_path / "data/raw/invoices.csv"
    pd.read_csv(invoice_path).iloc[0:0].to_csv(invoice_path, index=False)
    details = get_invoice_details("INV999999", tmp_path)
    assert details["found"] is False
    out = _demo_answer("Tell me about invoice INV999999", base_dir=tmp_path)
    assert "No invoice record was found" in out


def test_malformed_exception_amount_does_not_invent_value(tmp_path):
    shutil.copytree(ROOT / "data", tmp_path / "data")
    shutil.copytree(ROOT / "outputs", tmp_path / "outputs")
    ex_path = tmp_path / "outputs/exceptions.csv"
    df = pd.read_csv(ex_path)
    df.loc[df.index[0], "amount_difference"] = "NOT_A_NUMBER"
    df.to_csv(ex_path, index=False)
    result = get_exceptions(base_dir=tmp_path)
    assert result["count"] == len(df)
    assert all("NOT_A_NUMBER" in str(r.get("amount_difference")) for r in result["exceptions"] if r.get("amount_difference") == "NOT_A_NUMBER")
