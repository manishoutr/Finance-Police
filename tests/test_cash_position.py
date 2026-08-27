from pathlib import Path

from src.finance_tools import get_cash_position

BASE = Path(__file__).resolve().parents[1]


def test_cash_position_has_required_labels_and_real_values():
    result = get_cash_position(BASE)
    assert result["confirmed_cash"]["label"] == "CONFIRMED"
    assert result["pending_review_payments"]["label"] == "PENDING"
    assert result["unmatched_incoming_payments"]["label"] == "UNRESOLVED"
    assert result["unresolved_receivables"]["label"] == "UNRESOLVED"
    assert result["expected_incoming_cash"]["label"] == "EXPECTED"
    assert result["confirmed_cash"]["amount"] == 315501.19
    assert result["pending_review_payments"]["amount"] == 89283.63
    assert result["unmatched_incoming_payments"]["amount"] == 80223.22
    assert result["confirmed_receivables"]["amount"] == 315501.19
    assert result["unresolved_receivables"]["amount"] == 149404.37
    assert result["expected_incoming_cash"]["amount"] == 464905.56


def test_cash_position_returns_largest_pending_receivables():
    result = get_cash_position(BASE)
    rows = result["largest_pending_receivables"]
    assert rows
    amounts = [r["amount"] for r in rows]
    assert amounts == sorted(amounts, reverse=True)
    assert len(rows) <= 10


def test_cash_position_is_not_settled_bank_balance():
    result = get_cash_position(BASE)
    assert result["demo_model"] is True
    assert "not a bank balance" in result["methodology"].lower()
