import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))

from finance_assistant import ask_finance_controller


def test_demo_mode_does_not_require_api_key(monkeypatch):
    monkeypatch.setenv("FINANCE_ASSISTANT_MODE", "demo")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    answer, response_id = ask_finance_controller("What is our current reconciliation status?", base_dir=BASE)
    assert response_id is None
    assert "Records processed" in answer
    assert "Match rate" in answer


def test_demo_invoice_lookup(monkeypatch):
    monkeypatch.setenv("FINANCE_ASSISTANT_MODE", "demo")
    answer, _ = ask_finance_controller("Tell me everything about invoice INV-2024-00047.", base_dir=BASE)
    assert "INV-2024-00047" in answer
    assert "Status" in answer


def test_demo_top_exceptions(monkeypatch):
    monkeypatch.setenv("FINANCE_ASSISTANT_MODE", "demo")
    answer, _ = ask_finance_controller("Show me the five largest exceptions.", base_dir=BASE)
    assert "largest exceptions" in answer.casefold()
