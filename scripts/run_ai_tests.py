from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

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
    ("What is our current reconciliation status?", "evaluation_results.json / get_reconciliation_summary"),
    ("What is our match rate?", "evaluation_results.json / get_reconciliation_summary"),
    ("How many invoices are unresolved?", "evaluation_results.json / get_reconciliation_summary"),
    ("How much money is currently in exceptions?", "evaluation_results.json / get_reconciliation_summary"),
    ("Show me the five largest exceptions.", "exceptions.csv / get_top_exceptions"),
    ("Why was invoice INV001 not reconciled?", "invoices.csv + reconciliation_results.csv / get_invoice_details"),
    ("Find payments with discrepancies above ₹1,000.", "exceptions.csv / get_exceptions"),
    ("What are the most common exception types?", "exceptions.csv / get_exception_summary"),
    ("What is our current cash position?", "raw financial data + reconciliation + audit / get_cash_position"),
    ("Tell me about invoice INV999999.", "invoices.csv / get_invoice_details"),
]


def _check(question: str, source: str, answer: str) -> tuple[bool, str]:
    q = question.casefold()
    if "current reconciliation status" in q:
        d = get_reconciliation_summary(ROOT)
        ok = all(str(d[k]) in answer for k in ("total_records", "matched_count", "review_count", "unmatched_count")) and f"{d['match_rate']:.1%}" in answer
        return ok, "Compared counts and match rate against stored evaluation output."
    if "match rate" in q:
        d = get_reconciliation_summary(ROOT)
        return f"{d['match_rate']:.1%}" in answer, "Compared match rate against stored evaluation output."
    if "unresolved" in q and "invoices" in q:
        d = get_reconciliation_summary(ROOT)
        n = d["review_count"] + d["unmatched_count"]
        return str(n) in answer, "Derived REVIEW + UNMATCHED from stored evaluation output."
    if "exceptions" in q and "how much" in q:
        d = get_reconciliation_summary(ROOT)
        return f"₹{d['total_exception_value']:,.2f}" in answer, "Compared exception value against stored evaluation output."
    if "five largest" in q:
        d = get_top_exceptions(5, ROOT)
        invoice_ids = {str(x["invoice_id"]) for x in d["exceptions"]}
        return bool(d["exceptions"]) and all(i in answer for i in invoice_ids), "Checked that returned invoice IDs are actual top-exception records."
    if "inv001" in q:
        d = get_invoice_details("INV001", ROOT)
        expected = "No invoice record was found" if not d["found"] else "### Invoice INV001"
        return expected in answer, "Verified existence check against invoices.csv."
    if "discrepancies" in q:
        rows = [r for r in get_exceptions(base_dir=ROOT)["exceptions"] if abs(float(r.get("amount_difference") or 0)) > 1000]
        if not rows:
            return "No stored exceptions" in answer, "Verified no qualifying stored exceptions exist."
        return all(str(r["invoice_id"]) in answer and str(r["payment_id"]) in answer for r in rows), "Verified IDs against exceptions.csv."
    if "most common" in q:
        d = get_exception_summary(ROOT)
        rows = d.get("summary", d.get("by_type", []))
        return all(r["exception_type"] in answer for r in rows), "Verified every reported type against exception summary."
    if "cash position" in q:
        d = get_cash_position(ROOT)
        values = [amount for section in ("confirmed_cash", "pending_review_payments", "unmatched_incoming_payments", "expected_incoming_cash") for amount in d.get(section, {}).get("by_currency", {}).values()]
        return all(f"{float(v):,.2f}" in answer for v in values), "Compared cash values against get_cash_position()."
    if "inv999999" in q:
        return "No invoice record was found" in answer, "Verified nonexistent invoice is explicitly reported unavailable."
    return False, "No evaluator rule."


def main() -> int:
    rows = []
    for question, source in QUESTIONS:
        try:
            actual = _demo_answer(question, base_dir=ROOT)
            passed, notes = _check(question, source, actual)
        except Exception as exc:
            actual = f"ERROR: {type(exc).__name__}: {exc}"
            passed = False
            notes = "Assistant/tool execution raised an exception."
        rows.append({
            "question": question,
            "expected_data_source": source,
            "actual_result": actual,
            "passed_or_failed": "PASSED" if passed else "FAILED",
            "notes": notes,
        })

    report_dir = ROOT / "outputs"
    report_dir.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(report_dir / "ai_controller_test_report.csv", index=False)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_questions": len(rows),
        "passed": sum(r["passed_or_failed"] == "PASSED" for r in rows),
        "failed": sum(r["passed_or_failed"] == "FAILED" for r in rows),
        "tests": rows,
    }
    (report_dir / "ai_controller_test_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n## AI FINANCE CONTROLLER TEST REPORT")
    print(f"Total tests: {summary['total_questions']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Report: {report_dir / 'ai_controller_test_report.csv'}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
