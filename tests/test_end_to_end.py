"""End-to-end reliability tests.

The fixture creates fresh synthetic CSVs in a temporary project directory and
runs the same backend modules used by the application.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))

from classify_exceptions import classify_exceptions
from evaluate_reconciliation import evaluate, load_inputs, write_outputs
from finance_assistant import ask_finance_controller
from finance_tools import (
    get_cash_position,
    get_invoice_details,
    get_reconciliation_summary,
)
from human_review import approve_match, get_audit_log, get_operational_status, reject_match
from input_validation import validate_source_frames
from reconciler import reconcile


def _frames() -> dict[str, pd.DataFrame]:
    invoices = pd.DataFrame([
        ["INV-001", "Acme Pvt. Ltd.", "2026-08-01", "2026-08-05", 1000, "USD", "Invoice 001"],
        ["INV-002", "Jane Smith", "2026-08-01", "2026-08-05", 1200, "USD", "Invoice 002"],
        ["INV-003", "Beta Ltd", "2026-08-01", "2026-08-05", 1000, "USD", "Invoice 003"],
        ["INV-004", "Gamma Ltd", "2026-08-01", "2026-08-05", 1000, "USD", "Invoice 004"],
        ["INV-005", "ZZZZZZZ", "2025-01-01", "2025-01-05", 15000, "USD", "Invoice 005"],
        ["INV-006", "Duplicate Buyer", "2026-08-01", "2026-08-05", 900, "USD", "Invoice 006"],
        ["INV-007", "YYYYYYY", "2025-01-01", "2025-01-05", 25000, "USD", "Invoice 007"],
        ["INV-008", "Shared Customer", "2026-08-01", "2026-08-05", 700, "USD", "Invoice 008"],
        ["INV-009", "Shared Customer", "2026-08-10", "2026-08-15", 1800, "USD", "Invoice 009"],
        ["INV-010", "Split Customer", "2026-08-01", "2026-08-05", 1000, "USD", "Invoice 010"],
        ["INV-011", "Reference Missing Co", "2026-08-01", "2026-08-05", 1100, "USD", "Invoice 011"],
        ["INV-012", "Reject Customer", "2026-08-01", "2026-08-05", 1000, "USD", "Invoice 012"],
    ], columns=["invoice_id", "customer_name", "invoice_date", "due_date", "amount", "currency", "description"])

    payments = pd.DataFrame([
        ["PMT-001", "Acme Private Limited", "2026-08-05", 1000, "USD", "ACH", "INV-001 payment"],
        ["PMT-002", "Jane Smyth", "2026-08-05", 1200, "USD", "ACH", "INV-002 payment"],
        ["PMT-003", "Beta Ltd", "2026-08-05", 1300, "USD", "ACH", "INV-003 payment"],
        ["PMT-004", "Gamma Ltd", "2026-08-05", 600, "USD", "ACH", "INV-004 part 1"],
        ["PMT-005", "Gamma Ltd", "2026-08-06", 200, "USD", "ACH", "INV-004 part 2"],
        ["PMT-006", "Duplicate Buyer", "2026-08-05", 900, "USD", "ACH", "INV-006 payment"],
        ["PMT-007", "Duplicate Buyer", "2026-08-05", 900, "USD", "ACH", "duplicate payment"],
        ["PMT-008", "Unknown Sender", "2026-08-30", 500, "USD", "Wire", "UNMATCHED PAYMENT"],
        ["PMT-009", "Shared Customer", "2026-08-05", 700, "USD", "ACH", "INV-008 payment"],
        ["PMT-010", "Shared Customer", "2026-08-15", 1800, "USD", "ACH", "INV-009 payment"],
        ["PMT-011", "Split Customer", "2026-08-05", 600, "USD", "ACH", "INV-010 part 1"],
        ["PMT-012", "Split Customer", "2026-08-06", 200, "USD", "ACH", "INV-010 part 2"],
        ["PMT-013", "Reference Missing Co", "2026-08-15", 1100, "USD", "ACH", ""],
        ["PMT-014", "Reject Customer", "2026-08-05", 700, "USD", "ACH", "INV-012 payment"],
    ], columns=["payment_id", "payer_name", "payment_date", "amount", "currency", "method", "reference_note"])

    settlements = pd.DataFrame([
        [f"STL-{i:03d}", f"PMT-{i:03d}", "2026-08-10", float(amt), 0.0, float(amt), "settled"]
        for i, amt in [
            (1, 1000), (2, 1200), (3, 1300), (4, 600), (5, 200), (6, 900), (7, 900),
            (8, 500), (9, 700), (10, 1800), (11, 600), (12, 200), (13, 1100), (14, 700)
        ]
    ], columns=["settlement_id", "payment_id", "settlement_date", "gross_amount", "fee", "settled_amount", "status"])
    return {"invoices.csv": invoices, "payments.csv": payments, "settlements.csv": settlements}


def _ground_truth() -> pd.DataFrame:
    rows = [
        ["INV-001", "PMT-001", "exact_match", 1000, 1000, "exact"],
        ["INV-002", "PMT-002", "fuzzy_name", 1200, 1200, "fuzzy"],
        ["INV-003", "PMT-003", "amount_mismatch", 1000, 1300, "overpayment"],
        ["INV-004", "PMT-004;PMT-005", "partial_payment", 1000, 800, "two partial payments"],
        ["INV-005", "", "missing_payment", 15000, 0, "no payment"],
        ["INV-006", "PMT-006;PMT-007", "duplicate_payment", 900, 1800, "duplicate full payment"],
        ["INV-007", "", "missing_payment", 25000, 0, "no payment"],
        ["INV-008", "PMT-009", "exact_match", 700, 700, "same customer, separate invoice"],
        ["INV-009", "PMT-010", "exact_match", 1800, 1800, "same customer, separate invoice"],
        ["INV-010", "PMT-011;PMT-012", "partial_payment", 1000, 800, "multiple payments"],
        ["INV-011", "PMT-013", "missing_reference", 1100, 1100, "reference absent"],
        ["INV-012", "PMT-014", "amount_mismatch", 1000, 700, "review then reject"],
        [None, "PMT-008", "unmatched_payment", 0, 500, "unmatched incoming payment"],
    ]
    return pd.DataFrame(rows, columns=["invoice_id", "payment_ids", "match_type", "invoice_amount", "total_paid", "notes"])


def _project(tmp_path: Path) -> Path:
    base = tmp_path / "RZP"
    (base / "data" / "raw").mkdir(parents=True)
    (base / "data" / "ground_truth").mkdir(parents=True)
    (base / "outputs").mkdir(parents=True)
    frames = _frames()
    for name, df in frames.items():
        df.to_csv(base / "data" / "raw" / name, index=False)
    _ground_truth().to_csv(base / "data" / "ground_truth" / "ground_truth.csv", index=False)
    return base


def _run_workflow(base: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = {name: pd.read_csv(base / "data" / "raw" / name) for name in ["invoices.csv", "payments.csv", "settlements.csv"]}
    validate_source_frames(frames)
    results = reconcile(frames["invoices.csv"], frames["payments.csv"], frames["settlements.csv"])
    results.to_csv(base / "outputs" / "reconciliation_results.csv", index=False)
    gt = pd.read_csv(base / "data" / "ground_truth" / "ground_truth.csv")
    metrics = evaluate(results, gt)
    write_outputs(metrics, base / "outputs")
    exceptions = classify_exceptions(results, payments=frames["payments.csv"])
    exceptions.to_csv(base / "outputs" / "exceptions.csv", index=False)
    return results, exceptions


def test_end_to_end_core_workflow(tmp_path):
    base = _project(tmp_path)
    gt_path = base / "data" / "ground_truth" / "ground_truth.csv"
    gt_hash = hashlib.sha256(gt_path.read_bytes()).hexdigest()

    results, exceptions = _run_workflow(base)
    assert len(results) == 16
    assert (base / "outputs" / "reconciliation_results.csv").exists()
    assert (base / "outputs" / "evaluation_results.json").exists()
    assert (base / "outputs" / "evaluation_report.csv").exists()
    assert (base / "outputs" / "exceptions.csv").exists()
    assert hashlib.sha256(gt_path.read_bytes()).hexdigest() == gt_hash


def test_exact_and_fuzzy_matches(tmp_path):
    base = _project(tmp_path)
    results, _ = _run_workflow(base)
    r1 = results[results.invoice_id == "INV-001"].iloc[0]
    r2 = results[results.invoice_id == "INV-002"].iloc[0]
    assert r1.status == "MATCHED" and r1.payment_id == "PMT-001"
    assert r2.status == "MATCHED" and r2.payment_id == "PMT-002"
    assert r2.customer_similarity >= 85


def test_amount_mismatch_and_partial_payment(tmp_path):
    base = _project(tmp_path)
    results, exceptions = _run_workflow(base)
    assert results.loc[results.invoice_id == "INV-003", "status"].iloc[0] == "REVIEW"
    assert results.loc[results.invoice_id == "INV-004", "status"].iloc[0] == "REVIEW"
    types = dict(zip(exceptions.invoice_id.astype(str), exceptions.exception_type))
    assert types["INV-003"] == "AMOUNT_MISMATCH"
    assert types["INV-004"] == "PARTIAL_PAYMENT"


def test_missing_duplicate_unmatched_and_missing_reference(tmp_path):
    base = _project(tmp_path)
    results, exceptions = _run_workflow(base)
    types = set(exceptions.exception_type)
    assert "MISSING_PAYMENT" in types
    assert "DUPLICATE_PAYMENT" in types
    assert "UNMATCHED_TRANSACTION" in types
    assert "MISSING_REFERENCE" in types
    assert "PMT-008" in set(results.loc[results.invoice_id.isna(), "payment_id"].dropna())


def test_multiple_invoices_same_customer_and_multiple_payments(tmp_path):
    base = _project(tmp_path)
    results, _ = _run_workflow(base)
    r8 = results[results.invoice_id == "INV-008"].iloc[0]
    r9 = results[results.invoice_id == "INV-009"].iloc[0]
    r10 = results[results.invoice_id == "INV-010"].iloc[0]
    assert {r8.payment_id, r9.payment_id} == {"PMT-009", "PMT-010"}
    assert r10.payment_id in {"PMT-011", "PMT-012"}
    assert (results.payment_id == "PMT-011").sum() + (results.payment_id == "PMT-012").sum() == 2


def test_human_approval_preserves_original_and_logs(tmp_path):
    base = _project(tmp_path)
    results, _ = _run_workflow(base)
    review = results[results.status == "REVIEW"].iloc[0]
    recon_path = base / "outputs" / "reconciliation_results.csv"
    before = hashlib.sha256(recon_path.read_bytes()).hexdigest()
    record = approve_match(str(review.invoice_id), str(review.payment_id), "Human verified supporting evidence.", base)
    after = hashlib.sha256(recon_path.read_bytes()).hexdigest()
    assert before == after
    assert record["new_status"] == "APPROVED"
    assert get_operational_status(str(review.invoice_id), str(review.payment_id), base) == "APPROVED"
    assert len(get_audit_log(base)) == 1


def test_human_rejection_is_logged(tmp_path):
    base = _project(tmp_path)
    results, _ = _run_workflow(base)
    review = results[results.status == "REVIEW"].iloc[1]
    record = reject_match(str(review.invoice_id), str(review.payment_id), "Bank evidence does not support the match.", base)
    assert record["new_status"] == "REJECTED"
    audit = get_audit_log(base)
    assert len(audit) == 1
    assert audit.iloc[0]["decision"] == "REJECT MATCH"


def test_ai_existing_and_nonexistent_invoice_use_backend_data(tmp_path, monkeypatch):
    base = _project(tmp_path)
    _run_workflow(base)
    monkeypatch.setenv("FINANCE_ASSISTANT_MODE", "demo")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    answer, _ = ask_finance_controller("Tell me everything about invoice INV-001.", base_dir=base)
    assert "INV-001" in answer
    assert "USD 1,000.00" in answer
    missing, _ = ask_finance_controller("Tell me everything about invoice INV-999.", base_dir=base)
    assert "No invoice record was found" in missing


def test_cash_position_uses_actual_data(tmp_path):
    base = _project(tmp_path)
    results, _ = _run_workflow(base)
    cash = get_cash_position(base)
    assert cash["demo_model"] is True
    assert cash["confirmed_cash"]["amount"] is not None
    assert cash["expected_incoming_cash"]["amount"] is not None
    # Cash must derive from the generated payment/reconciliation data, not a hardcoded value.
    confirmed_ids = set(results.loc[results.status == "MATCHED", "payment_id"].dropna().astype(str))
    assert cash["confirmed_cash"]["payment_count"] == len(confirmed_ids)


def test_dashboard_data_products_are_loadable(tmp_path):
    base = _project(tmp_path)
    _run_workflow(base)
    recon = pd.read_csv(base / "outputs" / "reconciliation_results.csv")
    exc = pd.read_csv(base / "outputs" / "exceptions.csv")
    metrics = pd.read_json(base / "outputs" / "evaluation_results.json", typ="series")
    assert {"invoice_id", "payment_id", "confidence_score", "status"}.issubset(recon.columns)
    assert {"exception_type", "severity", "recommended_action"}.issubset(exc.columns)
    assert float(metrics["records_processed"]) == len(_frames()["invoices.csv"])


def test_invalid_empty_and_duplicate_inputs_are_rejected(tmp_path):
    frames = _frames()
    validate_source_frames(frames)  # baseline is valid

    empty = {k: v.copy() for k, v in frames.items()}
    empty["invoices.csv"] = empty["invoices.csv"].iloc[0:0]
    with pytest.raises(ValueError, match="invoices.csv is empty"):
        validate_source_frames(empty)

    invalid = {k: v.copy() for k, v in frames.items()}
    invalid["payments.csv"]["amount"] = invalid["payments.csv"]["amount"].astype(object)
    invalid["payments.csv"].loc[0, "amount"] = "NOT_A_NUMBER"
    with pytest.raises(ValueError, match="invalid numeric values"):
        validate_source_frames(invalid)

    dup_invoice = {k: v.copy() for k, v in frames.items()}
    dup_invoice["invoices.csv"] = pd.concat([dup_invoice["invoices.csv"], dup_invoice["invoices.csv"].iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate invoice_id"):
        validate_source_frames(dup_invoice)

    dup_payment = {k: v.copy() for k, v in frames.items()}
    dup_payment["payments.csv"] = pd.concat([dup_payment["payments.csv"], dup_payment["payments.csv"].iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate payment_id"):
        validate_source_frames(dup_payment)
