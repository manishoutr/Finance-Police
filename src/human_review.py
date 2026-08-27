"""Human-in-the-loop approval workflow for REVIEW reconciliation cases.

The original reconciliation_results.csv is immutable from this module. Human
choices are stored separately in outputs/audit_log.csv and exposed as an
operational status overlay for the dashboard.
"""
from __future__ import annotations

import csv
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

AUDIT_COLUMNS = [
    "audit_id",
    "invoice_id",
    "payment_id",
    "previous_status",
    "new_status",
    "decision",
    "timestamp",
    "reason",
]

VALID_DECISIONS = {"APPROVE MATCH", "REJECT MATCH"}
FINAL_STATUSES = {"APPROVED", "REJECTED"}


def audit_path(base_dir: Path | str | None = None) -> Path:
    base = Path(base_dir or Path(__file__).resolve().parents[1])
    return base / "outputs" / "audit_log.csv"


def _read_audit(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=AUDIT_COLUMNS)
    df = pd.read_csv(path, dtype=str)
    for col in AUDIT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[AUDIT_COLUMNS]


def get_audit_log(base_dir: Path | str | None = None) -> pd.DataFrame:
    """Return the append-only human decision history."""
    return _read_audit(audit_path(base_dir))


def get_operational_status_map(base_dir: Path | str | None = None) -> dict[tuple[str, str], str]:
    """Return latest human operational status for each invoice/payment pair.

    Only the latest audit decision is used for the operational overlay. The
    underlying reconciliation status remains untouched.
    """
    audit = get_audit_log(base_dir)
    if audit.empty:
        return {}
    audit = audit.copy()
    audit["timestamp_parsed"] = pd.to_datetime(audit["timestamp"], errors="coerce", utc=True)
    audit = audit.sort_values(["timestamp_parsed", "audit_id"], kind="stable")
    status_map: dict[tuple[str, str], str] = {}
    for _, row in audit.iterrows():
        key = (str(row["invoice_id"]), "" if pd.isna(row["payment_id"]) else str(row["payment_id"]))
        status_map[key] = str(row["new_status"])
    return status_map


def get_operational_status(invoice_id: str, payment_id: str | None = None, base_dir: Path | str | None = None) -> str | None:
    key = (str(invoice_id), "" if payment_id is None else str(payment_id))
    return get_operational_status_map(base_dir).get(key)


def _load_reconciliation(base_dir: Path | str | None = None) -> pd.DataFrame:
    base = Path(base_dir or Path(__file__).resolve().parents[1])
    path = base / "outputs" / "reconciliation_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Reconciliation results not found: {path}")
    return pd.read_csv(path)


def _find_review_case(invoice_id: str, payment_id: str | None, base_dir: Path | str | None) -> pd.Series:
    results = _load_reconciliation(base_dir)
    mask = results["invoice_id"].astype(str) == str(invoice_id)
    if payment_id is not None:
        mask &= results["payment_id"].fillna("").astype(str) == str(payment_id)
    rows = results[mask]
    if rows.empty:
        raise ValueError(f"No reconciliation record found for invoice {invoice_id} / payment {payment_id or 'N/A'}.")
    row = rows.iloc[0]
    if str(row.get("status", "")) != "REVIEW":
        raise ValueError("Human approval is only allowed for records whose original reconciliation status is REVIEW.")
    return row


def _append_audit(record: dict[str, Any], base_dir: Path | str | None = None) -> dict[str, Any]:
    path = audit_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({col: record.get(col) for col in AUDIT_COLUMNS})
    return record


def record_human_decision(
    invoice_id: str,
    payment_id: str | None,
    decision: str,
    reason: str,
    base_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Record an explicit human decision without modifying reconciliation results."""
    decision = str(decision).strip().upper()
    if decision not in VALID_DECISIONS:
        raise ValueError(f"Invalid decision. Choose one of: {sorted(VALID_DECISIONS)}")
    reason = str(reason or "").strip()
    if not reason:
        raise ValueError("A reason is required for every human approval or rejection.")

    row = _find_review_case(invoice_id, payment_id, base_dir)
    payment_value = row.get("payment_id")
    payment_value = None if pd.isna(payment_value) else str(payment_value)
    operational = get_operational_status(invoice_id, payment_value, base_dir)
    if operational in FINAL_STATUSES:
        raise ValueError(f"This REVIEW case already has a human decision: {operational}.")

    new_status = "APPROVED" if decision == "APPROVE MATCH" else "REJECTED"
    record = {
        "audit_id": f"AUD-{uuid.uuid4().hex[:12].upper()}",
        "invoice_id": str(invoice_id),
        "payment_id": payment_value,
        "previous_status": "REVIEW",
        "new_status": new_status,
        "decision": decision,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
    }
    _append_audit(record, base_dir)
    return record


def approve_match(invoice_id: str, payment_id: str | None, reason: str, base_dir: Path | str | None = None) -> dict[str, Any]:
    return record_human_decision(invoice_id, payment_id, "APPROVE MATCH", reason, base_dir)


def reject_match(invoice_id: str, payment_id: str | None, reason: str, base_dir: Path | str | None = None) -> dict[str, Any]:
    return record_human_decision(invoice_id, payment_id, "REJECT MATCH", reason, base_dir)
