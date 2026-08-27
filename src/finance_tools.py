"""Read-only finance data tools for the AI Finance Controller.

These functions are deliberately independent of the LLM. They read the existing
CSV/JSON outputs and return structured, source-of-truth data. They never mutate
financial records or make reconciliation decisions.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .cash_position import calculate_cash_position
except ImportError:
    from cash_position import calculate_cash_position


class FinanceDataStore:
    """Read-only access to the existing Finance Controller data products."""

    def __init__(self, base_dir: Path | str | None = None) -> None:
        self.base_dir = Path(base_dir or Path(__file__).resolve().parents[1])
        self.raw_dir = self.base_dir / "data" / "raw"
        self.outputs_dir = self.base_dir / "outputs"

    def _read_csv(self, path: Path, required: bool = True) -> pd.DataFrame:
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Required finance data file not found: {path}")
            return pd.DataFrame()
        return pd.read_csv(path)

    def invoices(self) -> pd.DataFrame:
        return self._read_csv(self.raw_dir / "invoices.csv")

    def payments(self) -> pd.DataFrame:
        return self._read_csv(self.raw_dir / "payments.csv")

    def settlements(self) -> pd.DataFrame:
        return self._read_csv(self.raw_dir / "settlements.csv")

    def reconciliation(self) -> pd.DataFrame:
        return self._read_csv(self.outputs_dir / "reconciliation_results.csv")

    def exceptions(self) -> pd.DataFrame:
        return self._read_csv(self.outputs_dir / "exceptions.csv")

    def evaluation(self) -> dict[str, Any]:
        path = self.outputs_dir / "evaluation_results.json"
        if not path.exists():
            raise FileNotFoundError(f"Required evaluation file not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, float):
        return round(value, 6)
    return value


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [{k: _clean(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


def _parse_bool(value: Any) -> bool | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().casefold()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def get_reconciliation_summary(base_dir: Path | str | None = None) -> dict[str, Any]:
    """Return the stored reconciliation benchmark summary."""
    store = FinanceDataStore(base_dir)
    metrics = store.evaluation()
    return {
        "total_records": int(metrics.get("records_processed", 0)),
        "matched_count": int(metrics.get("automatic_matches", 0)),
        "review_count": int(metrics.get("review_records", 0)),
        "unmatched_count": int(metrics.get("unmatched_records", 0)),
        "match_rate": float(metrics.get("automatic_match_rate", 0)),
        "precision": float(metrics.get("precision", 0)),
        "recall": float(metrics.get("recall", 0)),
        "f1_score": float(metrics.get("f1_score", 0)),
        "total_exception_value": float(metrics.get("total_value_exceptions", 0)),
    }


def get_exceptions(
    exception_type: str | None = None,
    severity: str | None = None,
    base_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Return exceptions filtered by deterministic exception type/severity."""
    df = FinanceDataStore(base_dir).exceptions().copy()
    if exception_type:
        df = df[df["exception_type"].astype(str).str.casefold() == exception_type.strip().casefold()]
    if severity:
        df = df[df["severity"].astype(str).str.casefold() == severity.strip().casefold()]
    return {"count": len(df), "exceptions": _records(df)}


def get_invoice_details(invoice_id: str, base_dir: Path | str | None = None) -> dict[str, Any]:
    """Return an invoice, its reconciliation record, settlement, and exception data."""
    if not invoice_id or not str(invoice_id).strip():
        return {"found": False, "error": "invoice_id is required"}

    store = FinanceDataStore(base_dir)
    query = str(invoice_id).strip().casefold()
    invoices = store.invoices()
    matches = invoices[invoices["invoice_id"].astype(str).str.casefold() == query]
    if matches.empty:
        # Permit a useful exact suffix/substring lookup, but return one clear record.
        matches = invoices[invoices["invoice_id"].astype(str).str.casefold().str.contains(query, regex=False)]
    if matches.empty:
        return {"found": False, "invoice_id": invoice_id}
    invoice = matches.iloc[0]
    actual_id = str(invoice["invoice_id"])

    recon = store.reconciliation()
    recon_rows = recon[recon["invoice_id"].astype(str) == actual_id].copy()
    recon_row = recon_rows.iloc[0].to_dict() if not recon_rows.empty else None

    payment_id = None
    if recon_row:
        payment_id = _clean(recon_row.get("payment_id"))
    payment = None
    if payment_id:
        payments = store.payments()
        p = payments[payments["payment_id"].astype(str) == str(payment_id)]
        if not p.empty:
            payment = p.iloc[0].to_dict()

    settlement = None
    if payment_id:
        settlements = store.settlements()
        s = settlements[settlements["payment_id"].astype(str) == str(payment_id)]
        if not s.empty:
            settlement = s.iloc[0].to_dict()

    exceptions = store.exceptions()
    ex = exceptions[exceptions["invoice_id"].astype(str) == actual_id].copy()
    if payment_id:
        payment_ex = exceptions[exceptions["payment_id"].astype(str) == str(payment_id)]
        ex = pd.concat([ex, payment_ex], ignore_index=True).drop_duplicates(subset=["exception_id"])

    return {
        "found": True,
        "invoice": {k: _clean(v) for k, v in invoice.to_dict().items()},
        "matched_payment": None if payment is None else {k: _clean(v) for k, v in payment.items()},
        "settlement": None if settlement is None else {k: _clean(v) for k, v in settlement.items()},
        "reconciliation": None if recon_row is None else {k: _clean(v) for k, v in recon_row.items()},
        "matching_signals": None if recon_row is None else {
            "customer_similarity": _clean(recon_row.get("customer_similarity")),
            "date_difference": _clean(recon_row.get("date_difference")),
            "reference_match": _parse_bool(recon_row.get("reference_match")),
            "amount_difference": _clean(recon_row.get("amount_difference")),
            "confidence_score": _clean(recon_row.get("confidence_score")),
        },
        "exceptions": _records(ex),
    }


def get_payment_details(payment_id: str, base_dir: Path | str | None = None) -> dict[str, Any]:
    """Return payment details and all reconciliation associations for a payment."""
    if not payment_id or not str(payment_id).strip():
        return {"found": False, "error": "payment_id is required"}

    store = FinanceDataStore(base_dir)
    query = str(payment_id).strip().casefold()
    payments = store.payments()
    p = payments[payments["payment_id"].astype(str).str.casefold() == query]
    if p.empty:
        p = payments[payments["payment_id"].astype(str).str.casefold().str.contains(query, regex=False)]
    if p.empty:
        return {"found": False, "payment_id": payment_id}

    payment = p.iloc[0]
    actual_id = str(payment["payment_id"])
    recon = store.reconciliation()
    associated = recon[recon["payment_id"].astype(str) == actual_id].copy()

    invoices = store.invoices()
    invoice_ids = associated["invoice_id"].dropna().astype(str).tolist() if not associated.empty else []
    invoice_rows = invoices[invoices["invoice_id"].astype(str).isin(invoice_ids)]

    settlements = store.settlements()
    settlement_rows = settlements[settlements["payment_id"].astype(str) == actual_id]

    exceptions = store.exceptions()
    ex = exceptions[exceptions["payment_id"].astype(str) == actual_id]

    return {
        "found": True,
        "payment": {k: _clean(v) for k, v in payment.to_dict().items()},
        "associated_invoices": _records(invoice_rows),
        "reconciliation_records": _records(associated),
        "settlements": _records(settlement_rows),
        "exceptions": _records(ex),
    }


def get_top_exceptions(limit: int = 10, base_dir: Path | str | None = None) -> dict[str, Any]:
    """Return highest-value unresolved exceptions by invoice exposure."""
    limit = max(1, min(int(limit), 100))
    df = FinanceDataStore(base_dir).exceptions().copy()
    df["exception_value"] = pd.to_numeric(df["invoice_amount"], errors="coerce").abs().fillna(
        pd.to_numeric(df["amount_difference"], errors="coerce").abs().fillna(0)
    )
    df = df.sort_values(["exception_value", "exception_id"], ascending=[False, True], kind="stable").head(limit)
    return {"count": len(df), "exceptions": _records(df)}


def get_exception_summary(base_dir: Path | str | None = None) -> dict[str, Any]:
    """Return counts and invoice-value exposure grouped by exception type."""
    df = FinanceDataStore(base_dir).exceptions().copy()
    if df.empty:
        return {"total_exceptions": 0, "by_type": []}
    df["exception_value"] = pd.to_numeric(df["invoice_amount"], errors="coerce").abs().fillna(
        pd.to_numeric(df["amount_difference"], errors="coerce").abs().fillna(0)
    )
    grouped = (
        df.groupby("exception_type", dropna=False)
        .agg(count=("exception_id", "count"), exception_value=("exception_value", "sum"))
        .reset_index()
        .sort_values(["exception_value", "count"], ascending=[False, False])
    )
    return {
        "total_exceptions": len(df),
        "total_exception_value": float(df["exception_value"].sum()),
        "by_type": _records(grouped),
    }


def search_transactions(query: str, limit: int = 25, base_dir: Path | str | None = None) -> dict[str, Any]:
    """Search invoices, payments, and reconciliation records by common identifiers/text."""
    if not query or not str(query).strip():
        return {"count": 0, "results": [], "error": "query is required"}
    query = str(query).strip().casefold()
    limit = max(1, min(int(limit), 100))
    store = FinanceDataStore(base_dir)

    invoices = store.invoices().copy()
    payments = store.payments().copy()
    recon = store.reconciliation().copy()

    invoice_mask = invoices.astype(str).apply(lambda col: col.str.casefold().str.contains(query, regex=False)).any(axis=1)
    payment_mask = payments.astype(str).apply(lambda col: col.str.casefold().str.contains(query, regex=False)).any(axis=1)
    recon_mask = recon.astype(str).apply(lambda col: col.str.casefold().str.contains(query, regex=False)).any(axis=1)

    results: list[dict[str, Any]] = []
    for _, row in invoices[invoice_mask].head(limit).iterrows():
        results.append({"record_type": "invoice", **{k: _clean(v) for k, v in row.to_dict().items()}})
    for _, row in payments[payment_mask].head(limit).iterrows():
        results.append({"record_type": "payment", **{k: _clean(v) for k, v in row.to_dict().items()}})
    for _, row in recon[recon_mask].head(limit).iterrows():
        results.append({"record_type": "reconciliation", **{k: _clean(v) for k, v in row.to_dict().items()}})

    # Keep the response compact and deterministic.
    return {"count": min(len(results), limit), "results": results[:limit]}


def get_cash_position(base_dir: Path | str | None = None) -> dict[str, Any]:
    """Return the synthetic cash-position model from the existing finance data."""
    return calculate_cash_position(base_dir)


TOOL_FUNCTIONS = {
    "get_reconciliation_summary": get_reconciliation_summary,
    "get_exceptions": get_exceptions,
    "get_invoice_details": get_invoice_details,
    "get_payment_details": get_payment_details,
    "get_top_exceptions": get_top_exceptions,
    "get_exception_summary": get_exception_summary,
    "search_transactions": search_transactions,
    "get_cash_position": get_cash_position,
}
