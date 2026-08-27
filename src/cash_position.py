"""Synthetic cash-position calculations for the Finance Controller.

This module is a demo finance model, not a production accounting or bank-
balance system. It uses only financial records already present in the project.
The original reconciliation results remain immutable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .human_review import get_operational_status_map
except ImportError:
    from human_review import get_operational_status_map


CATEGORIES = ("CONFIRMED", "PENDING", "UNRESOLVED", "EXPECTED")


def _base(base_dir: Path | str | None = None) -> Path:
    return Path(base_dir or Path(__file__).resolve().parents[1])


def _money_by_currency(df: pd.DataFrame, amount_col: str, currency_col: str = "currency") -> dict[str, float]:
    if df.empty or amount_col not in df.columns:
        return {}
    amounts = pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0)
    currencies = df[currency_col].fillna("UNKNOWN").astype(str) if currency_col in df.columns else pd.Series("UNKNOWN", index=df.index)
    grouped = pd.DataFrame({"currency": currencies, "amount": amounts}).groupby("currency")["amount"].sum()
    return {str(k): round(float(v), 2) for k, v in grouped.items()}


def _scalar_if_one(values: dict[str, float]) -> float | None:
    return next(iter(values.values())) if len(values) == 1 else None


def calculate_cash_position(base_dir: Path | str | None = None) -> dict[str, Any]:
    """Calculate a transparent synthetic receivables/cash position.

    Definitions:
    - CONFIRMED: payment amounts on original MATCHED records plus REVIEW cases
      explicitly approved by a human. These are incoming payments confirmed by
      the reconciliation workflow, not a bank balance.
    - PENDING: payment amounts on REVIEW records that have no final human
      decision yet.
    - UNRESOLVED: invoice value on REVIEW/UNMATCHED invoice records after
      excluding human-approved REVIEW cases, plus separately tracked incoming
      payments that have no invoice candidate.
    - EXPECTED: confirmed receivables plus unresolved receivables. In this
      synthetic model this equals the value of receivables currently represented
      by invoices, and is not a forecast.
    """
    base = _base(base_dir)
    raw = base / "data" / "raw"
    outputs = base / "outputs"

    invoices = pd.read_csv(raw / "invoices.csv")
    payments = pd.read_csv(raw / "payments.csv")
    reconciliation = pd.read_csv(outputs / "reconciliation_results.csv")

    for frame, cols, name in [
        (invoices, {"invoice_id", "amount", "currency"}, "invoices.csv"),
        (payments, {"payment_id", "amount", "currency"}, "payments.csv"),
        (reconciliation, {"invoice_id", "payment_id", "invoice_amount", "payment_amount", "status"}, "reconciliation_results.csv"),
    ]:
        missing = cols - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {sorted(missing)}")

    operational = get_operational_status_map(base)

    def op_status(row: pd.Series) -> str:
        invoice_id = row.get("invoice_id")
        payment_id = row.get("payment_id")
        if pd.isna(invoice_id):
            return "UNMATCHED"
        key = (str(invoice_id), "" if pd.isna(payment_id) else str(payment_id))
        return operational.get(key, str(row.get("status", "UNMATCHED")))

    reconciliation = reconciliation.copy()
    reconciliation["operational_status"] = reconciliation.apply(op_status, axis=1)
    reconciliation["invoice_amount"] = pd.to_numeric(reconciliation["invoice_amount"], errors="coerce")
    reconciliation["payment_amount"] = pd.to_numeric(reconciliation["payment_amount"], errors="coerce")

    payment_currency = payments[["payment_id", "currency"]].drop_duplicates("payment_id").set_index("payment_id")["currency"]
    invoice_currency = invoices[["invoice_id", "currency"]].drop_duplicates("invoice_id").set_index("invoice_id")["currency"]
    reconciliation["payment_key"] = reconciliation["payment_id"].where(reconciliation["payment_id"].notna(), "").astype(str)
    reconciliation["invoice_key"] = reconciliation["invoice_id"].where(reconciliation["invoice_id"].notna(), "").astype(str)
    reconciliation["payment_currency"] = reconciliation["payment_key"].map(payment_currency)
    reconciliation["invoice_currency"] = reconciliation["invoice_key"].map(invoice_currency)
    reconciliation["currency"] = reconciliation["payment_currency"].fillna(reconciliation["invoice_currency"]).fillna("UNKNOWN")


    # Original MATCHED plus human-approved REVIEW cases are confirmed.
    confirmed = reconciliation[
        (reconciliation["status"].astype(str) == "MATCHED")
        | (reconciliation["operational_status"] == "APPROVED")
    ].copy()
    confirmed = confirmed[confirmed["payment_id"].notna()].drop_duplicates("payment_id")

    # REVIEW cases still awaiting a human decision are pending.
    pending = reconciliation[
        (reconciliation["status"].astype(str) == "REVIEW")
        & (reconciliation["operational_status"] == "REVIEW")
        & reconciliation["payment_id"].notna()
    ].copy()
    pending = pending.drop_duplicates("payment_id")

    # Payments with no invoice candidate are unresolved incoming money.
    unmatched_incoming = reconciliation[
        reconciliation["invoice_id"].isna() & reconciliation["payment_id"].notna()
    ].copy().drop_duplicates("payment_id")

    # Invoice-side receivables. Human APPROVED REVIEW cases move into confirmed;
    # everything else that was not originally MATCHED remains unresolved.
    confirmed_receivables = confirmed[confirmed["invoice_id"].notna()].drop_duplicates("invoice_id")
    unresolved_receivables = reconciliation[
        reconciliation["invoice_id"].notna()
        & (reconciliation["status"].astype(str).isin(["REVIEW", "UNMATCHED"]))
        & (reconciliation["operational_status"] != "APPROVED")
    ].copy().drop_duplicates("invoice_id")

    confirmed_cash_by_currency = _money_by_currency(confirmed, "payment_amount", "currency")
    pending_by_currency = _money_by_currency(pending, "payment_amount", "currency")
    unmatched_by_currency = _money_by_currency(unmatched_incoming, "payment_amount", "currency")
    confirmed_receivable_by_currency = _money_by_currency(confirmed_receivables, "invoice_amount", "currency")
    unresolved_receivable_by_currency = _money_by_currency(unresolved_receivables, "invoice_amount", "currency")

    expected_by_currency = {}
    for currency in set(confirmed_receivable_by_currency) | set(unresolved_receivable_by_currency):
        expected_by_currency[currency] = round(
            confirmed_receivable_by_currency.get(currency, 0.0)
            + unresolved_receivable_by_currency.get(currency, 0.0),
            2,
        )

    pending_top = pending.sort_values("payment_amount", ascending=False).head(10)
    pending_receivables = [
        {
            "invoice_id": str(r["invoice_id"]),
            "payment_id": str(r["payment_id"]),
            "amount": round(float(r["payment_amount"]), 2),
            "currency": str(
                payments.loc[payments["payment_id"].astype(str) == str(r["payment_id"]), "currency"].iloc[0]
            ) if not payments.loc[payments["payment_id"].astype(str) == str(r["payment_id"])].empty else "UNKNOWN",
            "invoice_amount": round(float(r["invoice_amount"]), 2),
        }
        for _, r in pending_top.iterrows()
    ]

    confirmed_cash_total = round(float(confirmed["payment_amount"].sum()), 2)
    pending_total = round(float(pending["payment_amount"].sum()), 2)
    unmatched_total = round(float(unmatched_incoming["payment_amount"].sum()), 2)
    confirmed_receivable_total = round(float(confirmed_receivables["invoice_amount"].sum()), 2)
    unresolved_receivable_total = round(float(unresolved_receivables["invoice_amount"].sum()), 2)
    expected_total = round(confirmed_receivable_total + unresolved_receivable_total, 2)

    # Backward-compatible informational fields from the earlier cash tool.
    # These are settlement figures only and are deliberately not used as the
    # new CONFIRMED cash KPI because a settled payment is not automatically a
    # reconciled receivable.
    settlements_path = raw / "settlements.csv"
    settlements = pd.read_csv(settlements_path) if settlements_path.exists() else pd.DataFrame()
    if not settlements.empty and {"status", "settled_amount"}.issubset(set(settlements.columns)):
        settled_mask = settlements["status"].astype(str).str.casefold() == "settled"
        settled_cash_total = round(float(pd.to_numeric(settlements.loc[settled_mask, "settled_amount"], errors="coerce").fillna(0).sum()), 2)
        settled_payment_count = int(settled_mask.sum())
    else:
        settled_cash_total = 0.0
        settled_payment_count = 0

    return {
        "demo_model": True,
        "currency_scope": sorted(set(invoices["currency"].dropna().astype(str)) | set(payments["currency"].dropna().astype(str))),
        "confirmed_cash": {
            "label": "CONFIRMED",
            "amount": _scalar_if_one(confirmed_cash_by_currency),
            "by_currency": confirmed_cash_by_currency,
            "payment_count": int(len(confirmed)),
            "definition": "Incoming payment amounts from original MATCHED records and human-approved REVIEW records.",
        },
        "pending_review_payments": {
            "label": "PENDING",
            "amount": _scalar_if_one(pending_by_currency),
            "by_currency": pending_by_currency,
            "payment_count": int(len(pending)),
            "definition": "Payment amounts on REVIEW records still awaiting human approval or rejection.",
        },
        "unmatched_incoming_payments": {
            "label": "UNRESOLVED",
            "amount": _scalar_if_one(unmatched_by_currency),
            "by_currency": unmatched_by_currency,
            "payment_count": int(len(unmatched_incoming)),
            "definition": "Incoming payments with no invoice candidate in reconciliation results.",
        },
        "confirmed_receivables": {
            "label": "CONFIRMED",
            "amount": _scalar_if_one(confirmed_receivable_by_currency),
            "by_currency": confirmed_receivable_by_currency,
            "invoice_count": int(len(confirmed_receivables)),
        },
        "unresolved_receivables": {
            "label": "UNRESOLVED",
            "amount": _scalar_if_one(unresolved_receivable_by_currency),
            "by_currency": unresolved_receivable_by_currency,
            "invoice_count": int(len(unresolved_receivables)),
        },
        "expected_incoming_cash": {
            "label": "EXPECTED",
            "amount": _scalar_if_one(expected_by_currency),
            "by_currency": expected_by_currency,
            "definition": "Confirmed receivables plus unresolved receivables represented by the current invoice dataset; not a forecast.",
        },
        "largest_pending_receivables": pending_receivables,
        "settled_cash_total": settled_cash_total,
        "settled_payment_count": settled_payment_count,
        "methodology": (
            "Synthetic demo model using only invoices, payments, reconciliation_results.csv, "
            "and the human-review audit overlay. It is not a bank balance, cash forecast, "
            "or production accounting calculation."
        ),
    }
