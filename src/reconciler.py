#!/usr/bin/env python3
"""Deterministic invoice-to-payment reconciliation engine.

No LLMs are used for financial matching. Matching is based on explainable,
weighted signals and deterministic greedy assignment.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
from rapidfuzz.fuzz import ratio, WRatio

try:
    from financial_normalizer import normalize_dataframe
except ImportError:  # pragma: no cover
    from src.financial_normalizer import normalize_dataframe


MATCHED_THRESHOLD = 90.0
REVIEW_THRESHOLD = 70.0
DATE_SOFT_LIMIT_DAYS = 30

LOGGER = logging.getLogger("reconciler")

OUTPUT_COLUMNS = [
    "invoice_id", "payment_id", "settlement_id", "invoice_amount",
    "payment_amount", "amount_difference", "customer_similarity",
    "date_difference", "reference_match", "confidence_score", "status",
    "reason",
]


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _is_missing(value: Any) -> bool:
    return value is None or pd.isna(value)


def amount_similarity(invoice_amount: Any, payment_amount: Any) -> float:
    if _is_missing(invoice_amount) or _is_missing(payment_amount):
        return 0.0
    invoice_amount = abs(float(invoice_amount))
    payment_amount = abs(float(payment_amount))
    if invoice_amount == payment_amount:
        return 100.0
    if invoice_amount == 0:
        return 0.0
    # Ratio is deliberately symmetric and rewards partial payments without
    # treating them as exact matches.
    return max(0.0, 100.0 - abs(invoice_amount - payment_amount) / invoice_amount * 100.0)


def date_proximity_score(invoice_date: Any, payment_date: Any) -> float:
    if _is_missing(invoice_date) or _is_missing(payment_date):
        return 0.0
    days = abs((pd.Timestamp(payment_date) - pd.Timestamp(invoice_date)).days)
    if days <= 3:
        return 100.0
    if days <= 7:
        return 85.0
    if days <= 14:
        return 65.0
    if days <= DATE_SOFT_LIMIT_DAYS:
        return 35.0
    return 0.0


def date_difference_days(invoice_date: Any, payment_date: Any) -> int | None:
    if _is_missing(invoice_date) or _is_missing(payment_date):
        return None
    return abs((pd.Timestamp(payment_date) - pd.Timestamp(invoice_date)).days)


def reference_matches(invoice_id: str, payment_row: pd.Series) -> bool:
    if not invoice_id:
        return False
    invoice_id_norm = str(invoice_id).casefold()
    for column in ("reference_note", "description", "reference", "invoice_reference"):
        if column in payment_row.index and not _is_missing(payment_row[column]):
            candidate = str(payment_row[column]).casefold()
            if invoice_id_norm in candidate:
                return True
    return False


def customer_similarity(invoice_row: pd.Series, payment_row: pd.Series) -> float:
    left = invoice_row.get("customer_name_normalized")
    right = payment_row.get("payer_name_normalized")
    if not left or not right:
        return 0.0
    return float(WRatio(str(left), str(right)))


def description_similarity(invoice_row: pd.Series, payment_row: pd.Series) -> float:
    left = invoice_row.get("description_normalized")
    # Payment generator calls this reference_note. Support a generic
    # description field too so the engine works with other payment exports.
    right = payment_row.get("description_normalized")
    if not right:
        right = payment_row.get("reference_note_normalized")
    if not left or not right:
        return 0.0
    return float(ratio(str(left), str(right)))


def calculate_candidate(invoice: pd.Series, payment: pd.Series) -> dict[str, Any]:
    amount_sim = amount_similarity(invoice["amount_normalized"], payment["amount_normalized"])
    cust_sim = customer_similarity(invoice, payment)
    desc_sim = description_similarity(invoice, payment)
    # Payment timing is compared with the invoice due date because that is the
    # operational reconciliation anchor used by the generated finance data.
    date_score = date_proximity_score(invoice["due_date_normalized"], payment["payment_date_normalized"])
    ref_match = reference_matches(str(invoice["invoice_id"]), payment)

    # Explainable fixed weights. Reference is the strongest deterministic key;
    # amount remains the main economic signal.
    score = (
        amount_sim * 0.40
        + cust_sim * 0.30
        + desc_sim * 0.05
        + date_score * 0.20
        + (100.0 if ref_match else 0.0) * 0.05
    )

    return {
        "invoice_id": invoice["invoice_id"],
        "payment_id": payment["payment_id"],
        "amount_similarity": amount_sim,
        "customer_similarity": cust_sim,
        "description_similarity": desc_sim,
        "date_score": date_score,
        "date_difference": date_difference_days(
            invoice["due_date_normalized"], payment["payment_date_normalized"]
        ),
        "reference_match": ref_match,
        "confidence_score": round(score, 2),
    }


def classify(score: float) -> str:
    if score >= MATCHED_THRESHOLD:
        return "MATCHED"
    if score >= REVIEW_THRESHOLD:
        return "REVIEW"
    return "UNMATCHED"


def build_reason(result: dict[str, Any], *, missing_payment: bool = False, duplicate: bool = False) -> str:
    if missing_payment:
        return "No payment transaction is available for this invoice."

    score = result["confidence_score"]
    parts = [
        f"confidence {score:.2f}",
        f"amount similarity {result['amount_similarity']:.1f}%",
        f"customer similarity {result['customer_similarity']:.1f}%",
    ]
    if result["description_similarity"] > 0:
        parts.append(f"description similarity {result['description_similarity']:.1f}%")
    if result["date_difference"] is not None:
        parts.append(f"date difference {result['date_difference']} day(s)")
    parts.append("invoice reference matched" if result["reference_match"] else "invoice reference not found")
    if duplicate:
        parts.append("payment is already assigned to another invoice")
    return "; ".join(parts) + "."


def prepare_inputs(
    invoices: pd.DataFrame,
    payments: pd.DataFrame,
    settlements: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Normalize without changing the existing normalizer implementation."""
    inv = normalize_dataframe(
        invoices,
        customer_col="customer_name",
        description_col="description",
        date_col="invoice_date",
        amount_col="amount",
    )
    # The existing normalizer is intentionally left untouched. Its date parser
    # is useful for day-first exports, but ISO dates such as 2026-06-10 can be
    # ambiguous when dayfirst=True. Reconciliation therefore applies an
    # explicit mixed-format parse to the already-normalized working copy.
    inv["invoice_date_normalized"] = pd.to_datetime(
        invoices["invoice_date"], errors="coerce", format="mixed", dayfirst=True
    )
    inv["due_date_normalized"] = pd.to_datetime(
        invoices["due_date"], errors="coerce", format="mixed", dayfirst=True
    )
    pay = normalize_dataframe(
        payments,
        customer_col="payer_name",
        description_col="reference_note",
        date_col="payment_date",
        amount_col="amount",
    )
    pay["payment_date_normalized"] = pd.to_datetime(
        payments["payment_date"], errors="coerce", format="mixed", dayfirst=True
    )
    # Settlement schema has different field names, so preserve its IDs and
    # normalize the fields needed for lookup.
    stl = settlements.copy()
    stl["settlement_date_normalized"] = pd.to_datetime(
        stl.get("settlement_date"), errors="coerce", dayfirst=True
    )
    stl["settled_amount_normalized"] = pd.to_numeric(
        stl.get("settled_amount"), errors="coerce"
    )
    return inv, pay, stl


def reconcile(
    invoices: pd.DataFrame,
    payments: pd.DataFrame,
    settlements: pd.DataFrame,
) -> pd.DataFrame:
    inv, pay, stl = prepare_inputs(invoices, payments, settlements)

    assigned_payments: set[str] = set()
    assigned_by_invoice: dict[str, str] = {}
    rows: list[dict[str, Any]] = []

    # Stable sorting makes tie resolution deterministic across runs.
    inv = inv.sort_values("invoice_id", kind="stable").reset_index(drop=True)
    pay = pay.sort_values("payment_id", kind="stable").reset_index(drop=True)

    candidates_by_invoice: dict[str, list[dict[str, Any]]] = {}
    for _, invoice in inv.iterrows():
        candidates = [calculate_candidate(invoice, payment) for _, payment in pay.iterrows()]
        candidates.sort(key=lambda c: (-c["confidence_score"], str(c["payment_id"])))
        candidates_by_invoice[str(invoice["invoice_id"])] = candidates

    for _, invoice in inv.iterrows():
        invoice_id = str(invoice["invoice_id"])
        candidates = candidates_by_invoice[invoice_id]
        available = [c for c in candidates if str(c["payment_id"]) not in assigned_payments]

        if not available:
            rows.append({
                "invoice_id": invoice_id,
                "payment_id": None,
                "settlement_id": None,
                "invoice_amount": invoice["amount_normalized"],
                "payment_amount": None,
                "amount_difference": None,
                "customer_similarity": 0.0,
                "date_difference": None,
                "reference_match": False,
                "confidence_score": 0.0,
                "status": "UNMATCHED",
                "reason": "No unassigned payment transaction is available for this invoice.",
            })
            LOGGER.info("%s -> UNMATCHED: no available payment", invoice_id)
            continue

        best = available[0]
        status = classify(best["confidence_score"])
        payment = pay.loc[pay["payment_id"].astype(str) == str(best["payment_id"])].iloc[0]
        duplicate = str(best["payment_id"]) in assigned_payments
        # At this stage duplicate=False because unavailable candidates were removed.
        if status in {"MATCHED", "REVIEW"}:
            assigned_payments.add(str(best["payment_id"]))
            assigned_by_invoice[invoice_id] = str(best["payment_id"])

        amount_difference = round(
            float(invoice["amount_normalized"]) - float(payment["amount_normalized"]), 2
        )
        settlement_match = stl.loc[stl["payment_id"].astype(str) == str(best["payment_id"])]
        settlement_id = settlement_match.iloc[0]["settlement_id"] if not settlement_match.empty else None

        result = {
            "invoice_id": invoice_id,
            "payment_id": best["payment_id"],
            "settlement_id": settlement_id,
            "invoice_amount": invoice["amount_normalized"],
            "payment_amount": payment["amount_normalized"],
            "amount_difference": amount_difference,
            "customer_similarity": round(best["customer_similarity"], 2),
            "date_difference": best["date_difference"],
            "reference_match": best["reference_match"],
            "confidence_score": best["confidence_score"],
            "status": status,
            "reason": build_reason(best, duplicate=duplicate),
        }
        rows.append(result)
        LOGGER.info(
            "%s -> %s via %s | score=%.2f amount=%.1f customer=%.1f date=%s ref=%s",
            invoice_id, status, best["payment_id"], best["confidence_score"],
            best["amount_similarity"], best["customer_similarity"],
            best["date_difference"], best["reference_match"],
        )

    # Surface every payment that wasn't consumed by an invoice. This captures
    # duplicate second payments and genuinely unmatched transactions.
    for _, payment in pay.iterrows():
        pid = str(payment["payment_id"])
        if pid in assigned_payments:
            continue

        duplicate_invoice = next(
            (inv_id for inv_id, assigned_pid in assigned_by_invoice.items() if assigned_pid == pid),
            None,
        )
        reason = (
            f"Payment {pid} was not assigned because it is a duplicate/unallocated payment."
            if duplicate_invoice
            else f"Payment {pid} has no sufficiently strong invoice candidate."
        )
        settlement_match = stl.loc[stl["payment_id"].astype(str) == pid]
        settlement_id = settlement_match.iloc[0]["settlement_id"] if not settlement_match.empty else None
        rows.append({
            "invoice_id": None,
            "payment_id": pid,
            "settlement_id": settlement_id,
            "invoice_amount": None,
            "payment_amount": payment["amount_normalized"],
            "amount_difference": None,
            "customer_similarity": 0.0,
            "date_difference": None,
            "reference_match": False,
            "confidence_score": 0.0,
            "status": "UNMATCHED",
            "reason": reason,
        })
        LOGGER.info("%s -> UNMATCHED payment: %s", pid, reason)

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def run_reconciliation(base_dir: Path) -> Path:
    raw = base_dir / "data" / "raw"
    output_dir = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    invoices = pd.read_csv(raw / "invoices.csv")
    payments = pd.read_csv(raw / "payments.csv")
    settlements = pd.read_csv(raw / "settlements.csv")

    LOGGER.info("Loaded %d invoices, %d payments, %d settlements", len(invoices), len(payments), len(settlements))
    results = reconcile(invoices, payments, settlements)

    output = output_dir / "reconciliation_results.csv"
    results.to_csv(output, index=False)
    LOGGER.info("Wrote %d reconciliation rows to %s", len(results), output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic finance reconciliation.")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    run_reconciliation(args.base_dir)


if __name__ == "__main__":
    main()
