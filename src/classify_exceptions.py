#!/usr/bin/env python3
"""Deterministic exception classification for finance reconciliation.

Consumes reconciliation_results.csv and classifies REVIEW/UNMATCHED records
into actionable finance exceptions. No LLM is used.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger("exception_classifier")

EXCEPTION_COLUMNS = [
    "exception_id", "invoice_id", "payment_id", "exception_type",
    "invoice_amount", "payment_amount", "amount_difference",
    "confidence_score", "severity", "reason", "recommended_action",
]

AMOUNT_TOLERANCE = 0.01
PARTIAL_MIN_RATIO = 0.50
CUSTOMER_AMBIGUITY_THRESHOLD = 70.0
DATE_MISMATCH_DAYS = 14


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def missing(value: Any) -> bool:
    return value is None or pd.isna(value)


def money(value: Any) -> float | None:
    if missing(value):
        return None
    return float(value)


def _same_customer_amount_date(a: pd.Series, b: pd.Series) -> bool:
    """Detect likely duplicate payments from reconciliation signals.

    The reconciliation output does not retain the payer name, so duplicate
    detection uses the signals available there: same invoice candidate,
    amount, date proximity and high customer similarity. Exact payment IDs
    are excluded naturally by the caller.
    """
    if missing(a.get("payment_amount")) or missing(b.get("payment_amount")):
        return False
    if abs(float(a["payment_amount"]) - float(b["payment_amount"])) > AMOUNT_TOLERANCE:
        return False
    # Rows with the same invoice and same economic amount are especially strong.
    if not missing(a.get("invoice_id")) and not missing(b.get("invoice_id")):
        if str(a["invoice_id"]) == str(b["invoice_id"]):
            return True
    customer_close = float(a.get("customer_similarity", 0) or 0) >= 85 and float(b.get("customer_similarity", 0) or 0) >= 85
    dates_close = (
        not missing(a.get("date_difference")) and not missing(b.get("date_difference"))
        and float(a["date_difference"]) <= 3 and float(b["date_difference"]) <= 3
    )
    return customer_close and dates_close


def build_duplicate_payment_ids(results: pd.DataFrame, payments: pd.DataFrame | None = None) -> set[str]:
    """Return unassigned payment IDs that look like duplicate transfers.

    A payment-only row is considered duplicate when another result row has an
    invoice and the same payment amount, with strong customer/date signals.
    """
    assigned = results[results["invoice_id"].notna() & results["payment_id"].notna()].copy()
    unassigned = results[results["invoice_id"].isna() & results["payment_id"].notna()].copy()
    duplicate_ids: set[str] = set()

    if payments is None or payments.empty:
        for _, u in unassigned.iterrows():
            for _, a in assigned.iterrows():
                if _same_customer_amount_date(u, a):
                    duplicate_ids.add(str(u["payment_id"]))
                    break
        return duplicate_ids

    # The reconciliation output intentionally contains only the signals used
    # for the winning candidate. For duplicate payments, enrich the payment-only
    # rows with the same raw payment fields and compare them deterministically
    # against payments already assigned to invoices.
    pay = payments.copy()
    pay["payment_id"] = pay["payment_id"].astype(str)
    pay["amount"] = pd.to_numeric(pay["amount"], errors="coerce")
    pay["payment_date"] = pd.to_datetime(pay["payment_date"], errors="coerce", format="mixed", dayfirst=True)
    for _, u in unassigned.iterrows():
        pid = str(u["payment_id"])
        ur = pay[pay["payment_id"] == pid]
        if ur.empty:
            continue
        ur = ur.iloc[0]
        for _, a in assigned.iterrows():
            aid = str(a["payment_id"])
            ar = pay[pay["payment_id"] == aid]
            if ar.empty:
                continue
            ar = ar.iloc[0]
            amount_same = abs(float(ur["amount"]) - float(ar["amount"])) <= AMOUNT_TOLERANCE
            payer_sim = 0.0
            if pd.notna(ur.get("payer_name")) and pd.notna(ar.get("payer_name")):
                from rapidfuzz.fuzz import WRatio
                payer_sim = float(WRatio(str(ur["payer_name"]), str(ar["payer_name"])))
            date_close = False
            if pd.notna(ur.get("payment_date")) and pd.notna(ar.get("payment_date")):
                date_close = abs((ur["payment_date"] - ar["payment_date"]).days) <= 3
            if amount_same and payer_sim >= 85 and date_close:
                duplicate_ids.add(pid)
                break
    return duplicate_ids


def classify_exception(row: pd.Series, *, duplicate_payment: bool = False) -> tuple[str, str, str]:
    """Return (exception_type, severity, recommended_action)."""
    invoice_id = row.get("invoice_id")
    payment_id = row.get("payment_id")
    inv = money(row.get("invoice_amount"))
    pmt = money(row.get("payment_amount"))
    diff = money(row.get("amount_difference"))
    score = float(row.get("confidence_score", 0) or 0)
    customer = float(row.get("customer_similarity", 0) or 0)
    date_diff = row.get("date_difference")
    ref = bool(row.get("reference_match", False))

    # 1. No invoice/payment relationship at all.
    if missing(invoice_id) and not missing(payment_id):
        if duplicate_payment:
            return "DUPLICATE_PAYMENT", "HIGH", "VERIFY DUPLICATE"
        return "UNMATCHED_TRANSACTION", "MEDIUM", "CHECK BANK STATEMENT"

    # 2. Invoice with no payment candidate, or a very weak placeholder candidate.
    # A very weak candidate is treated as effectively missing rather than a
    # meaningful payment relationship.
    if not missing(invoice_id) and missing(payment_id):
        return "MISSING_PAYMENT", "HIGH", "CHECK BANK STATEMENT"
    if not missing(invoice_id) and not missing(payment_id) and score < 40 and customer < 50 and (inv is None or pmt is None or abs(diff or 0) >= max(1.0, abs(inv or 0) * 0.50)):
        return "MISSING_PAYMENT", "HIGH", "CHECK BANK STATEMENT"

    # 3. Amount-driven exceptions take precedence over softer signals.
    if inv is not None and pmt is not None and abs(diff or 0) > AMOUNT_TOLERANCE:
        ratio = abs(pmt) / abs(inv) if inv else 0
        if 0 < ratio < 1 and ratio < 0.95 and customer >= 80:
            return "PARTIAL_PAYMENT", "HIGH", "CONTACT CUSTOMER"
        return "AMOUNT_MISMATCH", "HIGH" if abs(diff or 0) >= 1000 else "MEDIUM", "REVIEW PAYMENT"

    # 4. Customer ambiguity: money/date can look plausible but payer identity is weak.
    if customer < CUSTOMER_AMBIGUITY_THRESHOLD:
        return "CUSTOMER_AMBIGUITY", "MEDIUM", "CONTACT CUSTOMER"

    # 5. Date mismatch where other evidence is comparatively strong.
    if not missing(date_diff) and float(date_diff) > DATE_MISMATCH_DAYS:
        return "DATE_MISMATCH", "MEDIUM", "CHECK BANK STATEMENT"

    # 6. Reference missing despite otherwise strong candidate.
    if not ref and customer >= 85 and (inv is None or pmt is None or abs(diff or 0) <= AMOUNT_TOLERANCE):
        return "MISSING_REFERENCE", "LOW", "REVIEW PAYMENT"

    return "OTHER", "LOW", "MANUAL RECONCILIATION"


def make_reason(row: pd.Series, exception_type: str, *, duplicate_payment: bool = False) -> str:
    inv = money(row.get("invoice_amount"))
    pmt = money(row.get("payment_amount"))
    diff = money(row.get("amount_difference"))
    score = float(row.get("confidence_score", 0) or 0)
    date_diff = row.get("date_difference")

    if exception_type == "AMOUNT_MISMATCH":
        direction = "lower" if (diff or 0) > 0 else "higher"
        return f"Payment is ₹{abs(diff or 0):,.2f} {direction} than invoice amount."
    if exception_type == "PARTIAL_PAYMENT":
        return f"Payment of ₹{abs(pmt or 0):,.2f} covers only part of the invoice of ₹{abs(inv or 0):,.2f}."
    if exception_type == "MISSING_PAYMENT":
        return "No payment candidate was found for this invoice."
    if exception_type == "DUPLICATE_PAYMENT":
        return "Payment appears to have been associated with more than one invoice or duplicated in the transaction feed."
    if exception_type == "CUSTOMER_AMBIGUITY":
        return f"Customer similarity is low ({float(row.get('customer_similarity', 0) or 0):.1f}%), so payer identity is ambiguous."
    if exception_type == "DATE_MISMATCH":
        return f"Payment date is {int(float(date_diff))} days from the invoice due date."
    if exception_type == "MISSING_REFERENCE":
        return "Payment amount and customer signals are strong, but the invoice reference is missing."
    if exception_type == "UNMATCHED_TRANSACTION":
        return "Payment transaction has no sufficiently strong invoice relationship."
    return f"No dominant exception rule applied; reconciliation confidence was {score:.2f}."


def classify_exceptions(results: pd.DataFrame, payments: pd.DataFrame | None = None) -> pd.DataFrame:
    required = {"invoice_id", "payment_id", "status", "invoice_amount", "payment_amount", "amount_difference", "customer_similarity", "date_difference", "reference_match", "confidence_score"}
    missing_cols = required - set(results.columns)
    if missing_cols:
        raise ValueError(f"reconciliation_results.csv is missing columns: {sorted(missing_cols)}")

    duplicate_ids = build_duplicate_payment_ids(results, payments)
    exceptions = []
    counter = 1

    candidates = results[results["status"].isin(["REVIEW", "UNMATCHED"])].copy()
    for _, row in candidates.iterrows():
        pid = None if missing(row.get("payment_id")) else str(row["payment_id"])
        is_dup = pid in duplicate_ids if pid else False
        exception_type, severity, action = classify_exception(row, duplicate_payment=is_dup)
        reason = make_reason(row, exception_type, duplicate_payment=is_dup)
        exceptions.append({
            "exception_id": f"EXC-{counter:05d}",
            "invoice_id": None if missing(row.get("invoice_id")) else row["invoice_id"],
            "payment_id": None if missing(row.get("payment_id")) else row["payment_id"],
            "exception_type": exception_type,
            "invoice_amount": row.get("invoice_amount"),
            "payment_amount": row.get("payment_amount"),
            "amount_difference": row.get("amount_difference"),
            "confidence_score": row.get("confidence_score", 0),
            "severity": severity,
            "reason": reason,
            "recommended_action": action,
        })
        LOGGER.info("%s | %s | invoice=%s payment=%s | %s", f"EXC-{counter:05d}", exception_type, row.get("invoice_id"), row.get("payment_id"), reason)
        counter += 1

    return pd.DataFrame(exceptions, columns=EXCEPTION_COLUMNS)


def run_exception_classification(base_dir: Path) -> Path:
    input_path = base_dir / "outputs" / "reconciliation_results.csv"
    output_dir = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(input_path)
    payments_path = base_dir / "data" / "raw" / "payments.csv"
    payments = pd.read_csv(payments_path) if payments_path.exists() else None
    exceptions = classify_exceptions(results, payments=payments)
    output_path = output_dir / "exceptions.csv"
    exceptions.to_csv(output_path, index=False)
    LOGGER.info("Wrote %d exceptions to %s", len(exceptions), output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify deterministic reconciliation exceptions.")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    run_exception_classification(args.base_dir)


if __name__ == "__main__":
    main()
