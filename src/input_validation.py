"""Validation for uploaded Finance Controller source CSVs.

This module is deliberately small and deterministic. It validates structure and
basic data integrity before uploaded files replace the current raw dataset.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

REQUIRED_COLUMNS: dict[str, set[str]] = {
    "invoices.csv": {"invoice_id", "customer_name", "invoice_date", "due_date", "amount", "currency", "description"},
    "payments.csv": {"payment_id", "payer_name", "payment_date", "amount", "currency", "method", "reference_note"},
    "settlements.csv": {"settlement_id", "payment_id", "settlement_date", "gross_amount", "fee", "settled_amount", "status"},
}


def validate_source_frames(frames: Mapping[str, pd.DataFrame]) -> None:
    """Validate all required source frames and raise clear ValueErrors."""
    missing_files = set(REQUIRED_COLUMNS) - set(frames)
    if missing_files:
        raise ValueError(f"Missing required files: {sorted(missing_files)}")

    for name, required in REQUIRED_COLUMNS.items():
        df = frames[name]
        if df.empty:
            raise ValueError(f"{name} is empty.")

        missing_columns = required - set(df.columns)
        if missing_columns:
            raise ValueError(f"{name} is missing required columns: {sorted(missing_columns)}")

        id_column = "invoice_id" if name == "invoices.csv" else "payment_id" if name == "payments.csv" else "settlement_id"
        if df[id_column].isna().any() or df[id_column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{name} contains missing/blank {id_column} values.")
        if df[id_column].duplicated().any():
            duplicates = df.loc[df[id_column].duplicated(keep=False), id_column].astype(str).unique().tolist()
            raise ValueError(f"{name} contains duplicate {id_column} values: {duplicates}")

    for name, column in [
        ("invoices.csv", "amount"),
        ("payments.csv", "amount"),
        ("settlements.csv", "gross_amount"),
        ("settlements.csv", "fee"),
        ("settlements.csv", "settled_amount"),
    ]:
        numeric = pd.to_numeric(frames[name][column], errors="coerce")
        if numeric.isna().any():
            bad_rows = numeric[numeric.isna()].index.tolist()[:5]
            raise ValueError(f"{name} contains invalid numeric values in {column} at rows: {bad_rows}")

    for name, column in [
        ("invoices.csv", "invoice_date"),
        ("invoices.csv", "due_date"),
        ("payments.csv", "payment_date"),
        ("settlements.csv", "settlement_date"),
    ]:
        parsed = pd.to_datetime(frames[name][column], errors="coerce", format="mixed", dayfirst=True)
        if parsed.isna().any():
            bad_rows = parsed[parsed.isna()].index.tolist()[:5]
            raise ValueError(f"{name} contains invalid dates in {column} at rows: {bad_rows}")

    settlement_payment_ids = set(frames["settlements.csv"]["payment_id"].astype(str))
    payment_ids = set(frames["payments.csv"]["payment_id"].astype(str))
    unknown_settlement_payments = settlement_payment_ids - payment_ids
    if unknown_settlement_payments:
        raise ValueError(
            "settlements.csv references unknown payment_id values: "
            f"{sorted(unknown_settlement_payments)[:10]}"
        )


def validate_source_files(base_dir: Path | str) -> None:
    """Read and validate the three raw CSV files in a project directory."""
    base = Path(base_dir)
    raw = base / "data" / "raw"
    frames = {name: pd.read_csv(raw / name) for name in REQUIRED_COLUMNS}
    validate_source_frames(frames)
