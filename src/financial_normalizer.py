"""
financial_normalizer.py

Normalization utilities for financial reconciliation:
- Customer names
- Transaction descriptions
- Dates
- Monetary amounts

Designed for messy CSV/Excel-style financial data and pandas DataFrames.
"""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any

import pandas as pd


# Common legal-entity suffixes. We normalize these to "ltd" so that:
# "ABC Pvt Ltd", "ABC Private Limited", and "ABC Ltd." can match.
LEGAL_SUFFIX_RE = re.compile(
    r"""
    \b(
        private\s+limited|
        pvt\.?\s*ltd\.?|
        pvt\.?|
        limited|
        ltd\.?
    )\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

WHITESPACE_RE = re.compile(r"\s+")
PUNCTUATION_RE = re.compile(r"[^\w\s]", re.UNICODE)

# Indian/international currency symbols and common accounting notation.
CURRENCY_RE = re.compile(r"[₹$€£¥]|(?:INR|USD|EUR|GBP|JPY)\b", re.IGNORECASE)


def normalize_text(value: Any) -> str | None:
    """General text cleanup: Unicode normalization, case folding, punctuation,
    and repeated whitespace removal.
    """
    if pd.isna(value):
        return None

    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    text = PUNCTUATION_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()

    return text or None


def normalize_customer_name(value: Any) -> str | None:
    """Normalize customer/vendor names for matching.

    Examples:
        '  Acme Pvt. Ltd.  ' -> 'acme ltd'
        'ACME PRIVATE LIMITED' -> 'acme ltd'
        'Acme, Ltd.' -> 'acme ltd'
    """
    text = normalize_text(value)
    if text is None:
        return None

    # Normalize common legal suffix variants.
    text = LEGAL_SUFFIX_RE.sub(" ltd", text)
    text = WHITESPACE_RE.sub(" ", text).strip()

    return text or None


def normalize_description(value: Any) -> str | None:
    """Normalize transaction descriptions while preserving meaningful words.

    Examples:
        '  INV-001 / Payment   Received ' -> 'inv 001 payment received'
        'PAYMENT RECEIVED' -> 'payment received'
    """
    return normalize_text(value)


def normalize_date(value: Any) -> pd.Timestamp | pd.NaT:
    """Parse dates into pandas timestamps.

    dayfirst=True is useful for common Indian/accounting exports such as
    26/08/2026, while ISO dates are also handled correctly by pandas.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NaT

    parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)

    if isinstance(parsed, pd.DatetimeIndex):
        return parsed[0] if len(parsed) else pd.NaT

    return parsed


def normalize_amount(value: Any) -> float | None:
    """Convert messy monetary values to a numeric amount.

    Handles:
        ₹1,234.50
        INR 1,234.50
        $1,234.50
        (1,234.50)       -> -1234.50
        -1,234.50        -> -1234.50
        1234,50           -> 1234.50
        1,234             -> 1234.00
    """
    if value is None or pd.isna(value):
        return None

    # Already numeric.
    if isinstance(value, (int, float, Decimal)):
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    text = str(value).strip()

    if not text:
        return None

    negative = (
        (text.startswith("(") and text.endswith(")"))
        or text.startswith("-")
    )

    text = text.replace("(", "").replace(")", "").replace("-", "")
    text = CURRENCY_RE.sub("", text)
    text = text.replace(" ", "")

    # Determine decimal convention.
    # 1.234,56 -> 1234.56
    # 1,234.56 -> 1234.56
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        # Treat a final comma followed by 1-2 digits as decimal comma.
        if re.search(r",\d{1,2}$", text):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")

    try:
        amount = Decimal(text)
    except InvalidOperation:
        return None

    if negative:
        amount = -abs(amount)

    return float(amount)


def normalize_dataframe(
    df: pd.DataFrame,
    *,
    customer_col: str = "customer_name",
    description_col: str = "description",
    date_col: str = "date",
    amount_col: str = "amount",
) -> pd.DataFrame:
    """Return a normalized copy of a financial DataFrame.

    Original columns are preserved and normalized columns are added with
    the suffix '_normalized'.
    """
    result = df.copy()

    if customer_col in result.columns:
        result[f"{customer_col}_normalized"] = result[customer_col].map(
            normalize_customer_name
        )

    if description_col in result.columns:
        result[f"{description_col}_normalized"] = result[description_col].map(
            normalize_description
        )

    if date_col in result.columns:
        result[f"{date_col}_normalized"] = result[date_col].map(
            normalize_date
        )

    if amount_col in result.columns:
        result[f"{amount_col}_normalized"] = result[amount_col].map(
            normalize_amount
        )

    return result


def normalize_and_select(
    df: pd.DataFrame,
    *,
    customer_col: str = "customer_name",
    description_col: str = "description",
    date_col: str = "date",
    amount_col: str = "amount",
) -> pd.DataFrame:
    """Normalize and return a clean reconciliation-ready schema."""
    normalized = normalize_dataframe(
        df,
        customer_col=customer_col,
        description_col=description_col,
        date_col=date_col,
        amount_col=amount_col,
    )

    output = pd.DataFrame(index=normalized.index)

    if customer_col in normalized:
        output["customer_name"] = normalized[f"{customer_col}_normalized"]

    if description_col in normalized:
        output["description"] = normalized[f"{description_col}_normalized"]

    if date_col in normalized:
        output["date"] = normalized[f"{date_col}_normalized"].dt.normalize()

    if amount_col in normalized:
        output["amount"] = normalized[f"{amount_col}_normalized"]

    return output.reset_index(drop=True)


if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "customer_name": [
                "  ACME Pvt. Ltd. ",
                "Acme PRIVATE LIMITED",
                "Acme, Ltd.",
                "Globex   LTD.",
            ],
            "description": [
                "INV-001 / Payment   Received",
                "invoice #001 payment received",
                "INV 002 - Subscription",
                "Subscription payment",
            ],
            "date": [
                "26/08/2026",
                "2026-08-26",
                "26-08-2026",
                "Aug 26, 2026",
            ],
            "amount": [
                "₹1,234.50",
                "1,234.50",
                "(500.00)",
                "INR 2,000",
            ],
        }
    )

    clean = normalize_and_select(sample)
    print(clean.to_string(index=False))
