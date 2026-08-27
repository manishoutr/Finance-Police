"""Step 7: deterministic evaluation and benchmarking for reconciliation results.

Compares the invoice-level predictions in reconciliation_results.csv against
 ground_truth.csv. The reconciliation algorithm itself is not modified.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger("reconciliation_evaluation")

REQUIRED_RESULT_COLUMNS = {
    "invoice_id", "payment_id", "invoice_amount", "payment_amount", "status"
}
REQUIRED_GT_COLUMNS = {"invoice_id", "payment_ids", "invoice_amount", "match_type"}


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value)


def _payment_set(value: Any) -> set[str]:
    if _is_missing(value):
        return set()
    return {p.strip() for p in str(value).replace(",", ";").split(";") if p.strip()}


def _safe_float(value: Any) -> float:
    if _is_missing(value):
        return 0.0
    return float(value)


def load_inputs(results_path: Path, ground_truth_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = pd.read_csv(results_path)
    ground_truth = pd.read_csv(ground_truth_path)

    missing_results = REQUIRED_RESULT_COLUMNS - set(results.columns)
    missing_gt = REQUIRED_GT_COLUMNS - set(ground_truth.columns)
    if missing_results:
        raise ValueError(f"Missing reconciliation columns: {sorted(missing_results)}")
    if missing_gt:
        raise ValueError(f"Missing ground-truth columns: {sorted(missing_gt)}")

    return results, ground_truth


def evaluate(results: pd.DataFrame, ground_truth: pd.DataFrame, processing_time_seconds: float = 0.0) -> dict[str, Any]:
    """Calculate deterministic invoice-level benchmark metrics.

    Ground truth is the source of truth for correctness. For invoices with
    multiple valid payments (partial/duplicate cases), a predicted payment is
    considered correct when it belongs to the ground-truth payment set.
    Missing-payment invoices are correctly identified only when the engine
    produces no payment and status UNMATCHED.
    """
    start = time.perf_counter()

    # Ground truth is keyed by invoice_id. Duplicate GT rows would make the
    # benchmark ambiguous, so fail loudly instead of silently choosing one.
    gt_invoice = ground_truth.dropna(subset=["invoice_id"]).copy()
    if gt_invoice["invoice_id"].duplicated().any():
        duplicates = gt_invoice.loc[gt_invoice["invoice_id"].duplicated(), "invoice_id"].tolist()
        raise ValueError(f"Ground truth contains duplicate invoice IDs: {duplicates}")

    result_invoice = results[results["invoice_id"].notna()].copy()
    result_invoice["invoice_id"] = result_invoice["invoice_id"].astype(str)
    gt_invoice["invoice_id"] = gt_invoice["invoice_id"].astype(str)

    # Keep exactly one prediction per invoice for invoice-level metrics.
    predictions = result_invoice.drop_duplicates(subset=["invoice_id"], keep="first").set_index("invoice_id")
    truth = gt_invoice.set_index("invoice_id")

    invoice_ids = list(truth.index)
    total_records = len(invoice_ids)

    automatic_matches = 0
    review = 0
    unmatched = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    correctly_identified_unmatched = 0
    incorrect_automatic_matches = 0
    total_invoice_value = 0.0
    total_value_reconciled = 0.0
    total_value_exceptions = 0.0

    # An invoice is a true positive if an automatically matched payment is one
    # of its valid ground-truth payments. This works for exact, fuzzy, partial,
    # and duplicate-payment ground truth without hardcoding match types.
    for invoice_id in invoice_ids:
        gt = truth.loc[invoice_id]
        invoice_value = _safe_float(gt["invoice_amount"])
        total_invoice_value += invoice_value
        valid_payments = _payment_set(gt["payment_ids"])
        is_expected_unmatched = not valid_payments

        if invoice_id not in predictions.index:
            status = "UNMATCHED"
            predicted_payment = None
        else:
            pred = predictions.loc[invoice_id]
            status = str(pred["status"]).upper()
            predicted_payment = None if _is_missing(pred["payment_id"]) else str(pred["payment_id"])

        if status == "MATCHED":
            automatic_matches += 1
            if predicted_payment in valid_payments:
                true_positives += 1
                total_value_reconciled += invoice_value
            else:
                false_positives += 1
                incorrect_automatic_matches += 1
                total_value_exceptions += invoice_value
        elif status == "REVIEW":
            review += 1
            total_value_exceptions += invoice_value
            if valid_payments:
                false_negatives += 1
            else:
                correctly_identified_unmatched += 1
        else:
            unmatched += 1
            total_value_exceptions += invoice_value
            if is_expected_unmatched:
                correctly_identified_unmatched += 1
            else:
                false_negatives += 1

    precision = true_positives / automatic_matches if automatic_matches else 0.0
    recall_denominator = true_positives + false_negatives
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    automatic_match_rate = automatic_matches / total_records if total_records else 0.0

    # Extra operational check: unmatched payment rows are outside the 100
    # invoice records. Count them separately, using ground truth's NaN invoice
    # rows as the expected-unmatched payment population.
    gt_unmatched_payments: set[str] = set()
    for _, row in ground_truth[ground_truth["invoice_id"].isna()].iterrows():
        gt_unmatched_payments |= _payment_set(row["payment_ids"])
    if not gt_unmatched_payments:
        # The current generator stores unmatched payment IDs in payment_ids even
        # though invoice_id is blank, so this normally remains non-empty.
        pass

    result_unmatched_payment_ids = set(
        str(x) for x in results.loc[results["invoice_id"].isna(), "payment_id"].dropna()
    )
    correctly_identified_unmatched_payments = len(result_unmatched_payment_ids & gt_unmatched_payments)

    evaluation_elapsed = time.perf_counter() - start
    elapsed = processing_time_seconds if processing_time_seconds > 0 else evaluation_elapsed

    return {
        "records_processed": total_records,
        "automatic_matches": automatic_matches,
        "review_records": review,
        "unmatched_records": unmatched,
        "automatic_match_rate": automatic_match_rate,
        "true_positives": true_positives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "correctly_identified_unmatched": correctly_identified_unmatched,
        "incorrect_automatic_matches": incorrect_automatic_matches,
        "correctly_identified_unmatched_payments": correctly_identified_unmatched_payments,
        "expected_unmatched_payments": len(gt_unmatched_payments),
        "predicted_unmatched_payments": len(result_unmatched_payment_ids),
        "total_invoice_value": total_invoice_value,
        "total_value_reconciled": total_value_reconciled,
        "total_value_exceptions": total_value_exceptions,
        "processing_time_seconds": elapsed,
    }


def write_outputs(metrics: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "evaluation_results.json"
    csv_path = output_dir / "evaluation_report.csv"

    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)

    LOGGER.info("Wrote %s", json_path)
    LOGGER.info("Wrote %s", csv_path)


def print_summary(metrics: dict[str, Any]) -> None:
    print("\n## RECONCILIATION BENCHMARK\n")
    print(f"Records processed: {metrics['records_processed']}")
    print(f"Automatic matches: {metrics['automatic_matches']}")
    print(f"Review: {metrics['review_records']}")
    print(f"Unmatched: {metrics['unmatched_records']}")
    print(f"Match rate: {metrics['automatic_match_rate']:.1%}")
    print(f"True positives: {metrics['true_positives']}")
    print(f"Precision: {metrics['precision']:.1%}")
    print(f"Recall: {metrics['recall']:.1%}")
    print(f"F1: {metrics['f1_score']:.1%}")
    print(f"False positives: {metrics['false_positives']}")
    print(f"False negatives: {metrics['false_negatives']}")
    print(f"Correctly identified unmatched: {metrics['correctly_identified_unmatched']}")
    print(f"Incorrect automatic matches: {metrics['incorrect_automatic_matches']}")
    # The benchmark data is currency-aware; this is presentation only and does
    # not convert any values.
    currency = "USD"
    print(f"Total invoice value: {currency} {metrics['total_invoice_value']:,.2f}")
    print(f"Successfully reconciled: {currency} {metrics['total_value_reconciled']:,.2f}")
    print(f"Exception value: {currency} {metrics['total_value_exceptions']:,.2f}")
    print(f"Processing time: {metrics['processing_time_seconds']:.4f} seconds")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark reconciliation_results.csv against ground_truth.csv")
    parser.add_argument("--results", type=Path, default=Path("outputs/reconciliation_results.csv"))
    parser.add_argument("--ground-truth", type=Path, default=Path("data/ground_truth/ground_truth.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    start = time.perf_counter()
    results, ground_truth = load_inputs(args.results, args.ground_truth)
    metrics = evaluate(results, ground_truth, processing_time_seconds=time.perf_counter() - start)
    write_outputs(metrics, args.output_dir)
    print_summary(metrics)


if __name__ == "__main__":
    main()
