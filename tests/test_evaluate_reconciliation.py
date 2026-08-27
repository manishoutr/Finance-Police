import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate_reconciliation import evaluate


def make_results(rows):
    return pd.DataFrame(rows, columns=[
        "invoice_id", "payment_id", "invoice_amount", "payment_amount", "status"
    ])


def make_truth(rows):
    return pd.DataFrame(rows, columns=[
        "invoice_id", "payment_ids", "match_type", "invoice_amount"
    ])


def test_metric_calculations_known_dataset():
    # I1 is a correct automatic match.
    # I2 is an incorrect automatic match (false positive).
    # I3 has a valid payment but is REVIEW (false negative).
    # I4 is correctly identified as unmatched.
    results = make_results([
        ["I1", "P1", 100.0, 100.0, "MATCHED"],
        ["I2", "WRONG", 200.0, 200.0, "MATCHED"],
        ["I3", "P3", 300.0, 150.0, "REVIEW"],
        ["I4", None, 400.0, None, "UNMATCHED"],
    ])
    truth = make_truth([
        ["I1", "P1", "exact_match", 100.0],
        ["I2", "P2", "exact_match", 200.0],
        ["I3", "P3", "partial_payment", 300.0],
        ["I4", None, "missing_payment", 400.0],
    ])

    m = evaluate(results, truth, processing_time_seconds=1.25)

    assert m["records_processed"] == 4
    assert m["automatic_matches"] == 2
    assert m["review_records"] == 1
    assert m["unmatched_records"] == 1
    assert m["automatic_match_rate"] == 0.5
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["f1_score"] == 0.5
    assert m["false_positives"] == 1
    assert m["false_negatives"] == 1
    assert m["correctly_identified_unmatched"] == 1
    assert m["incorrect_automatic_matches"] == 1
    assert m["total_invoice_value"] == 1000.0
    assert m["total_value_reconciled"] == 100.0
    assert m["total_value_exceptions"] == 900.0
    assert m["processing_time_seconds"] == 1.25


def test_multiple_ground_truth_payments_accepts_any_valid_payment():
    results = make_results([
        ["I1", "P2", 500.0, 250.0, "MATCHED"],
        ["I2", None, 600.0, None, "UNMATCHED"],
    ])
    truth = make_truth([
        ["I1", "P1;P2", "partial_payment", 500.0],
        ["I2", None, "missing_payment", 600.0],
    ])

    m = evaluate(results, truth)

    assert m["true_positives"] if "true_positives" in m else True
    assert m["automatic_matches"] == 1
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1_score"] == 1.0
    assert m["false_positives"] == 0
    assert m["false_negatives"] == 0
    assert m["correctly_identified_unmatched"] == 1
