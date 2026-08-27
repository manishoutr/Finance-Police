#!/usr/bin/env python3
"""Generate and benchmark a completely fresh held-out reconciliation dataset.

This script never changes the existing reconciliation thresholds or algorithm.
It writes only under evaluation/final_benchmark and calls the existing reconcile()
function unchanged.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
from faker import Faker

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate_reconciliation import evaluate, write_outputs  # noqa: E402
from reconciler import reconcile  # noqa: E402

SEED = 20260827
OUT = ROOT / "evaluation"
DATA = OUT / "final_benchmark"
RAW = DATA / "data" / "raw"
GT_DIR = DATA / "data" / "ground_truth"
RESULT_DIR = DATA / "outputs"


def money(x: float) -> float:
    return round(float(x), 2)


def make_name_variation(name: str, mode: str) -> str:
    if mode == "case":
        return name.upper()
    if mode == "punctuation":
        return name.replace(" & ", " and ")
    if mode == "suffix":
        if name.endswith(" Private Limited"):
            return name.replace(" Private Limited", " Pvt Ltd")
        if name.endswith(" Limited"):
            return name.replace(" Limited", " Ltd")
        return name + " Pvt Ltd"
    if mode == "spacing":
        return "  ".join(name.split())
    if mode == "token":
        parts = name.split()
        return " ".join(list(reversed(parts[:2])) + parts[2:]) if len(parts) > 2 else name
    return name


def generate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fake = Faker("en_US")
    fake.seed_instance(SEED)
    rng = __import__("random").Random(SEED)

    customers = []
    seen = set()
    while len(customers) < 80:
        suffix = rng.choice(["Private Limited", "Limited", "LLP", "Inc"])
        base = fake.company().replace(",", "")
        name = f"{base} {suffix}"
        if name not in seen:
            seen.add(name)
            customers.append(name)

    invoices = []
    payments = []
    settlements = []
    gt_rows = []
    payment_counter = 100000
    settlement_counter = 200000

    # 120 invoices, with deliberate held-out scenario mix.
    scenarios = (
        ["exact"] * 28
        + ["fuzzy"] * 18
        + ["date"] * 12
        + ["amount"] * 12
        + ["partial"] * 10
        + ["missing"] * 10
        + ["duplicate"] * 10
        + ["ambiguous"] * 10
        + ["missing_ref"] * 10
    )
    rng.shuffle(scenarios)

    base_date = pd.Timestamp("2026-01-05")

    def new_payment(customer: str, date: pd.Timestamp, amount: float, reference: str, method: str = "Bank Transfer") -> str:
        nonlocal payment_counter, settlement_counter
        pid = f"HOP-{payment_counter}"
        sid = f"HST-{settlement_counter}"
        payment_counter += 1
        settlement_counter += 1
        payments.append({
            "payment_id": pid,
            "payer_name": customer,
            "payment_date": date.strftime("%Y-%m-%d"),
            "amount": money(amount),
            "currency": "USD",
            "method": method,
            "reference_note": reference,
        })
        fee = money(amount * 0.006)
        settlements.append({
            "settlement_id": sid,
            "payment_id": pid,
            "settlement_date": (date + pd.Timedelta(days=rng.randint(0, 2))).strftime("%Y-%m-%d"),
            "gross_amount": money(amount),
            "fee": fee,
            "settled_amount": money(amount - fee),
            "status": "settled",
        })
        return pid

    for i, scenario in enumerate(scenarios, start=1):
        invoice_id = f"HINV-{i:04d}"
        customer = customers[(i * 7) % len(customers)]
        invoice_date = base_date + pd.Timedelta(days=rng.randint(0, 150))
        due_date = invoice_date + pd.Timedelta(days=rng.choice([15, 30, 45]))
        amount = money(rng.uniform(250, 12000))
        description = f"Invoice {invoice_id} for professional services"
        invoices.append({
            "invoice_id": invoice_id,
            "customer_name": customer,
            "invoice_date": invoice_date.strftime("%Y-%m-%d"),
            "due_date": due_date.strftime("%Y-%m-%d"),
            "amount": amount,
            "currency": "USD",
            "description": description,
        })

        valid_ids: list[str] = []
        payment_date = due_date + pd.Timedelta(days=rng.randint(-2, 3))
        pay_customer = customer
        reference = invoice_id
        pay_amount = amount

        if scenario == "exact":
            valid_ids.append(new_payment(pay_customer, payment_date, pay_amount, reference))

        elif scenario == "fuzzy":
            pay_customer = make_name_variation(customer, rng.choice(["case", "suffix", "punctuation", "spacing", "token"]))
            valid_ids.append(new_payment(pay_customer, payment_date, pay_amount, reference))

        elif scenario == "date":
            payment_date = due_date + pd.Timedelta(days=rng.choice([5, 8, 12, 20, 28]))
            valid_ids.append(new_payment(pay_customer, payment_date, pay_amount, reference))

        elif scenario == "amount":
            pay_amount = money(amount * rng.choice([0.72, 0.85, 1.12, 1.25]))
            valid_ids.append(new_payment(pay_customer, payment_date, pay_amount, reference))

        elif scenario == "partial":
            first = money(amount * rng.choice([0.35, 0.45, 0.60]))
            second = money(amount - first)
            valid_ids.append(new_payment(pay_customer, payment_date, first, reference))
            valid_ids.append(new_payment(pay_customer, payment_date + pd.Timedelta(days=rng.randint(1, 5)), second, reference))

        elif scenario == "missing":
            # No payment created. This is the intended missing-payment case.
            valid_ids = []

        elif scenario == "duplicate":
            primary = new_payment(pay_customer, payment_date, pay_amount, reference)
            duplicate = new_payment(pay_customer, payment_date + pd.Timedelta(days=1), pay_amount, reference)
            # Either payment is economically attributable to the invoice; the second is a duplicate.
            valid_ids.extend([primary, duplicate])

        elif scenario == "ambiguous":
            # Two payments with nearly identical customer/amount/date but no invoice reference.
            # Ground truth selects the first; the second belongs to another synthetic invoice later.
            reference = ""
            valid_ids.append(new_payment(pay_customer, payment_date, pay_amount, reference))
            # distractor is slightly different and intentionally lacks reference
            distractor = new_payment(pay_customer, payment_date + pd.Timedelta(days=1), money(amount * 1.01), "")
            gt_rows.append({
                "invoice_id": pd.NA,
                "payment_ids": distractor,
                "invoice_amount": 0.0,
                "match_type": "unmatched_payment",
            })

        elif scenario == "missing_ref":
            reference = "BANK CREDIT"
            valid_ids.append(new_payment(pay_customer, payment_date, pay_amount, reference))

        gt_rows.append({
            "invoice_id": invoice_id,
            "payment_ids": ";".join(valid_ids),
            "invoice_amount": amount,
            "match_type": scenario,
        })

    # Add independent unmatched incoming payments, including several near-customer distractors.
    for j in range(22):
        customer = customers[rng.randrange(len(customers))]
        date = base_date + pd.Timedelta(days=rng.randint(0, 180))
        amount = money(rng.uniform(100, 7000))
        pid = new_payment(customer, date, amount, f"UNRELATED-{j+1:02d}")
        gt_rows.append({
            "invoice_id": pd.NA,
            "payment_ids": pid,
            "invoice_amount": 0.0,
            "match_type": "unmatched_payment",
        })

    return pd.DataFrame(invoices), pd.DataFrame(payments), pd.DataFrame(settlements), pd.DataFrame(gt_rows)


def main() -> None:
    # Fresh directory every run. Existing project outputs are untouched.
    if DATA.exists():
        shutil.rmtree(DATA)
    RAW.mkdir(parents=True)
    GT_DIR.mkdir(parents=True)
    RESULT_DIR.mkdir(parents=True)

    invoices, payments, settlements, gt = generate()
    invoices.to_csv(RAW / "invoices.csv", index=False)
    payments.to_csv(RAW / "payments.csv", index=False)
    settlements.to_csv(RAW / "settlements.csv", index=False)
    gt.to_csv(GT_DIR / "ground_truth_test.csv", index=False)

    # Verify the held-out invoice population and ID uniqueness before running.
    assert len(invoices) == 120
    assert invoices["invoice_id"].is_unique
    assert payments["payment_id"].is_unique
    assert len(payments) >= 100

    start = time.perf_counter()
    results = reconcile(invoices, payments, settlements)
    processing_time = time.perf_counter() - start
    result_path = RESULT_DIR / "reconciliation_results.csv"
    results.to_csv(result_path, index=False)

    metrics = evaluate(results, gt, processing_time_seconds=processing_time)
    metrics["benchmark"] = "FINAL HELD-OUT TEST"
    metrics["seed"] = SEED
    metrics["invoice_records"] = len(invoices)
    metrics["payment_records"] = len(payments)
    metrics["settlement_records"] = len(settlements)
    metrics["exception_rate"] = (
        (metrics["review_records"] + metrics["unmatched_records"]) / metrics["records_processed"]
        if metrics["records_processed"] else 0.0
    )

    (RESULT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(RESULT_DIR / "metrics.csv", index=False)

    print("\n## HELD-OUT BENCHMARK")
    print("FINAL HELD-OUT TEST")
    print(f"Records: {metrics['records_processed']}")
    print(f"Match rate: {metrics['automatic_match_rate']:.1%}")
    print(f"Precision: {metrics['precision']:.1%}")
    print(f"Recall: {metrics['recall']:.1%}")
    print(f"F1: {metrics['f1_score']:.1%}")
    print(f"False positives: {metrics['false_positives']}")
    print(f"False negatives: {metrics['false_negatives']}")
    print(f"Exception rate: {metrics['exception_rate']:.1%}")
    print(f"Exception value: ${metrics['total_value_exceptions']:,.2f}")
    print(f"Processing time: {metrics['processing_time_seconds']:.4f} seconds")
    print(f"\nHeld-out data: {DATA}")
    print(f"Results: {result_path}")
    print(f"Metrics: {RESULT_DIR / 'metrics.json'}")
    print(f"Report: {RESULT_DIR / 'metrics.csv'}")


if __name__ == "__main__":
    main()
