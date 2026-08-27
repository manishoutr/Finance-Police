#!/usr/bin/env python3
"""
Synthetic Finance Reconciliation Data Generator
=================================================

Generates a realistic-but-fake dataset for testing invoice <-> payment
reconciliation logic:

    invoices.csv      - 100 invoices
    payments.csv      - 125 payment transactions (100+)
    settlements.csv   - one settlement record per payment
    ground_truth.csv  - the correct invoice-to-payment mapping / match type

Reconciliation edge cases are deliberately injected:
    - exact matches               (name, amount, date all line up)
    - fuzzy customer-name variants (nicknames, typos, reordered, abbreviations)
    - partial payments            (one or two payments that don't cover the full amount)
    - duplicate payments          (invoice paid twice in full, in error)
    - date-mismatched payments    (correct amount/name, but paid very late)
    - missing payments            (invoice never gets paid)
    - unmatched payments          (payments with no corresponding invoice at all)

Run:
    pip install pandas faker
    python generate_reconciliation_data.py

Everything is driven off SEED, so re-running produces identical output.
"""

import random
import string
import pandas as pd
from datetime import timedelta
from faker import Faker

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
NUM_INVOICES = 100
NUM_CUSTOMERS = 55           # some customers will have multiple invoices
CURRENCY = "USD"

BUCKET_SIZES = {
    "exact_match":     50,   # 1 payment, exact name/amount, date within a few days
    "fuzzy_name":      15,   # 1 payment, exact amount, name spelled/formatted differently
    "partial_payment": 10,   # 2 payments that together are LESS than the invoice amount
    "duplicate_payment": 10, # 2 payments, both for the full invoice amount
    "date_mismatch":   10,   # 1 payment, correct amount/name, paid very late
    "missing_payment":  5,   # 0 payments
}
assert sum(BUCKET_SIZES.values()) == NUM_INVOICES

NUM_UNMATCHED_PAYMENTS = 10  # extra payments with no invoice at all

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def random_amount(low=50.0, high=9500.0):
    return round(random.uniform(low, high), 2)


def make_invoice_id(i):
    return f"INV-{2024 + (i // 400)}-{i:05d}"


def make_payment_id(i):
    return f"PMT-{100000 + i}"


def make_settlement_id(i):
    return f"STL-{100000 + i}"


def fuzz_name(name):
    """Return a plausible, imperfect variant of a customer's name."""
    parts = name.split()
    style = random.choice(
        ["nickname", "initials", "reorder", "typo", "suffix", "case",
         "abbreviation", "missing_middle"]
    )

    nicknames = {
        "Robert": "Rob", "Richard": "Rick", "William": "Bill", "James": "Jim",
        "Michael": "Mike", "Elizabeth": "Liz", "Katherine": "Kate",
        "Jennifer": "Jen", "Christopher": "Chris", "Alexander": "Alex",
        "Nicholas": "Nick", "Matthew": "Matt", "Daniel": "Dan",
        "Samantha": "Sam", "Benjamin": "Ben", "Joseph": "Joe",
        "Anthony": "Tony", "Patricia": "Pat", "Margaret": "Meg",
        "Jonathan": "Jon",
    }

    if style == "nickname" and parts[0] in nicknames:
        parts[0] = nicknames[parts[0]]
        return " ".join(parts)

    if style == "initials" and len(parts) >= 2:
        return f"{parts[0][0]}. {' '.join(parts[1:])}"

    if style == "reorder" and len(parts) >= 2:
        return f"{parts[-1]}, {' '.join(parts[:-1])}"

    if style == "typo" and len(name) > 4:
        pos = random.randint(1, len(name) - 2)
        chars = list(name)
        # swap two adjacent characters
        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        return "".join(chars)

    if style == "suffix":
        suffix = random.choice(["Jr.", "Sr.", "II", "LLC", "& Co."])
        return f"{name} {suffix}"

    if style == "case":
        return name.upper() if random.random() < 0.5 else name.lower()

    if style == "abbreviation" and len(parts) >= 2:
        abbr = {"Company": "Co.", "Corporation": "Corp.", "Incorporated": "Inc.",
                "Limited": "Ltd.", "Enterprises": "Ent."}
        for full, short in abbr.items():
            if full in name:
                return name.replace(full, short)
        return f"{parts[0]} {parts[-1][0]}."

    if style == "missing_middle" and len(parts) >= 3:
        return f"{parts[0]} {parts[-1]}"

    # fallback: just return uppercase-first-letter variant
    return name.replace(" ", "  ")  # double space quirk


def random_business_name():
    """~30% of customers are companies rather than individuals."""
    if random.random() < 0.30:
        return fake.company()
    return fake.name()


# --------------------------------------------------------------------------
# 1. Customers
# --------------------------------------------------------------------------
customers = [random_business_name() for _ in range(NUM_CUSTOMERS)]

# --------------------------------------------------------------------------
# 2. Invoices
# --------------------------------------------------------------------------
invoice_rows = []
invoice_bucket = {}  # invoice_id -> bucket name

# Build a shuffled list of bucket labels, one per invoice
bucket_labels = []
for bucket, count in BUCKET_SIZES.items():
    bucket_labels.extend([bucket] * count)
random.shuffle(bucket_labels)

for i in range(NUM_INVOICES):
    invoice_id = make_invoice_id(i)
    customer = random.choice(customers)
    invoice_date = fake.date_between(start_date="-1y", end_date="-30d")
    due_date = invoice_date + timedelta(days=random.choice([15, 30, 45, 60]))
    amount = random_amount()
    status_bucket = bucket_labels[i]
    invoice_bucket[invoice_id] = status_bucket

    invoice_rows.append({
        "invoice_id": invoice_id,
        "customer_name": customer,
        "invoice_date": invoice_date,
        "due_date": due_date,
        "amount": amount,
        "currency": CURRENCY,
        "description": fake.bs().capitalize(),
    })

invoices_df = pd.DataFrame(invoice_rows)

# --------------------------------------------------------------------------
# 3. Payments (+ ground truth as we go)
# --------------------------------------------------------------------------
payment_rows = []
ground_truth_rows = []
payment_counter = 0
payment_methods = ["ACH", "Wire", "Credit Card", "Check", "PayPal"]


def add_payment(customer_name, payment_date, amount, method=None):
    """Append a payment row and return its payment_id."""
    global payment_counter
    pid = make_payment_id(payment_counter)
    payment_counter += 1
    payment_rows.append({
        "payment_id": pid,
        "payer_name": customer_name,
        "payment_date": payment_date,
        "amount": round(amount, 2),
        "currency": CURRENCY,
        "method": method or random.choice(payment_methods),
        "reference_note": fake.sentence(nb_words=4).rstrip("."),
    })
    return pid


for row in invoice_rows:
    inv_id = row["invoice_id"]
    customer = row["customer_name"]
    amount = row["amount"]
    invoice_date = row["invoice_date"]
    due_date = row["due_date"]
    bucket = invoice_bucket[inv_id]

    if bucket == "exact_match":
        pay_date = due_date - timedelta(days=random.randint(-2, 3))
        pid = add_payment(customer, pay_date, amount)
        ground_truth_rows.append({
            "invoice_id": inv_id, "payment_ids": pid, "match_type": "exact_match",
            "invoice_amount": amount, "total_paid": amount,
            "notes": "Name, amount, and date all align closely.",
        })

    elif bucket == "fuzzy_name":
        pay_date = due_date - timedelta(days=random.randint(-2, 3))
        variant_name = fuzz_name(customer)
        pid = add_payment(variant_name, pay_date, amount)
        ground_truth_rows.append({
            "invoice_id": inv_id, "payment_ids": pid, "match_type": "fuzzy_name",
            "invoice_amount": amount, "total_paid": amount,
            "notes": f"Payer name '{variant_name}' is a variant of '{customer}'.",
        })

    elif bucket == "partial_payment":
        # two installments that together fall short of the full amount
        shortfall_pct = random.uniform(0.10, 0.35)
        total_to_pay = round(amount * (1 - shortfall_pct), 2)
        first = round(total_to_pay * random.uniform(0.4, 0.6), 2)
        second = round(total_to_pay - first, 2)
        pay_date_1 = due_date - timedelta(days=random.randint(0, 10))
        pay_date_2 = pay_date_1 + timedelta(days=random.randint(5, 20))
        pid1 = add_payment(customer, pay_date_1, first)
        pid2 = add_payment(customer, pay_date_2, second)
        ground_truth_rows.append({
            "invoice_id": inv_id, "payment_ids": f"{pid1};{pid2}",
            "match_type": "partial_payment",
            "invoice_amount": amount, "total_paid": round(first + second, 2),
            "notes": "Two installments received; balance remains outstanding.",
        })

    elif bucket == "duplicate_payment":
        pay_date_1 = due_date - timedelta(days=random.randint(-2, 5))
        pay_date_2 = pay_date_1 + timedelta(days=random.randint(1, 6))
        pid1 = add_payment(customer, pay_date_1, amount)
        pid2 = add_payment(customer, pay_date_2, amount)
        ground_truth_rows.append({
            "invoice_id": inv_id, "payment_ids": f"{pid1};{pid2}",
            "match_type": "duplicate_payment",
            "invoice_amount": amount, "total_paid": round(amount * 2, 2),
            "notes": "Invoice paid in full twice; second payment is a duplicate/refund candidate.",
        })

    elif bucket == "date_mismatch":
        pay_date = due_date + timedelta(days=random.randint(35, 75))
        pid = add_payment(customer, pay_date, amount)
        ground_truth_rows.append({
            "invoice_id": inv_id, "payment_ids": pid, "match_type": "date_mismatch",
            "invoice_amount": amount, "total_paid": amount,
            "notes": f"Payment arrived {(pay_date - due_date).days} days after the due date.",
        })

    elif bucket == "missing_payment":
        ground_truth_rows.append({
            "invoice_id": inv_id, "payment_ids": "", "match_type": "missing_payment",
            "invoice_amount": amount, "total_paid": 0.0,
            "notes": "No payment received for this invoice.",
        })

# --- unmatched payments: money in, but no invoice behind it ------------
for _ in range(NUM_UNMATCHED_PAYMENTS):
    stray_customer = random.choice(
        customers + [fake.name(), fake.company()]  # sometimes a totally new payer
    )
    stray_date = fake.date_between(start_date="-11m", end_date="-1d")
    stray_amount = random_amount(20, 4000)
    pid = add_payment(stray_customer, stray_date, stray_amount,
                       method=random.choice(payment_methods))
    ground_truth_rows.append({
        "invoice_id": "", "payment_ids": pid, "match_type": "unmatched_payment",
        "invoice_amount": None, "total_paid": stray_amount,
        "notes": "Payment has no corresponding invoice (overpayment, stray transfer, "
                 "or invoice missing from the system).",
    })

payments_df = pd.DataFrame(payment_rows)
ground_truth_df = pd.DataFrame(ground_truth_rows)

# --------------------------------------------------------------------------
# 4. Settlements — one row per payment transaction
# --------------------------------------------------------------------------
settlement_rows = []
for i, prow in enumerate(payment_rows):
    settlement_id = make_settlement_id(i)
    lag_days = random.choice([0, 1, 1, 2, 2, 3, 5])
    settlement_date = prow["payment_date"] + timedelta(days=lag_days)

    roll = random.random()
    if roll < 0.04:
        status = "failed"
        settled_amount = 0.0
        fee = 0.0
    elif roll < 0.08:
        status = "pending"
        settled_amount = None
        fee = None
    else:
        status = "settled"
        fee_rate = {"Credit Card": 0.029, "PayPal": 0.034, "Wire": 0.0,
                    "ACH": 0.008, "Check": 0.0}.get(prow["method"], 0.01)
        fee = round(prow["amount"] * fee_rate, 2)
        settled_amount = round(prow["amount"] - fee, 2)

    settlement_rows.append({
        "settlement_id": settlement_id,
        "payment_id": prow["payment_id"],
        "settlement_date": settlement_date,
        "gross_amount": prow["amount"],
        "fee": fee,
        "settled_amount": settled_amount,
        "status": status,
    })

settlements_df = pd.DataFrame(settlement_rows)

# --------------------------------------------------------------------------
# 5. Shuffle output row order (so it doesn't read "bucket by bucket")
#    but keep everything reproducible since we already seeded RNGs.
# --------------------------------------------------------------------------
invoices_df = invoices_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
payments_df = payments_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
settlements_df = settlements_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# --------------------------------------------------------------------------
# 6. Write CSVs
# --------------------------------------------------------------------------
invoices_df.to_csv("invoices.csv", index=False)
payments_df.to_csv("payments.csv", index=False)
settlements_df.to_csv("settlements.csv", index=False)
ground_truth_df.to_csv("ground_truth.csv", index=False)

print(f"invoices.csv       -> {len(invoices_df)} rows")
print(f"payments.csv       -> {len(payments_df)} rows")
print(f"settlements.csv    -> {len(settlements_df)} rows")
print(f"ground_truth.csv   -> {len(ground_truth_df)} rows")
print("\nMatch-type breakdown:")
print(ground_truth_df["match_type"].value_counts().to_string())
