# AI Finance Controller

**Razorpay Buildathon | Track 04: Run the books and the cash position**

AI Finance Controller is a finance-operations prototype that reconciles invoices, payments, and settlements from a batch of synthetic records. It produces deterministic match decisions, measures those decisions against held-out ground truth, surfaces unresolved cases for review, and rolls the results into a simple cash-position view.

The project is intentionally designed around a simple principle: **financial decisions should be reproducible and auditable**. The reconciliation engine and scoring logic are deterministic. The optional language-model layer is read-only and is used to query and explain results, not to change records or approve financial actions.

## What the project covers

- Batch reconciliation across invoices, payments, and settlements
- Normalization of customer names, amounts, dates, and references
- Weighted matching with confidence scoring
- `MATCHED`, `REVIEW`, and `UNMATCHED` outcomes
- Exception classification for cases that cannot be resolved automatically
- Human-review workflow with an audit log
- Cash-position summary using confirmed, pending, unresolved, and expected states
- Held-out benchmark with precision, recall, F1, match rate, and error counts
- Read-only finance Q&A interface with explicit guardrails

## How it works

```text
Raw finance records
        |
        v
Normalization
        |
        v
Deterministic reconciliation + confidence scoring
        |
        +--------------------+
        |                    |
        v                    v
    MATCHED          REVIEW / UNMATCHED
        |                    |
        v                    v
Evaluation          Exception classification
        |                    |
        +---------+----------+
                  |
                  v
             Human review
                  |
                  v
             Cash position
                  |
                  v
      Read-only results assistant
```

## Final held-out benchmark

The final evaluation uses a synthetic dataset generated separately from the development data. The benchmark is intended to show how the controller behaves on previously unseen records, including its trade-off between automatic resolution and cautious escalation.

| Metric | Result |
|---|---:|
| Invoices | 120 |
| Payments | 162 |
| Settlements | 162 |
| Automatic matches | 77 |
| Automatic match rate | 64.17% |
| Precision | 100.00% |
| Recall | 70.00% |
| F1 score | 82.35% |
| False positives | 0 |
| REVIEW cases | 35 |
| UNMATCHED cases | 8 |

The reconciliation strategy favors precision. When the available evidence is not strong enough for an automatic match, the record is routed to `REVIEW` or `UNMATCHED` instead of forcing a decision.

For the full methodology and results, see `evaluation/FINAL_BENCHMARK.md`. The canonical benchmark artifacts are stored in `evaluation/final_benchmark/`.

## Running the project

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the core pipeline:

```bash
python src/reconciler.py
python src/evaluate_reconciliation.py
python src/classify_exceptions.py
```

Run the held-out benchmark:

```bash
python scripts/run_held_out_benchmark.py
```

Run the application:

```bash
streamlit run src/app.py
```

Run the test suite:

```bash
pytest tests/
```

The application can run in demo mode without an API key. When `OPENAI_API_KEY` is configured, the optional assistant can answer questions over the generated results. It remains read-only and does not have tools for editing records, approving payments, or deleting data.

## Project structure

```text
src/                         Core application and finance-ops logic
  reconciler.py              Deterministic reconciliation engine
  financial_normalizer.py    Record normalization
  evaluate_reconciliation.py Benchmark metrics
  classify_exceptions.py     Exception categorization
  human_review.py            Review workflow and audit trail
  cash_position.py           Cash-position rollup
  finance_assistant.py       Read-only assistant layer
  app.py                     Streamlit interface

scripts/                     Reproducible project runners
  run_held_out_benchmark.py  Regenerates the final benchmark
  run_system_tests.py        Runs the broader system test workflow
  run_ai_tests.py            Runs assistant-focused tests

tests/                       Unit, integration, safety, and UI tests

data/                        Development synthetic dataset and ground truth

evaluation/                  Final held-out benchmark and methodology

evaluation/final_benchmark/  Canonical benchmark data and outputs

outputs/                     Latest pipeline and validation reports
```

## Safety and control boundaries

The project treats the language model as an interface layer rather than a financial decision-maker:

- Reconciliation outcomes and confidence scores are produced by deterministic code.
- The assistant is read-only and has no tool for editing, approving, rejecting, or deleting financial records.
- A preflight guard rejects recognized write and approval requests before they reach the assistant layer.
- Benchmark ground truth is used for evaluation, not as an operational input to the reconciliation engine.
- Cash-position labels are a synthetic model for the prototype and should not be interpreted as a production bank balance or accounting statement.

Related validation artifacts:

- `tests/test_ai_safety.py`
- `outputs/ai_safety_validation_report.md`
- `outputs/system_test_report.md`

## Notes

This is a prototype built for a synthetic-data hackathon setting. The benchmark results demonstrate the behavior of this implementation on the included held-out synthetic data and should not be treated as production performance or accounting advice.
