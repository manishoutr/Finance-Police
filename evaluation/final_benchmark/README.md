# Final Held-Out Benchmark

This folder is the canonical final evaluation artifact for the AI Finance Controller.

## What is included

- `data/raw/` — freshly generated synthetic invoices, payments, and settlements
- `data/ground_truth/` — ground-truth labels used only for evaluation
- `outputs/reconciliation_results.csv` — reconciliation engine output
- `outputs/metrics.json` — machine-readable benchmark metrics
- `outputs/metrics.csv` — tabular benchmark metrics

## Reproduce

From the project root:

```bash
python scripts/run_held_out_benchmark.py
```

The runner regenerates this directory using seed `20260827` and does not modify the reconciliation logic, confidence scoring, thresholds, exception rules, ground truth formulas, or evaluation formulas.

## Interpretation

The benchmark is synthetic and held out from the main project workflow. It is not a claim of production accuracy. The reconciliation engine deliberately prioritizes precision, sending uncertain records to REVIEW or UNMATCHED rather than forcing automatic matches.
