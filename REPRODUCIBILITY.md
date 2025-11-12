# Reproducibility Guide

## Quick Start

To reproduce all paper results:

```bash
python multi_seed_analysis.py --base-seed 1495790106
python plot_results.py
```

This runs:
- 1000 seeds per scenario
- 3 cluster sizes: Small (8 GPUs), Medium (32 GPUs), Large (64 GPUs)
- 3 load regimes: Light (0.6), Medium (0.9), Heavy (1.2)
- 6 schedulers: FCFS, EASY, Conservative, PriorityQoS, RunAI, TTCC
- **Total: 54,000 simulator runs** (~5–8 hours on modern CPU)

## Output Files

After running the analysis:

- `multi_seed_results.json` – Raw per-seed results
- `multi_seed_stats.csv` – Aggregated statistics with 95% CI
- `multi_seed_metadata.json` – Seed configuration metadata
- 4 figures (PNG, 600 dpi, publication-ready)
- `summary_across_all_scenarios.csv` – Aggregate metrics table

## Your Exact Seeds

- **Base seed**: 1495790106
- **Seeds per scenario**: 1000
- **Total scenarios**: 9 (3 clusters × 3 loads)
- **Seed range**: [1495790106, 1495791105]

## Customization

### Different seed count:
```bash
python multi_seed_analysis.py --base-seed 1495790106 --seeds 100
```

### Different base seed:
```bash
python multi_seed_analysis.py --base-seed 42
```

### Edit configuration:
Open `multi_seed_analysis.py` and modify:
- Cluster configs (line ~40)
- Load targets (line ~75)
- Workload parameters (line ~65–100)

## Scenario Details

| Scenario | Cluster | Load Regime | Description |
|----------|---------|------------|-------------|
| 1–3 | Small (8 GPUs) | Light / Medium / Heavy | 0.6 / 0.9 / 1.2 |
| 4–6 | Medium (32 GPUs) | Light / Medium / Heavy | 0.6 / 0.9 / 1.2 |
| 7–9 | Large (64 GPUs) | Light / Medium / Heavy | 0.6 / 0.9 / 1.2 |

## Workload Parameters

| Parameter | Value |
|-----------|-------|
| Arrival process | Poisson |
| Job demand | 1–4 GPUs (Small), 1–4 or 8 GPUs (Medium/Large; 5% are 8-GPU) |
| QoS class mix | Gold 20%, Silver 50%, Bronze 30% |
| Deadline multiplier | Gold 1.0, Silver 1.5, Bronze 3.0 |
| Simulation duration | 10,000 time units |

## Schedulers

| Scheduler | Configuration |
|-----------|---------------|
| FCFS | No backfilling |
| EASY | Head-of-queue reservation |
| Conservative | Full reservation |
| PriorityQoS | Kubernetes-style priority |
| RunAI | Quota/fair-share |
| TTCC | Matching-theoretic reallocation |

## Metrics

Three metrics per scheduler/scenario:

1. **Utilization** – Fraction of GPU-time used
2. **Average Utility** – Preference-aware satisfaction
3. **Bounded Slowdown (Tail)** – P95/P99 percentiles for responsiveness

All metrics use 95% confidence intervals (bootstrap, 10,000 replicates).

## Expected Results

Aggregate across all 9 scenarios:

| Scheduler | Utilization | Avg Utility | P95 Slow | P99 Slow |
|-----------|------------|-------------|----------|----------|
| FCFS | 0.761 | 1.057 | 39 | 77 |
| EASY | 0.777 | 1.175 | 19 | 39 |
| Conservative | 0.770 | 1.153 | 16 | 31 |
| PriorityQoS | 0.766 | 1.537 | 34 | 114 |
| RunAI | 0.764 | 1.275 | 37 | 87 |
| **TTCC** | **0.765** | **1.283** | **12** | **31** |

If your results match these (within ±2%), reproducibility is confirmed.

## Verification

After running the analysis, check:

- [ ] `multi_seed_results.json` exists (54,000 results: 1000 × 9 × 6)
- [ ] `multi_seed_metadata.json` shows base_seed=1495790106
- [ ] `multi_seed_stats.csv` has 54 rows (9 scenarios × 6 schedulers)
- [ ] All 4 figures generated
- [ ] `summary_across_all_scenarios.csv` matches expected results table

## Troubleshooting

**Slow execution (>8 hours):**
- Use fewer seeds: `python multi_seed_analysis.py --base-seed 1495790106 --seeds 100`

**Out of memory:**
- Reduce cluster sizes in `multi_seed_analysis.py`

**Different results than paper:**
- Verify you used `--base-seed 1495790106`
- Check Python version (3.8+) and NumPy version match
- Minor differences (±0.5%) are expected across platforms

**Figures don't render:**
- Ensure matplotlib installed: `pip install -e .`
- Check files exist: `ls multi_seed_*.json multi_seed_*.csv`

## Common Random Numbers

All schedulers tested on identical workloads per seed:
- Same job arrivals
- Same job deadlines
- Same job preferences

This ensures differences reflect **scheduler behavior**, not randomness, and reduces confidence interval width by 50–70%.

## Reproducibility Statement

This artifact adheres to:

1. **Deterministic Execution** – All randomness seeded with base seed 1495790106
2. **Complete Configuration** – All parameters documented in this guide and code
3. **Open Code** – All scheduler code provided, unmodified from publication
4. **Result Verification** – Raw results and metadata exported for verification
5. **CI Quantification** – 95% confidence intervals via bootstrap for all metrics

## For More Details

- See [README.md](README.md) for project overview
- Check `multi_seed_analysis.py` for exact parameter values

---

**Last Updated**: November 2025
