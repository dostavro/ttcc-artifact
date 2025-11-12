# Preference-Aware Scheduling for 6G Testbeds - Simulator and Artifacts
_Implementation of Top Trading Cycles and Chains (TTCC) for GPU/Edge Scheduling_


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains a discrete-event simulator implementing **Top Trading Cycles and Chains (TTCC)**, a matching-theoretic scheduler for preference-aware resource allocation in heterogeneous 6G testbeds. The simulator evaluates TTCC against six representative schedulers (FCFS, EASY, Conservative, PriorityQoS, RunAI) under synthetic workloads.

**Associated submission:**
Artifact corresponding to the paper *“Preference-Aware Scheduling for 6G Testbeds Using Matching Theory”* (WoNS 2026 submission).


## Key Features

- **Matching-Theoretic Allocation**: TTCC performs global, Pareto-efficient reallocations on each resource-release event (job completion).
- **QoS-Aware Preferences**: Jobs declare strict rankings over heterogeneous resources (edge vs. cloud GPUs) weighted by Quality-of-Service class (Gold, Silver, Bronze).
- **Utility Maximization**: Per-job utility combines deadline adherence, QoS compliance, and preference satisfaction.
- **Bounded Slowdown Metrics**: P95/P99 tail responsiveness for latency-sensitive workloads.
- **Event-Driven Simulation**: Discrete-event simulator with Common Random Numbers (CRN) for low-variance comparisons.
- **Reproducibility**: Multi-seed analysis with configurable base seeds, metadata export, and full artifact traceability.

## Publication Results

TTCC achieves (aggregate across all scenarios):
- **Utilization**: 0.765 (comparable to backfilling baselines)
- **Average Utility**: 1.283 (+20% over FCFS, +10-12% over EASY/Conservative)
- **P95 Slowdown**: Median 12 (best among all schedulers)
- **P99 Slowdown**: Median 31 (matches Conservative, −73% vs. Priority/QoS)

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for detailed instructions.

---

## Quick Start

### 1. Setup

```bash
# Clone the repository
git clone https://github.com/dostavro/ttcc-artifact.git
cd ttcc-artifact

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### 2. Run Reproducible Analysis (1000 seeds, fixed)

To reproduce the paper results exactly:

```bash
# Exact reproduction with paper seeds
python multi_seed_analysis.py --base-seed 1495790106
```

This runs across all 9 scenarios (3 cluster sizes × 3 load regimes) with:
- **1000 seeds** per scenario (Common Random Numbers for low-variance comparisons)
- **6 schedulers**: FCFS, EASY, Conservative, PriorityQoS, RunAI, TTCC
- **Total**: 54,000 simulator runs (~5–8 hours on modern CPU)

**Output files:**
- `multi_seed_results.json`: Raw per-seed results (all schedulers, all scenarios)
- `multi_seed_stats.csv`: Pre-computed statistics (mean, 95% CI, median)
- `multi_seed_metadata.json`: Seed metadata for reproducibility

### 3. Generate Publication-Ready Figures

```bash
python plot_results.py
```

**Output figures:**
- `fig1_utilization.png` – Cluster utilization versus offered load (averaged across cluster sizes)
- `fig2_utility.png` – Average preference-aware utility by scheduler and load
- `fig3_tail_slowdown.png` – P95/P99 bounded slowdown by scheduler and load (lower is better)
- `fig4_utility_tail_tradeoff.png` – Mean utility vs. mean P99 slowdown trade-off (scatter)

**Output table:**
- `summary_across_all_scenarios.csv` – Aggregate metrics (all 9 scenarios) with 95% CIs

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for detailed reproduction instructions.

---

## Project Structure

```
ttcc-artifact/
├── README.md                    # This file
├── REPRODUCIBILITY.md           # Detailed reproduction & seed control guide
├── LICENSE                      # MIT License
├── pyproject.toml               # Python package dependencies
│
├── multi_seed_analysis.py       # 🚀 Main entry point: 1000-seed analysis with CRN
├── plot_results.py              # Publication-ready figure generation
│
├── simulator.py                 # Discrete-event simulator (core loop)
├── cluster.py                   # Cluster model (edge/cloud GPU inventory)
├── workload.py                  # Synthetic job generation (Poisson arrivals, QoS mix)
├── jobs.py                      # Job class with utility calculation
├── metrics.py                   # Performance metrics (utilization, utility, bounded slowdown)
│
├── schedulers/
│   ├── base.py                  # Base Scheduler class
│   ├── fcfs.py                  # FCFS (no backfilling)
│   ├── easy.py                  # EASY backfilling
│   ├── conservative.py          # Conservative backfilling
│   ├── priority.py              # Priority/QoS (Kubernetes-style)
│   ├── runai.py                 # RunAI (quota/fair-share)
│   └── ttcc.py                  # 🎯 TTCC (matching-theoretic reallocation)
│
├── .gitignore                   # Exclude cache, results, venv
└── multi_seed_results.json      # [Generated] Raw results
└── multi_seed_stats.csv         # [Generated] Pre-computed statistics
└── multi_seed_metadata.json     # [Generated] Seed metadata
```

---

## System Model

### Jobs & Resources
- **Jobs**: Arrive Poisson; demand 1–4 GPUs (Small cluster) or 1–4, 8 GPUs (Medium/Large, 5% are 8-GPU large training tasks)
- **Resources**: Edge (low-latency) and Cloud (high-capacity) GPUs; indivisible and exclusive
- **QoS Classes**: Gold (20%), Silver (50%), Bronze (30%) with distinct priorities and deadline tightness

### Per-Job Utility

$$U_j = U_{\mathrm{base}}(j) \cdot S_j$$

where:
- **Base utility** (deadline-aware): $U_{\mathrm{base}}(j) = w_{q_j} \exp(-T_j / \beta_{q_j})$
  - $T_j = \max(0, c_j - D_j)$ = tardiness
  - QoS weights: Gold 1.0, Silver 0.75, Bronze 0.5
  - Decay scales: Gold 0.5, Silver 1.0, Bronze 2.0
- **Preference satisfaction**: $S_j = \frac{1}{d_j} \sum_{k=1}^{d_j} \gamma^{\rho_j(r_{j,k})-1}$
  - $\rho_j(r)$ = job's rank of resource $r$ (1=top choice)
  - $\gamma = 0.9$ = geometric decay for lower-ranked resources

### Evaluation Metrics

1. **Utilization**: $\frac{\sum_{j \in J_{\text{fin}}} d_j t_j}{M \cdot T_{\text{MS}}}$ — Fraction of total resource capacity actively used
2. **Average Utility**: $\bar{U} = \frac{1}{|J_{\text{fin}}|} \sum_{j \in J_{\text{fin}}} U_j$ — System-wide preference-aware performance
3. **Bounded Slowdown (Tail)**: $\text{BSLD}_j = \frac{R_j}{\max(t_j, \tau)}$ with P95/P99 percentiles — High-percentile responsiveness

---

## TTCC Algorithm

### Overview

TTCC treats each **resource-release event** (job completion) as a trigger for global reallocation:

1. **Graph Construction**: Build directed graph of jobs and resources
   - Resources point to highest-priority (QoS-weighted) compatible jobs
   - Waiting jobs point to most-preferred feasible resource
   - Running jobs point to currently-held resource (optional migration)

2. **Cycle Resolution**: Find and execute all cycles simultaneously
   - Each job receives the resource it points to

3. **Chain Resolution**: Free resources initiate chains
   - Resource → Job → Resource → Job → ...
   - Chain terminates when reaching a waiting job or revisiting a node

4. **Reallocation**: Shift assignments along all paths for Pareto improvement

**Complexity**: $O(n^2)$ per event (tractable for hundreds of jobs).

### Example

Initial:
- Edge GPU (r₁) ← Bronze job B (prefers cloud)
- Cloud GPU (r₂) ← Silver job A
- Waiting: Gold job C (prefers edge)

When A completes, TTCC constructs chain: r₂ → B → r₁ → C
- B migrates to cloud (preferred)
- C starts on edge (preferred)
- **Result**: Pareto improvement for both jobs



---

## File Descriptions

### Core Simulation

| File | Purpose |
|------|---------|
| `simulator.py` | Discrete-event kernel; event loop, clock advancement, scheduler invocation |
| `cluster.py` | GPU cluster model (edge/cloud inventory, allocation tracking) |
| `workload.py` | Synthetic workload generation (Poisson arrivals, QoS mix, demand distribution) |
| `jobs.py` | Job class with utility calculation and preference scoring |
| `metrics.py` | Utilization, average utility, bounded slowdown computation |

### Schedulers

| File | Scheduler | Key Feature |
|------|-----------|-------------|
| `base.py` | Base class | Common interface for all schedulers |
| `fcfs.py` | FCFS | First-come, first-served (no backfilling) |
| `easy.py` | EASY | Backfilling with head-of-queue reservation |
| `conservative.py` | Conservative | Backfilling with full reservation |
| `priority.py` | Priority/QoS | Kubernetes-style priority + QoS classes |
| `runai.py` | RunAI | Quota/fair-share allocation |
| `ttcc.py` | **TTCC** | Matching-theoretic, preference-aware reallocation |

### Analysis & Plotting

| File | Purpose |
|------|---------|
| `multi_seed_analysis.py` | 🚀 **Main entry point**: 1000-seed CRN analysis, statistics, CI computation |
| `plot_results.py` | Generate 6 publication-ready figures + aggregate table |

---

## Key Results

### Performance Summary (Table, Aggregate Across All Scenarios)

| Scheduler | Utilization | Avg Utility | P95 Slow | P99 Slow |
|-----------|------------|-------------|----------|----------|
| FCFS | 0.761 | 1.057 | 39 | 77 |
| EASY | **0.777** | 1.175 | 19 | 39 |
| Conservative | 0.770 | 1.153 | 16 | **31** |
| PriorityQoS | 0.766 | **1.537** | 34 | 114 |
| RunAI | 0.764 | 1.275 | 37 | 87 |
| **TTCC** | 0.765 | 1.283 | **12** | **31** |

**Key observations:**
- TTCC achieves **utilization** comparable to backfilling baselines (0.765 vs. 0.777 for EASY)
- TTCC improves **utility** by ~20% over FCFS and ~10-12% over EASY/Conservative
- TTCC achieves **best P95 slowdown** (12) among all schedulers
- TTCC matches Conservative on **P99 slowdown** (31) while delivering higher utility

---

## Citation

This artifact implements research on Top Trading Cycles and Chains (TTCC) for preference-aware GPU scheduling in 6G testbeds.

**For now, please cite this GitHub repository:**
[https://github.com/dostavro/ttcc-artifact](https://github.com/dostavro/ttcc-artifact)

Once the paper is published, we will provide a formal BibTeX citation.

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit changes with clear messages
4. Submit a pull request

For bugs or questions, open an issue.

---

## License

This project is licensed under the **MIT License** – see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Matching theory foundations: Shapley & Scarf (TTC), Roth et al. (TTCC, kidney exchange)
- Inspiration from HPC schedulers (EASY, Conservative), Kubernetes QoS, and RunAI quota systems
- Simulation built with discrete-event techniques common in systems research

---

## Contact

**Author**: Donatos Stavropoulos
**Affiliation**: Department of Electrical and Computer Engineering, University of Thessaly, Greece
**Email**: `dostavro@gmail.com`

---

**Last Updated**: November 2025
**Status**: Publication-ready artifact
