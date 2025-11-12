"""
Tau sensitivity analysis: robustness check for bounded slowdown metric.

Tests P95 and P99 bounded slowdown across multiple tau values:
  - τ=60s   (very strict, penalizes small jobs heavily)
  - τ=300s  (canonical, ~order of magnitude below median runtime)
  - τ=600s  (lenient, smooths out short-job effects)

Reports results with stability metrics across seeds to detect ranking changes.
"""

from workload import generate_jobs
from cluster import Cluster
from simulator import Simulator
from schedulers.fcfs import FCFS
from schedulers.easy import EASY
from schedulers.conservative import Conservative
from schedulers.priority import PriorityQoS
from schedulers.runai import RunAI
from schedulers.ttcc import TTCC
import metrics
import numpy as np
from copy import deepcopy

DEBUG = False

# Tau values to test (in seconds)
TAU_VALUES = [60, 300, 600]

# Cluster configuration
cluster_config = {"name": "Medium", "edge": 4, "cloud": 4}
num_edge = cluster_config["edge"]
num_cloud = cluster_config["cloud"]
total_res = num_edge + num_cloud

# Workload scenario: medium load
scenario = {
    "name": "Medium Load",
    "num_jobs": 80,
    "arrival_rate": 8,
    "horizon": 120
}

scheduler_names = ["FCFS", "EASY", "Conservative", "PriorityQoS", "RunAI", "TTCC"]


def create_scheduler(name):
    """Create a fresh scheduler instance."""
    if name == "FCFS":
        return FCFS()
    elif name == "EASY":
        return EASY()
    elif name == "Conservative":
        return Conservative()
    elif name == "PriorityQoS":
        return PriorityQoS()
    elif name == "RunAI":
        return RunAI()
    elif name == "TTCC":
        return TTCC()


print("=" * 160)
print(f"TAU SENSITIVITY ANALYSIS: Bounded Slowdown Robustness")
print(f"Cluster: {cluster_config['name']} ({total_res} GPUs)")
print(f"Workload: {scenario['name']} ({scenario['num_jobs']} jobs)")
print(f"Multiple seeds to assess P95/P99 stability")
print("=" * 160)

# Test multiple seeds to assess stability
num_seeds = 3
results_by_tau = {tau: {} for tau in TAU_VALUES}

for seed_idx in range(num_seeds):
    print(f"\n{'─' * 160}")
    print(f"Seed {seed_idx + 1}/{num_seeds}")
    print(f"{'─' * 160}")

    for name in scheduler_names:
        cluster = Cluster(num_edge=num_edge, num_cloud=num_cloud)
        jobs = generate_jobs(num_jobs=scenario['num_jobs'],
                             arrival_rate=scenario['arrival_rate'],
                             seed=42 + seed_idx)  # Different seed per trial
        scheduler = create_scheduler(name)

        # Run simulation once, use for all tau values
        sim = Simulator(cluster, jobs, scheduler, debug=DEBUG)
        finished = sim.run(horizon=scenario['horizon'])

        if name not in results_by_tau[TAU_VALUES[0]]:
            results_by_tau[TAU_VALUES[0]][name] = []
            for _ in TAU_VALUES[1:]:
                results_by_tau[_][name] = []

        # Compute P95/P99 for each tau
        for tau_s in TAU_VALUES:
            p95 = metrics.p95_slowdown(finished, tau_seconds=float(tau_s), sim_seconds_per_unit=3600.0)
            p99 = metrics.p99_slowdown(finished, tau_seconds=float(tau_s), sim_seconds_per_unit=3600.0)
            results_by_tau[tau_s][name].append((p95, p99))

# Report results
print(f"\n\n{'=' * 160}")
print("RESULTS BY TAU VALUE")
print("=" * 160)

for tau_s in TAU_VALUES:
    tau_h = tau_s / 3600.0
    print(f"\n{'─' * 160}")
    print(f"τ = {tau_s}s ({tau_h:.4f}h)")
    print(f"{'─' * 160}")
    print(f"{'Scheduler':<15} {'P95 Mean':>12} {'P95 Std':>12} {'P95 Range':>16} | {'P99 Mean':>12} {'P99 Std':>12} {'P99 Range':>16}")
    print(f"{'-' * 160}")

    rankings_p95 = {}
    rankings_p99 = {}

    for name in scheduler_names:
        p95_vals = [r[0] for r in results_by_tau[tau_s][name]]
        p99_vals = [r[1] for r in results_by_tau[tau_s][name]]

        p95_mean = np.mean(p95_vals)
        p95_std = np.std(p95_vals)
        p95_min = min(p95_vals)
        p95_max = max(p95_vals)

        p99_mean = np.mean(p99_vals)
        p99_std = np.std(p99_vals)
        p99_min = min(p99_vals)
        p99_max = max(p99_vals)

        rankings_p95[name] = p95_mean
        rankings_p99[name] = p99_mean

        p95_range = f"[{p95_min:.1f}, {p95_max:.1f}]"
        p99_range = f"[{p99_min:.1f}, {p99_max:.1f}]"

        print(f"{name:<15} {p95_mean:>12.2f} {p95_std:>12.3f} {p95_range:>16} | {p99_mean:>12.2f} {p99_std:>12.3f} {p99_range:>16}")

    # Show rankings
    ranked_p95 = sorted(rankings_p95.items(), key=lambda x: x[1])
    ranked_p99 = sorted(rankings_p99.items(), key=lambda x: x[1])

    print(f"\n  ├─ P95 Ranking (lower is better):")
    for rank, (name, val) in enumerate(ranked_p95, 1):
        print(f"  │  {rank}. {name:<15} {val:.2f}")

    print(f"  └─ P99 Ranking (lower is better):")
    for rank, (name, val) in enumerate(ranked_p99, 1):
        print(f"     {rank}. {name:<15} {val:.2f}")

# Stability check: compare rankings across tau values
print(f"\n\n{'=' * 160}")
print("RANKING STABILITY ACROSS TAU VALUES")
print("=" * 160)

rankings_all_tau = {}
for tau_s in TAU_VALUES:
    tau_rankings = {}
    for name in scheduler_names:
        p99_vals = [r[1] for r in results_by_tau[tau_s][name]]
        tau_rankings[name] = np.mean(p99_vals)
    rankings_all_tau[tau_s] = sorted(tau_rankings.items(), key=lambda x: x[1])

print("\nP99 Rankings by tau:\n")
for tau_s in TAU_VALUES:
    tau_h = tau_s / 3600.0
    print(f"τ={tau_s}s ({tau_h:.4f}h):")
    for rank, (name, val) in enumerate(rankings_all_tau[tau_s], 1):
        print(f"  {rank}. {name:<15} P99={val:7.2f}")
    print()

# Check if rankings are stable
print("Ranking Stability:")
first_ranking = [name for name, _ in rankings_all_tau[TAU_VALUES[0]]]
all_stable = all([name for name, _ in rankings_all_tau[tau]] == first_ranking for tau in TAU_VALUES[1:])

if all_stable:
    print("  ✓ Rankings STABLE across tau values → claims are ROBUST")
else:
    print("  ⚠ Rankings CHANGED across tau values → results are tau-sensitive")
    for i, tau1 in enumerate(TAU_VALUES[:-1]):
        for tau2 in TAU_VALUES[i+1:]:
            rank1 = [name for name, _ in rankings_all_tau[tau1]]
            rank2 = [name for name, _ in rankings_all_tau[tau2]]
            if rank1 != rank2:
                print(f"    └─ τ={tau1}s vs τ={tau2}s: rankings differ")

print(f"\n{'=' * 160}")
print("INTERPRETATION")
print("=" * 160)
print("""
If rankings are stable:
  → Your scheduler comparisons are robust to tau choice
  → Main paper can use canonical τ=300s with confidence

If rankings change:
  → Report sensitivity in appendix
  → Investigate which schedulers are affected
  → Use P95 (more stable) or CVaR alongside P99

Recommended tau choice:
  τ should be ~0.5–1.0× median job runtime to avoid smoothing bias
  For GPU workload: τ=300s (5 min) is typical
  Avoid τ >> mean(R) which masks short-job scheduling problems
""")
