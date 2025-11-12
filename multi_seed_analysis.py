"""
Multi-seed statistical analysis of scheduler performance.

Uses Common Random Numbers (CRN) for variance reduction across schedulers.
Runs a fixed 50 seeds per cluster–load configuration with 95% confidence
intervals for all metrics. Per-seed results and analysis scripts are
included in the artifact for reproducibility.

Usage:
    python multi_seed_analysis.py                    # Run with default 50 seeds
    python multi_seed_analysis.py --seeds 100        # Override to 100 seeds
    python multi_seed_analysis.py --base-seed 42     # Use specific base seed for reproducibility
    python multi_seed_analysis.py --help             # Show options
"""

import argparse
import csv
import json
import numpy as np
from copy import deepcopy
import sys
from datetime import datetime

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

DEBUG = False

# Cluster configurations
cluster_configs = [
    {"name": "Small", "edge": 4, "cloud": 4},
    {"name": "Medium", "edge": 16, "cloud": 16},
    {"name": "Large", "edge": 32, "cloud": 32},
]


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-seed scheduler performance analysis with CRN and statistical rigor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python multi_seed_analysis.py                 # Default: 50 seeds, random base
  python multi_seed_analysis.py --seeds 100     # Run 100 seeds
  python multi_seed_analysis.py --base-seed 42  # Reproducible: seeds 42-91
  python multi_seed_analysis.py --help          # Show this help
        """
    )
    parser.add_argument('--seeds', type=int, default=50,
                        help='Number of seeds to run (default: 50)')
    parser.add_argument('--base-seed', type=int, default=None,
                        help='Base seed for reproducibility (default: random). Seeds will be base_seed to base_seed+N-1')

    return parser.parse_args()


def get_scenarios(cluster_size):
    total_gpus = cluster_size["edge"] + cluster_size["cloud"]
    mean_runtime = 2.0

    # NEW: expected demand and realized runtime
    expected_demand = 1.0*0.40 + 2.0*0.35 + 3.0*0.20 + 4.0*0.05  # = 1.9
    p_early, mean_U = 0.3, 0.55
    expected_runtime = mean_runtime * ((1 - p_early) + p_early * mean_U)  # = 1.73

    target_loads = {"light": 0.6, "medium": 0.9, "heavy": 1.2}

    scenarios = []
    for load_name, target_load in target_loads.items():
        jobs_per_gpu = {"light": 6, "medium": 10, "heavy": 15}[load_name]
        num_jobs = total_gpus * jobs_per_gpu

        # FIXED: offered load calibration (GPU-hrs/hr)
        arrival_rate = (target_load * total_gpus) / (expected_demand * expected_runtime)

        last_arrival_time = num_jobs / arrival_rate
        queue_drain_time = jobs_per_gpu * mean_runtime * 3.0
        horizon = last_arrival_time + queue_drain_time

        scenario_name = {
            "light": "Light Load",
            "medium": "Medium Load",
            "heavy": "Heavy Load",
        }[load_name]

        scenarios.append({
            "name": scenario_name,
            "num_jobs": num_jobs,
            "arrival_rate": arrival_rate,
            "horizon": horizon,
            "target_load": target_load,
        })

    return scenarios


def create_scheduler(name):
    """Create scheduler instance."""
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
        return TTCC(enable_migrations=False)


def run_single_scenario_trial(cluster_config, scenario, seed):
    """Run all schedulers on same workload (CRN)."""
    num_edge = cluster_config["edge"]
    num_cloud = cluster_config["cloud"]
    total_res = num_edge + num_cloud

    # Select demand distribution based on cluster size
    # Small: {1, 2, 3, 4} GPUs
    # Medium/Large: {1, 2, 3, 4, 8} GPUs with 8-GPU jobs (5%) for gang pressure
    if cluster_config["name"] in ["Medium", "Large"]:
        demand_distribution = {1: 0.35, 2: 0.30, 3: 0.20, 4: 0.10, 8: 0.05}
    else:  # Small
        demand_distribution = {1: 0.4, 2: 0.35, 3: 0.2, 4: 0.05}

    # Generate workload ONCE with seed (Common Random Numbers)
    # Use the trial seed directly - no need for complex hashing
    workload_seed = seed
    base_jobs = generate_jobs(
        num_jobs=scenario['num_jobs'],
        arrival_rate=scenario['arrival_rate'],
        demand_distribution=demand_distribution,
        seed=workload_seed
    )

    results = {}
    scheduler_names = ["FCFS", "EASY", "Conservative", "PriorityQoS", "RunAI", "TTCC"]

    for sched_name in scheduler_names:
        cluster = Cluster(num_edge=num_edge, num_cloud=num_cloud)
        jobs = deepcopy(base_jobs)
        scheduler = create_scheduler(sched_name)

        sim = Simulator(cluster, jobs, scheduler, debug=DEBUG)
        finished = sim.run(horizon=scenario['horizon'])

        # Calculate metrics
        util = metrics.utilization(finished, total_res)
        utility = metrics.avg_utility(finished, gamma=0.9)
        p95_slow = metrics.p95_slowdown(finished, tau_seconds=300.0, sim_seconds_per_unit=3600.0)
        p99_slow = metrics.p99_slowdown(finished, tau_seconds=300.0, sim_seconds_per_unit=3600.0)

        results[sched_name] = {
            'util': util,
            'utility': utility,
            'p95_slow': p95_slow,
            'p99_slow': p99_slow,
            'finished': len(finished),
            'total': scenario['num_jobs']
        }

    return results


def bootstrap_ci(data, func=np.median, n_bootstrap=10000, ci=0.95):
    """
    Compute bootstrap confidence interval.

    Returns: (point_estimate, lower, upper)
    """
    data = np.asarray(data)
    point_est = func(data)

    bootstrap_stats = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(data), size=len(data), replace=True)
        bootstrap_stats.append(func(data[indices]))

    bootstrap_stats = np.array(bootstrap_stats)
    alpha = (1 - ci) / 2
    lower = np.percentile(bootstrap_stats, alpha * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha) * 100)

    return point_est, lower, upper


def paired_ttest(x, y):
    """Paired t-test: x vs y. Returns (t_stat, p_value)."""
    x = np.asarray(x)
    y = np.asarray(y)
    diff = x - y
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    se_diff = std_diff / np.sqrt(len(diff))
    t_stat = mean_diff / (se_diff + 1e-10)

    # Approximate p-value using normal distribution (good for n >= 10)
    # For n < 10, could use t-distribution but normal is conservative estimate
    from math import erfc
    p_value = erfc(abs(t_stat) / np.sqrt(2))

    return t_stat, p_value


print("=" * 160)
print("Multi-Seed Scheduler Analysis: 50 Seeds with Common Random Numbers (CRN)")
print("95% Confidence Intervals for All Metrics")
print("=" * 160)

# Parse command-line arguments
args = parse_args()
max_seeds = args.seeds
base_seed = args.base_seed if args.base_seed is not None else np.random.randint(0, 2**31)

print(f"\n📊 Configuration:")
print(f"  Max seeds: {max_seeds}")
print(f"  Base seed: {base_seed}")
print(f"  Seed range: {base_seed} to {base_seed + max_seeds - 1}")
print(f"  Timestamp: {datetime.now().isoformat()}")
print()

# Collect results across seeds and scenarios
all_results = {}  # {(cluster_name, scenario_name): {scheduler: [metrics_across_seeds]}}
seed_metadata = {
    'max_seeds': max_seeds,
    'base_seed': base_seed,
    'timestamp': datetime.now().isoformat(),
    'scenarios': {}
}

scheduler_names = ["FCFS", "EASY", "Conservative", "PriorityQoS", "RunAI", "TTCC"]

for cluster_config in cluster_configs:
    cluster_name = cluster_config['name']
    scenarios = get_scenarios(cluster_config)

    for scenario in scenarios:
        scenario_key = (cluster_name, scenario['name'])
        all_results[scenario_key] = {sched: {
            'util': [], 'utility': [],
            'p95_slow': [], 'p99_slow': [], 'finished': []
        } for sched in scheduler_names}

        print(f"\n{'=' * 160}")
        print(f"{cluster_name} - {scenario['name']}")
        print(f"Jobs: {scenario['num_jobs']}, Arrival: {scenario['arrival_rate']:.2f}/h, "
              f"Horizon: {scenario['horizon']:.1f}h")
        print("=" * 160)
        print(f"{'Seed':>4}  ", end="")
        for sched in scheduler_names:
            print(f"{sched:>10} ", end="")
        print()
        print("-" * 160)

        # Run 50 seeds with Common Random Numbers (CRN)
        seeds_used = []
        for seed_idx in range(max_seeds):
            trial_seed = base_seed + seed_idx
            seeds_used.append(trial_seed)

            trial_results = run_single_scenario_trial(cluster_config, scenario, trial_seed)

            # Store results
            for sched_name in scheduler_names:
                all_results[scenario_key][sched_name]['util'].append(
                    trial_results[sched_name]['util'])
                all_results[scenario_key][sched_name]['utility'].append(
                    trial_results[sched_name]['utility'])
                all_results[scenario_key][sched_name]['p95_slow'].append(
                    trial_results[sched_name]['p95_slow'])
                all_results[scenario_key][sched_name]['p99_slow'].append(
                    trial_results[sched_name]['p99_slow'])
                all_results[scenario_key][sched_name]['finished'].append(
                    trial_results[sched_name]['finished'])

            # Print seed row (avg utilization across schedulers)
            print(f"{seed_idx+1:>4}  ", end="")
            for sched_name in scheduler_names:
                util = trial_results[sched_name]['util']
                print(f"{util:>10.3f} ", end="")
            print()

        n_seeds = max_seeds
        print(f"\n[Completed all {n_seeds} seeds with CRN]")
        seed_metadata['scenarios'][str(scenario_key)] = {
            'n_seeds': n_seeds,
            'seeds_used': seeds_used[:n_seeds],
            'base_seed': base_seed,
        }

        # Compute summary statistics
        print("-" * 160)
        print(f"\n📊 Summary Statistics ({n_seeds} seeds, CRN)")
        print("-" * 100)
        print(f"{'Scheduler':<15}  {'Util':>10}  {'Avg Util':>12}  "
              f"{'P95 Slow':>12}  {'P99 Slow':>12}  {'Compl %':>10}")
        print("-" * 100)

        for sched_name in scheduler_names:
            data = all_results[scenario_key][sched_name]
            util_vals = np.array(data['util'])
            utility_vals = np.array(data['utility'])

            p95_vals = np.array(data['p95_slow'])
            p99_vals = np.array(data['p99_slow'])
            finished_vals = np.array(data['finished'])

            # Mean ± 95% CI for continuous metrics
            util_mean = np.mean(util_vals)
            util_se = np.std(util_vals, ddof=1) / np.sqrt(len(util_vals))
            util_ci = 1.96 * util_se

            utility_mean = np.mean(utility_vals)
            utility_se = np.std(utility_vals, ddof=1) / np.sqrt(len(utility_vals))
            utility_ci = 1.96 * utility_se

            # Bootstrap CI for slowdown (median more robust for tail metrics)
            p95_med, p95_lower, p95_upper = bootstrap_ci(p95_vals, func=np.median)
            p99_med, p99_lower, p99_upper = bootstrap_ci(p99_vals, func=np.median)

            # Completion rate
            compl_pct = np.mean(finished_vals / scenario['num_jobs']) * 100

            print(f"{sched_name:<15}  "
                  f"{util_mean:.3f}±{util_ci:.3f}  "
                  f"{utility_mean:.3f}±{utility_ci:.3f}  "
                  f"{p95_med:.1f}({p95_lower:.1f}-{p95_upper:.1f})  "
                  f"{p99_med:.1f}({p99_lower:.1f}-{p99_upper:.1f})  "
                  f"{compl_pct:>9.1f}%")

        # Paired comparisons: Conservative vs others (as reference)
        print("\n📈 Paired t-tests (Conservative as reference):")
        print("-" * 160)

        cons_util = np.array(all_results[scenario_key]['Conservative']['util'])
        cons_utility = np.array(all_results[scenario_key]['Conservative']['utility'])

        print(f"{'Scheduler':<15}  {'Util vs Cons':>15}  {'Utility vs Cons':>15}  {'Interpretation':<30}")
        print("-" * 160)

        for sched_name in scheduler_names:
            if sched_name == 'Conservative':
                print(f"{'Conservative':<15}  {'(reference)':>15}  {'(reference)':>15}  {'---':<30}")
                continue

            sched_util = np.array(all_results[scenario_key][sched_name]['util'])
            sched_utility = np.array(all_results[scenario_key][sched_name]['utility'])

            t_util, p_util = paired_ttest(sched_util, cons_util)
            t_util_str = f"t={t_util:+.2f}, p={p_util:.3f}"

            t_util_better = "↑ Better" if t_util > 0 else "↓ Worse"
            util_sig = "***" if p_util < 0.01 else "**" if p_util < 0.05 else "*" if p_util < 0.10 else "ns"

            t_util_str += f" {util_sig}"

            t_utility, p_utility = paired_ttest(sched_utility, cons_utility)
            t_utility_str = f"t={t_utility:+.2f}, p={p_utility:.3f}"

            t_util_better_u = "↑ Better" if t_utility > 0 else "↓ Worse"
            utility_sig = "***" if p_utility < 0.01 else "**" if p_utility < 0.05 else "*" if p_utility < 0.10 else "ns"

            t_utility_str += f" {utility_sig}"

            interp = f"{t_util_better} (util), {t_util_better_u} (util)"

            print(f"{sched_name:<15}  {t_util_str:>15}  {t_utility_str:>15}  {interp:<30}")

        print()

print("\n" + "=" * 160)
print("Multi-Seed Analysis Complete")
print("=" * 160)
print("\nSignificance levels: *** p<0.01, ** p<0.05, * p<0.10, ns = not significant")
print("CI notation: median(lower-upper) for slowdown percentiles")

# Save results to JSON for plotting

# Convert numpy arrays to lists for JSON serialization


# Convert numpy arrays to lists for JSON serialization
def convert_to_serializable(obj):
    """Convert numpy arrays to lists and tuple keys to strings recursively."""
    if isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    else:
        return obj


json_data = convert_to_serializable(all_results)

with open('multi_seed_results.json', 'w') as f:
    json.dump(json_data, f, indent=2)
print("\n✓ Results saved to multi_seed_results.json")

# Save seed metadata for reproducibility
with open('multi_seed_metadata.json', 'w') as f:
    json.dump(seed_metadata, f, indent=2)
print("✓ Metadata saved to multi_seed_metadata.json")
print(f"  Base seed: {base_seed}")
print(f"  Seed range: {base_seed} to {base_seed + max_seeds - 1}")
print(f"  Reproducibility: Run with --base-seed {base_seed} to recreate")

# Also export pre-computed statistics to CSV for fast plotting

# Compute all stats once
stats_data = {}  # (cluster, load, scheduler) -> {mean, ci_lower, ci_upper, ...}
for (cluster, load), schedulers_data in all_results.items():
    for sched, metrics in schedulers_data.items():
        for metric_name in ['util', 'utility', 'p95_slow', 'p99_slow']:
            metric_vals = np.array(metrics[metric_name])
            mean = np.mean(metric_vals)
            std_err = np.std(metric_vals, ddof=1) / np.sqrt(len(metric_vals))
            ci_lower = mean - 1.96 * std_err
            ci_upper = mean + 1.96 * std_err
            median = np.median(metric_vals)

            stats_data[(cluster, load, sched, metric_name)] = {
                'mean': mean,
                'std_err': std_err,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'median': median,
            }

# Export to CSV
with open('multi_seed_stats.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Cluster', 'Load', 'Scheduler', 'Metric', 'Mean', 'StdErr', 'CILower', 'CIUpper', 'Median'])

    for (cluster, load, sched, metric), stat in sorted(stats_data.items()):
        writer.writerow([
            cluster, load, sched, metric,
            f"{stat['mean']:.6f}",
            f"{stat['std_err']:.6f}",
            f"{stat['ci_lower']:.6f}",
            f"{stat['ci_upper']:.6f}",
            f"{stat['median']:.6f}",
        ])

print("✓ Pre-computed stats saved to multi_seed_stats.csv")
print("  Run: python plot_results.py (now loads CSV instead of recomputing)")
