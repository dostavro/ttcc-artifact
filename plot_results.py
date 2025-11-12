#!/usr/bin/env python3
"""
Load multi-seed results from CSV and generate publication-ready tables and figures.

Implements the complete 4-figure publication roadmap:
1. Cluster Utilization vs. Load
2. Average Utility by Load & Cluster
3. Tail Slowdown (P95 + P99)
4. Utility–Tail Trade-off
"""
import json
import csv
import os
import sys
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# Global styling — consistent across all figures
COLORS = {
    'FCFS': '#808080',           # gray
    'EASY': '#1f77b4',           # blue
    'Conservative': '#aec7e8',   # light blue
    'PriorityQoS': '#ff7f0e',    # orange
    'RunAI': '#2ca02c',          # green
    'TTCC': '#d62728',           # red
}

SCHEDULER_ORDER = ["FCFS", "EASY", "Conservative", "PriorityQoS", "RunAI", "TTCC"]
LOAD_ORDER = ['Light Load', 'Medium Load', 'Heavy Load']
CLUSTER_ORDER = ['Small', 'Medium', 'Large']


def load_results(json_file="multi_seed_results.json", csv_file="multi_seed_stats.csv"):
    """Load results from JSON file and pre-computed stats from CSV."""
    if not os.path.exists(csv_file):
        print(f"✗ {csv_file} not found")
        print(f"  Run: python multi_seed_analysis.py")
        sys.exit(1)

    # Load pre-computed stats from CSV (much faster than recomputing bootstrap)
    stats = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cluster, load, sched, metric = row['Cluster'], row['Load'], row['Scheduler'], row['Metric']
            key = (cluster, load, sched, metric)
            stats[key] = {
                'mean': float(row['Mean']),
                'std_err': float(row['StdErr']),
                'ci_lower': float(row['CILower']),
                'ci_upper': float(row['CIUpper']),
                'median': float(row['Median']),
            }

    print(f"✓ Loaded pre-computed stats from {csv_file}")
    return stats


def convert_scenario_keys(data):
    """Not needed anymore — CSV already has cluster/load/scheduler separated."""
    return data


def get_results(data):
    """CSV stats dict is already in the right format."""
    return data


def get_stat(stats, cluster, load, sched, metric):
    """Retrieve pre-computed stat from CSV data."""
    key = (cluster, load, sched, metric)
    if key in stats:
        return stats[key]
    return {'mean': 0, 'median': 0, 'ci_lower': 0, 'ci_upper': 0}


def print_tables(stats):
    """Print comprehensive summary tables using pre-computed stats from CSV."""
    clusters = CLUSTER_ORDER
    loads = LOAD_ORDER
    schedulers = SCHEDULER_ORDER

    print("\n" + "="*100)
    print("PLOT DATA TABLES")
    print("="*100)

    # Table 1: Utilization
    print("\n" + "="*100)
    print("TABLE 1: UTILIZATION BY LOAD AND CLUSTER")
    print("="*100)
    header = "Load".ljust(16) + "Cluster".ljust(18)
    for sched in schedulers:
        header += sched.rjust(12)
    print(header)
    print("-" * 100)

    for load in loads:
        for cluster in clusters:
            row = load.ljust(16) + cluster.ljust(18)
            for sched in schedulers:
                stat = get_stat(stats, cluster, load, sched, 'util')
                row += f"{stat['mean']:.3f}".rjust(12)
            print(row)

    # Table 2: Average Utility
    print("\n" + "="*100)
    print("TABLE 2: AVERAGE UTILITY BY LOAD AND CLUSTER")
    print("="*100)
    header = "Load".ljust(16) + "Cluster".ljust(18)
    for sched in schedulers:
        header += sched.rjust(12)
    print(header)
    print("-" * 100)

    for load in loads:
        for cluster in clusters:
            row = load.ljust(16) + cluster.ljust(18)
            for sched in schedulers:
                stat = get_stat(stats, cluster, load, sched, 'utility')
                row += f"{stat['mean']:.3f}".rjust(12)
            print(row)

    # Table 3: P95 Slowdown
    print("\n" + "="*100)
    print("TABLE 3: P95 SLOWDOWN BY LOAD AND CLUSTER (median, 95% CI)")
    print("="*100)
    header = "Load".ljust(16) + "Cluster".ljust(35)
    for sched in schedulers:
        header += sched.rjust(18)
    print(header)
    print("-" * 130)

    for load in loads:
        for cluster in clusters:
            row = load.ljust(16) + cluster.ljust(35)
            for sched in schedulers:
                stat = get_stat(stats, cluster, load, sched, 'p95_slow')
                row += f"{stat['median']:.1f}({stat['ci_lower']:5.1f}-{stat['ci_upper']:5.1f})".rjust(18)
            print(row)

    # Table 4: P99 Slowdown
    print("\n" + "="*100)
    print("TABLE 4: P99 SLOWDOWN BY LOAD AND CLUSTER (median, 95% CI)")
    print("="*100)
    header = "Load".ljust(16) + "Cluster".ljust(35)
    for sched in schedulers:
        header += sched.rjust(18)
    print(header)
    print("-" * 130)

    for load in loads:
        for cluster in clusters:
            row = load.ljust(16) + cluster.ljust(35)
            for sched in schedulers:
                stat = get_stat(stats, cluster, load, sched, 'p99_slow')
                row += f"{stat['median']:.1f}({stat['ci_lower']:5.1f}-{stat['ci_upper']:5.1f})".rjust(18)
            print(row)

    # Utility-Tail Tradeoff Summary
    print("\n" + "="*100)
    print("UTILITY-TAIL TRADEOFF SUMMARY (across all scenarios)")
    print("="*100)

    # Aggregate across all scenarios (3 clusters × 3 loads = 9 scenarios)
    agg_util = {sched: [] for sched in schedulers}
    agg_p99 = {sched: [] for sched in schedulers}
    agg_utility = {sched: [] for sched in schedulers}
    agg_p95 = {sched: [] for sched in schedulers}

    for load in loads:
        for cluster in clusters:
            for sched in schedulers:
                util_stat = get_stat(stats, cluster, load, sched, 'util')
                utility_stat = get_stat(stats, cluster, load, sched, 'utility')
                p95_stat = get_stat(stats, cluster, load, sched, 'p95_slow')
                p99_stat = get_stat(stats, cluster, load, sched, 'p99_slow')
                agg_util[sched].append(util_stat['mean'])
                agg_utility[sched].append(utility_stat['mean'])
                agg_p95[sched].append(p95_stat['median'])
                agg_p99[sched].append(p99_stat['median'])

    print("Scheduler".ljust(16) + "Avg Utility".rjust(15) + "P99 Slow".rjust(15) + "vs Conservative".rjust(20))
    print("-" * 70)

    cons_util = np.mean(agg_utility['Conservative'])
    cons_p99 = np.mean(agg_p99['Conservative'])

    for sched in schedulers:
        util_val = np.mean(agg_utility[sched])
        p99_val = np.mean(agg_p99[sched])
        util_delta = (util_val - cons_util) / cons_util * 100 if cons_util > 0 else 0
        p99_delta = (p99_val - cons_p99) / cons_p99 * 100 if cons_p99 > 0 else 0

        marker = "Baseline" if sched == "Conservative" else f"↑ util({util_delta:+.0f}%), ↓ slow({p99_delta:+.0f}%)"
        row = sched.ljust(16)
        row += f"{util_val:.3f}".rjust(15)
        row += f"{p99_val:.1f}".rjust(15)
        row += marker.rjust(20)
        print(row)

    # Save aggregated results to CSV with 95% CI
    with open('summary_across_all_scenarios.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Scheduler',
                         'Mean_Utilization', 'Util_CI_Lower', 'Util_CI_Upper',
                         'Mean_Avg_Utility', 'Utility_CI_Lower', 'Utility_CI_Upper',
                         'Median_P95_Slowdown', 'P95_CI_Lower', 'P95_CI_Upper',
                         'Median_P99_Slowdown', 'P99_CI_Lower', 'P99_CI_Upper'])

        for sched in schedulers:
            # Compute mean and 95% CI for each metric
            util_mean = np.mean(agg_util[sched])
            util_se = np.std(agg_util[sched], ddof=1) / np.sqrt(len(agg_util[sched]))
            util_ci_lower = util_mean - 1.96 * util_se
            util_ci_upper = util_mean + 1.96 * util_se

            utility_mean = np.mean(agg_utility[sched])
            utility_se = np.std(agg_utility[sched], ddof=1) / np.sqrt(len(agg_utility[sched]))
            utility_ci_lower = utility_mean - 1.96 * utility_se
            utility_ci_upper = utility_mean + 1.96 * utility_se

            p95_mean = np.mean(agg_p95[sched])
            p95_se = np.std(agg_p95[sched], ddof=1) / np.sqrt(len(agg_p95[sched]))
            p95_ci_lower = p95_mean - 1.96 * p95_se
            p95_ci_upper = p95_mean + 1.96 * p95_se

            p99_mean = np.mean(agg_p99[sched])
            p99_se = np.std(agg_p99[sched], ddof=1) / np.sqrt(len(agg_p99[sched]))
            p99_ci_lower = p99_mean - 1.96 * p99_se
            p99_ci_upper = p99_mean + 1.96 * p99_se

            writer.writerow([
                sched,
                f"{util_mean:.6f}", f"{util_ci_lower:.6f}", f"{util_ci_upper:.6f}",
                f"{utility_mean:.6f}", f"{utility_ci_lower:.6f}", f"{utility_ci_upper:.6f}",
                f"{p95_mean:.6f}", f"{p95_ci_lower:.6f}", f"{p95_ci_upper:.6f}",
                f"{p99_mean:.6f}", f"{p99_ci_lower:.6f}", f"{p99_ci_upper:.6f}",
            ])

    print("✓ Saved aggregated metrics with 95% CI to summary_across_all_scenarios.csv")

    # CSV Export
    print("\n" + "="*100)
    print("CSV EXPORT (copy to Excel for plotting)")
    print("="*100)
    print("\n# Utilization by Load (averaged across clusters)")
    print("Load," + ",".join(schedulers) + ",")
    for load in loads:
        row = load
        for sched in schedulers:
            vals = []
            for cluster in clusters:
                stat = get_stat(stats, cluster, load, sched, 'util')
                vals.append(stat['mean'])
            row += "," + f"{np.mean(vals):.3f}"
        print(row + ",")

    print("\n# Average Utility by Load (averaged across clusters)")
    print("Load," + ",".join(schedulers) + ",")
    for load in loads:
        row = load
        for sched in schedulers:
            vals = []
            for cluster in clusters:
                stat = get_stat(stats, cluster, load, sched, 'utility')
                vals.append(stat['mean'])
            row += "," + f"{np.mean(vals):.3f}"
        print(row + ",")


def fig1_utilization(stats, clusters, loads, schedulers):
    """Figure 1: Cluster Utilization vs. Load (grouped bar chart with 95% CI error bars)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax_idx, cluster in enumerate(clusters):
        ax = axes[ax_idx]
        x_pos = np.arange(len(loads))
        width = 0.13

        for sched_idx, sched in enumerate(schedulers):
            vals = []
            ci_lower = []
            ci_upper = []
            for load in loads:
                stat = get_stat(stats, cluster, load, sched, 'util')
                vals.append(stat['mean'])
                ci_lower.append(stat['ci_lower'])
                ci_upper.append(stat['ci_upper'])

            # Compute error bar heights
            errors = [np.array(vals) - np.array(ci_lower), np.array(ci_upper) - np.array(vals)]

            offset = (sched_idx - 2.5) * width
            linewidth = 1.5 if sched == 'TTCC' else 0.8
            ax.bar(x_pos + offset, vals, width, label=sched, color=COLORS[sched], alpha=0.8,
                   edgecolor='black', linewidth=linewidth, yerr=errors, capsize=3,
                   error_kw={'elinewidth': 0.8, 'alpha': 0.6})

        ax.set_ylabel("Utilization", fontsize=11, fontweight='bold')
        ax.set_title(f"{cluster} Cluster", fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(loads, fontsize=10)
        ax.set_ylim([0.0, 1.0])
        ax.grid(True, alpha=0.2, axis='y', linestyle='--')
        ax.legend(fontsize=9, loc='upper left', framealpha=0.95)

    plt.tight_layout()
    plt.savefig('fig1_utilization.png', dpi=300, bbox_inches='tight')
    print("✓ Generated fig1_utilization.png")
    plt.close()


def fig2_utility(stats, clusters, loads, schedulers):
    """Figure 2: Average Utility by Load & Cluster (grouped bar chart with 95% CI error bars)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax_idx, cluster in enumerate(clusters):
        ax = axes[ax_idx]
        x_pos = np.arange(len(loads))
        width = 0.13

        for sched_idx, sched in enumerate(schedulers):
            vals = []
            ci_lower = []
            ci_upper = []
            for load in loads:
                stat = get_stat(stats, cluster, load, sched, 'utility')
                vals.append(stat['mean'])
                ci_lower.append(stat['ci_lower'])
                ci_upper.append(stat['ci_upper'])

            # Compute error bar heights
            errors = [np.array(vals) - np.array(ci_lower), np.array(ci_upper) - np.array(vals)]

            offset = (sched_idx - 2.5) * width
            linewidth = 1.5 if sched == 'TTCC' else 0.8
            ax.bar(x_pos + offset, vals, width, label=sched, color=COLORS[sched], alpha=0.8,
                   edgecolor='black', linewidth=linewidth, yerr=errors, capsize=3,
                   error_kw={'elinewidth': 0.8, 'alpha': 0.6})

        ax.set_ylabel("Average Utility", fontsize=11, fontweight='bold')
        ax.set_title(f"{cluster} Cluster", fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(loads, fontsize=10)
        ax.grid(True, alpha=0.2, axis='y', linestyle='--')
        ax.legend(fontsize=9, loc='lower left', framealpha=0.95)

    plt.tight_layout()
    plt.savefig('fig2_utility.png', dpi=300, bbox_inches='tight')
    print("✓ Generated fig2_utility.png")
    plt.close()


def fig3_tail_slowdown(stats, clusters, loads, schedulers):
    """Figure 3: Tail Slowdown (P95 vs P99) with 95% CI error bars — compact dual-metric design.

    Per-cluster subplots showing P95 (light) and P99 (dark) side-by-side for each scheduler.
    Reduces visual clutter and enables direct P95/P99 comparison.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Pre-build legend handles/labels so we can place the legend on all subplots
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[s], alpha=0.6, edgecolor='black', linewidth=1.5 if s == 'TTCC' else 0.8)
               for s in schedulers]
    labels = [s for s in schedulers]

    # Add P95/P99 explanation handles
    p95_handle = plt.Rectangle((0, 0), 1, 1, fc='gray', alpha=0.6, edgecolor='black', linewidth=0.8)
    p99_handle = plt.Rectangle((0, 0), 1, 1, fc='gray', alpha=0.95, edgecolor='black', linewidth=0.8)
    handles.extend([p95_handle, p99_handle])
    labels.extend(['Lighter: P95', 'Darker: P99'])

    for ax_idx, cluster in enumerate(clusters):
        ax = axes[ax_idx]
        x_pos = np.arange(len(loads))
        width = 0.065  # Narrower bars to fit P95+P99 pairs

        for sched_idx, sched in enumerate(schedulers):
            p95_vals = []
            p95_ci_lower = []
            p95_ci_upper = []
            p99_vals = []
            p99_ci_lower = []
            p99_ci_upper = []

            for load in loads:
                p95_stat = get_stat(stats, cluster, load, sched, 'p95_slow')
                p99_stat = get_stat(stats, cluster, load, sched, 'p99_slow')
                p95_vals.append(p95_stat['median'])
                p95_ci_lower.append(max(0, p95_stat['ci_lower']))  # Clip to 0
                p95_ci_upper.append(p95_stat['ci_upper'])
                p99_vals.append(p99_stat['median'])
                p99_ci_lower.append(max(0, p99_stat['ci_lower']))  # Clip to 0
                p99_ci_upper.append(p99_stat['ci_upper'])

            # Compute error bar heights (clip negative values)
            p95_errors = [np.maximum(np.array(p95_vals) - np.array(p95_ci_lower), 0),
                          np.array(p95_ci_upper) - np.array(p95_vals)]
            p99_errors = [np.maximum(np.array(p99_vals) - np.array(p99_ci_lower), 0),
                          np.array(p99_ci_upper) - np.array(p99_vals)]

            # Position offset for this scheduler
            sched_offset = (sched_idx - 2.5)
            linewidth = 1.5 if sched == 'TTCC' else 0.8

            # P95 bars (lighter, left side of pair)
            x_p95 = x_pos + sched_offset * width * 2.1 - width/2
            ax.bar(x_p95, p95_vals, width, label=sched if sched_idx == 0 else None,
                   color=COLORS[sched], alpha=0.6, edgecolor='black', linewidth=linewidth,
                   yerr=p95_errors, capsize=2, error_kw={'elinewidth': 0.6, 'alpha': 0.5})

            # P99 bars (darker, right side of pair)
            x_p99 = x_pos + sched_offset * width * 2.1 + width/2
            ax.bar(x_p99, p99_vals, width,
                   color=COLORS[sched], alpha=0.95, edgecolor='black', linewidth=linewidth,
                   yerr=p99_errors, capsize=2, error_kw={'elinewidth': 0.6, 'alpha': 0.5})

        ax.set_ylabel("Bounded Slowdown", fontsize=11, fontweight='bold')
        ax.set_title(f"{cluster} Cluster", fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(loads, fontsize=10)
        ax.grid(True, alpha=0.2, axis='y', linestyle='--')

        # Place the same legend on every subplot for consistency
        ax.legend(handles, labels, fontsize=8, loc='upper left', ncol=2, framealpha=0.98)

    plt.tight_layout()
    plt.savefig('fig3_tail_slowdown.png', dpi=300, bbox_inches='tight')
    print("✓ Generated fig3_tail_slowdown.png")
    plt.close()


def generate_figures(stats):
    """Generate all 4 publication-ready figures using pre-computed stats."""
    if not MATPLOTLIB_AVAILABLE:
        return

    clusters = CLUSTER_ORDER
    loads = LOAD_ORDER
    schedulers = SCHEDULER_ORDER

    # Aggregate stats for scatter plots
    agg_util = {sched: [] for sched in schedulers}
    agg_p99 = {sched: [] for sched in schedulers}

    for load in loads:
        for cluster in clusters:
            for sched in schedulers:
                util_stat = get_stat(stats, cluster, load, sched, 'utility')
                p99_stat = get_stat(stats, cluster, load, sched, 'p99_slow')
                agg_util[sched].append(util_stat['mean'])
                agg_p99[sched].append(p99_stat['median'])

    # Generate figures
    fig1_utilization(stats, clusters, loads, schedulers)
    fig2_utility(stats, clusters, loads, schedulers)
    fig3_tail_slowdown(stats, clusters, loads, schedulers)
    fig4_utility_tail_tradeoff(agg_util, agg_p99, schedulers)

    print("✓ All 4 figures generated successfully!")


def fig4_utility_tail_tradeoff(agg_util, agg_p99, schedulers):
    """Figure 4: Utility vs P99 Slowdown tradeoff scatter plot with smart label positioning.

    Shows the Pareto frontier between average utility (user satisfaction) and tail latency (fairness).
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define label offsets for each scheduler to avoid collisions
    label_offsets = {
        'FCFS': (8, -12),         # bottom-right
        'EASY': (8, 8),           # top-right
        'Conservative': (-20, 8),  # top-left, moved right (was -50)
        'PriorityQoS': (-50, -20),  # bottom-left, moved lower
        'RunAI': (-50, 12),       # top-left, higher
        'TTCC': (8, -20),         # bottom-right
    }

    for sched in schedulers:
        util = np.mean(agg_util[sched])
        p99 = np.mean(agg_p99[sched])
        marker = 'D' if sched == 'TTCC' else 'o'
        size = 250 if sched == 'TTCC' else 150
        ax.scatter(p99, util, s=size, color=COLORS[sched], marker=marker,
                   alpha=0.7, edgecolors='black', linewidth=2, label=sched, zorder=3)

        # Place label with scheduler-specific offset
        offset = label_offsets.get(sched, (8, 8))
        ax.annotate(sched, (p99, util), xytext=offset, textcoords='offset points',
                    fontsize=9, fontweight='bold')

    ax.set_xlabel("P99 Slowdown (Lower is Better ←)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Average Utility (Higher is Better →)", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('fig4_utility_tail_tradeoff.png', dpi=300, bbox_inches='tight')
    print("✓ Generated fig4_utility_tail_tradeoff.png")
    plt.close()


def fig6_utility_fairness_frontier(agg_util, agg_prs, schedulers):
    """Figure 6: Utility-PRS Frontier scatter plot with smart label positioning."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define label offsets for each scheduler to avoid collisions
    label_offsets = {
        'FCFS': (-20, 8),         # top-left, moved right (was -50)
        'EASY': (8, 8),           # top-right (no collision with Conservative)
        'Conservative': (-50, 12),  # top-left, higher
        'PriorityQoS': (-50, -20),  # bottom-left, moved lower
        'RunAI': (8, -12),        # bottom-right
        'TTCC': (8, 12),          # top-right, higher
    }

    for sched in schedulers:
        util = np.mean(agg_util[sched])
        prs = np.mean(agg_prs[sched])
        marker = 'D' if sched == 'TTCC' else 'o'
        size = 250 if sched == 'TTCC' else 150
        ax.scatter(util, prs, s=size, color=COLORS[sched], marker=marker,
                   alpha=0.7, edgecolors='black', linewidth=2, label=sched, zorder=3)

        # Place label with scheduler-specific offset
        offset = label_offsets.get(sched, (8, 8))
        ax.annotate(sched, (util, prs), xytext=offset, textcoords='offset points',
                    fontsize=9, fontweight='bold')

    ax.set_xlabel("Average Utility (Higher is Better →)", fontsize=12, fontweight='bold')
    ax.set_ylabel("PRS (Higher is Better →)", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('fig6_utility_prs_frontier.png', dpi=300, bbox_inches='tight')
    print("✓ Generated fig6_utility_prs_frontier.png")
    plt.close()


def main():
    """Main entry point."""
    stats = load_results()  # Loads from multi_seed_stats.csv (instant)

    print("\n" + "="*100)
    print("PLOT DATA TABLES")
    print("="*100)

    print_tables(stats)

    if MATPLOTLIB_AVAILABLE:
        print("\n" + "="*100)
        print("GENERATING PUBLICATION-READY FIGURES (using pre-computed stats)")
        print("="*100)
        generate_figures(stats)
    else:
        print("\n⚠ matplotlib not available - tables generated only")
        print("  Install: pip install matplotlib")

    print("\n" + "="*100)
    print("✓ Plot generation complete!")
    print("="*100)


if __name__ == "__main__":
    main()
