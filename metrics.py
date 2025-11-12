"""
Performance metrics for scheduler evaluation.

Implements four key metrics:
1. Utilization: % of GPU time used (0–1, higher is better)
2. Average Utility: Preference-aware satisfaction accounting for deadline adherence (0–3, higher is better)
3. P95 Bounded Slowdown: 95th percentile tail latency (lower is better)
4. P99 Bounded Slowdown: 99th percentile tail latency (lower is better)
"""
import numpy as np


def utilization(jobs, total_resources):
    """
    Utilization = (sum of all (GPUs per job × runtime per job)) / (Total GPUs × Makespan)
    Returns a value in [0, 1].
    """
    if not jobs or total_resources <= 0:
        return 0.0
    # Numerator: total GPU-hours used
    total_gpu_time = sum(j.demand * j.actual_time for j in jobs)
    # Denominator: total GPU-hours available during makespan
    finished_jobs = [j for j in jobs if j.finish is not None]
    if not finished_jobs:
        return 0.0
    first_arrival = min(j.arrival for j in finished_jobs)
    last_finish = max(j.finish for j in finished_jobs)
    makespan = max(1.0, last_finish - first_arrival)  # Avoid division by zero
    total_capacity = total_resources * makespan
    return total_gpu_time / total_capacity if total_capacity > 0 else 0.0


def avg_utility(jobs, gamma=0.9):
    """
    Calculate average system utility (preference-aware) across completed jobs.

    Per-job utility combines deadline adherence, QoS class, and preference satisfaction:
      U_j = U_base(j) * S_j

    where:
      U_base(j) = w_{q_j} * exp(-T_j / beta_{q_j})
        T_j = max(0, c_j - D_j)  (tardiness)
        w_q: priority weight (Gold > Silver > Bronze)
        beta_q: decay scale (Gold < Silver < Bronze)

      S_j = (1/d_j) * sum_{k=1}^{d_j} gamma^(rho_j(r_k) - 1)
        rho_j(r): preference rank of resource r for job j
        gamma in (0, 1]: decay factor for non-preferred resources

    Args:
        jobs: List of Job objects
        gamma: Preference decay factor in (0, 1]. Default 0.9.
               gamma=1 means preference doesn't matter; gamma~0.9 means ~10% penalty per rank

    Returns:
        Average utility across all completed jobs, in [0, max(w_q)]
    """
    finished_jobs = [j for j in jobs if j.finish is not None]
    if not finished_jobs:
        return 0.0

    utilities = [j.utility(gamma=gamma) for j in finished_jobs]
    return np.mean(utilities)


def p95_slowdown(jobs, tau_seconds=60.0, sim_seconds_per_unit=3600.0):
    """
    Calculate the 95th percentile of bounded slowdown across all jobs.

    More stable than P99 with small sample sizes (N < 100).
    See p99_slowdown() for metric definition.

    Args:
        jobs: List of Job objects
        tau_seconds: Floor on job runtime in real seconds (default 60 seconds)
        sim_seconds_per_unit: Seconds per simulator time unit. Default 3600 (sim unit = hours)

    Returns:
        P95 bounded slowdown as float, or 0.0 if no finished jobs
    """
    tau = tau_seconds / sim_seconds_per_unit
    finished_jobs = [j for j in jobs if j.start is not None and j.finish is not None]
    if not finished_jobs:
        return 0.0

    slowdowns = []
    for job in finished_jobs:
        response_time = job.finish - job.arrival
        effective_runtime = max(job.actual_time, tau)
        slowdown = response_time / effective_runtime
        slowdowns.append(slowdown)

    return float(np.percentile(slowdowns, 95))


def p99_slowdown(jobs, tau_seconds=60.0, sim_seconds_per_unit=3600.0):
    """
    Calculate the 99th percentile of bounded slowdown across all jobs.

    Bounded slowdown (stretch factor) is a standard fairness metric in scheduling literature.
    For each job j:
      - response_time_j = finish_j - arrival_j
      - tau = minimum threshold (prevents tiny jobs from skewing the metric)
      - bounded_slowdown_j = response_time_j / max(actual_time_j, tau)

    Args:
        jobs: List of Job objects
        tau_seconds: Floor on job runtime in real seconds (default 60 seconds)
        sim_seconds_per_unit: Seconds per simulator time unit. Default 3600 (sim unit = hours)

    Returns:
        P99 bounded slowdown as float, or 0.0 if no finished jobs
    """
    # Convert tau to simulator time units
    tau = tau_seconds / sim_seconds_per_unit

    finished_jobs = [j for j in jobs if j.start is not None and j.finish is not None]
    if not finished_jobs:
        return 0.0

    slowdowns = []
    for job in finished_jobs:
        response_time = job.finish - job.arrival
        effective_runtime = max(job.actual_time, tau)
        slowdown = response_time / effective_runtime
        slowdowns.append(slowdown)

    return float(np.percentile(slowdowns, 99))
