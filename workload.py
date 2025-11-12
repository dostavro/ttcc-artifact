"""
Synthetic workload generation for GPU job simulation.

Generates jobs with:
- Poisson arrivals (specified arrival rate)
- Exponential runtimes (parameterizable mean)
- QoS mix (Gold: deadline-sensitive, Silver: medium, Bronze: loose deadlines)
- GPU demand distribution (1–4 GPUs small clusters, 1–4, 8 GPUs medium/large)
- Deadline tightness per QoS class

Supports Common Random Numbers (CRN) via configurable seed for low-variance
cross-scheduler comparisons.
"""
import numpy as np
import random
from jobs import Job


def generate_jobs(
    num_jobs=200,
    arrival_rate=10,         # jobs per hour
    mean_runtime=2.0,        # mean runtime (hours)
    p_early=0.3,             # probability job ends early
    qos_distribution={"Gold": 0.2, "Silver": 0.5, "Bronze": 0.3},
    deadline_alpha_gold=1.0,  # deadline tightness for Gold jobs
    deadline_alpha_silver=1.5,  # deadline tightness for Silver jobs
    demand_distribution=None,  # GPU demand distribution: {demand: probability}
    seed=42
):
    """
    Generate a synthetic workload of jobs for simulation.
    Returns a list of Job objects.

    Args:
        demand_distribution: dict mapping GPU counts to probabilities.
            Default: {1: 0.4, 2: 0.35, 3: 0.2, 4: 0.05} (Small clusters)
            Extended: {1: 0.35, 2: 0.30, 3: 0.20, 4: 0.10, 8: 0.05} (Medium/Large)
    """
    np.random.seed(seed)
    random.seed(seed)

    # Default demand distribution (1-4 GPUs)
    if demand_distribution is None:
        demand_distribution = {1: 0.4, 2: 0.35, 3: 0.2, 4: 0.05}

    jobs = []
    t = 0.0
    for jid in range(num_jobs):
        # Interarrival time ~ Exponential(lambda = arrival_rate)
        interarrival = np.random.exponential(1.0 / arrival_rate)
        t += interarrival
        arrival = t

        # Runtime
        reserved_time = np.random.exponential(mean_runtime)
        if np.random.rand() < p_early:
            actual_time = reserved_time * np.random.uniform(0.3, 0.8)
        else:
            actual_time = reserved_time

        # QoS class
        qos = random.choices(
            list(qos_distribution.keys()),
            weights=list(qos_distribution.values()),
            k=1
        )[0]

        # GPU demand from specified distribution
        demand_values = list(demand_distribution.keys())
        demand_probs = list(demand_distribution.values())
        demand = np.random.choice(demand_values, p=demand_probs)

        job = Job(
            jid=jid,
            arrival=arrival,
            demand=demand,
            reserved_time=reserved_time,
            actual_time=actual_time,
            qos=qos
        )

        # Set deadline based on QoS class and alpha parameters
        if qos == "Gold":
            job.deadline = arrival + reserved_time * deadline_alpha_gold
        elif qos == "Silver":
            job.deadline = arrival + reserved_time * deadline_alpha_silver
        else:  # Bronze - no deadline or very loose
            job.deadline = arrival + reserved_time * 3.0

        jobs.append(job)

    return jobs
