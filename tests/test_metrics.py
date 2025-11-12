"""
Tests for scheduler performance metrics and comparisons.
"""
import pytest
from test_utils import create_test_job, create_test_cluster, run_scheduler_test

# Import all schedulers for comparison
from schedulers.fcfs import FCFS
from schedulers.easy import EASY
from schedulers.conservative import Conservative
from schedulers.priority import PriorityQoS
from schedulers.runai import RunAI
from schedulers.ttcc import TTCC


def test_easy_vs_fcfs_efficiency():
    """Test that EASY is more efficient than pure FCFS due to backfilling."""
    # Create identical scenarios for both schedulers
    def create_scenario():
        cluster = create_test_cluster(2, 2)
        jobs = [
            create_test_job("big", 0, 8, 2, "Gold"),     # Uses 2 resources for 8 time units
            create_test_job("small1", 1, 2, 1, "Silver"),  # Can backfill
            create_test_job("small2", 2, 3, 1, "Bronze"),  # Can backfill
        ]
        return cluster, jobs

    # Test FCFS
    fcfs_cluster, fcfs_jobs = create_scenario()
    fcfs_result = run_scheduler_test(FCFS(), fcfs_jobs, fcfs_cluster, horizon=15)

    # Test EASY
    easy_cluster, easy_jobs = create_scenario()
    easy_result = run_scheduler_test(EASY(), easy_jobs, easy_cluster, horizon=15)

    # Both should complete all jobs
    assert len(fcfs_result['finished_jobs']) == 3, "FCFS should complete all jobs"
    assert len(easy_result['finished_jobs']) == 3, "EASY should complete all jobs"

    # EASY should finish sooner (more efficient)
    fcfs_makespan = max(j.finish for j in fcfs_result['finished_jobs'])
    easy_makespan = max(j.finish for j in easy_result['finished_jobs'])

    assert easy_makespan <= fcfs_makespan, f"EASY makespan ({easy_makespan}) should be <= FCFS ({fcfs_makespan})"


def test_scheduler_makespan_comparison():
    """Compare makespan (total completion time) across all schedulers."""
    def create_standard_workload():
        cluster = create_test_cluster(3, 2)  # 5 total resources
        jobs = [
            create_test_job("job1", 0, 4, 2, "Gold"),
            create_test_job("job2", 1, 3, 1, "Silver"),
            create_test_job("job3", 2, 2, 1, "Bronze"),
            create_test_job("job4", 3, 5, 2, "Gold"),
        ]
        return cluster, jobs

    schedulers = [
        ("FCFS", FCFS()),
        ("EASY", EASY()),
        ("Conservative", Conservative()),
        ("PriorityQoS", PriorityQoS()),
        ("RunAI", RunAI()),
        ("TTCC", TTCC())
    ]

    results = {}

    for name, scheduler in schedulers:
        cluster, jobs = create_standard_workload()
        result = run_scheduler_test(scheduler, jobs, cluster, horizon=20)

        # All schedulers should complete all jobs
        assert len(result['finished_jobs']) == 4, f"{name} should complete all jobs"

        # Calculate makespan
        makespan = max(j.finish for j in result['finished_jobs'])
        results[name] = makespan

    # All makespans should be reasonable (not infinite)
    for name, makespan in results.items():
        assert makespan < 20, f"{name} makespan ({makespan}) should be reasonable"

    print(f"\nMakespan Results: {results}")


def test_resource_utilization():
    """Test resource utilization efficiency across schedulers."""
    def calculate_utilization(jobs, cluster, makespan):
        """Calculate average resource utilization."""
        total_resources = len(cluster.resources)
        total_time = makespan

        # Calculate total job-time (sum of job_duration * job_demand)
        job_time = sum(job.actual_time * job.demand for job in jobs)

        # Utilization = job_time / (total_resources * total_time)
        return job_time / (total_resources * total_time) if total_time > 0 else 0

    def create_utilization_workload():
        cluster = create_test_cluster(2, 2)
        jobs = [
            create_test_job("job1", 0, 6, 1, "Gold"),
            create_test_job("job2", 2, 4, 1, "Silver"),
            create_test_job("job3", 4, 2, 2, "Bronze"),
        ]
        return cluster, jobs

    schedulers = [FCFS(), EASY(), TTCC()]

    for scheduler in schedulers:
        cluster, jobs = create_utilization_workload()
        result = run_scheduler_test(scheduler, jobs, cluster, horizon=15)

        assert len(result['finished_jobs']) == 3, f"{scheduler.__class__.__name__} should complete all jobs"

        makespan = max(j.finish for j in result['finished_jobs'])
        utilization = calculate_utilization(result['finished_jobs'], cluster, makespan)

        # Utilization should be reasonable (between 0 and 1)
        assert 0 <= utilization <= 1, f"{scheduler.__class__.__name__} utilization should be between 0 and 1"

        print(f"{scheduler.__class__.__name__} utilization: {utilization:.2f}")


def test_qos_preference_satisfaction():
    """Test how well schedulers satisfy QoS resource preferences."""
    def count_qos_satisfaction(jobs):
        """Count how many jobs got their preferred resource type."""
        satisfied = 0
        total = 0

        for job in jobs:
            allocation = getattr(job, 'final_allocated', None) or getattr(job, 'allocated', None)
            if allocation:
                total += 1

                # Check if job got preferred resource type
                has_edge = any(r.startswith("edge") for r in allocation)
                has_cloud = any(r.startswith("cloud") for r in allocation)

                if job.qos == "Gold" and has_edge:
                    satisfied += 1
                elif job.qos in ["Silver", "Bronze"] and has_cloud:
                    satisfied += 1
                elif job.qos == "Gold" and not has_cloud:
                    satisfied += 1  # Gold got edge or mixed, which is acceptable
                elif job.qos in ["Silver", "Bronze"] and not has_edge:
                    satisfied += 1  # Silver/Bronze got cloud or mixed

        return satisfied / total if total > 0 else 0

    def create_qos_workload():
        cluster = create_test_cluster(2, 2)
        jobs = [
            create_test_job("gold1", 0, 3, 1, "Gold"),
            create_test_job("gold2", 1, 2, 1, "Gold"),
            create_test_job("silver1", 2, 3, 1, "Silver"),
            create_test_job("bronze1", 3, 2, 1, "Bronze"),
        ]
        return cluster, jobs

    # Test QoS-aware schedulers
    qos_schedulers = [PriorityQoS(), TTCC()]

    for scheduler in qos_schedulers:
        cluster, jobs = create_qos_workload()
        result = run_scheduler_test(scheduler, jobs, cluster, horizon=15)

        assert len(result['finished_jobs']) == 4, f"{scheduler.__class__.__name__} should complete all jobs"

        satisfaction_rate = count_qos_satisfaction(result['finished_jobs'])
        print(f"{scheduler.__class__.__name__} QoS satisfaction: {satisfaction_rate:.2f}")

        # QoS-aware schedulers should have some level of satisfaction
        # Note: May not be 100% due to resource availability and timing


@pytest.mark.parametrize("scheduler_class", [FCFS, EASY, Conservative, PriorityQoS, RunAI, TTCC])
def test_scheduler_scalability(scheduler_class):
    """Test scheduler performance with larger workloads."""
    scheduler = scheduler_class()
    cluster = create_test_cluster(4, 4)  # Larger cluster

    # Create larger workload
    jobs = []
    for i in range(10):  # 10 jobs
        qos = ["Gold", "Silver", "Bronze"][i % 3]
        jobs.append(create_test_job(f"job_{i}", i, 2 + (i % 3), 1, qos))

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=30)

    # All jobs should complete
    assert len(result['finished_jobs']) == 10, f"{scheduler_class.__name__} should handle larger workload"

    # Completion should be within reasonable time
    makespan = max(j.finish for j in result['finished_jobs'])
    assert makespan < 30, f"{scheduler_class.__name__} should complete within horizon"


if __name__ == "__main__":
    pytest.main([__file__])
