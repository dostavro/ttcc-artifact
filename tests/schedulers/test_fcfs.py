"""
Tests for FCFS (First Come First Serve) scheduler.
"""
from test_utils import *
from schedulers.fcfs import FCFS


def test_fcfs_basic():
    """Test basic FCFS functionality."""
    scheduler = FCFS()
    cluster = create_test_cluster(2, 2)
    jobs = [
        create_test_job("job1", 0, 3, 1, "Gold"),
        create_test_job("job2", 1, 2, 1, "Silver"),
        create_test_job("job3", 2, 4, 1, "Bronze")
    ]

    result = run_scheduler_test(scheduler, jobs, cluster)
    assert len(result['finished_jobs']) == 3, "All jobs should complete"


def test_fcfs_arrival_order():
    """Test that FCFS respects arrival order with different arrival times."""
    scheduler = FCFS()
    cluster = create_test_cluster(1, 0)  # Only 1 edge GPU, forcing competition
    jobs = [
        create_test_job("early", 0, 2, 1, "Gold"),  # Arrives at t=0
        create_test_job("late", 1, 1, 1, "Gold"),   # Arrives at t=1, shorter but later
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=5)

    # Both jobs should complete
    assert len(result['finished_jobs']) == 2, "Both jobs should complete"

    # Early job should start first despite being longer
    early_job = next(j for j in result['finished_jobs'] if j.jid == "early")
    late_job = next(j for j in result['finished_jobs'] if j.jid == "late")

    assert early_job.start < late_job.start, "FCFS should start jobs in arrival order"
    assert late_job.start >= early_job.finish, "Late job should wait for early job"


def test_fcfs_simultaneous_arrival():
    """Test FCFS behavior when jobs arrive simultaneously."""
    scheduler = FCFS()
    cluster = create_test_cluster(2, 2)  # Sufficient resources for parallel execution
    jobs = [
        create_test_job("job1", 0, 3, 1, "Gold"),    # Same arrival time
        create_test_job("job2", 0, 2, 1, "Silver"),  # Same arrival time, different QoS
        create_test_job("job3", 0, 1, 1, "Bronze"),  # Same arrival time, shortest
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=10)

    # All jobs should complete
    assert len(result['finished_jobs']) == 3, "All jobs should complete"

    # With sufficient resources, jobs can run in parallel
    # FCFS doesn't guarantee specific ordering for simultaneous arrivals
    # but all should get resources quickly
    for job in result['finished_jobs']:
        assert job.start == job.arrival, "Jobs should start immediately with sufficient resources"
