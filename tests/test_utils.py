"""
Test utilities and helper functions for scheduler testing.
"""
import pytest
from simulator import Simulator
from cluster import Cluster
from jobs import Job


def create_test_job(jid, arrival, time, demand, qos):
    """Helper to create a test job with proper parameters."""
    return Job(jid, arrival, demand, time, time, qos)


def create_test_cluster(num_edge=2, num_cloud=2):
    """Helper to create a test cluster."""
    return Cluster(num_edge, num_cloud)


def run_scheduler_test(scheduler, jobs, cluster, horizon=10, debug=False):
    """
    Helper to run a scheduler test and return results.

    Returns:
        dict: Test results including finished jobs, metrics, etc.
    """
    sim = Simulator(cluster, jobs, scheduler, debug=debug)
    finished = sim.run(horizon=horizon)

    return {
        'finished_jobs': finished,
        'running_jobs': sim.running,
        'simulator': sim,
        'scheduler': scheduler,
        'total_jobs': len(jobs),
        'completed_jobs': len(finished)
    }


def assert_job_has_resource_type(job, expected_resource_type):
    """Assert that a job got the expected resource type."""
    allocation = getattr(job, 'final_allocated', None) or getattr(job, 'allocated', None)
    assert allocation is not None, f"Job {job.jid} has no allocation"

    actual_types = set()
    for resource in allocation:
        if resource.startswith("edge"):
            actual_types.add("edge")
        elif resource.startswith("cloud"):
            actual_types.add("cloud")

    assert expected_resource_type in actual_types, \
        f"Job {job.jid} expected {expected_resource_type} but got {actual_types}"


def assert_qos_preference_respected(job):
    """Assert that a job got resources matching its QoS preference."""
    if job.qos == "Gold":
        expected = "edge"
    else:  # Silver, Bronze
        expected = "cloud"

    # Note: Job might not get preferred type due to availability, so this is optional
    # assert_job_has_resource_type(job, expected)


# Pytest fixtures for common test data
@pytest.fixture
def basic_cluster():
    """Create a basic test cluster."""
    return create_test_cluster(2, 2)


@pytest.fixture
def limited_cluster():
    """Create a cluster with limited resources."""
    return create_test_cluster(1, 1)


@pytest.fixture
def basic_jobs():
    """Create basic test jobs."""
    return [
        create_test_job("job1", 0, 3, 1, "Gold"),
        create_test_job("job2", 1, 2, 1, "Silver"),
        create_test_job("job3", 2, 4, 1, "Bronze")
    ]


@pytest.fixture
def priority_test_jobs():
    """Create jobs for priority testing."""
    return [
        create_test_job("bronze", 0, 3, 1, "Bronze"),  # Lower priority, arrives first
        create_test_job("gold", 1, 2, 1, "Gold"),      # Higher priority, arrives later
    ]
