"""
Pytest configuration and shared fixtures for scheduler tests.
"""
import pytest
from test_utils import create_test_job, create_test_cluster


@pytest.fixture
def basic_cluster():
    """Create a basic test cluster with 2 edge and 2 cloud GPUs."""
    return create_test_cluster(2, 2)


@pytest.fixture
def limited_cluster():
    """Create a cluster with limited resources."""
    return create_test_cluster(1, 1)


@pytest.fixture
def basic_jobs():
    """Create basic test jobs for general testing."""
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


@pytest.fixture
def edge_only_cluster():
    """Create a cluster with only edge GPUs."""
    return create_test_cluster(2, 0)


@pytest.fixture
def cloud_only_cluster():
    """Create a cluster with only cloud GPUs."""
    return create_test_cluster(0, 2)
