"""
Tests for RunAI scheduler.

RunAI implements quota-based fair-share scheduling across QoS classes.
Default quotas: Gold=3, Silver=2, Bronze=1
Jobs are scheduled based on current usage/quota ratio (lower ratio = higher priority).
"""
import pytest
from test_utils import create_test_job, create_test_cluster, run_scheduler_test
from schedulers.runai import RunAI


def test_runai_basic():
    """Test basic RunAI functionality with no resource contention."""
    scheduler = RunAI()
    cluster = create_test_cluster(2, 2)  # 4 total resources
    jobs = [
        create_test_job("gold", 0, 2, 1, "Gold"),
        create_test_job("silver", 1, 2, 1, "Silver"),
        create_test_job("bronze", 2, 2, 1, "Bronze")
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=10)

    # All jobs should complete
    assert len(result['finished_jobs']) == 3, "All jobs should complete"

    # All jobs should have proper allocations
    for job in result['finished_jobs']:
        allocation = getattr(job, 'final_allocated', None) or getattr(job, 'allocated', None)
        assert allocation is not None, f"Job {job.jid} had no allocation"
        assert len(allocation) == job.demand, f"Job {job.jid} should get {job.demand} resources"
        assert job.start >= job.arrival, "Jobs shouldn't start before arrival"


def test_runai_quota_based_fairness():
    """
    Test RunAI's core quota-based scheduling: classes with higher cumulative
    usage/quota ratios should yield to classes with lower ratios.

    Also verifies no starvation: even with many Gold jobs, lower-quota classes
    still get scheduled based on their accumulated usage ratios.
    """
    scheduler = RunAI()  # Default quotas: Gold=3, Silver=2, Bronze=1
    cluster = create_test_cluster(2, 0)  # 2 resources

    jobs = [
        # Bronze arrives first, will accumulate high usage ratio
        create_test_job("bronze1", 0, 3, 1, "Bronze"),
        # Multiple Gold jobs arrive - could monopolize, but shouldn't
        create_test_job("gold1", 1, 3, 1, "Gold"),
        create_test_job("gold2", 1, 3, 1, "Gold"),
        create_test_job("gold3", 2, 3, 1, "Gold"),
        # Bronze and Silver should still get scheduled despite Gold dominance
        create_test_job("bronze2", 2, 3, 1, "Bronze"),
        create_test_job("silver1", 3, 3, 1, "Silver"),
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=20)
    assert len(result['finished_jobs']) == 6, "All jobs should complete"

    bronze1 = next(j for j in result['finished_jobs'] if j.jid == "bronze1")
    bronze2 = next(j for j in result['finished_jobs'] if j.jid == "bronze2")
    silver1 = next(j for j in result['finished_jobs'] if j.jid == "silver1")
    gold_jobs = [j for j in result['finished_jobs'] if j.jid.startswith("gold")]

    # Bronze1 starts immediately (first arrival, ratio=0)
    assert bronze1.start == 0, "Bronze1 should start immediately"

    # After Bronze1 finishes, Bronze has high usage ratio (3/1=3.0)
    # Gold/Silver should be prioritized over Bronze2
    jobs_after_bronze1 = [j for j in result['finished_jobs']
                          if j.start >= bronze1.finish and j.jid != "bronze1"]
    next_job = min(jobs_after_bronze1, key=lambda j: j.start)
    assert next_job.jid != "bronze2", \
        "After Bronze1 finishes, Gold/Silver should be prioritized over Bronze2"

    # Verify no starvation: Silver and Bronze2 should start before all Gold jobs finish
    max_gold_finish = max(j.finish for j in gold_jobs)
    assert bronze2.start < max_gold_finish, "Bronze2 should not be starved by Gold"
    assert silver1.start < max_gold_finish, "Silver should not be starved by Gold"


def test_runai_respects_quota_weights():
    """
    Test that RunAI respects quota weights when distributing resources over time.

    With default quotas (Gold=3, Silver=2, Bronze=1), when all classes submit
    the SAME total workload (demand × duration), the scheduler should allocate
    resources proportionally based on quotas.

    Key principle: Quota-based fairness means that over time, resource allocation
    should reflect quota ratios. This is measured by average job completion times:
    classes with higher quotas should have better (lower) average completion times.
    """
    scheduler = RunAI()
    cluster = create_test_cluster(2, 0)  # 2 resources, forces competition

    # CRITICAL: All classes submit same TOTAL workload
    # Each class submits 18 resource-time units of work (3 jobs × 6 duration × 1 demand)
    # This ensures we're testing scheduler fairness, not workload imbalance
    jobs = [
        # Gold jobs (quota=3): 3 jobs × 6 duration × 1 demand = 18 resource-time
        create_test_job("gold1", 0, 6, 1, "Gold"),
        create_test_job("gold2", 1, 6, 1, "Gold"),
        create_test_job("gold3", 2, 6, 1, "Gold"),

        # Silver jobs (quota=2): 3 jobs × 6 duration × 1 demand = 18 resource-time
        create_test_job("silver1", 0, 6, 1, "Silver"),
        create_test_job("silver2", 1, 6, 1, "Silver"),
        create_test_job("silver3", 2, 6, 1, "Silver"),

        # Bronze jobs (quota=1): 3 jobs × 6 duration × 1 demand = 18 resource-time
        create_test_job("bronze1", 0, 6, 1, "Bronze"),
        create_test_job("bronze2", 1, 6, 1, "Bronze"),
        create_test_job("bronze3", 2, 6, 1, "Bronze"),
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=40)

    # All jobs should complete
    assert len(result['finished_jobs']) == 9, "All jobs should complete"

    gold_jobs = [j for j in result['finished_jobs'] if j.jid.startswith("gold")]
    silver_jobs = [j for j in result['finished_jobs'] if j.jid.startswith("silver")]
    bronze_jobs = [j for j in result['finished_jobs'] if j.jid.startswith("bronze")]

    # Verify equal workload was submitted (sanity check)
    # Workload = actual_time (job duration) * demand (resources needed)
    gold_workload = sum(j.actual_time * j.demand for j in gold_jobs)
    silver_workload = sum(j.actual_time * j.demand for j in silver_jobs)
    bronze_workload = sum(j.actual_time * j.demand for j in bronze_jobs)
    assert gold_workload == silver_workload == bronze_workload == 18, \
        "Test setup error: all classes should submit equal workload"

    # Measure average completion time per class
    # With quota-based fairness and equal workloads:
    # - Higher quota → better service → lower average completion time
    # - Lower quota → worse service → higher average completion time
    gold_avg_completion = sum(j.finish for j in gold_jobs) / len(gold_jobs)
    silver_avg_completion = sum(j.finish for j in silver_jobs) / len(silver_jobs)
    bronze_avg_completion = sum(j.finish for j in bronze_jobs) / len(bronze_jobs)

    # Verify quota-based prioritization: Gold < Silver < Bronze (approximately)
    # Gold (quota=3) should have the best (lowest) average completion time
    assert gold_avg_completion < bronze_avg_completion, \
        f"Gold (quota=3) should have better avg completion than Bronze (quota=1): " \
        f"Gold avg={gold_avg_completion:.1f}, Bronze avg={bronze_avg_completion:.1f}"

    # Silver (quota=2) should have better or equal service than Bronze (quota=1)
    # Use <= because with discrete arrivals and finite resources, exact ordering may vary
    assert silver_avg_completion <= bronze_avg_completion, \
        f"Silver (quota=2) should have better/equal avg completion than Bronze (quota=1): " \
        f"Silver avg={silver_avg_completion:.1f}, Bronze avg={bronze_avg_completion:.1f}"

    # Gold (quota=3) should have strictly better service than Silver (quota=2)
    assert gold_avg_completion < silver_avg_completion, \
        f"Gold (quota=3) should have better avg completion than Silver (quota=2): " \
        f"Gold avg={gold_avg_completion:.1f}, Silver avg={silver_avg_completion:.1f}"

    # Print debug info to show quota-based fairness in action
    print(f"\n=== Quota-Based Fairness Metrics (Equal Workload=18 each) ===")
    print(f"Gold   (quota=3): avg completion = {gold_avg_completion:.1f}")
    print(f"Silver (quota=2): avg completion = {silver_avg_completion:.1f}")
    print(f"Bronze (quota=1): avg completion = {bronze_avg_completion:.1f}")
    print(f"Expected: Gold < Silver ≤ Bronze")
    print(f"✓ Quota-based prioritization verified")


def test_runai_custom_quotas():
    """
    Test RunAI with custom quota configuration that inverts priority.

    With custom quotas Bronze=10, Silver=1, Gold=1, Bronze jobs should
    be strongly favored when competing for the same resources, resulting
    in Bronze completing its work faster overall.

    This verifies that RunAI actually uses the quota values, not hardcoded QoS priorities.
    """
    # Give Bronze a very high quota, make Silver and Gold equal with low quotas
    custom_quotas = {"Gold": 1, "Silver": 1, "Bronze": 10}
    scheduler = RunAI(quotas=custom_quotas)

    cluster = create_test_cluster(1, 0)  # 1 resource - forces clear competition

    # All jobs arrive at the same time to create maximum competition
    jobs = [
        # Gold jobs (quota=1) - 3 jobs
        create_test_job("gold1", 0, 3, 1, "Gold"),
        create_test_job("gold2", 0, 3, 1, "Gold"),
        create_test_job("gold3", 0, 3, 1, "Gold"),

        # Silver jobs (quota=1) - 3 jobs
        create_test_job("silver1", 0, 3, 1, "Silver"),
        create_test_job("silver2", 0, 3, 1, "Silver"),
        create_test_job("silver3", 0, 3, 1, "Silver"),

        # Bronze jobs (quota=10) - 3 jobs
        create_test_job("bronze1", 0, 3, 1, "Bronze"),
        create_test_job("bronze2", 0, 3, 1, "Bronze"),
        create_test_job("bronze3", 0, 3, 1, "Bronze"),
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=30)

    # All should complete
    assert len(result['finished_jobs']) == 9, "All jobs should complete"

    gold_jobs = [j for j in result['finished_jobs'] if j.jid.startswith("gold")]
    bronze_jobs = [j for j in result['finished_jobs'] if j.jid.startswith("bronze")]

    # With quota-based scheduling, Bronze (quota=10) should be heavily favored
    # All Bronze jobs should complete before most Gold/Silver jobs
    # At least some Bronze jobs should finish before any Gold/Silver jobs start finishing
    # (showing Bronze gets prioritized due to high quota)
    bronze_finishes = sorted([j.finish for j in bronze_jobs])
    gold_finishes = sorted([j.finish for j in gold_jobs])

    # With 10:1 quota ratio, Bronze should dominate early execution
    # The first Bronze job should finish very early
    assert bronze_finishes[0] <= gold_finishes[0], \
        f"First Bronze should finish before or with first Gold: Bronze={bronze_finishes[0]}, Gold={gold_finishes[0]}"

    # Most Bronze jobs should complete before most Gold jobs
    # (due to sustained quota-based prioritization)
    assert bronze_finishes[-1] < gold_finishes[-1], \
        f"All Bronze should complete before all Gold: Last Bronze={bronze_finishes[-1]}, Last Gold={gold_finishes[-1]}"


def test_runai_mixed_demands():
    """
    Test RunAI with mixed resource demands (fragmentation scenario).

    With 2 resources, Gold jobs (demand=2) can only run when both resources are free,
    while Bronze jobs (demand=1) can run in pairs. This tests:
    - Correct resource allocation (Gold gets 2, Bronze gets 1)
    - No overlapping execution (Gold monopolizes both resources)
    - Batched execution pattern (Bronze → Gold → Bronze)
    - Fragmentation allows lower-quota class early execution opportunities
    """
    scheduler = RunAI()
    cluster = create_test_cluster(2, 0)  # 2 resources

    jobs = [
        create_test_job("gold_large1", 0, 4, 2, "Gold"),
        create_test_job("gold_large2", 1, 4, 2, "Gold"),
        create_test_job("bronze1", 0, 4, 1, "Bronze"),
        create_test_job("bronze2", 0, 4, 1, "Bronze"),
        create_test_job("bronze3", 0, 4, 1, "Bronze"),
        create_test_job("bronze4", 0, 4, 1, "Bronze"),
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=30)
    assert len(result['finished_jobs']) == 6, "All jobs should complete"

    gold_jobs = [j for j in result['finished_jobs'] if j.jid.startswith("gold")]
    bronze_jobs = [j for j in result['finished_jobs'] if j.jid.startswith("bronze")]

    # Verify correct resource allocations
    for gold in gold_jobs:
        assert len(gold.final_allocated or gold.allocated) == 2, f"{gold.jid} should get 2 resources"
    for bronze in bronze_jobs:
        assert len(bronze.final_allocated or bronze.allocated) == 1, f"{bronze.jid} should get 1 resource"

    # Verify no overlapping: Gold (demand=2) uses both resources, Bronze cannot run concurrently
    for gold in gold_jobs:
        for bronze in bronze_jobs:
            overlap = not (gold.finish <= bronze.start or bronze.finish <= gold.start)
            assert not overlap, f"{gold.jid} should not overlap with {bronze.jid}"

    # Verify batched execution: Some Bronze before Gold, some after
    earliest_gold_start = min(j.start for j in gold_jobs)
    latest_gold_finish = max(j.finish for j in gold_jobs)

    bronze_before_gold = [j for j in bronze_jobs if j.finish <= earliest_gold_start]
    bronze_after_gold = [j for j in bronze_jobs if j.start >= latest_gold_finish]

    assert len(bronze_before_gold) >= 1, "Bronze should run before Gold"
    assert len(bronze_after_gold) >= 1, "Bronze should run after Gold"

    # Verify fragmentation effect: Bronze gets early execution despite lower quota
    bronze_completed_before_gold_finishes = [j for j in bronze_jobs if j.finish <= latest_gold_finish]
    assert len(bronze_completed_before_gold_finishes) >= 2, \
        "Fragmentation should allow Bronze early execution opportunities"


if __name__ == "__main__":
    pytest.main([__file__])
