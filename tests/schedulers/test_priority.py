"""
Tests for Priority QoS scheduler.
"""
from test_utils import *
from schedulers.priority import PriorityQoS


def test_priority_basic():
    """Test basic Priority QoS functionality."""
    scheduler = PriorityQoS()
    cluster = create_test_cluster(2, 2)
    jobs = [
        create_test_job("bronze", 0, 1, 1, "Bronze"),   # arrival, time, demand, qos
        create_test_job("silver", 1, 1, 1, "Silver"),
        create_test_job("gold", 2, 1, 1, "Gold")
    ]

    result = run_scheduler_test(scheduler, jobs, cluster)
    assert len(result['finished_jobs']) == 3, "All jobs should complete"


def test_priority_ordering():
    """Test that Priority QoS with preemption allows higher-priority jobs to preempt."""
    # With preemption enabled, Gold should preempt Bronze even if Bronze started first
    scheduler = PriorityQoS(enable_preemption=True)
    cluster = create_test_cluster(2, 0)  # 2 GPUs
    jobs = [
        create_test_job("bronze", 0, 5, 2, "Bronze"),  # Needs 2 GPUs, starts first
        create_test_job("gold", 2, 2, 2, "Gold"),      # Higher priority, should preempt Bronze
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=20)

    finished_jobs = {j.jid: j for j in result['finished_jobs']}
    bronze_job = finished_jobs["bronze"]
    gold_job = finished_jobs["gold"]

    # Bronze starts first at t=0
    assert bronze_job.start == 0, "Bronze job should start at time 0"

    # Gold arrives at t=2 and should preempt Bronze immediately (at t=2)
    assert gold_job.start == 2, "Gold job should start at time 2 (by preempting Bronze)"

    # Gold finishes quickly (duration 2), so at t=4
    assert gold_job.finish == 4, f"Gold job should finish at time 2+2=4, got {gold_job.finish}"

    # Bronze had 5 units of work, ran for 2 units (t=0 to t=2), so 3 remaining
    # Bronze resumes at t=4 and finishes at t=4+3=7
    assert bronze_job.finish == 7, f"Bronze job should finish at time 4+3=7, got {bronze_job.finish}"
    # Both should complete
    assert len(finished_jobs) == 2, "Both jobs should complete"


def test_priority_qos_preferences():
    """Test that Priority QoS considers resource preferences."""
    scheduler = PriorityQoS()
    cluster = create_test_cluster(1, 1)
    jobs = [
        create_test_job("gold", 0, 1, 1, "Gold"),    # Should prefer edge
        create_test_job("silver", 1, 1, 1, "Silver")  # Should prefer cloud
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=10)
    assert len(result['finished_jobs']) == 2, "Both jobs should complete"

    finished_jobs = {j.jid: j for j in result['finished_jobs']}
    gold_job = finished_jobs["gold"]
    silver_job = finished_jobs["silver"]

    if gold_job.final_allocated:
        assert any(r.startswith("edge") for r in gold_job.final_allocated), "Gold job should get edge resource"
    if silver_job.final_allocated:
        assert any(r.startswith("cloud") for r in silver_job.final_allocated), "Silver job should get cloud resource"


def test_priority_preemption_enabled():
    """Test that preemption works when enabled (default)."""
    scheduler = PriorityQoS(enable_preemption=True)
    cluster = create_test_cluster(2, 0)  # Only 2 GPUs
    jobs = [
        create_test_job("bronze", 0, 5, 2, "Bronze"),  # arrival=0, time=5, demand=2
        create_test_job("gold", 2, 2, 1, "Gold"),      # arrival=2, time=2, demand=1 (should preempt)
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=20)
    assert len(result['finished_jobs']
               ) >= 2, f"Both jobs should eventually complete, got {len(result['finished_jobs'])}"

    finished_jobs = {j.jid: j for j in result['finished_jobs']}
    assert "gold" in finished_jobs, "Gold job should finish"
    assert "bronze" in finished_jobs, "Bronze job should finish after preemption"

    gold_job = finished_jobs["gold"]
    bronze_job = finished_jobs["bronze"]

    # Gold should start soon after arrival (at t=2) via preemption
    assert gold_job.start == 2, f"Gold job should start at time 2 (by preemption), got {gold_job.start}"

    # Bronze should be preempted, then finish later after Gold completes
    assert bronze_job.finish > gold_job.finish, "Bronze should finish after Gold due to preemption"


def test_priority_preemption_disabled():
    """Test that preemption respects disable flag."""
    scheduler = PriorityQoS(enable_preemption=False)
    cluster = create_test_cluster(2, 0)
    jobs = [
        create_test_job("bronze", 0, 5, 2, "Bronze"),  # arrival=0, time=5, demand=2
        create_test_job("gold", 2, 2, 1, "Gold"),      # arrival=2, time=2, demand=1
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=20)
    assert len(result['finished_jobs']) >= 2, "Both jobs should complete"

    finished_jobs = {j.jid: j for j in result['finished_jobs']}
    gold_job = finished_jobs["gold"]
    bronze_job = finished_jobs["bronze"]

    # Without preemption, Gold must wait for Bronze to finish
    assert gold_job.start >= bronze_job.finish, "Gold should wait for Bronze without preemption"


def test_priority_silver_bronze_preemption_policy():
    """Test Silver->Bronze preemption policy setting."""
    # With preemption enabled and allow_silver_preempt_bronze=True
    scheduler_allow = PriorityQoS(enable_preemption=True, allow_silver_preempt_bronze=True)
    cluster = create_test_cluster(1, 0)
    jobs = [
        create_test_job("bronze", 0, 5, 1, "Bronze"),
        create_test_job("silver", 1, 2, 1, "Silver"),
    ]

    result = run_scheduler_test(scheduler_allow, jobs, cluster, horizon=20)
    silver_job = next(j for j in result['finished_jobs'] if j.jid == "silver")
    # Silver should preempt Bronze when allowed
    assert silver_job.start == 1, "Silver should start at arrival with preemption allowed"

    # With preemption enabled but allow_silver_preempt_bronze=False
    scheduler_disallow = PriorityQoS(enable_preemption=True, allow_silver_preempt_bronze=False)
    cluster2 = create_test_cluster(1, 0)
    jobs2 = [
        create_test_job("bronze", 0, 5, 1, "Bronze"),
        create_test_job("silver", 1, 2, 1, "Silver"),
    ]

    result2 = run_scheduler_test(scheduler_disallow, jobs2, cluster2, horizon=20)
    finished_jobs2 = {j.jid: j for j in result2['finished_jobs']}
    silver_job2 = finished_jobs2.get("silver")
    bronze_job2 = finished_jobs2.get("bronze")

    if silver_job2 and bronze_job2:
        # Silver should NOT preempt Bronze when disallowed
        assert silver_job2.start >= bronze_job2.finish, "Silver should wait for Bronze when preemption disallowed"


def test_priority_work_remaining_after_preemption():
    """Test that preempted jobs track work remaining correctly."""
    scheduler = PriorityQoS(enable_preemption=True)
    cluster = create_test_cluster(2, 0)
    jobs = [
        create_test_job("bronze", 0, 10, 2, "Bronze"),  # Long-running
        create_test_job("gold", 5, 3, 2, "Gold"),       # Higher priority
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=30)

    finished_jobs = {j.jid: j for j in result['finished_jobs']}
    bronze_job = finished_jobs.get("bronze")
    gold_job = finished_jobs.get("gold")

    if gold_job:
        # Gold should finish quickly after start
        assert gold_job.finish - \
            gold_job.start <= 4, f"Gold job should finish quickly after start, took {gold_job.finish - gold_job.start}"

    if bronze_job and gold_job:
        # Bronze should be preempted, then resume and finish later
        assert bronze_job.finish > gold_job.finish, "Bronze should finish after being preempted"


def test_priority_multiple_qos_levels():
    """Test scheduling with all three QoS levels."""
    scheduler = PriorityQoS(enable_preemption=True)
    cluster = create_test_cluster(3, 0)  # 3 GPUs
    jobs = [
        create_test_job("bronze1", 0, 5, 1, "Bronze"),
        create_test_job("silver1", 1, 4, 1, "Silver"),
        create_test_job("gold1", 2, 3, 1, "Gold"),
        create_test_job("gold2", 3, 2, 1, "Gold"),
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=30)
    assert len(result['finished_jobs']) == 4, "All 4 jobs should complete"

    gold_jobs = [j for j in result['finished_jobs'] if j.qos == "Gold"]
    silver_jobs = [j for j in result['finished_jobs'] if j.qos == "Silver"]
    bronze_jobs = [j for j in result['finished_jobs'] if j.qos == "Bronze"]

    if gold_jobs and bronze_jobs:
        # Gold jobs should generally finish before lower priority jobs
        max_gold_finish = max(j.finish for j in gold_jobs)
        min_bronze_finish = min(j.finish for j in bronze_jobs)
        assert max_gold_finish <= min_bronze_finish, "Gold jobs should finish before Bronze jobs"


def test_priority_no_resources_available():
    """Test priority scheduler when no resources are available."""
    scheduler = PriorityQoS()
    cluster = create_test_cluster(1, 0)  # Only 1 GPU
    jobs = [
        create_test_job("job1", 0, 5, 2, "Gold"),   # Needs 2 GPUs, can't fit
        create_test_job("job2", 1, 2, 1, "Silver"),  # Needs 1 GPU
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=20)

    # job2 (Silver) should start when job1 can't fit
    started_jobs = [j for j in result['finished_jobs'] if j.start is not None]
    assert len(started_jobs) >= 1, "At least one job should start"


def test_priority_simultaneous_arrivals():
    """Test priority ordering when multiple jobs arrive at same time."""
    scheduler = PriorityQoS()
    cluster = create_test_cluster(2, 0)
    jobs = [
        create_test_job("gold", 0, 1, 1, "Gold"),
        create_test_job("silver", 0, 1, 1, "Silver"),
        create_test_job("bronze", 0, 1, 1, "Bronze"),
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=20)
    assert len(result['finished_jobs']) == 3, "All jobs should complete"

    finished_jobs = {j.jid: j for j in result['finished_jobs']}
    gold_job = finished_jobs["gold"]
    silver_job = finished_jobs["silver"]
    bronze_job = finished_jobs["bronze"]

    # Higher priority should finish first
    assert gold_job.finish <= silver_job.finish, "Gold should finish before or with Silver"
    assert silver_job.finish <= bronze_job.finish, "Silver should finish before or with Bronze"


def test_priority_resource_preference_respected():
    """Test that QoS resource preferences are respected even during preemption."""
    scheduler = PriorityQoS(enable_preemption=True)
    cluster = create_test_cluster(2, 2)  # 2 edge, 2 cloud
    jobs = [
        create_test_job("gold", 0, 5, 2, "Gold"),   # Prefers edge
        create_test_job("silver", 2, 3, 2, "Silver"),  # Prefers cloud
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=20)

    finished_jobs = {j.jid: j for j in result['finished_jobs']}
    gold_job = finished_jobs.get("gold")
    silver_job = finished_jobs.get("silver")

    # Gold should get edge resources
    if gold_job and gold_job.final_allocated:
        edge_count = sum(1 for r in gold_job.final_allocated if r.startswith("edge"))
        assert edge_count > 0, "Gold job should get at least some edge resources"

    # Silver should get cloud resources
    if silver_job and silver_job.final_allocated:
        cloud_count = sum(1 for r in silver_job.final_allocated if r.startswith("cloud"))
        assert cloud_count > 0, "Silver job should get at least some cloud resources"


def test_priority_starvation_prevention():
    """Test that low-priority jobs don't starve indefinitely."""
    scheduler = PriorityQoS(enable_preemption=True)
    cluster = create_test_cluster(2, 0)
    jobs = [
        create_test_job("bronze", 0, 2, 1, "Bronze"),
        create_test_job("gold1", 1, 1, 1, "Gold"),
        create_test_job("gold2", 2, 1, 1, "Gold"),
        create_test_job("gold3", 3, 1, 1, "Gold"),
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=50)

    finished_jobs = {j.jid: j for j in result['finished_jobs']}
    bronze_job = finished_jobs.get("bronze")

    # Bronze should eventually finish (not starve)
    assert bronze_job is not None, "Bronze job should exist"
    assert bronze_job.finish is not None, "Bronze job should eventually finish"
    # All jobs should finish
    assert len(result['finished_jobs']) == 4, "All jobs should complete"


def test_priority_qos_preferences():
    """Test that Priority QoS considers resource preferences."""
    scheduler = PriorityQoS()
    cluster = create_test_cluster(1, 1)
    jobs = [
        create_test_job("gold", 0, 3, 1, "Gold"),    # Should prefer edge
        create_test_job("silver", 1, 2, 1, "Silver")  # Should prefer cloud
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=10)
    assert len(result['finished_jobs']) == 2, "Both jobs should complete"

    assert all(r.startswith("edge") for r in result['finished_jobs']
               [0].final_allocated), "Gold job should get edge resource"
    assert all(r.startswith("cloud") for r in result['finished_jobs']
               [1].final_allocated), "Silver job should get cloud resource"


def test_priority_preemption_enabled():
    """Test that preemption works when enabled (default)."""
    scheduler = PriorityQoS(enable_preemption=True)
    cluster = create_test_cluster(2, 0)  # Only 2 GPUs
    jobs = [
        create_test_job("bronze", 0, 5, 2, "Bronze"),  # arrival=0, time=5, demand=2
        create_test_job("gold", 2, 2, 1, "Gold"),      # arrival=2, time=2, demand=1 (should preempt)
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=20)
    assert len(result['finished_jobs']
               ) >= 2, f"Both jobs should eventually complete, got {len(result['finished_jobs'])}"

    finished_jobs = {j.jid: j for j in result['finished_jobs']}
    assert "gold" in finished_jobs, "Gold job should finish"
    assert "bronze" in finished_jobs, "Bronze job should finish after preemption"

    gold_job = finished_jobs["gold"]
    bronze_job = finished_jobs["bronze"]

    # Gold should start soon after arrival (at t=2) via preemption
    assert gold_job.start == 2, f"Gold job should start at time 2 (by preemption), got {gold_job.start}"

    # Bronze should be preempted, then finish later after Gold completes
    assert bronze_job.finish > gold_job.finish, "Bronze should finish after Gold due to preemption"


def test_priority_preemption_disabled():
    """Test that preemption respects disable flag."""
    scheduler = PriorityQoS(enable_preemption=False)
    cluster = create_test_cluster(2, 0)
    jobs = [
        create_test_job("bronze", 0, 5, 2, "Bronze"),  # arrival=0, time=5, demand=2
        create_test_job("gold", 2, 2, 1, "Gold"),      # arrival=2, time=2, demand=1
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=20)
    assert len(result['finished_jobs']) >= 2, "Both jobs should complete"

    gold_job = next(j for j in result['finished_jobs'] if j.jid == "gold")
    bronze_job = next(j for j in result['finished_jobs'] if j.jid == "bronze")

    # Without preemption, Gold must wait for Bronze to finish
    assert gold_job.start >= bronze_job.finish, "Gold should wait for Bronze without preemption"


def test_priority_silver_bronze_preemption_policy():
    """Test Silver->Bronze preemption policy setting."""
    # With preemption enabled and allow_silver_preempt_bronze=True
    scheduler_allow = PriorityQoS(enable_preemption=True, allow_silver_preempt_bronze=True)
    cluster = create_test_cluster(1, 0)  # 1 GPU
    jobs = [
        create_test_job("bronze", 0, 5, 1, "Bronze"),  # arrival=0, time=5, demand=1
        create_test_job("silver", 1, 2, 1, "Silver"),  # arrival=1, time=2, demand=1
    ]

    result = run_scheduler_test(scheduler_allow, jobs, cluster, horizon=20)
    silver_job = next(j for j in result['finished_jobs'] if j.jid == "silver")
    # Silver should preempt Bronze when allowed
    assert silver_job.start == 1, "Silver should start at arrival with preemption allowed"

    # With preemption enabled but allow_silver_preempt_bronze=False
    scheduler_disallow = PriorityQoS(enable_preemption=True, allow_silver_preempt_bronze=False)
    cluster2 = create_test_cluster(1, 0)  # 1 GPU
    jobs2 = [
        create_test_job("bronze", 0, 5, 1, "Bronze"),
        create_test_job("silver", 1, 2, 1, "Silver"),
    ]

    result2 = run_scheduler_test(scheduler_disallow, jobs2, cluster2, horizon=20)
    silver_job2 = next(j for j in result2['finished_jobs'] if j.jid == "silver")
    bronze_job2 = next(j for j in result2['finished_jobs'] if j.jid == "bronze")
    # Silver should NOT preempt Bronze when disallowed
    assert silver_job2.start >= bronze_job2.finish, "Silver should wait for Bronze when preemption disallowed"


def test_priority_work_remaining_after_preemption():
    """Test that preempted jobs track work remaining correctly."""
    scheduler = PriorityQoS(enable_preemption=True)
    cluster = create_test_cluster(2, 0)
    jobs = [
        create_test_job("bronze", 0, 10, 2, "Bronze"),  # Long-running
        create_test_job("gold", 5, 3, 2, "Gold"),       # Higher priority
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=30)

    bronze_job = next(j for j in result['finished_jobs'] if j.jid == "bronze")
    gold_job = next(j for j in result['finished_jobs'] if j.jid == "gold")

    # Gold should preempt and finish quickly
    assert gold_job.finish - gold_job.start <= 4, "Gold job should finish quickly after start"
    # Bronze should be preempted, then resume and finish later
    assert bronze_job.finish > gold_job.finish, "Bronze should finish after being preempted"


def test_priority_multiple_qos_levels():
    """Test scheduling with all three QoS levels."""
    scheduler = PriorityQoS(enable_preemption=True)
    cluster = create_test_cluster(3, 0)  # 3 GPUs
    jobs = [
        create_test_job("bronze1", 0, 5, 1, "Bronze"),
        create_test_job("silver1", 1, 4, 1, "Silver"),
        create_test_job("gold1", 2, 3, 1, "Gold"),
        create_test_job("gold2", 3, 2, 1, "Gold"),
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=30)
    assert len(result['finished_jobs']) == 4, "All 4 jobs should complete"

    gold_jobs = [j for j in result['finished_jobs'] if j.qos == "Gold"]
    silver_jobs = [j for j in result['finished_jobs'] if j.qos == "Silver"]
    bronze_jobs = [j for j in result['finished_jobs'] if j.qos == "Bronze"]

    # Gold jobs should generally finish before lower priority jobs
    max_gold_finish = max(j.finish for j in gold_jobs)
    min_bronze_finish = min(j.finish for j in bronze_jobs)
    assert max_gold_finish <= min_bronze_finish, "Gold jobs should finish before Bronze jobs"


def test_priority_no_resources_available():
    """Test priority scheduler when no resources are available."""
    scheduler = PriorityQoS()
    cluster = create_test_cluster(1, 0)  # Only 1 GPU
    jobs = [
        create_test_job("job1", 0, 2, 5, "Gold"),   # Needs 5 GPUs, can't fit
        create_test_job("job2", 1, 1, 1, "Silver"),  # Needs 1 GPU
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=20)

    # job2 (Silver) should start when job1 can't fit
    started_jobs = [j for j in result['finished_jobs'] if j.start is not None]
    assert len(started_jobs) >= 1, "At least one job should start"


def test_priority_simultaneous_arrivals():
    """Test priority ordering when multiple jobs arrive at same time."""
    scheduler = PriorityQoS()
    cluster = create_test_cluster(2, 0)
    jobs = [
        create_test_job("gold", 0, 1, 2, "Gold"),
        create_test_job("silver", 0, 1, 2, "Silver"),
        create_test_job("bronze", 0, 1, 2, "Bronze"),
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=20)
    assert len(result['finished_jobs']) == 3, "All jobs should complete"

    gold_job = next(j for j in result['finished_jobs'] if j.jid == "gold")
    silver_job = next(j for j in result['finished_jobs'] if j.jid == "silver")
    bronze_job = next(j for j in result['finished_jobs'] if j.jid == "bronze")

    # Higher priority should finish first
    assert gold_job.finish <= silver_job.finish, "Gold should finish before or with Silver"
    assert silver_job.finish <= bronze_job.finish, "Silver should finish before or with Bronze"


def test_priority_resource_preference_respected():
    """Test that QoS resource preferences are respected even during preemption."""
    scheduler = PriorityQoS(enable_preemption=True)
    cluster = create_test_cluster(2, 2)  # 2 edge, 2 cloud
    jobs = [
        create_test_job("gold", 0, 2, 2, "Gold"),   # Prefers edge
        create_test_job("silver", 2, 2, 2, "Silver"),  # Prefers cloud
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=20)

    gold_job = next(j for j in result['finished_jobs'] if j.jid == "gold")
    silver_job = next(j for j in result['finished_jobs'] if j.jid == "silver")

    # Gold should get edge resources
    if gold_job.final_allocated:
        edge_count = sum(1 for r in gold_job.final_allocated if r.startswith("edge"))
        assert edge_count > 0, "Gold job should get at least some edge resources"

    # Silver should get cloud resources
    if silver_job.final_allocated:
        cloud_count = sum(1 for r in silver_job.final_allocated if r.startswith("cloud"))
        assert cloud_count > 0, "Silver job should get at least some cloud resources"


def test_priority_starvation_prevention():
    """Test that low-priority jobs don't starve indefinitely."""
    scheduler = PriorityQoS(enable_preemption=True)
    cluster = create_test_cluster(2, 0)
    jobs = [
        create_test_job("bronze", 0, 1, 2, "Bronze"),
        create_test_job("gold1", 1, 2, 1, "Gold"),
        create_test_job("gold2", 2, 2, 1, "Gold"),
        create_test_job("gold3", 3, 1, 1, "Gold"),
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=50)

    bronze_job = next(j for j in result['finished_jobs'] if j.jid == "bronze")
    gold_jobs = [j for j in result['finished_jobs'] if j.qos == "Gold"]

    # Bronze should eventually finish (not starve)
    assert bronze_job.finish is not None, "Bronze job should eventually finish"
    # All jobs should finish
    assert len(result['finished_jobs']) == 4, "All jobs should complete"


def test_priority_multiple_preemptions_work_remaining():
    """Test that work_remaining is tracked correctly with multiple preemptions.

    This is a regression test for the bug where work_done was calculated as
    (sim.time - start), which doesn't account for gaps when a job is preempted.

    Timeline:
      t=0-2:  bronze runs (2 work done)
      t=2-4:  silver preempts bronze
      t=4-6:  bronze resumes (2 more done, total 4)
      t=6-8:  gold preempts bronze (2 done, should have 6 remaining)
      t=8-18: bronze resumes (6 more, total 10)

    Expected: bronze finishes at t=18
    Bug would cause: bronze finishes at t=10 (if work_done calculated as 6-0=6 instead of 4)
    """
    scheduler = PriorityQoS(enable_preemption=True)
    cluster = create_test_cluster(1, 0)  # 1 GPU

    jobs = [
        create_test_job("bronze", 0, 10, 1, "Bronze"),  # 10 units of work
        create_test_job("silver", 2, 2, 1, "Silver"),   # Preempts at t=2
        create_test_job("gold", 6, 2, 1, "Gold"),       # Preempts at t=6
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=30, debug=False)

    finished_jobs = {j.jid: j for j in result['finished_jobs']}
    bronze = finished_jobs["bronze"]
    silver = finished_jobs["silver"]
    gold = finished_jobs["gold"]

    # Verify Silver finishes at t=4 (starts at t=2, runs 2 units)
    assert silver.start == 2, f"Silver should start at t=2, got {silver.start}"
    assert silver.finish == 4, f"Silver should finish at t=4, got {silver.finish}"

    # Verify Gold finishes at t=8 (starts at t=6, runs 2 units)
    assert gold.start == 6, f"Gold should start at t=6, got {gold.start}"
    assert gold.finish == 8, f"Gold should finish at t=8, got {gold.finish}"

    # Verify Bronze finishes at t=18 (not t=10 which would be the bug)
    # Bronze: starts at 0, preempted at 2 (2 done), resumes at 4, preempted at 6 (4 done total)
    # Bronze: resumes at 8, needs 6 more, finishes at 14
    # Wait, let me recalculate:
    # t=0: bronze starts
    # t=2: silver arrives, preempts bronze (bronze had 2 units done, 8 remaining)
    # t=2-4: silver runs (consumes 2 units)
    # t=4: silver finishes, bronze resumes
    # t=4-6: bronze runs (2 more units done, 6 remaining)
    # t=6: gold arrives, preempts bronze (bronze had 4 total done, 6 remaining)
    # t=6-8: gold runs (consumes 2 units)
    # t=8: gold finishes, bronze resumes
    # t=8-14: bronze runs (6 more units, total 10)
    assert bronze.finish == 14, f"Bronze should finish at t=14, got {bronze.finish}. Bug: work_done not tracking multiple preemptions correctly"

    # Verify Bronze has correct work_remaining (should be 0 or close to it after finishing)
    assert bronze.work_remaining <= 0.01, f"Bronze work_remaining should be ~0, got {bronze.work_remaining}"
