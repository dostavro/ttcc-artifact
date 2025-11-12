"""
Tests for Conservative scheduler.
"""
import pytest
from test_utils import create_test_job, create_test_cluster, run_scheduler_test
from schedulers.conservative import Conservative


def test_conservative_basic(basic_cluster, basic_jobs):
    """Test basic Conservative scheduler functionality."""
    scheduler = Conservative()
    result = run_scheduler_test(scheduler, basic_jobs, basic_cluster)

    # All jobs should complete
    assert len(result['finished_jobs']) == 3, "All jobs should complete"

    # All jobs should have had allocations
    for job in result['finished_jobs']:
        allocation = getattr(job, 'final_allocated', None) or getattr(job, 'allocated', None)
        assert allocation is not None, f"Job {job.jid} had no allocation"
        assert job.finish is not None
        assert job.start is not None


def test_conservative_backfilling_with_reservations():
    """Test Conservative scheduler does backfilling but respects reservations to prevent starvation."""
    scheduler = Conservative()
    cluster = create_test_cluster(2, 0)  # Only 2 edge resources
    jobs = [
        create_test_job("big_job", 0, 8, 1, "Gold"),        # Uses 1 resource, leaves 1 free for backfilling
        create_test_job("medium_waiting", 1, 6, 2, "Silver"),  # Needs 2 resources - must wait for big_job
        create_test_job("backfill1", 2, 2, 1, "Bronze"),    # Can backfill - uses the 1 free resource
        create_test_job("backfill2", 4, 4, 1, "Bronze"),    # Can backfill - arrives later but still fits
        create_test_job("blocked", 7, 1, 1, "Bronze"),      # Arrives late - should be blocked by reservation
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=20)

    # All jobs should complete
    assert len(result['finished_jobs']) == 5, "All jobs should complete"

    # Get job results
    big_job = next(j for j in result['finished_jobs'] if j.jid == "big_job")
    medium_waiting = next(j for j in result['finished_jobs'] if j.jid == "medium_waiting")
    backfill1 = next(j for j in result['finished_jobs'] if j.jid == "backfill1")
    backfill2 = next(j for j in result['finished_jobs'] if j.jid == "backfill2")
    blocked = next(j for j in result['finished_jobs'] if j.jid == "blocked")

    # Big job uses 1 resource from t=0 to t=8
    assert big_job.start == 0, "Big job should start immediately"
    assert big_job.finish == 8, "Big job should finish at t=8"

    # Medium waiting job needs 2 resources, must wait for big_job to finish
    assert medium_waiting.start >= big_job.finish, "Medium job must wait for big_job to finish"
    assert medium_waiting.start == 8, "Medium job should start immediately when big_job finishes"

    # Conservative ALLOWS backfilling: early jobs can use the 1 free resource
    assert backfill1.start >= 2, "backfill1 can't start before arrival"
    assert backfill1.start < big_job.finish, "backfill1 should backfill during big_job execution"
    assert backfill1.finish <= 4, "backfill1 should finish by t=4"

    assert backfill2.start >= 4, "backfill2 can't start before arrival"
    assert backfill2.start < big_job.finish, "backfill2 should backfill during big_job execution"
    assert backfill2.finish <= 8, "backfill2 should finish by t=8"

    # Conservative RESPECTS reservations: blocked job cannot backfill
    # blocked arrives at t=7 and should wait for medium job to start at t=8
    assert blocked.start >= 7, "blocked job can't start before arrival"
    assert blocked.start >= medium_waiting.finish, "blocked job should start after medium job finishes"
    assert medium_waiting.start == 8, "Medium job should still start at t=8 as planned"


def test_conservative_fifo_ordering():
    """Test that Conservative scheduler respects FIFO ordering for jobs with same requirements."""
    scheduler = Conservative()
    cluster = create_test_cluster(1, 0)  # Only 1 resource
    jobs = [
        create_test_job("first", 0, 3, 1, "Bronze"),   # Arrives first
        create_test_job("second", 1, 3, 1, "Bronze"),  # Arrives second
        create_test_job("third", 2, 3, 1, "Bronze"),   # Arrives third
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=15)

    first = next(j for j in result['finished_jobs'] if j.jid == "first")
    second = next(j for j in result['finished_jobs'] if j.jid == "second")
    third = next(j for j in result['finished_jobs'] if j.jid == "third")

    # Should execute in FIFO order
    assert first.start == 0, "First job should start immediately"
    assert second.start == 3, "Second job should start when first finishes"
    assert third.start == 6, "Third job should start when second finishes"


def test_conservative_no_starvation():
    """Test that large jobs don't get starved by continuous small jobs."""
    scheduler = Conservative()
    cluster = create_test_cluster(2, 0)  # Only 2 resources
    jobs = [
        # First wave: small jobs that start immediately
        create_test_job("small1", 0, 4, 1, "Bronze"),    # Uses 1 resource t=0→4
        create_test_job("small2", 0, 4, 1, "Bronze"),    # Uses 1 resource t=0→4

        # Big job arrives early but can't start yet (all resources busy)
        create_test_job("large", 1, 6, 2, "Gold"),       # Needs BOTH resources, gets reservation

        # Continuous stream of small jobs that could starve the large job
        create_test_job("small3", 3, 3, 1, "Bronze"),    # Arrives while large waits
        create_test_job("small4", 4, 3, 1, "Bronze"),    # Could grab resources when small1/2 finish
        create_test_job("small5", 5, 3, 1, "Bronze"),    # More pressure
        create_test_job("small6", 6, 3, 1, "Bronze"),    # Continuous pressure
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=25)

    large = next(j for j in result['finished_jobs'] if j.jid == "large")
    small1 = next(j for j in result['finished_jobs'] if j.jid == "small1")
    small2 = next(j for j in result['finished_jobs'] if j.jid == "small2")
    small3 = next(j for j in result['finished_jobs'] if j.jid == "small3")
    small4 = next(j for j in result['finished_jobs'] if j.jid == "small4")
    small5 = next(j for j in result['finished_jobs'] if j.jid == "small5")
    small6 = next(j for j in result['finished_jobs'] if j.jid == "small6")

    # Initial small jobs start immediately and occupy all resources
    assert small1.start == 0, "small1 starts immediately"
    assert small2.start == 0, "small2 starts immediately"
    assert small1.finish == 4, "small1 finishes at t=4"
    assert small2.finish == 4, "small2 finishes at t=4"

    # Large job should get its reservation respected despite later small jobs
    assert large.start == 4, "Large job should start when small1/small2 finish"
    assert large.finish == 10, "Large job should finish at t=10"

    # Key starvation prevention test: large job starts before later small jobs
    assert large.start < small3.start, "Large job should start before small3 (starvation prevention)"
    assert large.start < small4.start, "Large job should start before small4 (starvation prevention)"
    assert large.start < small5.start, "Large job should start before small5 (starvation prevention)"
    assert large.start < small6.start, "Large job should start before small6 (starvation prevention)"

    # Later small jobs must wait for large job to finish (can't jump ahead)
    assert small3.start >= large.finish, "small3 should wait for large job"
    assert small4.start >= large.finish, "small4 should wait for large job"
    assert small5.start >= large.finish, "small5 should wait for large job"
    assert small6.start >= large.finish, "small6 should wait for large job"

    assert small3.start <= small4.start <= small5.start <= small6.start, "Later small jobs should execute in order"


def test_conservative_resource_efficiency():
    """Test that Conservative scheduler efficiently uses available resources."""
    scheduler = Conservative()
    cluster = create_test_cluster(3, 0)  # 3 resources
    jobs = [
        create_test_job("big", 0, 10, 2, "Gold"),      # Uses 2 resources, leaves 1 free
        create_test_job("medium", 1, 8, 2, "Silver"),  # Needs 2, must wait
        create_test_job("small1", 2, 3, 1, "Bronze"),  # Should backfill immediately
        create_test_job("small2", 3, 4, 1, "Bronze"),  # Should backfill after small1
        create_test_job("small3", 5, 2, 1, "Bronze"),  # Can start with medium at t=10
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=25)

    big = next(j for j in result['finished_jobs'] if j.jid == "big")
    medium = next(j for j in result['finished_jobs'] if j.jid == "medium")
    small1 = next(j for j in result['finished_jobs'] if j.jid == "small1")
    small2 = next(j for j in result['finished_jobs'] if j.jid == "small2")
    small3 = next(j for j in result['finished_jobs'] if j.jid == "small3")

    # Verify efficient backfilling
    assert small1.start == 2, "small1 should backfill immediately"
    assert small2.start == 5, "small2 should start when small1 finishes"

    # Verify medium job gets its reservation respected
    assert medium.start == big.finish, "medium should start when big finishes"

    # small3 can start at the same time as medium (enough resources)
    assert small3.start == 10, "small3 can start when big finishes (3 total resources)"
    assert small3.start == medium.start, "small3 and medium can run concurrently"


def test_conservative_mixed_qos_levels():
    """Test Conservative scheduler with different QoS levels."""
    scheduler = Conservative()
    cluster = create_test_cluster(2, 0)
    jobs = [
        create_test_job("gold1", 0, 5, 1, "Gold"),
        create_test_job("silver1", 1, 4, 2, "Silver"),   # Must wait
        create_test_job("bronze1", 2, 2, 1, "Bronze"),   # Can backfill
        create_test_job("gold2", 3, 3, 1, "Gold"),       # Should respect silver1's reservation
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=20)

    gold1 = next(j for j in result['finished_jobs'] if j.jid == "gold1")
    silver1 = next(j for j in result['finished_jobs'] if j.jid == "silver1")
    bronze1 = next(j for j in result['finished_jobs'] if j.jid == "bronze1")
    gold2 = next(j for j in result['finished_jobs'] if j.jid == "gold2")

    # Conservative should respect arrival order regardless of QoS
    assert gold1.start == 0
    assert bronze1.start >= 2 and bronze1.start < gold1.finish  # Can backfill
    assert silver1.start == gold1.finish  # Gets reservation respected
    assert gold2.start >= silver1.finish  # Must wait for earlier silver job


def test_conservative_edge_cases():
    """Test Conservative scheduler edge cases."""
    scheduler = Conservative()
    cluster = create_test_cluster(1, 0)

    # Test with single job
    jobs = [create_test_job("solo", 0, 5, 1, "Gold")]
    result = run_scheduler_test(scheduler, jobs, cluster, horizon=10)
    solo = result['finished_jobs'][0]
    assert solo.start == 0, "Single job should start immediately"

    # Test with zero-duration jobs
    cluster = create_test_cluster(2, 0)
    jobs = [
        create_test_job("instant1", 0, 0, 1, "Gold"),
        create_test_job("instant2", 0, 0, 1, "Silver"),
    ]
    result = run_scheduler_test(scheduler, jobs, cluster, horizon=5)
    assert len(result['finished_jobs']) == 2, "Both instant jobs should complete"


def test_conservative_reservation_recalculation():
    """Test that reservations are computed correctly when jobs finish."""
    scheduler = Conservative()
    cluster = create_test_cluster(2, 0)
    jobs = [
        create_test_job("first", 0, 4, 2, "Gold"),      # Uses all resources t=0→4
        create_test_job("second", 1, 3, 1, "Silver"),   # Gets reservation at t=4
        create_test_job("third", 2, 3, 1, "Bronze"),    # Gets reservation at t=4 (can run with second)
        create_test_job("fourth", 3, 5, 2, "Gold"),     # Gets reservation at t=7 (after second+third)
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=20)

    first = next(j for j in result['finished_jobs'] if j.jid == "first")
    second = next(j for j in result['finished_jobs'] if j.jid == "second")
    third = next(j for j in result['finished_jobs'] if j.jid == "third")
    fourth = next(j for j in result['finished_jobs'] if j.jid == "fourth")

    # First uses all resources
    assert first.start == 0, "first starts immediately"
    assert first.finish == 4, "first finishes at t=4"

    # Second and third can start when first finishes (both fit in 2 resources)
    assert second.start == 4, "second starts when first finishes"
    assert third.start == 4, "third starts when first finishes (runs parallel with second)"
    assert second.finish == 7, "second finishes at t=7"
    assert third.finish == 7, "third finishes at t=7"

    # Fourth waits for both second and third to finish
    assert fourth.start == 7, "fourth starts when second+third finish"
    assert fourth.finish == 12, "fourth finishes at t=12"


def test_conservative_complex_overlapping_reservations():
    """Test Conservative with complex overlapping reservation scenarios."""
    scheduler = Conservative()
    cluster = create_test_cluster(4, 0)  # 4 resources
    jobs = [
        # Multiple jobs arrive at different times with different demands
        create_test_job("job1", 0, 6, 2, "Gold"),      # Uses 2, t=0→6
        create_test_job("job2", 1, 4, 1, "Silver"),    # Uses 1, should start at t=1 (2 free)
        create_test_job("job3", 2, 3, 1, "Bronze"),    # Uses 1, should start at t=2 (1 free)
        create_test_job("job4", 3, 8, 4, "Gold"),      # Uses 4, must wait until t=6
        create_test_job("job5", 4, 2, 1, "Silver"),    # Uses 1, should wait for job4 reservation
        create_test_job("job6", 5, 1, 1, "Bronze"),    # Uses 1, can backfill at t=5
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=25)

    job1 = next(j for j in result['finished_jobs'] if j.jid == "job1")
    job2 = next(j for j in result['finished_jobs'] if j.jid == "job2")
    job3 = next(j for j in result['finished_jobs'] if j.jid == "job3")
    job4 = next(j for j in result['finished_jobs'] if j.jid == "job4")
    job5 = next(j for j in result['finished_jobs'] if j.jid == "job5")
    job6 = next(j for j in result['finished_jobs'] if j.jid == "job6")

    # Job1 starts immediately
    assert job1.start == 0, "job1 starts immediately"

    # Jobs 2 and 3 should start early (resources available)
    assert job2.start == 1, "job2 should start soon (resources available)"
    assert job3.start == 2, "job3 should start soon (resources available)"

    # Job4 needs 3 resources, must wait for enough to free up
    assert job4.start == job1.finish, "job4 should start at t=6 when enough resources free up"

    # Job5 should wait because job4 has reservation
    assert job5.start == job4.finish, "job5 can't start before job4 reservation"

    # Job6 can backfill before job4 starts
    assert job6.start == 5, "job6 can backfill at t=5"

    # All jobs complete
    assert len(result['finished_jobs']) == 6, "All jobs should complete"


def test_conservative_multi_gpu_jobs():
    """Test Conservative with various multi-GPU job configurations."""
    scheduler = Conservative()
    cluster = create_test_cluster(4, 2)  # 4 edge, 2 cloud = 6 total
    jobs = [
        create_test_job("large1", 0, 5, 4, "Gold"),     # Uses 4 edge GPUs, t=0→5, leaves 2 cloud free
        create_test_job("large2", 1, 6, 4, "Gold"),     # Needs 4 GPUs, gets reservation, t=5→11
        create_test_job("medium", 2, 3, 2, "Silver"),   # Needs 2 GPUs, can BACKFILL using 2 cloud GPUs, t=2→5
        create_test_job("small1", 3, 2, 1, "Bronze"),   # Needs 1 GPU, backfills after medium, t=5→7
        create_test_job("medium2", 4, 2, 3, "Silver"),  # Needs 3 GPUs, must wait for large2 to finish
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=25)

    large1 = next(j for j in result['finished_jobs'] if j.jid == "large1")
    large2 = next(j for j in result['finished_jobs'] if j.jid == "large2")
    medium = next(j for j in result['finished_jobs'] if j.jid == "medium")
    small1 = next(j for j in result['finished_jobs'] if j.jid == "small1")
    medium2 = next(j for j in result['finished_jobs'] if j.jid == "medium2")

    # Large1 starts immediately (uses 4 edge GPUs, leaves 2 cloud GPUs free)
    assert large1.start == 0, "large1 starts immediately"
    assert large1.finish == 5, "large1 finishes at t=5"

    # Large2 gets reservation and starts when large1 finishes (needs 4 GPUs)
    assert large2.start == large1.finish, "large2 starts when large1 finishes"
    assert large2.finish == 11, "large2 finishes at t=11"

    # Medium can BACKFILL during large1 execution (uses the 2 free cloud GPUs)
    assert medium.start == 2, "medium backfills at t=2 using 2 cloud GPUs"
    assert medium.finish == 5, "medium finishes at t=5"
    assert medium.finish <= large1.finish, "medium backfills without delaying large1"

    # Small1 backfills after medium finishes (when medium's 2 cloud GPUs become free at t=5)
    # At t=5: large2 starts using 4 GPUs, 2 cloud GPUs are free
    assert small1.start == medium.finish, "small1 starts when medium finishes (t=5)"
    assert small1.finish == 7, "small1 finishes at t=7"

    # Medium2 needs 3 GPUs - cannot backfill (only 2 cloud GPUs free during large2)
    # Must wait for large2 to finish
    assert medium2.start == large2.finish, "medium2 waits for large2 to finish (needs 3 GPUs)"
    assert medium2.finish == 13, "medium2 finishes at t=13 (11+2)"

    # All jobs complete
    assert len(result['finished_jobs']) == 5, "All jobs should complete"


def test_conservative_empty_queue_handling():
    """Test Conservative handles queue becoming empty and refilling."""
    scheduler = Conservative()
    cluster = create_test_cluster(2, 0)
    jobs = [
        # First batch completes - both can run in parallel
        create_test_job("early1", 0, 3, 1, "Gold"),
        create_test_job("early2", 0, 3, 1, "Silver"),
        # Gap - queue becomes empty from t=3 to t=6
        # Second batch arrives later
        create_test_job("late1", 6, 4, 1, "Bronze"),   # Uses 1 GPU starts at t=6, finishes t=10
        create_test_job("late2", 7, 2, 2, "Gold"),     # Needs 2 GPUs, must wait for late1 to finish
        create_test_job("late3", 8, 1, 1, "Silver"),   # Can backfill before late2 starts
        create_test_job("late4", 9, 2, 1, "Bronze"),   # Should wait because late2 has reservation
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=20)

    early1 = next(j for j in result['finished_jobs'] if j.jid == "early1")
    early2 = next(j for j in result['finished_jobs'] if j.jid == "early2")
    late1 = next(j for j in result['finished_jobs'] if j.jid == "late1")
    late2 = next(j for j in result['finished_jobs'] if j.jid == "late2")
    late3 = next(j for j in result['finished_jobs'] if j.jid == "late3")
    late4 = next(j for j in result['finished_jobs'] if j.jid == "late4")

    # Early jobs both start immediately (2 GPUs available, each needs 1)
    assert early1.start == 0, "early1 starts immediately"
    assert early2.start == 0, "early2 starts immediately"
    assert early1.finish == 3, "early1 finishes at t=3"
    assert early2.finish == 3, "early2 finishes at t=3"

    # Queue is empty from t=3 to t=6

    # Late1 starts when it arrives (queue is empty, resources available)
    assert late1.start == 6, "late1 starts immediately when it arrives"
    assert late1.finish == 10, "late1 finishes at t=10"

    # Late2 arrives while late1 is running (late1 uses both GPUs)
    # Late2 must wait for late1 to finish
    assert late2.start >= 7, "late2 can't start before arrival"
    assert late2.start == late1.finish, "late2 starts when late1 finishes"
    assert late2.finish == 12, "late2 finishes at t=12 (10+2)"

    # Late3 can backfill while late1 is running (uses 1 GPU)
    assert late3.start == 8, "late3 starts when it arrives"
    assert late3.finish == 9, "late3 finishes at t=9"

    # Late4 must wait for late2 to finish (late2 has reservation for 2 GPUs)
    assert late4.start == late2.finish, "late4 starts when late2 finishes"
    assert late4.finish == 14, "late4 finishes at t=14 (12+2)"

    assert len(result['finished_jobs']) == 6, "All jobs complete"


def test_conservative_simultaneous_arrivals():
    """Test Conservative with many jobs arriving at the same time."""
    scheduler = Conservative()
    cluster = create_test_cluster(3, 0)
    jobs = [
        # All arrive at t=0 - first 3 in arrival order get resources
        create_test_job("a", 0, 5, 1, "Gold"),      # 1st: starts at t=0, finishes at t=5
        create_test_job("b", 0, 4, 1, "Silver"),    # 2nd: starts at t=0, finishes at t=4
        create_test_job("c", 0, 3, 1, "Bronze"),    # 3rd: starts at t=0, finishes at t=3
        create_test_job("d", 0, 6, 2, "Gold"),      # 4th: needs 2 GPUs, waits for 2 to be free
        create_test_job("e", 0, 2, 1, "Bronze"),    # 5th: needs 1 GPU, can use 3rd GPU when a finishes
        create_test_job("f", 0, 3, 2, "Silver"),    # 6th: needs 2 GPUs, must wait for d to finish
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=25)

    a = next(j for j in result['finished_jobs'] if j.jid == "a")
    b = next(j for j in result['finished_jobs'] if j.jid == "b")
    c = next(j for j in result['finished_jobs'] if j.jid == "c")
    d = next(j for j in result['finished_jobs'] if j.jid == "d")
    e = next(j for j in result['finished_jobs'] if j.jid == "e")
    f = next(j for j in result['finished_jobs'] if j.jid == "f")

    # First 3 jobs start immediately (3 resources available, FIFO order)
    assert a.start == 0, "a starts immediately (1st in queue)"
    assert b.start == 0, "b starts immediately (2nd in queue)"
    assert c.start == 0, "c starts immediately (3rd in queue)"

    # Finish times for first 3 jobs
    assert c.finish == 3, "c finishes at t=3"
    assert b.finish == 4, "b finishes at t=4"
    assert a.finish == 5, "a finishes at t=5"

    # Job d needs 2 GPUs, gets reservation for when 2 GPUs become available
    # At t=3: 1 GPU free (c done). At t=4: 2 GPUs free (b and c done)
    assert d.start == 4, "d starts at t=4 when 2 GPUs are free"
    assert d.finish == 10, "d finishes at t=10 (4+6)"

    # Job e needs 1 GPU - can start when a finishes at t=5 (3rd GPU becomes free)
    # At t=5: d is using 2 GPUs, a's GPU becomes free
    assert e.start == 5, "e starts at t=5 when a finishes (3rd GPU free)"
    assert e.finish == 7, "e finishes at t=7 (5+2)"

    # Job f needs 2 GPUs - must wait for d to finish (d has earlier reservation)
    assert f.start == d.finish, "f waits for d to finish (needs 2 GPUs)"
    assert f.finish == 13, "f finishes at t=13 (10+3)"

    # All jobs complete
    assert len(result['finished_jobs']) == 6, "All jobs should complete"


def test_conservative_reservation_integrity():
    """Test multiple reservations with interleaved backfilling."""
    scheduler = Conservative()
    cluster = create_test_cluster(4, 0)
    jobs = [
        # Long running job that leaves resources free
        create_test_job("long", 0, 20, 2, "Gold"),      # Uses 2, leaves 2 free, t=0→20

        # First blocked job with reservation
        create_test_job("blocked1", 1, 6, 3, "Silver"),  # Needs 3, gets reservation at t=20, t=20→26

        # Backfill jobs that can run during long
        create_test_job("backfill1", 2, 3, 2, "Bronze"),  # Can backfill with 2 free GPUs, t=2→5
        create_test_job("backfill2", 3, 4, 2, "Bronze"),  # Can backfill after backfill1, t=5→9

        # Second blocked job with reservation
        create_test_job("blocked2", 4, 8, 4, "Gold"),    # Needs 4, gets reservation at t=26, t=26→34

        # More backfill jobs
        create_test_job("backfill3", 6, 2, 2, "Bronze"),  # Can backfill, t=9→11
        create_test_job("backfill4", 8, 3, 2, "Bronze"),  # Can backfill, t=11→14

        # Third blocked job with reservation
        create_test_job("blocked3", 10, 5, 3, "Silver"),  # Needs 3, gets reservation at t=34, t=34→39

        # Late backfill that should respect all reservations
        create_test_job("backfill5", 12, 2, 2, "Bronze"),  # Can backfill, t=14→16
        create_test_job("backfill6", 15, 3, 2, "Bronze"),  # Can backfill, t=16→19
        create_test_job("blocked4", 18, 2, 3, "Bronze"),  # Needs 3, gets reservation at t=39, t=39→41
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=50)

    long = next(j for j in result['finished_jobs'] if j.jid == "long")
    blocked1 = next(j for j in result['finished_jobs'] if j.jid == "blocked1")
    backfill1 = next(j for j in result['finished_jobs'] if j.jid == "backfill1")
    backfill2 = next(j for j in result['finished_jobs'] if j.jid == "backfill2")
    blocked2 = next(j for j in result['finished_jobs'] if j.jid == "blocked2")
    backfill3 = next(j for j in result['finished_jobs'] if j.jid == "backfill3")
    backfill4 = next(j for j in result['finished_jobs'] if j.jid == "backfill4")
    blocked3 = next(j for j in result['finished_jobs'] if j.jid == "blocked3")
    backfill5 = next(j for j in result['finished_jobs'] if j.jid == "backfill5")
    backfill6 = next(j for j in result['finished_jobs'] if j.jid == "backfill6")
    blocked4 = next(j for j in result['finished_jobs'] if j.jid == "blocked4")

    # Long running job uses 2 GPUs, leaves 2 free for backfilling
    assert long.start == 0, "long starts immediately"
    assert long.finish == 20, "long finishes at t=20"

    # First blocked job gets reservation when long finishes
    assert blocked1.start == long.finish, "blocked1 starts when long finishes"
    assert blocked1.finish == 26, "blocked1 finishes at t=26"

    # Backfill jobs run during long's execution using the 2 free GPUs
    assert backfill1.start == 2, "backfill1 starts at t=2"
    assert backfill1.finish == 5, "backfill1 finishes at t=5"
    assert backfill1.finish <= long.finish, "backfill1 finishes before long"

    assert backfill2.start == backfill1.finish, "backfill2 starts when backfill1 finishes"
    assert backfill2.finish == 9, "backfill2 finishes at t=9"
    assert backfill2.finish <= long.finish, "backfill2 finishes before long"

    # Second blocked job gets reservation after blocked1
    assert blocked2.start == blocked1.finish, "blocked2 starts when blocked1 finishes"
    assert blocked2.finish == 34, "blocked2 finishes at t=34"

    # More backfills during long's execution
    assert backfill3.start == backfill2.finish, "backfill3 starts when backfill2 finishes"
    assert backfill3.finish == 11, "backfill3 finishes at t=11"
    assert backfill3.finish <= long.finish, "backfill3 finishes before long"

    assert backfill4.start == backfill3.finish, "backfill4 starts when backfill3 finishes"
    assert backfill4.finish == 14, "backfill4 finishes at t=14"
    assert backfill4.finish <= long.finish, "backfill4 finishes before long"

    # Third blocked job gets reservation after blocked2
    assert blocked3.start == blocked2.finish, "blocked3 starts when blocked2 finishes"
    assert blocked3.finish == 39, "blocked3 finishes at t=39"

    # Late backfills continue during long's execution
    assert backfill5.start == backfill4.finish, "backfill5 starts when backfill4 finishes"
    assert backfill5.finish == 16, "backfill5 finishes at t=16"
    assert backfill5.finish <= long.finish, "backfill5 finishes before long"

    assert backfill6.start == backfill5.finish, "backfill6 starts when backfill5 finishes"
    assert backfill6.finish == 19, "backfill6 finishes at t=19"
    assert backfill6.finish <= long.finish, "backfill6 finishes before long"

    # Fourth blocked job gets reservation after blocked3
    assert blocked4.start == blocked3.finish, "blocked4 starts when blocked3 finishes"
    assert blocked4.finish == 41, "blocked4 finishes at t=41"

    # Verify reservation order is maintained: blocked1 → blocked2 → blocked3 → blocked4
    assert blocked1.finish <= blocked2.start, "blocked1 finishes before blocked2 starts"
    assert blocked2.finish <= blocked3.start, "blocked2 finishes before blocked3 starts"
    assert blocked3.finish <= blocked4.start, "blocked3 finishes before blocked4 starts"

    # All jobs complete
    assert len(result['finished_jobs']) == 11, "All jobs should complete"


def test_conservative_long_running_jobs():
    """Test Conservative with long-running jobs and many short jobs."""
    scheduler = Conservative()
    cluster = create_test_cluster(2, 0)
    jobs = [
        create_test_job("long", 0, 20, 1, "Gold"),     # Long job using 1 resource
        create_test_job("short1", 1, 2, 1, "Bronze"),  # Short jobs that can use the free resource
        create_test_job("blocking", 2, 10, 2, "Silver"),  # Needs both resources, must wait for long
        create_test_job("short2", 3, 2, 1, "Bronze"),
        create_test_job("short3", 5, 2, 1, "Bronze"),
        create_test_job("short4", 7, 2, 1, "Bronze"),
        create_test_job("short5", 9, 2, 1, "Bronze"),
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=35)

    long = next(j for j in result['finished_jobs'] if j.jid == "long")
    blocking = next(j for j in result['finished_jobs'] if j.jid == "blocking")
    short1 = next(j for j in result['finished_jobs'] if j.jid == "short1")
    short2 = next(j for j in result['finished_jobs'] if j.jid == "short2")
    short3 = next(j for j in result['finished_jobs'] if j.jid == "short3")
    short4 = next(j for j in result['finished_jobs'] if j.jid == "short4")
    short5 = next(j for j in result['finished_jobs'] if j.jid == "short5")

    # Long job runs
    assert long.start == 0, "long starts immediately"
    assert long.finish == 20, "long finishes at t=20"

    # Blocking job must wait for long to finish
    assert blocking.start == 20, "blocking waits for long job"

    # Short jobs should be able to use the free resource during long job
    for short in [short1, short2, short3, short4, short5]:
        # They should either run before blocking's reservation OR after blocking finishes
        if short.start < blocking.start:
            assert short.finish <= blocking.start, f"{short.jid} shouldn't delay blocking's reservation"

    assert len(result['finished_jobs']) == 7, "All jobs complete"


if __name__ == "__main__":
    pytest.main([__file__])
