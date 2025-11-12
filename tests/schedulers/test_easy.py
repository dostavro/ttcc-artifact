"""
Tests for EASY (Extensible Argonne Scheduling System) scheduler.
"""
from schedulers.easy import EASY
from schedulers.conservative import Conservative
from test_utils import create_test_job, create_test_cluster, run_scheduler_test
from metrics import utilization


def test_easy_basic():
    """Test basic EASY functionality."""
    scheduler = EASY()
    cluster = create_test_cluster(2, 2)
    jobs = [
        create_test_job("job1", 0, 3, 1, "Gold"),
        create_test_job("job2", 1, 2, 1, "Silver"),
        create_test_job("job3", 2, 1, 1, "Bronze")
    ]

    result = run_scheduler_test(scheduler, jobs, cluster)
    assert len(result['finished_jobs']) == 3, "All jobs should complete"


def test_easy_backfilling():
    """Test EASY backfilling - short jobs can start while longer jobs wait."""
    scheduler = EASY()
    cluster = create_test_cluster(2, 1)  # 3 total resources
    jobs = [
        # Big job starts first, uses 2 resources, runs for 8 time units
        create_test_job("big_job", 0, 8, 2, "Gold"),

        # Medium job arrives later, needs 2 resources, must wait for big_job to finish
        create_test_job("medium_job", 1, 6, 2, "Silver"),

        # Small job arrives even later but only needs 1 resource - can backfill!
        create_test_job("small_job", 3, 2, 1, "Bronze"),
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=20)

    # All jobs should complete
    assert len(result['finished_jobs']) == 3, "All jobs should complete"

    # Get job results
    big_job = next(j for j in result['finished_jobs'] if j.jid == "big_job")
    medium_job = next(j for j in result['finished_jobs'] if j.jid == "medium_job")
    small_job = next(j for j in result['finished_jobs'] if j.jid == "small_job")

    # Big job should start immediately
    assert big_job.start == 0, "Big job should start at time 0"
    assert big_job.finish == 8, "Big job should finish at time 8"

    # Medium job must wait for big job (both need 2 resources, only 3 total, big job uses 2)
    assert medium_job.start >= big_job.finish, "Medium job must wait for big job to finish"

    # BACKFILLING: Small job should start BEFORE medium job even though it arrived later
    # This is the essence of backfilling - reorder jobs to fill resource gaps
    assert small_job.start < medium_job.start, "Small job should backfill and start before medium job"

    # Small job can use the 1 remaining resource while big job is still running
    assert small_job.start >= 3, "Small job can't start before arrival time 3"
    assert small_job.start < big_job.finish, "Small job should start while big job is still running (backfilling)"
    assert small_job.finish == small_job.start + 2, "Small job should finish 2 time units after starting"


def test_easy_backfill_no_starvation():
    """Test that EASY backfilling doesn't starve the reserved job."""
    scheduler = EASY()
    cluster = create_test_cluster(2, 0)  # 2 edge resources
    jobs = [
        create_test_job("running", 0, 10, 1, "Gold"),      # Uses 1 resource, runs for long time
        create_test_job("reserved", 1, 8, 2, "Silver"),    # Head of queue, gets reservation
        create_test_job("backfill1", 2, 3, 1, "Bronze"),   # Can backfill (finishes before reserved starts)
        create_test_job("backfill2", 4, 4, 1, "Bronze"),   # Can backfill (finishes before reserved starts)
        create_test_job("blocked", 6, 6, 1, "Bronze"),     # Would delay reserved job - should NOT backfill
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=25)

    # All jobs should complete
    assert len(result['finished_jobs']) == 5, "All jobs should complete"

    # Get job results
    running = next(j for j in result['finished_jobs'] if j.jid == "running")
    reserved = next(j for j in result['finished_jobs'] if j.jid == "reserved")
    backfill1 = next(j for j in result['finished_jobs'] if j.jid == "backfill1")
    backfill2 = next(j for j in result['finished_jobs'] if j.jid == "backfill2")
    blocked = next(j for j in result['finished_jobs'] if j.jid == "blocked")

    # Running job starts immediately
    assert running.start == 0, "Running job should start immediately"
    assert running.finish == 10, "Running job should finish at t=10"

    # Reserved job should start when running job finishes (has reservation)
    assert reserved.start == running.finish, "Reserved job should start when running job finishes"
    assert reserved.finish == 18, "Reserved job should finish at t=18"

    # Backfill jobs should run before reserved job (they don't delay it)
    assert backfill1.start >= 2, "backfill1 can't start before arrival"
    assert backfill1.start < reserved.start, "backfill1 should start before reserved job"
    assert backfill1.finish <= reserved.start, "backfill1 should finish before reserved job starts"

    assert backfill2.start >= 4, "backfill2 can't start before arrival"
    assert backfill2.start < reserved.start, "backfill2 should start before reserved job"
    assert backfill2.finish <= reserved.start, "backfill2 should finish before reserved job starts"

    # Blocked job should NOT backfill (would delay reserved job)
    # It should wait until after reserved job finishes
    assert blocked.start == reserved.finish, "blocked job should wait for reserved job to finish"

    # Key EASY behavior: reserved job is NOT delayed by other jobs
    assert reserved.start == 10, "Reserved job should not be delayed by backfilling jobs"


def test_easy_reservation_update():
    """Test that EASY updates reservation when reserved job finishes."""
    scheduler = EASY()
    cluster = create_test_cluster(3, 0)  # 3 GPUs so small job can backfill
    jobs = [
        create_test_job("first_reserved", 0, 5, 2, "Gold"),   # First head job, gets reservation, uses 2 GPUs
        create_test_job("second_reserved", 1, 6, 2, "Silver"),  # Becomes head after first finishes
        create_test_job("small", 2, 2, 1, "Bronze"),          # Can backfill with 1 remaining GPU
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=20)

    first_reserved = next(j for j in result['finished_jobs'] if j.jid == "first_reserved")
    second_reserved = next(j for j in result['finished_jobs'] if j.jid == "second_reserved")
    small = next(j for j in result['finished_jobs'] if j.jid == "small")

    # First reserved job starts immediately
    assert first_reserved.start == 0, "First reserved job should start immediately"
    assert first_reserved.finish == 5, "First reserved job should finish at t=5"

    # Second reserved job becomes head and starts when first finishes
    assert second_reserved.start == first_reserved.finish, "Second job should start when first finishes"
    assert second_reserved.finish == 11, "Second reserved job should finish at t=11"

    # Small job should backfill during first_reserved execution (1 GPU left)
    assert small.start >= 2 and small.start < first_reserved.finish, "Small job should backfill"


def test_easy_multi_gpu_jobs():
    """Test EASY with multi-GPU jobs."""
    scheduler = EASY()
    cluster = create_test_cluster(3, 1)  # 4 total resources
    jobs = [
        create_test_job("large", 0, 10, 3, "Gold"),      # Uses 3 GPUs, leaves 1 free
        create_test_job("medium", 1, 8, 2, "Silver"),    # Needs 2 GPUs, must wait
        create_test_job("small1", 2, 3, 1, "Bronze"),    # Can backfill with the 1 free GPU
        create_test_job("small2", 4, 4, 1, "Bronze"),    # Can backfill after small1
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=25)

    large = next(j for j in result['finished_jobs'] if j.jid == "large")
    medium = next(j for j in result['finished_jobs'] if j.jid == "medium")
    small1 = next(j for j in result['finished_jobs'] if j.jid == "small1")
    small2 = next(j for j in result['finished_jobs'] if j.jid == "small2")

    # Large job uses 3 GPUs
    assert large.start == 0, "Large job should start immediately"
    assert large.finish == 10, "Large job should finish at t=10"
    assert large.demand == 3, "Large job should use 3 GPUs"

    # Medium job must wait for large job (needs 2 GPUs, only 1 available)
    assert medium.start >= large.finish, "Medium job must wait for large job"

    # Small jobs can backfill using the 1 available GPU
    assert small1.start >= 2 and small1.start < large.finish, "Small1 should backfill"
    assert small2.start >= 4 and small2.start < large.finish, "Small2 should backfill"


def test_easy_multiple_backfill_candidates():
    """Test EASY chooses backfill candidates correctly when multiple jobs could fit."""
    scheduler = EASY()
    cluster = create_test_cluster(2, 0)
    jobs = [
        create_test_job("running", 0, 10, 1, "Gold"),       # Uses 1 GPU
        create_test_job("reserved", 1, 8, 2, "Silver"),     # Head of queue, reserved
        create_test_job("backfill1", 2, 3, 1, "Bronze"),    # Arrives first, can backfill
        create_test_job("backfill2", 2, 3, 1, "Bronze"),    # Arrives same time, can backfill
        create_test_job("backfill3", 3, 2, 1, "Bronze"),    # Arrives later, can backfill
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=25)

    running = next(j for j in result['finished_jobs'] if j.jid == "running")
    reserved = next(j for j in result['finished_jobs'] if j.jid == "reserved")
    backfill1 = next(j for j in result['finished_jobs'] if j.jid == "backfill1")
    backfill2 = next(j for j in result['finished_jobs'] if j.jid == "backfill2")
    backfill3 = next(j for j in result['finished_jobs'] if j.jid == "backfill3")

    # Reserved job should start when running finishes
    assert reserved.start == running.finish, "Reserved job starts when running finishes"

    # All backfill jobs should start before reserved job
    assert backfill1.start < reserved.start, "backfill1 should backfill"
    assert backfill2.start < reserved.start, "backfill2 should backfill"
    assert backfill3.start < reserved.start, "backfill3 should backfill"

    # Backfill jobs should execute in arrival order (FIFO within backfill candidates)
    assert backfill1.finish <= backfill2.start or backfill2.finish <= backfill1.start, \
        "backfill1 and backfill2 should not overlap (only 1 free GPU)"
    assert backfill3.start >= 3, "backfill3 can't start before arrival time"


def test_easy_empty_queue_refill():
    """Test EASY handles queue becoming empty and refilling."""
    scheduler = EASY()
    cluster = create_test_cluster(2, 0)
    jobs = [
        # First batch - both can run in parallel
        create_test_job("early1", 0, 3, 1, "Gold"),
        create_test_job("early2", 0, 3, 1, "Silver"),
        # Gap in arrivals - queue becomes empty at t=3
        # Second batch - new jobs arrive after queue was empty
        create_test_job("late1", 5, 4, 2, "Bronze"),  # Uses both GPUs
        create_test_job("late2", 6, 2, 1, "Gold"),     # Must wait for late1
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=15)

    early1 = next(j for j in result['finished_jobs'] if j.jid == "early1")
    early2 = next(j for j in result['finished_jobs'] if j.jid == "early2")
    late1 = next(j for j in result['finished_jobs'] if j.jid == "late1")
    late2 = next(j for j in result['finished_jobs'] if j.jid == "late2")

    # Early jobs should both run (we have 2 GPUs and each needs 1)
    assert early1.start == 0, "early1 starts immediately"
    assert early2.start == 0, "early2 starts immediately"
    assert early1.finish == 3, "early1 finishes at t=3"
    assert early2.finish == 3, "early2 finishes at t=3"

    # Queue should be empty between t=3 and t=5
    # Late1 starts when it arrives (queue empty, resources available)
    assert late1.start == 5, "late1 starts immediately upon arrival (queue was empty)"
    assert late1.finish == 9, "late1 finishes at t=9"

    # Late2 must wait for late1 (late1 uses all resources)
    assert late2.start >= late1.finish, "late2 waits for late1 to finish"

    # All jobs should complete
    assert len(result['finished_jobs']) == 4, "All jobs should complete"


def test_complex_overlapping_backfills():
    """Test complex scenario with overlapping backfill opportunities."""
    scheduler = EASY()
    cluster = create_test_cluster(2, 0)  # 2 GPUs
    jobs = [
        create_test_job("running", 0, 5, 1, "Gold"),        # Uses 1 GPU
        create_test_job("reserved", 1, 2, 2, "Silver"),      # Head of queue, needs all 2 GPUs, gets reservation t=5
        create_test_job("blocked", 2, 5, 1, "Bronze"),       # Would delay reserved job - should NOT backfill
        create_test_job("backfill1", 2, 3, 1, "Bronze"),     # Can backfill before reserved starts
        # Would delay blocked job - should NOT backfill gets reservation after reserved finishes t=7
        create_test_job("blocked2", 3, 5, 2, "Bronze"),
        create_test_job("backfill2", 4, 2, 1, "Bronze"),     # Can backfill before blocked2 starts
        create_test_job("blocked3", 4, 4, 1, "Bronze"),     # Would delay blocked2 - should NOT backfill
        create_test_job("backfill3", 5, 2, 1, "Bronze"),     # Can backfill before blocked3 starts
    ]

    result = run_scheduler_test(scheduler, jobs, cluster, horizon=30)

    running = next(j for j in result['finished_jobs'] if j.jid == "running")
    reserved = next(j for j in result['finished_jobs'] if j.jid == "reserved")
    blocked = next(j for j in result['finished_jobs'] if j.jid == "blocked")
    blocked2 = next(j for j in result['finished_jobs'] if j.jid == "blocked2")
    blocked3 = next(j for j in result['finished_jobs'] if j.jid == "blocked3")
    backfill1 = next(j for j in result['finished_jobs'] if j.jid == "backfill1")
    backfill2 = next(j for j in result['finished_jobs'] if j.jid == "backfill2")
    backfill3 = next(j for j in result['finished_jobs'] if j.jid == "backfill3")

    # Reserved job should start when running finishes
    assert reserved.start == running.finish, "Reserved job starts when running finishes"

    # Blocked jobs should start when their dependencies finish
    assert blocked.start == reserved.finish, "Blocked job starts when reserved finishes"
    assert blocked2.start == blocked.finish, "Blocked2 job starts when blocked1 finishes"
    assert blocked3.start == blocked2.finish, "Blocked3 job starts when blocked2 finishes"

    # Backfill jobs should start before reserved job
    assert backfill1.start < reserved.start, "backfill1 should start before reserved job"
    assert backfill2.start < blocked2.start, "backfill2 should start before blocked2 job"
    assert backfill3.start < blocked3.start, "backfill3 should start before blocked3 job"

    # Backfill jobs should execute in arrival order (FIFO within backfill candidates)
    assert backfill1.finish <= backfill2.start or backfill2.finish <= backfill1.start, \
        "backfill1 and backfill2 should not overlap (only 1 free GPU)"
    assert backfill2.finish <= backfill3.start or backfill3.finish <= backfill2.start, \
        "backfill2 and backfill3 should not overlap (only 1 free GPU)"
