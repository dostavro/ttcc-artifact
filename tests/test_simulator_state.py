"""
Test to verify that the Simulator doesn't cause state pollution between runs.
Checks if jobs are properly reset or if modifications persist.
"""
from simulator import Simulator
from cluster import Cluster
from schedulers.easy import EASY
from schedulers.conservative import Conservative
from jobs import Job


def test_simulator_resets_job_state():
    """Test that one Simulator can be reused with different schedulers without state pollution."""

    # Create a job scenario that produces different schedules in EASY vs Conservative
    # This is based on the test_easy_vs_conservative_performance scenario:
    # - 3 GPUs total
    # - running: arrives at 0, demand=2, duration=10 (occupies t=0-10, leaves 1 GPU free)
    # - wait1: arrives at 1, demand=2, duration=5 (reserved at t=10 in both schedulers)
    # - wait2: arrives at 2, demand=3, duration=10 (reserved at t=15 in Conservative, later in EASY)
    # - small: arrives at 3, demand=1, duration=20 (KEY: EASY allows backfill at t=3, Conservative blocks until after wait2)
    jobs = [
        Job("running", 0, 2, 10, 10, "Gold"),  # Takes 2 of 3 GPUs from t=0-10
        Job("wait1", 1, 2, 5, 5, "Gold"),      # Needs 2 GPUs, reserved at t=10
        Job("wait2", 2, 3, 10, 10, "Gold"),    # Needs ALL 3 GPUs, reserved later
        Job("small", 3, 1, 20, 20, "Silver"),  # Needs 1 GPU - can backfill in EASY but not Conservative
    ]

    print("\n=== Initial Job State ===")
    for job in jobs:
        print(f"{job.jid}: start={job.start}, finish={job.finish}, allocated={job.allocated}")

    # Create ONE cluster and ONE simulator instance
    cluster = Cluster(3, 0)  # 3 GPUs

    # Run with EASY scheduler
    print("\n=== Running with EASY Scheduler ===")
    simulator = Simulator(cluster, jobs, EASY(), debug=False)
    result1 = simulator.run(horizon=20)

    print("After EASY:")
    for job in jobs:
        print(f"{job.jid}: start={job.start}, finish={job.finish}, allocated={job.allocated}")

    # Save the state after first run
    state_after_easy = {}
    for job in jobs:
        state_after_easy[job.jid] = {
            'start': job.start,
            'finish': job.finish,
            'allocated': job.allocated,
        }

    # NOW: Reuse the SAME simulator with a DIFFERENT scheduler (Conservative)
    # The Simulator.run() method should reset job state and simulator state
    print("\n=== Running with Conservative Scheduler ===")
    simulator.scheduler = Conservative()
    result2 = simulator.run(horizon=20)

    print("After Conservative:")
    for job in jobs:
        print(f"{job.jid}: start={job.start}, finish={job.finish}, allocated={job.allocated}")

    # Save the state after second run
    state_after_conservative = {}
    for job in jobs:
        state_after_conservative[job.jid] = {
            'start': job.start,
            'finish': job.finish,
            'allocated': job.allocated,
        }

    print("\n=== State Comparison ===")
    print(f"EASY scheduling had {len(result1)} finished jobs")
    print(f"Conservative scheduling had {len(result2)} finished jobs")

    # Check if job states changed
    for jid in state_after_easy:
        state_easy = state_after_easy[jid]
        state_conservative = state_after_conservative[jid]

        print(f"\n{jid}:")
        print(f"  After EASY: start={state_easy['start']}, finish={state_easy['finish']}")
        print(f"  After Conservative: start={state_conservative['start']}, finish={state_conservative['finish']}")

        if state_easy == state_conservative:
            print(f"  ⚠️  SAME STATE - Jobs may not be getting reset between scheduler runs!")
        else:
            print(f"  ✓ DIFFERENT STATE - Jobs were re-scheduled differently")

    # Verify that jobs have different start times with different schedulers
    print("\n=== Validation ===")
    different_start = any(
        state_after_easy[jid]['start'] != state_after_conservative[jid]['start']
        for jid in state_after_easy
    )

    if different_start:
        print("✓ Good: Jobs have different start times between EASY and Conservative")
    else:
        print("⚠️  WARNING: All jobs have the same start times in both schedulers!")
        print("This could indicate state pollution where the second scheduler run")
        print("inherits state from the first without proper reset.\n")

        # Assert that schedulers should produce different schedules for this scenario
        assert False, (
            f"ASSERTION FAILURE: Same job start times in both scheduler runs!\n"
            f"EASY and Conservative schedulers should produce different schedules.\n"
            f"If they don't, it indicates state pollution where job state is not being "
            f"properly reset between runs.\n"
            f"Job states after EASY: {state_after_easy}\n"
            f"Job states after Conservative: {state_after_conservative}"
        )
