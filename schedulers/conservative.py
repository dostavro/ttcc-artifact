from .base import Scheduler
from collections import deque


class Conservative(Scheduler):
    """
    Conservative Backfilling Scheduler.

    Computes explicit resource reservations for all queued jobs and only allows backfilling
    if it doesn't violate any existing reservation. More conservative than EASY which only
    reserves time for the first queued job.

    Properties: Guarantees no job is delayed by backfilling. Provides stronger fairness
    than EASY (all jobs get reservations) but may have lower utilization due to stricter
    constraints. FIFO order is preserved for reservations.
    """

    def __init__(self):
        self.queue = deque()
        self.reservations = {}  # job -> reservation time

    def on_job_arrival(self, sim, job):
        self.queue.append(job)
        self.reservations[job] = self._compute_reservation(sim, job)

    def on_job_finish(self, sim, job):
        if job in self.reservations:
            del self.reservations[job]

    def try_schedule(self, sim):
        if not self.queue:
            return

        for job in list(self.queue):
            if sim.time >= self.reservations[job]:
                alloc = sim.cluster.allocate(job, job.demand)
                if alloc:
                    self.queue.remove(job)
                    del self.reservations[job]
                    job.start = sim.time
                    sim.running.append(job)
                    sim.schedule_event(sim.time + job.actual_time, "finish", job)
                    sim.log(f"Job {job.jid} STARTED (conservative, qos={job.qos}, demand={job.demand}, alloc={list(alloc)})")
                else:
                    sim.log(f"WARNING: Conservative job {job.jid} cannot be allocated despite meeting reservation time")

    def _compute_reservation(self, sim, job):
        """Compute when this job can start, respecting existing reservations."""

        candidate_time = sim.time

        while True:
            # Start with total cluster capacity
            available = sim.cluster.num_edge + sim.cluster.num_cloud

            # Debug output for this specific test case
            if job.jid == "small3":
                print(f"\nDEBUG small3 at candidate_time={candidate_time}")
                print(f"Total resources: {available}")

            # Subtract resources used by running jobs that will still be running
            for running_job in sim.running:
                if hasattr(running_job, 'start') and running_job.start is not None:
                    job_end = running_job.start + running_job.actual_time
                    if job_end > candidate_time:  # Still running at candidate_time
                        available -= running_job.demand
                        if job.jid == "small3":
                            print(
                                f"Running job {running_job.jid} blocks {running_job.demand} resources (ends at {job_end})")

            # Subtract resources used by reserved jobs during our execution window
            our_end_time = candidate_time + job.actual_time
            if job.jid == "small3":
                print(f"Our execution window: {candidate_time} → {our_end_time}")

            for other_job, reservation_time in self.reservations.items():
                if other_job != job:
                    other_end_time = reservation_time + other_job.actual_time
                    # Check if execution windows overlap
                    overlap = (reservation_time < our_end_time and other_end_time > candidate_time)
                    if job.jid == "small3":
                        print(f"Reserved job {other_job.jid}: {reservation_time} → {other_end_time}, overlap={overlap}")
                    if overlap:
                        available -= other_job.demand

            if job.jid == "small3":
                print(f"Available resources after conflicts: {available}")
                print(f"small3 needs: {job.demand}")

            # If we have enough resources, we can start at this time
            if available >= job.demand:
                return candidate_time

            # Find next time when resources might become available
            next_time = float('inf')

            # Check when running jobs finish
            for running_job in sim.running:
                if hasattr(running_job, 'start') and running_job.start is not None:
                    finish_time = running_job.start + running_job.actual_time
                    if finish_time > candidate_time:
                        next_time = min(next_time, finish_time)

            # Check when reserved jobs finish
            for other_job, reservation_time in self.reservations.items():
                if other_job != job:
                    finish_time = reservation_time + other_job.actual_time
                    if finish_time > candidate_time:
                        next_time = min(next_time, finish_time)

            if next_time == float('inf'):
                return candidate_time

            candidate_time = next_time
