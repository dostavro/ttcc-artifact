from collections import deque
from .base import Scheduler


class EASY(Scheduler):
    """
    EASY Backfilling Scheduler (Extensible Argonne Scheduling sYstem).

    Allows smaller jobs to backfill idle resources as long as they don't delay the first
    job in the queue. The head job has a guaranteed reservation time, while jobs behind
    can start early if they complete before this reservation.

    Properties: Higher utilization than FCFS due to backfilling. Provides bounded wait
    time for the head job but may delay other jobs. More aggressive than Conservative
    backfilling which reserves time for all queued jobs.
    """

    def __init__(self):
        self.queue = deque()
        self.reserved_job = None
        self.reservation_time = None

    def on_job_arrival(self, sim, job):
        self.queue.append(job)

    def on_job_finish(self, sim, job):
        if job == self.reserved_job:
            self.reserved_job = None
            self.reservation_time = None

    def try_schedule(self, sim):
        # Keep trying to schedule jobs until nothing more can be scheduled
        while self.queue:
            # Always ensure head is reserved
            if self.reserved_job is None or self.reserved_job not in self.queue:
                self.reserved_job = self.queue[0]
                self.reservation_time = self._compute_reservation(sim, self.reserved_job)

            # Try to backfill jobs behind the reserved job
            leftovers = deque()
            backfilled_any = False
            for job in self.queue:
                if job == self.reserved_job:
                    leftovers.append(job)
                    continue

                # Backfill only if it won't delay the reserved job
                # Check if the backfill job will still be running at the reserved time
                # and if so, whether there will still be enough free resources for the reserved job
                backfill_finishes = sim.time + job.actual_time
                if backfill_finishes <= self.reservation_time:
                    # Backfill job finishes before reservation - always safe
                    can_backfill = True
                else:
                    # Backfill job overlaps with reservation time
                    # Check if reserved job can still fit when backfill is running
                    # Calculate GPUs that would be free at reservation_time if backfill is running
                    gpus_from_backfill = job.demand
                    gpus_from_running = sum(rj.demand for rj in sim.running
                                            if rj.start is not None
                                            and rj.start + rj.actual_time > self.reservation_time)
                    total_used = gpus_from_backfill + gpus_from_running
                    free_at_reservation = (sim.cluster.num_edge + sim.cluster.num_cloud) - total_used
                    can_backfill = free_at_reservation >= self.reserved_job.demand

                if can_backfill:
                    alloc = sim.cluster.allocate(job, job.demand)
                    if alloc:
                        job.start = sim.time
                        sim.running.append(job)
                        sim.schedule_event(sim.time + job.actual_time, "finish", job)
                        sim.log(f"Job {job.jid} STARTED (backfilled, qos={job.qos}, demand={job.demand}, alloc={list(alloc)})")
                        backfilled_any = True
                        continue  # job was scheduled, skip adding to leftovers

                leftovers.append(job)

            self.queue = leftovers

            # Try to start the reserved job
            started_reserved = False
            if self.reserved_job and self.reserved_job in self.queue:
                alloc = sim.cluster.allocate(self.reserved_job, self.reserved_job.demand)
                if alloc:
                    job = self.reserved_job
                    self.queue.popleft()  # reserved job is always at head
                    job.start = sim.time
                    sim.running.append(job)
                    sim.schedule_event(sim.time + job.actual_time, "finish", job)
                    sim.log(f"Job {job.jid} STARTED (reserved, qos={job.qos}, demand={job.demand}, alloc={list(alloc)})")
                    self.reserved_job = None
                    self.reservation_time = None
                    started_reserved = True
                else:
                    # This should not happen if reservation calculation was correct
                    sim.log(
                        f"WARNING: Reserved job {self.reserved_job.jid} cannot be allocated at reservation time {self.reservation_time} (current time: {sim.time}, demand: {self.reserved_job.demand}, free: {len(sim.cluster.free)})")

            # If we didn't start anything, stop trying
            # If we didn't start anything, stop trying
            if not backfilled_any and not started_reserved:
                break

    def _compute_reservation(self, sim, job):
        """
        Compute the earliest time the reserved job could start.
        Looks at free GPUs and finish times of running jobs.
        """
        free_now = len(sim.cluster.free)
        if job.demand <= free_now:
            return sim.time

        # Need to wait for running jobs to finish and free enough GPUs
        finish_times = []
        for running_job in sim.running:
            if hasattr(running_job, 'start') and running_job.start is not None:
                # Calculate finish time from start + actual_time
                job_finish_time = running_job.start + running_job.actual_time
                # Add multiple entries for multi-GPU jobs
                for _ in range(running_job.demand):
                    finish_times.append(job_finish_time)

        finish_times.sort()
        need = job.demand - free_now
        if len(finish_times) >= need:
            # The job can start when the need-th GPU becomes available
            return finish_times[need - 1]
        return float("inf")
