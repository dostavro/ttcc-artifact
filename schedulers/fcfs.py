from collections import deque
from .base import Scheduler


class FCFS(Scheduler):
    """
    First-Come, First-Served (FCFS) Scheduler.

    Maintains strict arrival order with no backfilling or preemption. Jobs are executed
    in the exact order they arrive. The head job blocks the queue until it can start,
    even if resources are idle and smaller jobs could run.

    Properties: Simple, predictable, fair ordering, but low utilization due to
    head-of-line blocking. No support for priorities or resource preferences.
    """

    def __init__(self):
        self.queue = deque()

    def on_job_arrival(self, sim, job):
        self.queue.append(job)

    def on_job_finish(self, sim, job):
        pass  # nothing special

    def try_schedule(self, sim):
        while self.queue:
            job = self.queue[0]  # peek head of line
            if job in sim.running or job.finish is not None:
                # Already running or finished → discard
                self.queue.popleft()
                continue

            alloc = sim.cluster.allocate(job, job.demand)
            if alloc:
                # Start the job
                self.queue.popleft()
                job.start = sim.time
                sim.running.append(job)
                sim.schedule_event(sim.time + job.actual_time, "finish", job)
                sim.log(f"Job {job.jid} STARTED (qos={job.qos}, demand={job.demand}, alloc={list(alloc)})")
                # continue loop → maybe next head fits now
            else:
                # Head-of-line blocking: stop here
                break
