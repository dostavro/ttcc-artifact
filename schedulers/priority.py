from collections import deque
from .base import Scheduler


class PriorityQoS(Scheduler):
    """
    Priority-based Scheduler with Preemption (Kubernetes-style).

    Maintains strict priority ordering (Gold > Silver > Bronze) with separate queues per
    QoS level. Higher-priority jobs can preempt lower-priority running jobs to obtain
    resources. Preempted jobs are requeued and their work progress is tracked.

    Properties: Strong priority guarantees with no starvation of high-priority jobs.
    Configurable preemption policies. Low-priority jobs may experience delays under heavy
    high-priority load. Respects QoS-aware resource preferences (Gold prefers edge,
    Silver/Bronze prefer cloud).
    """

    def __init__(self, enable_preemption=True, allow_silver_preempt_bronze=True):
        self.enable_preemption = enable_preemption
        self.allow_silver_preempt_bronze = allow_silver_preempt_bronze
        self.order = ["Gold", "Silver", "Bronze"]
        self.queues = {q: deque() for q in self.order}

    def on_job_arrival(self, sim, job):
        self.queues[job.qos].append(job)
        # Try to schedule immediately (including preemption)
        self.try_schedule(sim)

    def on_job_finish(self, sim, job):
        # When a job finishes, immediately try to schedule waiting jobs (including preempted ones)
        self.try_schedule(sim)

    def try_schedule(self, sim):
        """
        Try to start as many jobs as possible in strict priority order.
        If a head-of-queue job can't start and preemption is enabled,
        evict the minimum set of lower-priority jobs that frees enough GPUs.
        """
        # Walk priorities Gold -> Silver -> Bronze
        for p_idx, qos in enumerate(self.order):
            q = self.queues[qos]
            # Keep trying to place jobs of this priority while possible
            placed_something = True
            while placed_something and q:
                placed_something = False
                job = q[0]  # peek head
                if job in sim.running or job.finish is not None:
                    q.popleft()  # stale entry protection
                    continue

                # 1) Try to allocate without preemption
                alloc = sim.cluster.allocate(job, job.demand)
                if alloc:
                    q.popleft()
                    job.start = sim.time if job.start is None else job.start
                    job.last_resume_time = sim.time  # Track when this job started/resumed
                    sim.running.append(job)
                    # Schedule (or reschedule) finish event
                    # Use work_remaining for preempted jobs, actual_time for new jobs
                    sim.reschedule_job_finish(job, sim.time + job.work_remaining)
                    resource_types = self._get_resource_type_summary(alloc)
                    sim.log(f"Job {job.jid} STARTED (qos={job.qos}, demand={job.demand}, {resource_types})")
                    placed_something = True
                    continue

                # 2) Consider preemption if enabled and a higher-priority job is blocked
                if not self.enable_preemption:
                    break  # can't place head; move to next priority

                # Only allow preemption “downwards”
                victims = self._choose_victims(sim, p_idx, job)
                if victims is None:
                    # Not enough lower-priority capacity to satisfy this job
                    break

                # Evict the chosen victims
                for v in victims:
                    sim.cluster.release(v)
                    if v in sim.running:
                        sim.running.remove(v)
                    # Track how much work the victim had done in THIS running interval
                    if hasattr(v, 'last_resume_time') and v.last_resume_time is not None:
                        # work_done = time since this job last started or resumed
                        work_done = sim.time - v.last_resume_time
                        work_remaining_before = v.work_remaining
                        v.work_remaining = max(0, v.work_remaining - work_done)
                        sim.log(
                            f"Job {v.jid} PREEMPTED after {work_done} time units (was {work_remaining_before} remaining, now {v.work_remaining} remaining)")

                        # If job has no work remaining, finish it immediately
                        if v.work_remaining <= 0:
                            v.finish = sim.time
                            sim.finished.append(v)
                            sim.log(f"Job {v.jid} FINISHED (completed during preemption)")
                        else:
                            # Reschedule the finish event for when it resumes
                            new_finish_time = sim.time + v.work_remaining
                            sim.reschedule_job_finish(v, new_finish_time)
                            # Requeue preempted victim to *front* of its queue
                            self.queues[v.qos].appendleft(v)
                    sim.log(f"Job {v.jid} PREEMPTED (qos={v.qos}, demand={v.demand})")

                # Now retry allocation for the head job
                alloc = sim.cluster.allocate(job, job.demand)
                if alloc:
                    q.popleft()
                    job.start = sim.time if job.start is None else job.start
                    job.last_resume_time = sim.time  # Track when this job started/resumed
                    sim.running.append(job)
                    # Schedule (or reschedule) finish event
                    # Use work_remaining for preempted jobs, actual_time for new jobs
                    sim.reschedule_job_finish(job, sim.time + job.work_remaining)
                    resource_types = self._get_resource_type_summary(alloc)
                    sim.log(f"Job {job.jid} STARTED (preempted, qos={job.qos}, demand={job.demand}, {resource_types})")
                    placed_something = True
                else:
                    # Should be rare (race with another placement); stop trying this queue
                    break

    def _choose_victims(self, sim, higher_pri_index, waiting_job):
        """
        Choose a minimal set of running *lower-priority* jobs to evict
        to free enough GPUs to start `waiting_job`.

        Strategy:
          - Consider only jobs with lower priority (higher_pri_index+1 .. end).
          - Optionally disallow Silver->Bronze preemption via allow_silver_preempt_bronze.
          - Sort candidates by (priority ascending, demand descending)
            so we evict the lowest-priority and largest-demand jobs first,
            minimizing the number of evictions.
          - Return the smallest prefix whose total freed GPUs + free_now >= demand.
          - If insufficient, return None.
        """
        free_now = len(sim.cluster.free)
        need = waiting_job.demand - free_now
        if need <= 0:
            return []  # already enough

        # collect candidate victims (strictly lower priority)
        candidates = []
        for job in sim.running:
            # skip already finishing or same/higher priority
            j_idx = self.order.index(job.qos)
            if j_idx <= higher_pri_index:
                continue
            # optional policy: block Silver preempting Bronze
            if (not self.allow_silver_preempt_bronze and
                    self.order[higher_pri_index] == "Silver" and job.qos == "Bronze"):
                continue
            candidates.append(job)

        if not candidates:
            return None

        # sort: lowest priority first (larger index), then larger demand first
        candidates.sort(key=lambda j: (self.order.index(j.qos), -j.demand), reverse=True)

        chosen = []
        freed = 0
        for v in candidates:
            chosen.append(v)
            freed += v.demand
            if freed + free_now >= waiting_job.demand:
                return chosen

        return None  # even evicting all lower-priority jobs isn't enough

    def _get_resource_type_summary(self, alloc):
        """Helper method to summarize resource allocation by type."""
        edge_count = sum(1 for r in alloc if r.startswith("edge"))
        cloud_count = sum(1 for r in alloc if r.startswith("cloud"))
        return f"edge={edge_count}, cloud={cloud_count}"
