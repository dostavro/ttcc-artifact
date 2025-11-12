from collections import deque
from .base import Scheduler


class RunAI(Scheduler):
    """
    Run:AI-style Quota-based Fair-Share Scheduler.

    Ensures equitable resource distribution across QoS classes based on configurable quota
    weights. Tracks cumulative resource-time usage per class (usage += demand × duration)
    and schedules from the most under-quota class first (lowest usage/quota ratio).

    Default quotas: Gold=3, Silver=2, Bronze=1. Within each class, maintains FIFO ordering.
    Usage accumulates indefinitely and is updated when jobs complete. This prevents any
    single class from monopolizing resources and provides long-term fairness.

    Properties: Strong long-term fairness guarantees across classes. No preemption overhead.
    Known limitation: resource fragmentation can prevent ideal quota-based scheduling (small
    jobs may fit when large jobs with better ratio cannot). Simpler than real Run:AI (no
    three-tier priority system, no gang scheduling, no time-windowed usage decay).
    """

    def __init__(self, quotas=None):
        """
        Args:
            quotas: dict mapping QoS class -> quota weight
                   Default: {"Gold": 3, "Silver": 2, "Bronze": 1} (reflecting priority)
        """
        # Default quota weights for QoS classes
        if quotas is None:
            quotas = {"Gold": 3, "Silver": 2, "Bronze": 1}

        self.quotas = quotas
        self.queues = {qos_class: deque() for qos_class in quotas}

        # Track cumulative resource-time usage per QoS class for fair-share
        # Usage = sum of (demand × duration) for all completed jobs in this class
        self.qos_usage = {qos_class: 0.0 for qos_class in quotas}

    def on_job_arrival(self, sim, job):
        # Jobs are queued by their QoS class
        if job.qos in self.queues:
            self.queues[job.qos].append(job)
        else:
            # Fallback - shouldn't happen with proper QoS classes
            sim.log(f"Warning: Job {job.jid} has unknown QoS {job.qos}, assigning to Bronze")
            job.qos = "Bronze"
            self.queues["Bronze"].append(job)

    def on_job_finish(self, sim, job):
        """
        Update cumulative usage when job finishes.
        Usage accumulates resource-time: demand × actual_time
        This ensures classes that have consumed more resources yield to others.
        """
        if hasattr(job, 'qos') and job.qos in self.qos_usage:
            if job.demand is not None and job.actual_time is not None:
                # Accumulate resource-time consumption (demand × duration)
                resource_time = job.demand * job.actual_time
                self.qos_usage[job.qos] += resource_time
                sim.log(f"Job {job.jid} finished: QoS {job.qos} cumulative usage now {self.qos_usage[job.qos]:.2f}")

    def try_schedule(self, sim):
        """
        Try to schedule jobs using fair-share quota-based allocation across QoS classes.
        QoS classes are scheduled in order of their current usage vs. quota ratio.
        Within each class, jobs are dispatched in arrival order (FIFO).
        """
        scheduled_something = True
        while scheduled_something:
            scheduled_something = False

            # Calculate fair-share priorities: QoS classes with lowest usage/quota ratio go first
            qos_priorities = []
            for qos_class in self.queues:
                if self.queues[qos_class]:  # Only consider classes with waiting jobs
                    quota_weight = self.quotas[qos_class]
                    current_usage = self.qos_usage[qos_class]
                    # Calculate usage ratio (lower = higher priority for fair-share)
                    usage_ratio = current_usage / quota_weight if quota_weight > 0 else float('inf')
                    qos_priorities.append((usage_ratio, qos_class))

            # Sort by usage ratio (ascending - lowest ratio gets priority)
            qos_priorities.sort()

            # Try to schedule from the most under-quota QoS class first
            for usage_ratio, qos_class in qos_priorities:
                queue = self.queues[qos_class]

                if queue:
                    job = queue[0]  # peek at head (FIFO within class)

                    # Stale entry protection
                    if job in sim.running or job.finish is not None:
                        queue.popleft()
                        continue

                    # Try allocation
                    alloc = sim.cluster.allocate(job, job.demand)

                    if alloc:
                        # Successfully allocated
                        queue.popleft()
                        job.start = sim.time if job.start is None else job.start
                        sim.running.append(job)
                        sim.schedule_event(sim.time + job.actual_time, "finish", job)

                        # Note: Usage is updated on job finish, not on start
                        # This implements cumulative resource-time consumption tracking

                        # Debug logging
                        resource_types = self._get_resource_type_summary(alloc)
                        sim.log(
                            f"Job {job.jid} STARTED RunAI (qos={qos_class}, {resource_types}, current_usage_ratio={usage_ratio:.2f})")

                        scheduled_something = True
                        break  # Recalculate priorities after scheduling
                    else:
                        # Cannot allocate this job, try next QoS class
                        continue  # Keep trying other QoS classes in priority order

    def _get_resource_type_summary(self, alloc):
        """Helper method to summarize resource allocation by type."""
        edge_count = sum(1 for r in alloc if r.startswith("edge"))
        cloud_count = sum(1 for r in alloc if r.startswith("cloud"))
        return f"edge={edge_count}, cloud={cloud_count}"
