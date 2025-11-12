"""
Discrete-event simulation kernel for GPU scheduling.

Implements event-driven simulation with:
- Priority event queue (ordered by time, with tie-breaking)
- Job lifecycle tracking (pending → running → finished)
- Scheduler integration via callback hooks (on_arrival, on_finish, try_schedule)
- Cluster resource management and tracking

The simulator processes events chronologically and invokes scheduler callbacks
to enable scheduler-specific allocation policies.
"""
import heapq


class Simulator:
    def __init__(self, cluster, jobs, scheduler, debug=False):
        self.cluster = cluster
        self.jobs = jobs
        self.scheduler = scheduler
        self.time = 0
        self.event_queue = []  # (time, counter, type, job)
        self.event_counter = 0  # Unique counter to break ties
        self.pending = []
        self.running = []
        self.finished = []
        self.debug = debug
        self.job_scheduled_finish_time = {}  # Maps job -> current scheduled finish time

    def log(self, msg):
        if self.debug:
            print(f"[t={self.time:.2f}] {msg}")

    def schedule_event(self, t, etype, job):
        heapq.heappush(self.event_queue, (t, self.event_counter, etype, job))
        self.event_counter += 1
        # Track the current scheduled finish time for finish events
        if etype == "finish":
            self.job_scheduled_finish_time[job] = t

    def reschedule_job_finish(self, job, new_finish_time):
        """
        Reschedule a job's finish event to a new time.
        Used when a job is preempted and needs to resume later.
        """
        # Just track the new finish time; old events will be ignored
        self.job_scheduled_finish_time[job] = new_finish_time
        # Schedule the new finish event
        self.schedule_event(new_finish_time, "finish", job)

    def run(self, horizon=100):
        # Reset all job state to avoid pollution from previous simulations
        for job in self.jobs:
            job.start = None
            job.last_resume_time = None
            job.finish = None
            job.allocated = None
            job.final_allocated = None
            job.finish_event_version = 0
            job.work_remaining = job.actual_time  # Reset work remaining to original

        # Reset cluster state (restore all resources to free)
        self.cluster.free = set(self.cluster.resources)

        # Reset simulator state
        self.time = 0
        self.event_queue = []
        self.event_counter = 0
        self.pending = []
        self.running = []
        self.finished = []
        self.job_scheduled_finish_time = {}

        for job in self.jobs:
            self.schedule_event(job.arrival, "arrival", job)

        while self.event_queue and self.time < horizon:
            # Get current time from next event
            current_time = self.event_queue[0][0]
            self.time = current_time

            # Process ALL finish events at current time first
            while self.event_queue and self.event_queue[0][0] == current_time and self.event_queue[0][2] == "finish":
                _, _, etype, job = heapq.heappop(self.event_queue)

                # Check if this is a stale finish event (old scheduled time, not the current one)
                if job in self.job_scheduled_finish_time and self.job_scheduled_finish_time[job] != current_time:
                    self.log(
                        f"Stale finish event for job {job.jid} at t={current_time}, scheduled for t={self.job_scheduled_finish_time[job]}, skipping")
                    continue

                if job in self.running:
                    self.cluster.release(job)
                    job.finish = self.time
                    job.work_remaining = 0  # Job is done, all work complete
                    self.running.remove(job)
                    self.finished.append(job)
                    self.log(f"Job {job.jid} FINISHED (qos={job.qos})")
                    self.scheduler.on_job_finish(self, job)
                else:
                    self.log(f"Stale finish event for job {job.jid}, skipping")

            # Then process ALL arrival events at current time
            while self.event_queue and self.event_queue[0][0] == current_time and self.event_queue[0][2] == "arrival":
                _, _, etype, job = heapq.heappop(self.event_queue)

                self.pending.append(job)
                self.log(f"Job {job.jid} (qos={job.qos}, demand={job.demand}, rt={job.actual_time:.2f}) ARRIVED")
                self.scheduler.on_job_arrival(self, job)

            # Finally, give scheduler a chance to act after all events at this time
            self.scheduler.try_schedule(self)
            self.log(f"Running jobs: {[j.jid for j in self.running]}")

        return self.finished
