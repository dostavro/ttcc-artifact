"""
Abstract base class for GPU scheduling algorithms.

All schedulers implement three core lifecycle methods:
1. on_job_arrival: React when a job enters the system
2. on_job_finish: React when a job completes and releases resources
3. try_schedule: Attempt to allocate waiting jobs to available resources

Subclasses implement concrete policies (FCFS, EASY, backfilling, preference-aware, etc.).
"""
from abc import ABC, abstractmethod


class Scheduler(ABC):
    @abstractmethod
    def on_job_arrival(self, sim, job):
        """
        Handle job arrival event.

        Args:
            sim: Simulator instance (provides access to cluster, jobs, time)
            job: Job instance that just arrived
        """
        pass

    @abstractmethod
    def on_job_finish(self, sim, job):
        """
        Handle job completion event.

        Args:
            sim: Simulator instance
            job: Job instance that just completed
        """
        pass

    @abstractmethod
    def try_schedule(self, sim):
        """
        Attempt to schedule waiting jobs on available resources.

        Called after arrival or completion events to allocate resources.

        Args:
            sim: Simulator instance
        """
        pass
