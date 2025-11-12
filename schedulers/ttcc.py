from collections import deque
from .base import Scheduler


class TTCC(Scheduler):
    """
    Multi-GPU Top Trading Cycles and Chains (TTCC) Scheduler with Gang Semantics.

    Implements true multi-unit TTC matching with two-sided preferences:
    - Jobs have preferences over resources (Gold prefers edge, Silver/Bronze prefer cloud)
    - Resources have preferences over jobs (Edge prefers Gold, Cloud prefers Silver/Bronze)
    - Gang semantics: waiting jobs start only when they acquire d GPUs simultaneously

    Algorithm (Pattern A: TTC on units, then gang check):
    1. Build a unit-level graph including both running and waiting jobs and all GPUs
    2. Each job points to its most-preferred GPU (owned or free)
    3. Each GPU points to its current owner or most-preferred waiting job
    4. Find cycles in this bipartite graph
    5. Execute unit-level trades: jobs swap GPUs along cycles
    6. Gang completion: waiting jobs with ≥ demand GPUs start immediately
    7. Repeat until no more trades or new jobs start (convergence)

    With migrations enabled:
    - Running jobs can migrate to better resources via cycles
    - Waiting jobs can start when they accumulate d GPUs

    With migrations disabled:
    - Only waiting jobs participate (no endowed running jobs in the graph)
    - Pure matching problem: allocate free GPUs to waiting jobs
    """

    def __init__(self, enable_migrations=False):
        self.queue = deque()
        self.migration_count = 0
        self.enable_migrations = enable_migrations

    def on_job_arrival(self, sim, job):
        """Add job to waiting queue."""
        self.queue.append(job)

    def on_job_finish(self, sim, job):
        """Trigger full scheduling after job completion."""
        # Resources are automatically released by cluster
        # Try full scheduling: multi-GPU TTC with gang semantics
        self.try_schedule(sim)

    def try_schedule(self, sim):
        """Try to schedule waiting jobs using multi-GPU TTC matching with gang semantics."""
        # Clean up queue first
        self.queue = deque([job for job in self.queue if job not in sim.running and job not in sim.finished])

        if not self.queue and not sim.running:
            return
        if not sim.cluster.free:
            return

        # Run multi-GPU TTC with optional migrations
        self._ttc_multi_gpu(sim)

    def _ttc_multi_gpu(self, sim):
        """Multi-GPU TTC matching with gang semantics.

        Strategy: Try TTC cycles first. When no cycles exist, greedily allocate immediately.
        Always finish with greedy allocation to ensure remaining GPUs are used.
        """
        max_iterations = 20
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Collect jobs and GPUs for this round
            if self.enable_migrations:
                all_jobs = list(sim.running) + list(self.queue)
            else:
                all_jobs = list(self.queue)

            if not all_jobs:
                break

            # Collect all GPUs (both free and allocated)
            all_gpus = set(sim.cluster.free)
            if self.enable_migrations:
                for job in sim.running:
                    if hasattr(job, 'allocated') and job.allocated:
                        all_gpus.update(job.allocated)

            # Include GPUs that waiting jobs already acquired
            for job in self.queue:
                if hasattr(job, 'allocated') and job.allocated:
                    all_gpus.update(job.allocated)

            if not all_gpus:
                # No GPUs available at all
                sim.log(f"TTCC iteration {iteration}: No GPUs available")
                break

            sim.log(f"TTCC iteration {iteration}: {len(all_jobs)} jobs, {len(all_gpus)} GPUs")

            # Build pointers
            job_to_gpu, gpu_to_job = self._build_pointers(sim, all_jobs, all_gpus)

            if not job_to_gpu or not gpu_to_job:
                sim.log(f"TTCC iteration {iteration}: No valid pointers")
                break

            # Find cycles
            cycles = self._find_ttc_cycles(job_to_gpu, gpu_to_job)

            if not cycles:
                # No cycles found: stop TTC
                sim.log(f"TTCC iteration {iteration}: No cycles - stopping TTC")
                break

            sim.log(f"TTCC iteration {iteration}: Found {len(cycles)} cycles")

            # Execute ALL cycles atomically to avoid conflicts
            made_trade = self._execute_all_cycles(sim, cycles, job_to_gpu, gpu_to_job)

            # Gang completion
            new_starts = self._gang_complete(sim)

            # If no progress, stop TTC
            if not made_trade and not new_starts:
                sim.log(f"TTCC iteration {iteration}: No progress - stopping TTC")
                break

        # Final EASY backfilling pass: allocate any remaining free GPUs to waiting jobs
        if self.queue and sim.cluster.free:
            sim.log(
                f"TTCC: Final EASY backfill for {len(self.queue)} waiting jobs with {len(sim.cluster.free)} free GPUs")
            self._easy_backfill(sim)

    def _build_pointers(self, sim, all_jobs, all_gpus):
        """Build unit-level job→GPU and GPU→job pointers.

        Each job points to its most-preferred GPU.
        Each GPU points to its current owner (or to its favorite job if free).
        """
        job_to_gpu = {}
        gpu_to_job = {}

        # Jobs point to their favorite GPU
        for job in all_jobs:
            best_gpu = self._best_gpu_for_job(job, all_gpus)
            if best_gpu is not None:
                job_to_gpu[job] = best_gpu

        # GPUs point to their owner or favorite job
        for gpu in all_gpus:
            # Check if GPU is owned by a running job (only when migrations enabled)
            owner = None
            if self.enable_migrations:
                for job in sim.running:
                    if hasattr(job, 'allocated') and job.allocated and gpu in job.allocated:
                        owner = job
                        break

            if owner is not None:
                # GPU is owned - it points to its owner
                gpu_to_job[gpu] = owner
            else:
                # GPU is free - it points to its favorite job among all jobs
                best_job = self._best_job_for_gpu(gpu, all_jobs)
                if best_job is not None:
                    gpu_to_job[gpu] = best_job

        return job_to_gpu, gpu_to_job

    def _execute_all_cycles(self, sim, cycles, job_to_gpu, gpu_to_job):
        """Execute ALL TTC cycles atomically to prevent conflicts.

        This is critical for correctness - executing cycles one-by-one can cause
        double-allocation when multiple cycles compete for the same GPUs.
        """
        # Collect ALL trades from ALL cycles
        all_trades = []  # List of (job, gpu_gives, gpu_receives)

        for cycle in cycles:
            jobs_in_cycle = [node for node in cycle if not isinstance(node, str)]
            gpus_in_cycle = [node for node in cycle if isinstance(node, str)]

            if not jobs_in_cycle:
                continue

            for job in jobs_in_cycle:
                gpu_receives = job_to_gpu.get(job)
                if gpu_receives is None:
                    continue

                gpu_gives = None

                if hasattr(job, 'allocated') and job.allocated:
                    # Running job - give away non-preferred GPU
                    non_preferred_type = "cloud" if job.qos == "Gold" else "edge"
                    non_preferred_gpus = [g for g in job.allocated if g.startswith(non_preferred_type)]
                    if non_preferred_gpus:
                        gpu_gives = non_preferred_gpus[0]
                    else:
                        continue  # Can't improve
                else:
                    # Waiting job - find GPU that points to this job
                    for gpu in gpus_in_cycle:
                        if gpu_to_job.get(gpu) == job:
                            gpu_gives = gpu
                            break

                # Skip self-loops (job already has this GPU or would receive what it gives)
                # Self-loops don't help and cause double-allocation bugs
                if gpu_gives == gpu_receives:
                    continue

                all_trades.append((job, gpu_gives, gpu_receives))

        if not all_trades:
            return False

        # Build global ownership map and validate all trades atomically
        gpu_receivers = {}  # gpu -> job that will receive it
        gpus_to_release = set()  # GPUs being released
        gpus_currently_owned = {}  # gpu -> current owner

        # Map ALL currently owned GPUs
        for job in sim.running:
            if hasattr(job, 'allocated') and job.allocated:
                for gpu in job.allocated:
                    gpus_currently_owned[gpu] = job

        for job in self.queue:
            if hasattr(job, 'allocated') and job.allocated:
                for gpu in job.allocated:
                    gpus_currently_owned[gpu] = job

        # Validate each trade
        for job, gpu_gives, gpu_receives in all_trades:
            # Track releases
            if gpu_gives and hasattr(job, 'allocated') and job.allocated and gpu_gives in job.allocated:
                gpus_to_release.add(gpu_gives)

            # Check for conflicts in receivers
            if gpu_receives in gpu_receivers:
                sim.log(
                    f"⚠️  Conflict: Jobs {job.jid} and {gpu_receivers[gpu_receives].jid} both want {gpu_receives} - skipping {job.jid}")
                continue

            # Check availability
            gpu_owner = gpus_currently_owned.get(gpu_receives)
            is_available = (
                gpu_receives in sim.cluster.free or
                gpu_receives in gpus_to_release or
                gpu_owner == job  # Self-loop
            )

            if is_available:
                gpu_receivers[gpu_receives] = job
            else:
                sim.log(
                    f"⚠️  GPU {gpu_receives} unavailable for job {job.jid} (owner: {gpu_owner.jid if gpu_owner else 'none'})")

        # Execute all validated trades atomically
        # CRITICAL: We must not modify job.allocated until all trades are applied,
        # otherwise later trades will see modified allocations and double-allocate
        made_trade = False
        pending_changes = []  # List of (job, gpu_to_remove, gpu_to_add)

        for job, gpu_gives, gpu_receives in all_trades:
            if gpu_receivers.get(gpu_receives) != job:
                continue  # Skip unvalidated trades

            if not hasattr(job, 'allocated') or job.allocated is None:
                job.allocated = set()

            if not job.allocated:
                # Waiting job acquires GPU
                pending_changes.append((job, None, gpu_receives))
                made_trade = True
            else:
                # Running job swaps
                if gpu_gives and gpu_gives in job.allocated:
                    pending_changes.append((job, gpu_gives, gpu_receives))
                    made_trade = True

        # Now apply all changes atomically
        for job, gpu_to_remove, gpu_to_add in pending_changes:
            if gpu_to_remove:
                # Swap case
                job.allocated.discard(gpu_to_remove)
                job.allocated.add(gpu_to_add)

                if gpu_to_add in sim.cluster.free:
                    sim.cluster.free.remove(gpu_to_add)
                sim.cluster.free.add(gpu_to_remove)

                sim.log(f"TTCC: Job {job.jid} swapped {gpu_to_remove} → {gpu_to_add}")
                self.migration_count += 1
            else:
                # Waiting job acquiring GPU
                job.allocated.add(gpu_to_add)
                if gpu_to_add in sim.cluster.free:
                    sim.cluster.free.remove(gpu_to_add)
                sim.log(f"TTCC: Job {job.jid} acquired {gpu_to_add} (now has {len(job.allocated)})")

        return made_trade

    def _gang_complete(self, sim):
        """Check waiting jobs and start those that have accumulated enough GPUs from TTC.

        If TTC gave partial allocations, complete them greedily with available GPUs.
        This prioritizes what TTC matched (preference-based) over pure EASY backfill.
        """
        new_starts = 0
        jobs_to_remove = []

        # Build snapshot of current allocations BEFORE modifying anything
        # This prevents race conditions when completing multiple jobs
        initial_allocations = {}
        for job in self.queue:
            if hasattr(job, 'allocated') and job.allocated:
                initial_allocations[job] = job.allocated.copy()

        for job in list(self.queue):
            if job not in initial_allocations:
                continue

            current_allocation = initial_allocations[job]

            # Check if job has enough GPUs from TTC
            if len(current_allocation) >= job.demand:
                # TTC gave us a complete gang - start the job
                # IMPORTANT: Sort for deterministic behavior (sets have random order)
                gpus_to_use = sorted(current_allocation)[:job.demand]
                job.allocated = set(gpus_to_use)

                job.start = sim.time
                sim.running.append(job)
                sim.schedule_event(sim.time + job.actual_time, "finish", job)
                jobs_to_remove.append(job)

                new_starts += 1
                resource_type = "edge" if all(r.startswith("edge") for r in gpus_to_use) else \
                    "cloud" if all(r.startswith("cloud") for r in gpus_to_use) else "mixed"
                sim.log(f"Job {job.jid} STARTED (ttcc-gang, qos={job.qos}, demand={job.demand}, resources={resource_type})")

            # If job has partial allocation, try to complete it greedily
            elif len(current_allocation) < job.demand:
                needed = job.demand - len(current_allocation)

                # Build set of truly free GPUs using INITIAL allocations snapshot
                truly_free = set(sim.cluster.free)
                for other_job, other_allocation in initial_allocations.items():
                    # Skip jobs we've already started in this iteration
                    if other_job != job and other_job not in jobs_to_remove:
                        truly_free -= other_allocation

                if len(truly_free) >= needed:
                    # Complete the gang - prioritize TTC's preference-based partial allocation
                    # IMPORTANT: Sort for deterministic behavior
                    additional_gpus = sorted(truly_free)[:needed]
                    job.allocated = current_allocation | set(additional_gpus)

                    # Remove from free pool
                    for gpu in additional_gpus:
                        sim.cluster.free.discard(gpu)

                    job.start = sim.time
                    sim.running.append(job)
                    sim.schedule_event(sim.time + job.actual_time, "finish", job)
                    jobs_to_remove.append(job)

                    new_starts += 1
                    resource_type = "edge" if all(r.startswith("edge") for r in job.allocated) else \
                        "cloud" if all(r.startswith("cloud") for r in job.allocated) else "mixed"
                    sim.log(
                        f"Job {job.jid} STARTED (ttcc-partial+greedy, qos={job.qos}, demand={job.demand}, resources={resource_type})")
                else:
                    # Can't complete gang yet - keep partial allocation for next iteration
                    # Don't release it - TTC worked hard to get these preferred GPUs
                    pass

        # Remove all started jobs from queue at once
        for job in jobs_to_remove:
            self.queue.remove(job)

        return new_starts > 0

    def _easy_backfill(self, sim):
        """TTCC-aware backfilling with preference-based job reordering.

        Unlike pure EASY, TTCC can reorder jobs to maximize optimal allocations.
        Skip head job if it would get suboptimal allocation while a later job
        could get optimal allocation instead.
        """
        if not self.queue:
            return

        # Compute reservation time for head job
        head_job = self.queue[0]
        reservation_time = self._compute_reservation(sim, head_job)

        # Track whether we should skip head job
        should_skip_head = False

        # Check if head job can start
        if head_job.demand <= len(sim.cluster.free):
            # Count available resources by type
            edge_free = len([r for r in sim.cluster.free if r.startswith("edge")])
            cloud_free = len([r for r in sim.cluster.free if r.startswith("cloud")])

            # Check if head job would get FULLY optimal allocation
            head_gets_optimal = False
            if head_job.qos == "Gold":
                head_gets_optimal = edge_free >= head_job.demand  # All edge
            else:  # Silver/Bronze prefer cloud
                head_gets_optimal = cloud_free >= head_job.demand  # All cloud

            # If head would get suboptimal (mixed) allocation, check if we should skip it
            if not head_gets_optimal and len(self.queue) > 1:
                # Look for a later job that WOULD get fully optimal allocation
                for later_job in list(self.queue)[1:]:
                    if later_job.demand > len(sim.cluster.free):
                        continue  # Can't fit

                    later_gets_optimal = False
                    if later_job.qos == "Gold":
                        later_gets_optimal = edge_free >= later_job.demand
                    else:
                        later_gets_optimal = cloud_free >= later_job.demand

                    # Skip head and START the optimal job immediately
                    if later_gets_optimal:
                        sim.log(f"TTCC: Skipping head {head_job.jid} ({head_job.qos}, d={head_job.demand}, would get mixed) "
                                f"to start {later_job.jid} ({later_job.qos}, d={later_job.demand}, gets optimal)")

                        # Start the later job with optimal allocation
                        alloc = sim.cluster.allocate(later_job, later_job.demand)
                        if alloc:
                            later_job.start = sim.time
                            sim.running.append(later_job)
                            sim.schedule_event(sim.time + later_job.actual_time, "finish", later_job)
                            self.queue.remove(later_job)
                            sim.log(
                                f"Job {later_job.jid} STARTED (ttcc-optimal, qos={later_job.qos}, demand={later_job.demand}, alloc={list(alloc)})")
                            # Continue with backfill for remaining jobs
                            should_skip_head = True
                            break

            # Start head job if not skipping
            if not should_skip_head:
                alloc = sim.cluster.allocate(head_job, head_job.demand)
                if alloc:
                    head_job.start = sim.time
                    sim.running.append(head_job)
                    sim.schedule_event(sim.time + head_job.actual_time, "finish", head_job)
                    self.queue.popleft()
                    sim.log(
                        f"Job {head_job.jid} STARTED (easy-head, qos={head_job.qos}, demand={head_job.demand}, alloc={list(alloc)})")
                    return

        # Try to backfill jobs behind head
        jobs_to_remove = []

        for job in list(self.queue)[1:]:  # Skip head job
            # Check if backfill is safe
            backfill_finishes = sim.time + job.actual_time

            if backfill_finishes <= reservation_time:
                # Backfill finishes before reservation - always safe
                can_backfill = True
            else:
                # Backfill overlaps with reservation - check if head can still fit
                gpus_from_backfill = job.demand
                gpus_from_running = sum(rj.demand for rj in sim.running
                                        if hasattr(rj, 'start') and rj.start is not None
                                        and rj.start + rj.actual_time > reservation_time)
                total_used = gpus_from_backfill + gpus_from_running
                free_at_reservation = (sim.cluster.num_edge + sim.cluster.num_cloud) - total_used
                can_backfill = free_at_reservation >= head_job.demand

            if can_backfill:
                alloc = sim.cluster.allocate(job, job.demand)
                if alloc:
                    job.start = sim.time
                    sim.running.append(job)
                    sim.schedule_event(sim.time + job.actual_time, "finish", job)
                    jobs_to_remove.append(job)
                    sim.log(f"Job {job.jid} STARTED (easy-backfill, qos={job.qos}, demand={job.demand}, alloc={list(alloc)})")

        # Remove backfilled jobs from queue
        for job in jobs_to_remove:
            self.queue.remove(job)

    def _compute_reservation(self, sim, job):
        """Compute earliest time the reserved job could start."""
        free_now = len(sim.cluster.free)
        if job.demand <= free_now:
            return sim.time

        # Need to wait for running jobs to finish
        finish_times = []
        for running_job in sim.running:
            if hasattr(running_job, 'start') and running_job.start is not None:
                job_finish_time = running_job.start + running_job.actual_time
                for _ in range(running_job.demand):
                    finish_times.append(job_finish_time)

        finish_times.sort()
        need = job.demand - free_now
        if len(finish_times) >= need:
            return finish_times[need - 1]
        return float("inf")

    def _best_gpu_for_job(self, job, all_gpus):
        """Return the most-preferred GPU for a job among all available GPUs.

        For running jobs (with allocated GPUs), returns the most preferred GPU
        that is BETTER than any currently allocated GPU. This enables migrations.
        For waiting jobs, returns the most preferred available GPU.
        """
        edge = [r for r in all_gpus if r.startswith("edge")]
        cloud = [r for r in all_gpus if r.startswith("cloud")]

        # Preference order based on QoS
        if job.qos == "Gold":
            candidates = edge + cloud  # Gold prefers edge > cloud
        else:
            candidates = cloud + edge  # Silver/Bronze prefer cloud > edge

        if not candidates:
            return None

        # For running jobs with allocations, only point to GPUs that are better
        if hasattr(job, 'allocated') and job.allocated:
            # For Gold: edge is preferred, cloud is non-preferred
            # For Silver/Bronze: cloud is preferred, edge is non-preferred
            preferred_type = "edge" if job.qos == "Gold" else "cloud"
            non_preferred_type = "cloud" if job.qos == "Gold" else "edge"

            # Check if job has ANY non-preferred GPUs
            has_non_preferred = any(
                gpu.startswith(non_preferred_type) for gpu in job.allocated
            )

            # If job has non-preferred GPUs, point to a better GPU of preferred type
            # IMPORTANT: Must exclude GPUs already allocated to this job!
            if has_non_preferred:
                better_candidates = [
                    g for g in candidates
                    if g.startswith(preferred_type) and g not in job.allocated
                ]
                if better_candidates:
                    return better_candidates[0]

            # Job is fully allocated on preferred type, point to own GPU (self-loop)
            # Use min() for deterministic selection
            return min(job.allocated)

        # For waiting jobs, just return most preferred
        return candidates[0]

    def _best_job_for_gpu(self, gpu, all_jobs):
        """Return the most-preferred job for a GPU among all jobs."""
        is_edge = gpu.startswith("edge")

        def job_priority(job):
            if is_edge:
                # Edge prefers Gold > Silver > Bronze
                qos_priority = {"Gold": 0, "Silver": 1, "Bronze": 2}[job.qos]
            else:
                # Cloud prefers Silver/Bronze > Gold
                qos_priority = 0 if job.qos in ("Silver", "Bronze") else 1
            # Tie-break by arrival time (FCFS among same QoS)
            return (qos_priority, job.arrival)

        return min(all_jobs, key=job_priority) if all_jobs else None

    def _find_ttc_cycles(self, job_to_gpu, gpu_to_job):
        """Find cycles in the bipartite job↔GPU matching graph."""
        cycles = []
        visited_globally = set()

        # Start from each job that hasn't been visited
        for start_job in job_to_gpu:
            if start_job in visited_globally:
                continue

            path = []
            node = start_job
            visited_in_path = set()

            # Follow pointers: job → gpu → job → gpu → ...
            while node not in visited_in_path:
                visited_in_path.add(node)
                path.append(node)

                if isinstance(node, str):  # It's a GPU
                    node = gpu_to_job.get(node)
                    if node is None:
                        break
                else:  # It's a job
                    node = job_to_gpu.get(node)
                    if node is None:
                        break

            # Check if we found a cycle
            if node in visited_in_path:
                cycle_start_idx = path.index(node)
                cycle = path[cycle_start_idx:]
                cycles.append(cycle)
                visited_globally.update(cycle)

        return cycles
