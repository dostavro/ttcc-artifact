class Job:
    def __init__(self, jid, arrival, demand, reserved_time, actual_time, qos):
        self.jid = jid
        self.arrival = arrival
        self.demand = demand
        self.reserved_time = reserved_time
        self.actual_time = actual_time  # NEVER modify - original required work time (for metrics)
        self.work_remaining = actual_time  # Track work remaining for preemption (gets modified)
        self.qos = qos
        # self.deadline = arrival + (1.0 if qos == "Gold" else 1.5) * actual_time
        self.start = None  # First time job starts (preserved across preemptions for metrics)
        self.last_resume_time = None  # Track most recent resume/start time (for preemption work calculation)
        self.finish = None
        self.allocated = None
        self.final_allocated = None  # Track final allocation for utility calculations
        self.finish_event_version = 0  # Track which finish event is valid (for preemption)

    def _get_resource_type(self, resource):
        """Extract resource type from resource identifier."""
        return "edge" if resource.startswith("edge") else "cloud"

    def get_preferred_resource_type(self):
        """Return preferred resource type based on QoS class."""
        return "edge" if self.qos == "Gold" else "cloud"

    def get_resource_rank(self, resource):
        """
        Get preference rank for a specific resource.

        Args:
            resource: Resource identifier (e.g., 'edge_0', 'cloud_1')

        Returns:
            1 for preferred resource type, 2 for non-preferred resource type
        """
        preferred_type = self.get_preferred_resource_type()
        resource_type = self._get_resource_type(resource)
        return 1 if resource_type == preferred_type else 2

    def base_utility(self):
        """
        Calculate base (deadline-aware) utility based on QoS class and tardiness.

        U_base(j) = w_{q_j} * exp(-T_j / beta_{q_j})
        where T_j = max(0, c_j - D_j) is tardiness, and:
          - w_Gold=3.0,    beta_Gold=1.0   (high priority, tight deadline penalty)
          - w_Silver=2.0, beta_Silver=1.5 (medium priority)
          - w_Bronze=1.0,  beta_Bronze=3.0 (low priority, loose deadline penalty)

        Returns:
            Base utility value in [0, w_{q_j}], accounting for deadline adherence
        """
        if self.finish is None:
            return 0.0

        # QoS parameters: (weight, decay_scale)
        qos_params = {
            "Gold": (3.0, 1.0),
            "Silver": (2.0, 1.5),
            "Bronze": (1.0, 3.0),
        }

        weight, decay_scale = qos_params.get(self.qos, (0.0, 1.0))

        # Tardiness: how late the job finished relative to deadline
        tardiness = max(0.0, self.finish - self.deadline)

        # Exponential decay: full weight if on-time, decays with lateness
        import math
        base_util = weight * math.exp(-tardiness / decay_scale)

        return base_util

    def _get_current_allocation(self):
        """Get the current resource allocation (final if available, otherwise current)."""
        return self.final_allocated if self.final_allocated is not None else self.allocated

    def satisfaction_multiplier(self, gamma=0.9):
        """
        Calculate satisfaction multiplier based on allocated resource set.

        Computes S_j(A(j)) = (1/d_j) * Σ(γ^(r_j(r_j,k) - 1)), where:
        - A(j) is the set of allocated resources
        - r_j(r_j,k) is the preference rank of the k-th allocated resource
        - γ is the satisfaction parameter for non-preferred resources

        Args:
            gamma: Satisfaction parameter for non-preferred resources (default 0.9)

        Returns:
            Average satisfaction multiplier across all allocated resources
        """
        allocation = self._get_current_allocation()

        if not allocation:
            return 0.0

        # Calculate average satisfaction across all allocated resources
        total_satisfaction = sum(gamma ** (self.get_resource_rank(resource) - 1)
                                 for resource in allocation)

        return total_satisfaction / len(allocation)

    def utility(self, gamma=0.9):
        """
        Calculate preference-aware utility accounting for resource allocation satisfaction.

        The preference-aware utility is computed as U_pref(j) = U(j) * S_j(A(j)), where:
        - U(j) is the base utility based on QoS class and deadline satisfaction
        - S_j(A(j)) is the satisfaction multiplier based on the set of allocated resources

        Args:
            gamma: Satisfaction parameter for non-preferred resource allocation (default 0.9)

        Returns:
            Preference-aware utility value
        """
        return self.base_utility() * self.satisfaction_multiplier(gamma)
