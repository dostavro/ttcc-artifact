"""
GPU Cluster model with heterogeneous edge and cloud resources.

A cluster consists of:
- Edge GPUs: Low-latency, limited capacity
- Cloud GPUs: High-capacity, higher latency

The cluster tracks resource allocation and provides methods for:
- Allocating GPUs to jobs (QoS-aware preference satisfaction)
- Releasing resources when jobs complete
- Querying resource availability
"""


class Cluster:
    def __init__(self, num_edge, num_cloud):
        self.resources = ["edge_{}".format(i) for i in range(num_edge)] + \
                         ["cloud_{}".format(i) for i in range(num_cloud)]
        self.free = set(self.resources)
        self.num_edge = num_edge
        self.num_cloud = num_cloud

    def allocate(self, job, num):
        """
        Smart allocation that considers job QoS preferences:
        - Gold jobs prefer edge GPUs (low latency for deadline-sensitive work)
        - Silver/Bronze jobs prefer cloud GPUs to leave edge resources for Gold
        - Falls back gracefully if preferred resources unavailable
        """
        if len(self.free) < num:
            return None

        # Get available edge and cloud GPUs
        edge_free = [r for r in self.free if r.startswith("edge")]
        cloud_free = [r for r in self.free if r.startswith("cloud")]

        chosen = set()

        if job.qos == "Gold":
            # Gold jobs prefer edge GPUs first, then cloud if needed
            while len(chosen) < num and edge_free:
                chosen.add(edge_free.pop())
            while len(chosen) < num and cloud_free:
                chosen.add(cloud_free.pop())
        else:
            # Silver/Bronze jobs prefer cloud GPUs first, then edge if needed
            while len(chosen) < num and cloud_free:
                chosen.add(cloud_free.pop())
            while len(chosen) < num and edge_free:
                chosen.add(edge_free.pop())

        if len(chosen) == num:
            self.free -= chosen
            job.allocated = chosen
            return chosen
        else:
            return None

    def release(self, job):
        if job.allocated:
            # Store final allocation for utility calculations before clearing
            job.final_allocated = job.allocated.copy()
            self.free |= job.allocated
            job.allocated = None
