from typing import Optional
from datetime import timedelta
import os
import torch
import torch.distributed as dist
from enum import Enum


class Protocol(Enum):
    """Communication Protocol used
    See following for explination on communication backends: https://docs.pytorch.org/docs/stable/distributed.html
    """

    NCCL = "nccl"
    GLOO = "mpi"
    MPI = "gloo"


class Device(Enum):
    """Computation backend used"""

    GPU = "GPU"
    CPU = "CPU"


class CutQCDistributed:
    _instance = None

    def __init__(
        self,
        computation_device: Device = Device.GPU,
        communication_protocol: Protocol = Protocol.NCCL,
        world_rank: int = 0,
        world_size: int = 1,
        gpus_per_node: int = 1,
        timeout: Optional[int] = 600,
    ):
        """
        Initialize the distributed context.

        Args:
            computation_device: Device to run computations on
            communication_protocol: message passing backend internally used by pytorch for
                        sending data between nodes.
            world_rank:   Global Identifier.
            world_size:   Total number of nodes.
            gpus_per_node: Number of GPUs per node.
            timeout:      Max amount of time pytorch will let any one node wait on
                        a message before killing it.
        """
        self.computation_device = computation_device
        self.communication_protocol = communication_protocol
        self.world_rank = world_rank
        self.world_size = world_size
        self.gpus_per_node = gpus_per_node
        self.timeout = timeout

    @classmethod
    def initialize(
        cls,
        computation_device: Device,
        communication_protocol: Protocol,
        world_rank: int,
        world_size: int,
        gpus_per_node: int,
        timeout: Optional[int] = 600,
    ) -> None:
        """
        Sets up to call the distributed kernel. Worker nodes

        Args:
            computation_device: Device to run computations on
            communication_protocol: message passing backend internally used by pytorch for
                        sending data between nodes.
            world_rank:   Global Identifier.
            world_size:   Total number of nodes.
            gpus_per_node: Number of GPUs per node.
            timeout:      Max amount of time pytorch will let any one node wait on
                        a message before killing it.
        """
        instance = cls(
            computation_device,
            communication_protocol,
            world_rank,
            world_size,
            gpus_per_node,
            timeout,
        )
        instance._setup()
        cls._instance = instance

    @classmethod
    def terminate(cls):
        """
        Sends signal to workers to finish their execution.
        """
        if cls._instance:
            cls._instance._cleanup()
            cls._instance = None

    def __enter__(self):
        """
        Called when entering the 'with' block.
        Sets up the distributed environment.
        """
        self._setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Called when exiting the 'with' block.
        Cleans up the distributed environment.
        """
        self._cleanup()

    def _setup(self) -> None:
        """Internal setup method"""
        # Start the group
        print("Hello world")

        local_rank = self.world_rank - self.gpus_per_node * (
            self.world_rank // self.gpus_per_node
        )  # GPU identifier on local compute cluster

        print("My local rank: {}".format(local_rank))
        print("My global rank: {}".format(self.world_rank))

        timelimit = timedelta(
            seconds=self.timeout
        )  # Bounded wait time to prevent deadlock

        # Initializes pytorch for multiprocessing

        dist.init_process_group(
            backend=self.communication_protocol.value,
            rank=self.world_rank,
            world_size=self.world_size,
            timeout=timelimit,
        )

        # Referenced in distributed_graph_contractor
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["PYTORCH"] = "True"
        os.environ["COMPUTATION_DEVICE"] = self.computation_device.value
        os.environ["COMMUNICATION_PROTOCOL"] = self.communication_protocol.value

        if self.world_rank == 0:
            os.environ["HOST"] = "True"

    def _cleanup(self) -> None:
        """Send termination signal to workers via handshake """
        if os.environ["HOST"] == "True" and dist.is_initialized():
            print("Host: Exiting Now!")

            # Send workers termination signal
            termination_signal = torch.tensor([-1], dtype=torch.int64)
            for rank in range(1, dist.get_world_size()):
                dist.send(termination_signal, dst=rank)

            # Waits for workers to respond
            output_buff = torch.zeros(1, dtype=torch.float32)
            dist.reduce(output_buff, dst=0, op=dist.ReduceOp.SUM)
            dist.destroy_process_group()
