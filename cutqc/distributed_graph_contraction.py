"""
File: distributed_graph_contraction.py
Original Author: Wei Tang (tangwei13579@gmail.com)
Current Version Author: Charles "Chuck" Garcia (chuckgarcian@utexas.edu)
Description: Distributed implementation of Wei Tang's original TensorFlow CutQC implementation.
"""

import itertools
from typing import List
import numpy as np
import torch
import torch.distributed as dist
from functools import reduce

from cutqc.abstract_graph_contractor import AbstractGraphContractor
from cutqc import distributed_helper

__host_machine__ = 0


class DistributedGraphContractor(AbstractGraphContractor):
    """
    Distributed Graph Contractor Implementation

    Args:
           local_rank (int): Node identifier value
           compute_backend (str): Device used for compute (Default is GPU)

    """

    def __init__(
        self, local_rank: int, compute_backend: distributed_helper.Device
    ) -> None:
        print("Compute Backend: {}".format(compute_backend))
        self.local_rank = local_rank

        # Sets GPU-CUDA id if using GPU
        if compute_backend == distributed_helper.Device.GPU:
            self.compute_backend = torch.device(f"cuda:{local_rank}")
        else:
            self.compute_backend = distributed_helper.Device.CPU

        print("Format: {}".format(self.compute_backend))

        self.is_gpu = compute_backend == distributed_helper.Device.GPU

        print(
            "Worker {}, compute_backend: {}".format(
                dist.get_rank(), self.compute_backend
            ),
            flush=True,
        )

        self.times = {"compute": 0}
        self.compute_graph = None
        self.subcircuit_entry_probs = None
        self.reconstructed_prob = None
        self.worker_execution_state = True

        if dist.get_rank() != __host_machine__:
            self._initiate_worker_loop()

        # dist.barrier()
        # dist.destroy_process_group()

    def _get_paulibase_probability(self, edge_bases: tuple, edges: list):
        """
        Returns probability contribution for the basis 'edge_bases' in the circuit
        cutting decomposition.
        """
        with torch.no_grad():
            self.compute_graph.assign_bases_to_edges(edge_bases=edge_bases, edges=edges)

            # Create list of kronecker product terms
            flat_size = np.sum(self.subcircuit_entry_lengths)
            flat = torch.empty(flat_size)
            idx = 0

            # Store all probability tensors into single flattened tensor
            for size, subcircuit_idx in zip(
                self.subcircuit_entry_lengths, self.smart_order
            ):
                subcircuit_entry_prob = self._get_subcircuit_entry_prob(subcircuit_idx)
                flat[idx : idx + size] = torch.tensor(
                    subcircuit_entry_prob, dtype=torch.float32
                )
                idx += size

        return flat

    def _send_distributed(
        self, dataset: List[torch.Tensor], num_batches: int
    ) -> torch.Tensor:
        """
        Decomposes `dataset` list into 'num_batches' number of batches and distributes
        to worker processes.
        """
        torch.set_default_device(self.compute_backend)

        with torch.no_grad():
            print("LEN(DATASET): {}".format(len(dataset)), flush=True)
            print("NUMBER BATCHES: {}".format(num_batches), flush=True)
            if len(dataset) < num_batches:
                raise ValueError(
                    "Error 2000: Invalid number of requested batches -- Too many nodes allocated, for dataset length {} and {} number of batches".format(
                        len(dataset), num_batches
                    )
                )

            batches = torch.stack(dataset).tensor_split(num_batches)
            tensor_sizes = torch.tensor(
                self.subcircuit_entry_lengths, dtype=torch.int64
            )
            tensor_sizes_shape = torch.tensor(tensor_sizes.shape, dtype=torch.int64)

            if dist.get_backend() == "gloo":
                op_list = []
                # List of sending objects
                for dst, batch in enumerate(batches, start=1):
                    op_list.extend(
                        [
                            dist.P2POp(dist.isend, tensor_sizes_shape, dst),
                            dist.P2POp(dist.isend, tensor_sizes, dst),
                            dist.P2POp(
                                dist.isend,
                                torch.tensor(batch.shape, dtype=torch.int64),
                                dst,
                            ),
                            dist.P2POp(dist.isend, batch, dst),
                        ]
                    )
            else:
                # NCCL backend
                for dst_rank, batch in enumerate(batches, start=1):
                    # Non-Blocking send on NCCL
                    dist.isend(tensor_sizes_shape, dst=dst_rank)
                    dist.isend(tensor_sizes, dst=dst_rank)
                    dist.isend(torch.tensor(batch.shape), dst=dst_rank)
                    dist.isend(batch.to(self.compute_backend), dst=dst_rank)

            # Receive Results
            output_buff = torch.zeros(self.result_size, dtype=torch.float32)
            dist.reduce(output_buff, dst=0, op=dist.ReduceOp.SUM)

        return torch.mul(output_buff, (1 / 2**self.num_cuts))

    def _compute(self) -> np.ndarray:
        """
        Performs distributed graph contraction. Returns the reconstructed probability.
        """
        edges = self.compute_graph.get_edges(from_node=None, to_node=None)
        summation_terms_sequence = []

        # Assemble sequence of uncomputed kronecker products
        for edge_bases in itertools.product(["I", "X", "Y", "Z"], repeat=len(edges)):
            summation_terms = self._get_paulibase_probability(edge_bases, edges)
            summation_terms_sequence.append(summation_terms)

        self.compute_graph.remove_bases_from_edges(edges=self.compute_graph.edges)

        # Distribute and Execute reconstruction on nodes
        num_batches = dist.get_world_size() - 1  # No batch for host
        reconstructed_prob = self._send_distributed(
            summation_terms_sequence, num_batches
        )

        return reconstructed_prob.cpu().numpy()

    def _receive_from_host(self):
        """
        Receives tensors sent by host. Returns batch and unpadded sizes.
        """
        torch.set_default_device(self.compute_backend)
        torch.cuda.device(self.compute_backend)

        with torch.no_grad():
            tensor_sizes_shape = torch.empty([1], dtype=torch.int64)
            dist.recv(tensor=tensor_sizes_shape, src=0)

            # Check for termination signal and handle it
            self.termination_handler(tensor_sizes_shape)

            # Used to unflatten
            tensor_sizes = torch.empty(tensor_sizes_shape, dtype=torch.int64)
            dist.recv(tensor=tensor_sizes, src=0)

            # Get shape of the batch we are receiving
            batch_shape = torch.empty([2], dtype=torch.int64)
            dist.recv(tensor=batch_shape, src=0)

            # Create an empty batch tensor and receive its data
            batch = torch.empty(tuple(batch_shape), dtype=torch.float32)
            dist.recv(tensor=batch, src=0)

        return batch_shape[0], batch, tensor_sizes

    def _initiate_worker_loop(self):
        """
        Primary worker loop.

        Each worker receives a portion of the workload from the host/master node.
        Once done with computation, all nodes perform a collective reduction
        operation back to the host. Synchronization among nodes is provided via
        barriers and blocked message passing.
        """
        torch.cuda.device(self.compute_backend)
        num_batches, batch, tensor_sizes = self._receive_from_host()

        # Executes until host sends termination signal (An empty tensor)
        while self.worker_execution_state:
            # Ensure Enough Size
            gpu_free = torch.cuda.mem_get_info()[0]
            batch_mem_size = (
                batch.element_size() * torch.prod(tensor_sizes) * num_batches
            )
            assert batch_mem_size < gpu_free, ValueError(
                "Batch of size {}, to large for GPU device of size {}".format(
                    batch_mem_size, gpu_free
                )
            )

            # Execute kronecker products in parallel (vectorization)
            torch.cuda.memory._record_memory_history()

            def lambda_fn(x):
                return compute_kronecker_product(x, tensor_sizes)

            vec_fn = torch.func.vmap(lambda_fn)
            res = vec_fn(batch)
            torch.cuda.memory._dump_snapshot("compute_snap.pickle")

            del batch
            res = res.sum(dim=0)
            res = res

            # Send Back to host
            dist.reduce(
                res.to(self.compute_backend), dst=__host_machine__, op=dist.ReduceOp.SUM
            )

            # Next iteration data
            num_batches, batch, tensor_sizes = self._receive_from_host()

    def termination_handler(self, tensor_sig):
        """Checks the recieved tensor for an agreed upon termination value (-1)"""
        if tensor_sig.item() == -1:
            print(f"WORKER {dist.get_rank()} DYING", flush=True)
            self.worker_execution_state = False

            host_sig = torch.zeros(1, dtype=torch.float32)
            dist.reduce(
                host_sig.to(self.compute_backend),
                dst=__host_machine__,
                op=dist.ReduceOp.SUM,
            )

            dist.destroy_process_group()
            exit()


def compute_kronecker_product(
    flattened: torch.Tensor, sizes: torch.Tensor
) -> torch.Tensor:
    """
    Computes sequence of Kronecker products, where operands are tensors in 'components'.
    """
    tensors = torch.split(flattened, tuple(sizes))
    return reduce(torch.kron, tensors)
