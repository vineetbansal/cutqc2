from cutqc.cutqc_model import CutQCModel
from cutqc.dynamic_definition import DynamicDefinition
from cutqc.distributed_graph_contraction import DistributedGraphContractor
from cutqc.graph_contraction import GraphContractor
from typing import Optional

from cutqc import distributed_helper
import os


class CircuitReconstructor:
    """
    The main module for CutQC
    cut --> evaluate results --> verify (optional)
    """

    def __init__(
        self,
        cutqc_model: CutQCModel,
        mem_limit: int,
        recursion_depth: int,
        verbose: Optional[bool] = False,
    ):
        """
        --- Distributed Reconstruction Related Arguments ---

        cutqc_model: `CutQCModel` containing subcircuitt output vectors

        """
        self.mem_limit = mem_limit
        self.recursion_depth = recursion_depth
        self.times = {}
        self.cutqc_model = cutqc_model
        self.verbose = verbose

        self.pytorch_distributed = False
        self.local_rank = None
        self.compute_backend = None

        if os.environ["PYTORCH"] == "True":
            self.pytorch_distributed = True

        self._init_graph_contractor()

    def _init_graph_contractor(self):
        """Sets the graph contractor depenedening on passed arguments"""
        # if (os.environ["HOST"]=="True"): return 0

        # Setup distributed environment variables and initializes workers into a loop
        if self.pytorch_distributed:
            print("Local Rank: {}".format(int(os.environ["LOCAL_RANK"])))
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.compute_backend = distributed_helper.Device(
                os.environ["COMPUTATION_DEVICE"]
            )

            print("Computeational backend: {}".format(self.compute_backend))

            self.graph_contractor = DistributedGraphContractor(
                local_rank=self.local_rank, compute_backend=self.compute_backend
            )

        else:
            self.graph_contractor = GraphContractor()

    def build(self):
        """
        mem_limit: memory limit during post process. 2^mem_limit is the largest vector
        """

        if self.verbose:
            print("--> Build %s" % (self.name))

        self.dd = DynamicDefinition(
            cutqc_model=self.cutqc_model,
            mem_limit=self.mem_limit,
            recursion_depth=self.recursion_depth,
            graph_contractor=self.graph_contractor,
        )
        self.dd.build()

        self.cutqc_model.approximation_bins = self.dd.dd_bins
        self.cutqc_model._is_reconstructed = True
        self.num_recursions = len(self.dd.dd_bins)
        self.overhead = self.dd.overhead

        # self.times["build"] = perf_counter() - build_begin
        # self.times["build"] += self.times["cutter"]
        # self.times["build"] -= self.times["merge_states_into_bins"]

        if self.verbose:
            print("Overhead = {}".format(self.overhead))

        return self.dd.graph_contractor.times["compute"]
