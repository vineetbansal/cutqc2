"""
Title: reconstruction_example.py
Description: Example of how CutQC can be used to efficiently reconstruct subcircuits using GPUS
"""

import os
import cutqc
from cutqc import CircuitReconstructor
from cutqc.cutqc_model import CutQCModel
from cutqc.distributed_helper import CutQCDistributed


# Environment variables set by slurm script
GPUS_PER_NODE = int(os.environ["SLURM_GPUS_ON_NODE"])
WORLD_RANK = int(os.environ["SLURM_PROCID"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])

if __name__ == "__main__":
    model_path = "adder.cutqc_model"
    compute_device = cutqc.Device.GPU
    communication_protocol = cutqc.Protocol.NCCL

    CutQCDistributed.initialize(
        compute_device, communication_protocol, WORLD_RANK, WORLD_SIZE, GPUS_PER_NODE
    )

    # Load CutQC Instance from Pickle
    print(f"--- Running {model_path} ---")
    cutqc_model = CutQCModel.load_cutqc_model(model_path)

    # Initiate Reconstruct
    reconstructor = CircuitReconstructor(cutqc_model, mem_limit=32, recursion_depth=1)
    compute_time = reconstructor.build()
    print("Done!")
    cutqc_model.verify()

    CutQCDistributed.terminate()
