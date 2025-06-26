"""
Title: single_node_cpu_load_example.py
Description: Example of loading a cutqc_model file using a single node. 
Notice how cutqc distributed is not initialized
"""

import os
from cutqc import CircuitReconstructor
from cutqc.cutqc_model import CutQCModel

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


if __name__ == "__main__":
    # Load CutQC Instance from Pickle
    print("--- Running ---")
    filename = "adder.cutqc_model"
    cutqc_model = CutQCModel.load_cutqc_model(filename)

    # Initiate Reconstruct
    cqc = CircuitReconstructor(cutqc_model, 32, 1)
    compute_time = cqc.build()

    approximation_error = cutqc_model.verify()
