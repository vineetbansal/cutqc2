from __future__ import annotations
import pickle
from dataclasses import dataclass
from typing import Optional, Dict

from cutqc import post_process_helper
from cutqc.helper_functions import conversions, non_ibmq_functions, metrics

import numpy as np
import os

import qiskit


@dataclass
class CutQCModel:
    """Stores subcircuit outputs prior to reconstruction

    Sets up to call the distributed kernel. Worker nodes

    Args:
        comm_backend: message passing backend internally used by pytorch for
                    sending data between nodes
        world_rank:   Global Identifier
        world_size:   Total number of nodes
        timeout:      Max amount of time pytorch will let any one node wait on
                    a message before killing it.
    """

    compute_graph: post_process_helper.ComputeGraph
    complete_path_map: Dict
    attributed_shots: Dict[tuple, np.ndarray]
    entry_init_meas_ids: Dict[int, Dict[tuple, int]]
    num_cuts: int
    subcircuits: list
    circuit: qiskit.QuantumCircuit
    _is_reconstructed = False
    approximation_bins = None

    def save_cutqc_model(self, filename: Optional[str] = None):
        # Its important to use binary mode
        dbfile = open(filename, "wb")

        # source, destination
        pickle.dump(self, dbfile)

        dbfile.close()

    def reconstructed_probability(self):
        if not self._is_reconstructed:
            raise ValueError(
                "Circuit model must be in 'reconstructed' state before attempting to access reconstructed probability."
            )

        subcircuit_out_qubits = post_process_helper.get_reconstruction_qubit_order(
            full_circuit=self.full_circuit,
            complete_path_map=self.complete_path_map,
            subcircuits=self.subcircuits,
        )

        return post_process_helper.read_dd_bins(
            subcircuit_out_qubits=subcircuit_out_qubits, dd_bins=self.approximation_bins
        )

    def verify(self):
        # verify_begin = perf_counter()
        if not self._is_reconstructed:
            raise ValueError(
                "Circuit model must be in 'reconstructed' state when calling verify."
            )

        reconstructed_prob, approximation_error = full_verify(
            full_circuit=self.circuit,
            complete_path_map=self.complete_path_map,
            subcircuits=self.subcircuits,
            dd_bins=self.approximation_bins,
        )

        print(f"Approximate Error: {approximation_error}")
        # print("verify took %.3f" % (perf_counter() - verify_begin))

        return approximation_error

    @classmethod
    def load_cutqc_model(cls, filename: str) -> CutQCModel:
        # Distributed execution initializes data on a single node
        if os.environ["PYTORCH"] == "TRUE" and os.environ["HOST"] == "False":
            return cls(None, None, None, None)

        with open(filename, "rb") as dbfile:
            return pickle.load(dbfile)


def full_verify(full_circuit, complete_path_map, subcircuits, dd_bins):
    ground_truth = non_ibmq_functions.evaluate_circ(
        circuit=full_circuit, backend="statevector_simulator"
    )
    subcircuit_out_qubits = post_process_helper.get_reconstruction_qubit_order(
        full_circuit=full_circuit,
        complete_path_map=complete_path_map,
        subcircuits=subcircuits,
    )
    reconstructed_prob = post_process_helper.read_dd_bins(
        subcircuit_out_qubits=subcircuit_out_qubits, dd_bins=dd_bins
    )
    real_probability = conversions.quasi_to_real(
        quasiprobability=reconstructed_prob, mode="nearest"
    )
    # print (f"MSE: {MSE(target=ground_truth, obs=real_probability)}")
    # print ("real_probability: {}".format (real_probability))
    # print ("real_probability.shape: {}".format (real_probability.shape))
    # print ("ground_truth: {}".format (ground_truth))
    # print ("ground_truth.shape: {}".format (ground_truth.shape))

    approximation_error = (
        metrics.MSE(target=ground_truth, obs=real_probability)
        * 2**full_circuit.num_qubits
        / np.linalg.norm(ground_truth) ** 2
    )

    # print (f"Reconstructed Error: {reconstructed_prob}")
    # print (f"Real Error: {real_probability}")

    return reconstructed_prob, approximation_error
