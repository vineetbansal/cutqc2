import itertools, copy, math
from cutqc2.cutqc.cutqc.compute_graph import ComputeGraph


def get_cut_qubit_pairs(complete_path_map):
    """
    Get O-Rho cut qubit pairs
    """
    O_rho_pairs = []
    for input_qubit in complete_path_map:
        path = complete_path_map[input_qubit]
        if len(path) > 1:
            for path_ctr, item in enumerate(path[:-1]):
                O_qubit_tuple = item
                rho_qubit_tuple = path[path_ctr + 1]
                O_rho_pairs.append((O_qubit_tuple, rho_qubit_tuple))
    return O_rho_pairs


def get_reconstruction_qubit_order(full_circuit, complete_path_map, subcircuits):
    """
    Get the output qubit in the full circuit for each subcircuit
    Qiskit orders the full circuit output in descending order of qubits
    """
    subcircuit_out_qubits = {
        subcircuit_idx: [] for subcircuit_idx in range(len(subcircuits))
    }
    for input_qubit in complete_path_map:
        path = complete_path_map[input_qubit]
        output_qubit = path[-1]
        subcircuit_out_qubits[output_qubit["subcircuit_idx"]].append(
            (output_qubit["subcircuit_qubit"], full_circuit.qubits.index(input_qubit))
        )
    for subcircuit_idx in subcircuit_out_qubits:
        subcircuit_out_qubits[subcircuit_idx] = sorted(
            subcircuit_out_qubits[subcircuit_idx],
            key=lambda x: subcircuits[subcircuit_idx].qubits.index(x[0]),
            reverse=True,
        )
        subcircuit_out_qubits[subcircuit_idx] = [
            x[1] for x in subcircuit_out_qubits[subcircuit_idx]
        ]
    return subcircuit_out_qubits
