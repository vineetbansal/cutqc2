import warnings
import numpy as np


def distribute_load(
    load: dict[int, int], capacity: int | None = None
) -> dict[int, int]:
    indices = list(load.keys())
    values = np.array(list(load.values()))

    total_load = min(np.sum(values), capacity) if capacity else np.sum(values)
    loads = np.floor(values / np.sum(values) * total_load).astype(int)

    remainder = int(total_load - np.sum(loads))
    if remainder > 0:
        # Add remaining to largest capacities first
        largest_indices = np.argsort(values)[-remainder:]
        loads[largest_indices] += 1

    return dict(zip(indices, loads))


def merge_prob_vector(unmerged_prob_vector: list[float], qubit_mask: int) -> np.ndarray:
    """
    Compress quantum probability vector by merging specified qubits.

    Parameters
    ----------
    unmerged_prob_vector : numpy.ndarray
        Original probability vector (2^num_qubits,)
    qubit_mask : mask specifying which qubits are active (1) and which
       need to get merged (0).

    Returns
    -------
    numpy.ndarray
        Compressed vector (2^num_active,) preserving active qubit states
        while summing over merged qubit states.
    """
    if not isinstance(qubit_mask, int):
        warnings.warn("Please pass in qubit_mask as an integer.")
        qubit_mask = "".join(["1" if x == "active" else "0" for x in qubit_mask])
        qubit_mask = int(qubit_mask, 2)

    num_qubits = len(unmerged_prob_vector).bit_length() - 1
    num_active = bin(qubit_mask).count("1")
    if num_qubits == num_active:
        return np.copy(unmerged_prob_vector)

    merged_prob_vector = np.zeros(2**num_active, dtype="float32")

    # Iterate through all possible states
    for state in range(len(unmerged_prob_vector)):
        # Extract active qubit values using bitwise operations
        active_state = 0
        active_bit_pos = 0

        for qubit in range(num_qubits):
            # If qubit is active
            if (qubit_mask >> qubit) & 1:
                # and has value 1 in original state
                if (state >> qubit) & 1:
                    # Set the corresponding bit in active_state
                    active_state |= 1 << active_bit_pos
                active_bit_pos += 1

        merged_prob_vector[active_state] += unmerged_prob_vector[state]

    return merged_prob_vector
