import numpy as np


def permute_bits(n: int, permutation: list[int]) -> int:
    n_bits = len(permutation)
    binary_n = f"{n:0{n_bits}b}"
    # Get bit i from position permutation[i]
    binary_n_permuted = "".join(binary_n[permutation[i]] for i in range(n_bits))
    return int(binary_n_permuted, 2)


def distribute(
    specs: dict[int, str], capacity: int | None = None, zoom: int | None = None
) -> dict[int, str]:
    """
    Distribute (or redistribute) loads to fit within a specified total capacity.
    Parameters
    ----------
    specs: Dictionary from id to load spec (string of A/M/0/1).
        - "A": qubit is preserved in output
        - "M": qubit is summed over
        - "0"/"1": qubit is fixed to that value
    capacity: Total capacity to distribute loads across.
    zoom: Optional id to zoom into

    Returns
    -------
    Dictionary mapping id to new load spec
    New load spec distributes the available capacity to each id's 'M' qubits
    by changing a subset of them to 'A'.
    """
    indices = list(specs.keys())
    if zoom is None:
        loads = np.array([v.count("M") for v in specs.values()])
    else:
        loads = np.array(
            [
                specs[zoom].count("M") + specs[zoom].count("A") if k == zoom else 0
                for k in indices
            ]
        )

    total_load = np.sum(loads)
    if total_load == 0:
        return specs
    if capacity is None:
        capacity = total_load
    new_loads = np.floor(loads / total_load * capacity).astype(int)

    remainder = int(capacity - np.sum(new_loads))
    if remainder > 0:
        # Add remaining to largest capacities first
        largest_indices = np.argsort(loads)[-remainder:]
        new_loads[largest_indices] += 1

    new_specs = []
    for k, spec in specs.items():
        new_load = new_loads[k]
        new_spec = spec.replace("A", "M")  # change all active to merged
        new_spec = new_spec.replace(
            "M", "A", new_load
        )  # change first |new_load| merged to active
        new_specs.append(new_spec)

    result = {}
    for index, new_load, new_spec in zip(indices, new_loads, new_specs):
        result[index] = new_spec
    return result


def distribute2(qubit_spec: str, probabilities: np.array, capacity: int) -> str:
    if "M" in qubit_spec:
        n_qubits = qubit_spec.count("A")
        largest_bin = np.argmax(probabilities)
        largest_bin_str = f"{largest_bin:0{n_qubits}b}"
        for j in range(n_qubits):
            qubit_spec = qubit_spec.replace("A", largest_bin_str[j], 1)
        qubit_spec = qubit_spec.replace("M", "A", capacity)

    return qubit_spec


def merge_prob_vector(unmerged_prob_vector: np.ndarray, qubit_spec: str) -> np.ndarray:
    """
    Compress quantum probability vector by merging specified qubits
    and conditioning on fixed qubit values.

    Parameters
    ----------
    unmerged_prob_vector : np.ndarray
        Original probability vector (2^num_qubits,)
    qubit_spec : str
        String of length `num_qubits`, MSB to LSB, with each character
        indicating:
        - "A": qubit is preserved in output
        - "M": qubit is summed over
        - "0"/"1": qubit is fixed to that value

    Returns
    -------
    np.ndarray
        Compressed probability vector (2^num_active,) with marginalization and conditioning applied.
    """
    num_qubits = len(qubit_spec)
    assert len(unmerged_prob_vector) == 2**num_qubits, (
        "Mismatch in qubit count and vector length."
    )

    active_qubit_indices = [i for i, q in enumerate(qubit_spec) if q == "A"]
    num_active = len(active_qubit_indices)

    if num_active == num_qubits:
        return np.copy(unmerged_prob_vector)

    merged_prob_vector = np.zeros(2**num_active, dtype="float32")

    for state in range(len(unmerged_prob_vector)):
        match = True
        for i, spec in enumerate(qubit_spec):
            bit_index = num_qubits - 1 - i  # MSB-first mapping
            bit_val = (state >> bit_index) & 1
            if spec == "0" and bit_val != 0:
                match = False
                break
            elif spec == "1" and bit_val != 1:
                match = False
                break
        if not match:
            continue

        # Construct index for active qubits
        active_state = 0
        for out_pos, i in enumerate(active_qubit_indices):
            bit_index = num_qubits - 1 - i
            bit_val = (state >> bit_index) & 1
            if bit_val:
                active_state |= 1 << (num_active - 1 - out_pos)  # MSB-first output

        merged_prob_vector[active_state] += unmerged_prob_vector[state]

    return merged_prob_vector


def unmerge_prob_vector(merged_prob_vector: np.ndarray, qubit_spec: str) -> np.ndarray:
    """
    Expand a merged quantum probability vector back to a full vector
    by evenly distributing over merged qubits and conditioning on fixed ones.

    Parameters
    ----------
    merged_prob_vector : np.ndarray
        Compressed probability vector (2^num_active,)
    qubit_spec : str
        String of length num_qubits with characters:
        - "A": active (preserved)
        - "M": merged (marginalized out)
        - "0"/"1": fixed bits

    Returns
    -------
    np.ndarray
        Expanded full probability vector of shape (2^num_qubits,)
    """
    num_qubits = len(qubit_spec)
    active_qubit_indices = [i for i, q in enumerate(qubit_spec) if q == "A"]
    merged_qubit_indices = [i for i, q in enumerate(qubit_spec) if q == "M"]
    fixed_qubit_conditions = {
        i: int(q) for i, q in enumerate(qubit_spec) if q in ("0", "1")
    }

    num_active = len(active_qubit_indices)
    num_merged = len(merged_qubit_indices)

    expanded = np.zeros(2**num_qubits, dtype=np.float32)

    for full_state in range(2**num_qubits):
        match = True
        for i, val in fixed_qubit_conditions.items():
            bit_index = num_qubits - 1 - i  # MSB to LSB
            bit_val = (full_state >> bit_index) & 1
            if bit_val != val:
                match = False
                break
        if not match:
            continue

        # Build index into merged vector
        active_index = 0
        for out_pos, i in enumerate(active_qubit_indices):
            bit_index = num_qubits - 1 - i
            bit_val = (full_state >> bit_index) & 1
            if bit_val:
                active_index |= 1 << (num_active - 1 - out_pos)

        num_merge_combinations = 2**num_merged

        # Uniformly distribute merged prob
        expanded[full_state] = merged_prob_vector[active_index] / num_merge_combinations

    return expanded
