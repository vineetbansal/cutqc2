import cudaq
from qiskit.primitives import StatevectorSampler
from cudaqut.qiskit_to_cudaq import Kernel


def normalize(counts):
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def total_variation_distance(counts1, counts2):
    keys = set(counts1) | set(counts2)
    p = normalize({k: counts1.get(k, 0) for k in keys})
    q = normalize({k: counts2.get(k, 0) for k in keys})
    return 0.5 * sum(abs(p[k] - q[k]) for k in keys)


def test_kernel_distribution(figure_4_qiskit_circuit):
    shots = 1_000_000
    qiskit_result = StatevectorSampler().run([figure_4_qiskit_circuit], shots=shots).result()
    # Qiskit count bitstrings are MSB-first: the leftmost bit corresponds to
    # the highest qubit index (the 'bottom' wire)
    qiskit_counts = qiskit_result[0].data.meas.get_counts()

    cudaq_kernel = Kernel("cudaq_kernel", figure_4_qiskit_circuit)
    cudaq_results = cudaq.sample(cudaq_kernel, shots_count=shots)
    cudaq_counts = dict(cudaq_results.items())
    # CudaQ count bitstrings are LSB-first: the leftmost bit corresponds to
    # the lowest qubit index (the 'top' wire)
    # We reverse the order to match the qiskit convention
    cudaq_counts = {k[::-1]: v for k, v in cudaq_counts.items()}

    tvd = total_variation_distance(qiskit_counts, cudaq_counts)
    assert tvd < 0.01, f"Total variation distance is too high: {tvd:.4f}"