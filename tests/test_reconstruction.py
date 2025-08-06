import math
from cutqc2.cutqc.helper_functions.benchmarks import generate_circ
from cutqc2.core.cut_circuit import CutCircuit


def test_supremacy_reconstruction_with_increasing_capacity():
    circuit = generate_circ(
        num_qubits=6,
        depth=1,
        circuit_type="supremacy",
        reg_name="q",
        connected_only=True,
        seed=1234,
    )

    cut_circuit = CutCircuit(circuit)
    cut_circuit.cut(
        max_subcircuit_width=math.ceil(circuit.num_qubits / 4 * 3),
        max_subcircuit_cuts=10,
        subcircuit_size_imbalance=2,
        max_cuts=10,
        num_subcircuits=[3],
    )
    cut_circuit.run_subcircuits()

    error = cut_circuit.verify(capacity=0, raise_error=False)
    for capacity in (
        1,
        2,
        3,
        4,
        5,
        6,
    ):
        _error = cut_circuit.verify(capacity=capacity, raise_error=False)
        # error should decrease with increasing capacity
        assert _error <= error
    # The final error with full capacity should be very small
    assert _error < 1e-10
