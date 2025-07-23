"""
Tests for end-end verification of cut circuits.
"""

from cutqc2.cutqc.helper_functions.benchmarks import generate_circ
from cutqc2.core.cut_circuit import CutCircuit


def test_adder():
    circuit = generate_circ(
        num_qubits=4,
        depth=4,
        circuit_type="adder",
        reg_name="q",
        connected_only=True,
        seed=None,
    )

    cut_circuit = CutCircuit(circuit)
    cut_circuit.cut(
        max_subcircuit_width=20,
        max_subcircuit_cuts=20,
        subcircuit_size_imbalance=20,
        max_cuts=20,
        num_subcircuits=[3],
    )

    cut_circuit.run_subcircuits()
    cut_circuit.postprocess()
    cut_circuit.verify()


def test_figure4_verify(figure_4_qiskit_circuit):
    cut_circuit = CutCircuit(figure_4_qiskit_circuit)
    cut_circuit.cut(
        max_subcircuit_width=3,
        max_subcircuit_cuts=2,
        subcircuit_size_imbalance=3,
        max_cuts=1,
        num_subcircuits=[2],
    )

    cut_circuit.run_subcircuits()
    cut_circuit.postprocess()
    cut_circuit.verify()
