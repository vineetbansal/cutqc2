"""
Tests for end-end verification of cut circuits.
"""

from cutqc2.cutqc.helper_functions.benchmarks import generate_circ
from cutqc2.core.cut_circuit import CutCircuit


def test_adder_verify():
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


def test_supremacy_verify():
    num_qubits = 6
    circuit = generate_circ(
        num_qubits=num_qubits,
        depth=1,
        circuit_type="supremacy",
        reg_name="q",
        connected_only=True,
        seed=None,
    )
    import math

    cutter_constraints = {
        "max_subcircuit_width": math.ceil(circuit.num_qubits / 4 * 3),
        "max_subcircuit_cuts": 10,
        "subcircuit_size_imbalance": 2,
        "max_cuts": 10,
        "num_subcircuits": [3],
    }

    cut_circuit = CutCircuit(circuit)

    cut_circuit.cut(
        max_subcircuit_width=cutter_constraints["max_subcircuit_width"],
        max_subcircuit_cuts=cutter_constraints["max_subcircuit_cuts"],
        subcircuit_size_imbalance=cutter_constraints["subcircuit_size_imbalance"],
        max_cuts=cutter_constraints["max_cuts"],
        num_subcircuits=cutter_constraints["num_subcircuits"],
    )

    cut_circuit.run_subcircuits()
    cut_circuit.postprocess()
    cut_circuit.verify()
