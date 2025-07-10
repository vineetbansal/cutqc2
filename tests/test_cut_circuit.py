import textwrap
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister, Qubit
from cutqc2.core.cut_circuit import CutCircuit


@pytest.fixture(scope="module")
def simple_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(3)
    qc.reset(0)
    qc.reset(1)
    qc.reset(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)

    return qc


def test_cut_solution_str(simple_circuit):
    cut_circuit = CutCircuit(
        simple_circuit,
        cut_qubits_and_positions=[(Qubit(QuantumRegister(3, "q"), 0), 2)],
    )
    got_str = str(cut_circuit)
    expected_str = textwrap.dedent("""
                      ┌───┐     ┌────┐     
            q_0: ─|0>─┤ H ├──■──┤ ✂️ ├──■──
                      └───┘┌─┴─┐└────┘  │  
            q_1: ─|0>──────┤ X ├────────┼──
                           └───┘      ┌─┴─┐
            q_2: ─|0>─────────────────┤ X ├
                                      └───┘
    """).strip("\n")
    assert got_str == expected_str
