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


def test_cut_circuit_str(simple_circuit):
    cut_circuit = CutCircuit(
        simple_circuit,
        cut_qubits_and_positions=[(Qubit(QuantumRegister(3, "q"), 0), 2)],
        add_labels=False,
    )
    got_str = str(cut_circuit)
    expected_str = textwrap.dedent("""
                      ┌───┐     ┌────┐     
            q_0: ─|0>─┤ H ├──■──┤ // ├──■──
                      └───┘┌─┴─┐└────┘  │  
            q_1: ─|0>──────┤ X ├────────┼──
                           └───┘      ┌─┴─┐
            q_2: ─|0>─────────────────┤ X ├
                                      └───┘
    """).strip("\n")
    assert got_str == expected_str


def test_labeled_cut_circuit_str(simple_circuit):
    cut_circuit = CutCircuit(
        simple_circuit,
        cut_qubits_and_positions=[(Qubit(QuantumRegister(3, "q"), 0), 2)],
        add_labels=True,
    )
    got_str = str(cut_circuit)
    expected_str = textwrap.dedent("""
                ┌──────┐ 0004 ┌────┐ 0005 
        0: ─|0>─┤ 0003 ├──■───┤ // ├──■───
                └──────┘┌─┴─┐ └────┘  │   
        1: ─|0>─────────┤ X ├─────────┼───
                        └───┘       ┌─┴─┐ 
        2: ─|0>─────────────────────┤ X ├─
                                    └───┘ 
    """).strip("\n")
    assert got_str == expected_str


def test_cut_circuit_add_cut(simple_circuit):
    cut_circuit = CutCircuit(simple_circuit)
    cut_circuit.add_cut_at_label("0004")
    got_str = str(cut_circuit)
    expected_str = textwrap.dedent("""
                ┌──────┐ 0004 ┌────┐ 0005 
        0: ─|0>─┤ 0003 ├──■───┤ // ├──■───
                └──────┘┌─┴─┐ └────┘  │   
        1: ─|0>─────────┤ X ├─────────┼───
                        └───┘       ┌─┴─┐ 
        2: ─|0>─────────────────────┤ X ├─
                                    └───┘ 
    """).strip("\n")
    assert got_str == expected_str


def test_cut_circuit_generate_subcircuits(simple_circuit):
    cut_circuit = CutCircuit(simple_circuit)
    cut_circuit.add_cut_at_label("0004")
    cut_circuit.generate_subcircuits()

    assert len(cut_circuit) == 2

    first_subcircuit_str = str(cut_circuit[0])
    expected_str = textwrap.dedent("""
                ┌──────┐ 0004 
        0: ─|0>─┤ 0003 ├──■───
                └──────┘┌─┴─┐ 
        1: ─|0>─────────┤ X ├─
                        └───┘ 
    """).strip("\n")
    assert first_subcircuit_str == expected_str

    second_subcircuit_str = str(cut_circuit[1])
    expected_str = textwrap.dedent("""
                ┌───┐ 
        0: ─|0>─┤ X ├─
                └─┬─┘ 
        1: ───────■───
                 0005 
    """).strip("\n")
    assert second_subcircuit_str == expected_str
