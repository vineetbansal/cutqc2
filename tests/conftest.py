from math import pi
from qiskit import QuantumCircuit
import cudaq
import pytest


@pytest.fixture(scope="module")
def figure_4_qiskit_circuit():
    qc = QuantumCircuit(5)
    qc.reset(0)
    qc.reset(1)
    qc.reset(2)
    qc.reset(3)
    qc.reset(4)

    qc.h(0)
    qc.h(1)
    qc.h(2)
    qc.h(3)
    qc.h(4)

    qc.cz(0, 1)
    qc.t(2)
    qc.t(3)
    qc.t(4)

    qc.cz(0, 2)
    qc.rx(pi / 2, 4)

    qc.rx(pi / 2, 0)
    qc.rx(pi / 2, 1)
    qc.cz(2, 4)

    qc.t(0)
    qc.t(1)
    qc.cz(2, 3)
    qc.rx(pi / 2, 4)

    qc.h(0)
    qc.h(1)
    qc.h(2)
    qc.h(3)
    qc.h(4)

    qc.measure_all()

    return qc


@pytest.fixture(scope="module")
def figure_4_cudaq_kernel():
    def cudaq_kernel_function():
        qubits = cudaq.qvector(5)

        h(qubits[0])
        h(qubits[1])
        h(qubits[2])
        h(qubits[3])
        h(qubits[4])

        cz(qubits[0], qubits[1])

        t(qubits[2])
        t(qubits[3])
        t(qubits[4])

        cz(qubits[0], qubits[2])

        rx(math.pi / 2, qubits[4])

        rx(math.pi / 2, qubits[0])
        rx(math.pi / 2, qubits[1])

        cz(qubits[2], qubits[4])

        t(qubits[0])
        t(qubits[1])

        cz(qubits[2], qubits[3])

        rx(math.pi / 2, qubits[4])

        h(qubits[0])
        h(qubits[1])
        h(qubits[2])
        h(qubits[3])
        h(qubits[4])

    return cudaq_kernel_function
