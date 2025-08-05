import numpy as np
from cutqc2.core.utils import distribute


def test_distribute0():
    # 3 bins with all merged bits
    specs = {0: "MMM", 1: "MMMMM", 2: "MMMM"}  # 3, 5, 4 -> total 12
    result = distribute(specs, capacity=4)
    expected = {0: "AMM", 1: "AAMMM", 2: "AMMM"}
    assert result == expected


def test_distribute1():
    specs = {0: "AMM", 1: "1AMMM", 2: "AMMM"}
    # since we zoom into id 1, all available capacity is allocated to it
    result = distribute(specs, capacity=3, zoom=1)
    expected = {0: "MMM", 1: "1AAAM", 2: "MMMM"}
    assert result == expected


def test_distribute2():
    specs = {0: "AAA", 1: "MMMMMM", 2: "1010M"}
    # since we zoom into id 2, all available capacity is allocated to it
    result = distribute(specs, capacity=2, zoom=2)
    expected = {0: "MMM", 1: "MMMMMM", 2: "1010A"}
    assert result == expected


def test_distribute3():
    specs = {0: "AAA", 1: "MMMMMM", 2: "1010"}
    # since we zoom into id 2, all available capacity is allocated to it
    # but since it doesn't need it, nothing changes
    result = distribute(specs, capacity=7, zoom=2)
    expected = {0: "AAA", 1: "MMMMMM", 2: "1010"}
    assert result == expected


def test_distribute4():
    def get_prob(qubit_spec: str):
        n_qubits = qubit_spec.count("A")
        result = np.random.random(2**n_qubits)
        result /= np.sum(result)

        largest_bin = np.argmax(result)
        largest_bin_str = f"{largest_bin:010b}"
        for j in range(n_qubits):
            qubit_spec = qubit_spec.replace("A", largest_bin_str[j], 1)

        return result, qubit_spec

    specs = {0: "M" * 200}
    for recursion_depth in range(10):
        print(f"Recursion depth: {recursion_depth}; specs: {specs[0]}")
        result = distribute(specs, capacity=10, zoom=0)
        result0 = list(result.values())[0]
        prob, next_qubit_spec = get_prob(result0)
        specs = {0: next_qubit_spec}
