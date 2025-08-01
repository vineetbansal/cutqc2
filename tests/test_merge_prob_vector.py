import numpy as np
from cutqc2.cutqc.cutqc.dynamic_definition import merge_prob_vector


def test_all_active_qubits():
    # 2-qubit system, all active
    prob_vector = np.array([0.1, 0.2, 0.3, 0.4])
    result = merge_prob_vector(prob_vector, 0b11)
    np.testing.assert_array_equal(result, prob_vector)


def test_all_merged_qubits():
    # 2-qubit system, all merged
    prob_vector = np.array([0.1, 0.2, 0.3, 0.4])
    result = merge_prob_vector(prob_vector, 0b00)
    np.testing.assert_array_almost_equal(result, np.array([1.0]))


def test_mixed_active_merged():
    # 3-qubit system: qubit 0 active, qubit 1 merged, qubit 2 active
    prob_vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    qubit_mask = 0b101  # Qubits 0 and 2 active, qubit 1 merged

    result = merge_prob_vector(prob_vector, qubit_mask)

    # State 00: prob_vector[0b000] + prob_vector[0b010] = 0.1 + 0.3 = 0.4
    # State 01: prob_vector[0b001] + prob_vector[0b011] = 0.2 + 0.4 = 0.6
    # State 10: prob_vector[0b100] + prob_vector[0b110] = 0.5 + 0.7 = 1.2
    # State 11: prob_vector[0b101] + prob_vector[0b111] = 0.6 + 0.8 = 1.4

    expected = np.array([0.4, 0.6, 1.2, 1.4])
    np.testing.assert_array_almost_equal(result, expected)


def test_single_active_qubit():
    prob_vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    qubit_mask = 0b010  # capture indices (0, 1, 4, 5) and (2, 3, 6, 7)

    result = merge_prob_vector(prob_vector, qubit_mask)

    # State 0 = 0.1 + 0.2 + 0.5 + 0.6 = 1.4
    # State 1 = 0.3 + 0.4 + 0.7 + 0.8 = 2.2

    expected = np.array([1.4, 2.2])
    np.testing.assert_array_almost_equal(result, expected)


def test_probability_conservation():
    for num_qubits in range(2, 8):
        prob_vector = np.random.random(2**num_qubits)

        for qubit_mask in range(2**num_qubits):
            result = merge_prob_vector(prob_vector, qubit_mask)
            assert np.isclose(np.sum(result), np.sum(prob_vector))
