import numpy as np
from cutqc2.cutqc.cutqc.dynamic_definition import read_dd_bins


def test_reconstruction():
    """
    Test reconstruction for a 5-qubit circuit with 2 subcircuits (with 2 and 3 qubits respectively).
    """

    result = read_dd_bins(
        # Mapping from subcircuit index to qubit indices
        # This is the actual (and consistent) mapping for Figure 4 in the paper
        # TODO: Find out why.
        subcircuit_out_qubits={0: [1, 0], 1: [2, 4, 3]},
        # Mapping from recursion level to a dict of bin information
        dd_bins={
            0: {
                # Mapping from subcircuit index to the state of its qubits (active/merged)
                "subcircuit_state": {
                    0: ["active", "active"],
                    1: ["active", "active", "active"],
                },
                # Ordering of subcircuits
                "smart_order": [0, 1],
                # 2^5 = 32 probability values
                "bins": np.array(
                    [
                        0.10803459,
                        0.0078125,
                        0.0390625,
                        0.03259041,
                        0.07678459,
                        0.0390625,
                        0.0078125,
                        0.06384041,
                        0.00134041,
                        0.0078125,
                        0.0078125,
                        0.04553459,
                        0.00134041,
                        0.0078125,
                        0.0078125,
                        0.04553459,
                        0.00134041,
                        0.0078125,
                        0.0078125,
                        0.04553459,
                        0.00134041,
                        0.0078125,
                        0.0078125,
                        0.04553459,
                        0.07678459,
                        0.0390625,
                        0.0078125,
                        0.06384041,
                        0.10803459,
                        0.0078125,
                        0.0390625,
                        0.03259041,
                    ]
                ),
                "expanded_bins": [],
            }
        },
    )

    # 2^5 = 32 "reconstructed" probability values
    assert np.allclose(
        result,
        np.array(
            [
                0.10803459,
                0.00134041,
                0.00134041,
                0.07678459,
                0.07678459,
                0.00134041,
                0.00134041,
                0.10803459,
                0.0078125,
                0.0078125,
                0.0078125,
                0.0390625,
                0.0390625,
                0.0078125,
                0.0078125,
                0.0078125,
                0.0390625,
                0.0078125,
                0.0078125,
                0.0078125,
                0.0078125,
                0.0078125,
                0.0078125,
                0.0390625,
                0.03259041,
                0.04553459,
                0.04553459,
                0.06384041,
                0.06384041,
                0.04553459,
                0.04553459,
                0.03259041,
            ]
        ),
    )
