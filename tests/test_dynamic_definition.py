import numpy as np
from qiskit.circuit.quantumregister import QuantumRegister, Qubit
from cutqc2.cutqc.cutqc.dynamic_definition import DynamicDefinition


class ComputeGraph:
    """
    Mock object that provides just enough information for DynamicDefinition
    to do its job.
    """

    # Simulating 2 subcircuits with these no. of wires
    s0_wires = 14
    s1_wires = 15

    def __init__(self):
        self.nodes = {
            0: {"effective": self.s0_wires - 1},
            1: {"effective": self.s1_wires},
        }

        self.initial_measurements = {
            0: {},
            1: {},
        }  # populated by `populate_subcircuit_entry_probs`
        self.subcircuit_entry_probs = {
            0: {},
            1: {},
        }  # populated by `populate_subcircuit_entry_probs`
        self.populate_subcircuit_entry_probs()

        self.current_pauli = None  # set by `assign_bases_to_edges`
        self.edges = []  # This is only used to remove a 'bases' key for each edge - not technically needed

    @staticmethod
    def get_edges(from_node, to_node):
        assert from_node is None
        assert to_node is None
        return [
            (
                0,  # from subcircuit 0
                1,  # to subcircuit 1
                {
                    "O_qubit": Qubit(QuantumRegister(10, "q"), 9),
                    "rho_qubit": Qubit(QuantumRegister(14, "q"), 0),
                },
            )
        ]

    def assign_bases_to_edges(self, edge_bases, edges):
        pauli = edge_bases[0]  # incoming edge_bases is a single char tuple
        # The next `get_init_meas` call expects us to remember the pauli
        self.current_pauli = pauli

    def remove_bases_from_edges(self, edges):
        # This is only used to remove a 'bases' key for each edge - not technically needed
        pass

    def populate_subcircuit_entry_probs(self):
        """
        Assume single cut, so we have:
        subcircuit 0 with s0_wires-1 qubit measurements
        """
        n_wires = ComputeGraph.s0_wires
        n_qubits = n_wires - 1
        inputs = ("zero",) * n_wires

        for pauli in "IXYZ":
            measurement = ("comp",) * n_qubits + (pauli,)
            self.initial_measurements[0][pauli] = inputs, measurement

        prob_dict = {}
        dist = np.full(
            2**n_qubits, 1 / 2**n_qubits
        )  # uniform distribution across all qubits
        for _, initial_measurement in self.initial_measurements[0].items():
            prob_dict[initial_measurement] = dist  # same distribution for all keys
        self.subcircuit_entry_probs[0] = prob_dict

        """
            Assume single cut, so we have:
            subcircuit 1 with s1_wires qubit measurements
            """
        n_wires = ComputeGraph.s1_wires
        n_qubits = n_wires
        measurement = ("comp",) * n_qubits

        for pauli in "IXYZ":
            inputs = (pauli,) + ("zero",) * (n_qubits - 1)
            self.initial_measurements[1][pauli] = inputs, measurement

        prob_dict = {}
        dist = np.full(
            2**n_qubits, 1 / 2**n_qubits
        )  # uniform distribution across all qubits
        for _, initial_measurement in self.initial_measurements[1].items():
            prob_dict[initial_measurement] = dist  # same distribution for all keys
        self.subcircuit_entry_probs[1] = prob_dict

    def get_init_meas(self, subcircuit_idx):
        return self.initial_measurements[subcircuit_idx][self.current_pauli]


def test_dynamic_definition_build():
    compute_graph = ComputeGraph()

    dynamic_definition = DynamicDefinition(
        compute_graph=compute_graph,
        num_cuts=1,
        subcircuit_entry_probs=compute_graph.subcircuit_entry_probs,
        mem_limit=6,
        recursion_depth=3,
    )

    dynamic_definition.build()
