import itertools
from functools import reduce
import numpy as np


class GraphContractor(object):
    def __init__(self, compute_graph, subcircuit_entry_probs, num_cuts) -> None:
        self.times = {}
        self.compute_graph = compute_graph
        self.subcircuit_entry_probs = subcircuit_entry_probs
        self.num_cuts = num_cuts
        self.subcircuit_entry_lengths = {}
        for subcircuit_idx in subcircuit_entry_probs:
            first_entry_init_meas = list(subcircuit_entry_probs[subcircuit_idx].keys())[
                0
            ]
            length = len(subcircuit_entry_probs[subcircuit_idx][first_entry_init_meas])
            self.subcircuit_entry_lengths[subcircuit_idx] = length
        self.num_qubits = 0
        for subcircuit_idx in compute_graph.nodes:
            self.num_qubits += compute_graph.nodes[subcircuit_idx]["effective"]

        self.smart_order = sorted(
            self.subcircuit_entry_lengths.keys(),
            key=lambda subcircuit_idx: self.subcircuit_entry_lengths[subcircuit_idx],
        )
        self.reconstructed_prob = self.compute()

    def compute(self):
        edges = self.compute_graph.get_edges(from_node=None, to_node=None)

        reconstructed_prob = None
        for edge_bases in itertools.product(["I", "X", "Y", "Z"], repeat=len(edges)):
            self.compute_graph.assign_bases_to_edges(edge_bases=edge_bases, edges=edges)
            summation_term = []
            for subcircuit_idx in self.smart_order:
                subcircuit_entry_init_meas = self.compute_graph.get_init_meas(
                    subcircuit_idx=subcircuit_idx
                )
                subcircuit_entry_prob = self.subcircuit_entry_probs[subcircuit_idx][
                    subcircuit_entry_init_meas
                ]
                summation_term.append(subcircuit_entry_prob)
            if reconstructed_prob is None:
                reconstructed_prob = reduce(np.kron, summation_term)
            else:
                reconstructed_prob += reduce(np.kron, summation_term)
            self.compute_graph.remove_bases_from_edges(edges=self.compute_graph.edges)
        reconstructed_prob *= 1 / 2**self.num_cuts
        return reconstructed_prob
