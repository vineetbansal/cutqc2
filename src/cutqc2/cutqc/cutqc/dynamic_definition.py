from functools import reduce
import itertools, copy, pickle, subprocess
import numpy as np


class DynamicDefinition:
    def __init__(
        self,
        compute_graph,
        num_cuts,
        subcircuit_entry_probs,
        mem_limit,
        recursion_depth,
    ) -> None:
        super().__init__()
        self.compute_graph = compute_graph
        self.num_cuts = num_cuts
        self.subcircuit_entry_probs = subcircuit_entry_probs
        self.mem_limit = mem_limit
        self.recursion_depth = recursion_depth
        self.dd_bins = {}

        self._build()

    def _build(self):
        """
        Returns

        dd_bins[recursion_layer] =  {'subcircuit_state','upper_bin'}
        subcircuit_state[subcircuit_idx] = ['0','1','active','merged']
        """
        num_qubits = sum(
            [
                self.compute_graph.nodes[subcircuit_idx]["effective"]
                for subcircuit_idx in self.compute_graph.nodes
            ]
        )
        largest_bins = []  # [{recursion_layer, bin_id}]
        recursion_layer = 0
        while recursion_layer < self.recursion_depth:
            """Get qubit states"""
            if recursion_layer == 0:
                dd_schedule = self.initialize_dynamic_definition_schedule()
            elif len(largest_bins) == 0:
                break
            else:
                bin_to_expand = largest_bins.pop(0)
                dd_schedule = self.next_dynamic_definition_schedule(
                    recursion_layer=bin_to_expand["recursion_layer"],
                    bin_id=bin_to_expand["bin_id"],
                )

            merged_subcircuit_entry_probs = self.merge_states_into_bins(dd_schedule)

            subcircuit_entry_lengths = {}
            for subcircuit_idx in self.subcircuit_entry_probs:
                first_entry_init_meas = \
                list(self.subcircuit_entry_probs[subcircuit_idx].keys())[
                    0
                ]
                length = len(self.subcircuit_entry_probs[subcircuit_idx][
                                 first_entry_init_meas])
                subcircuit_entry_lengths[subcircuit_idx] = length
            num_qubits = 0
            for subcircuit_idx in self.compute_graph.nodes:
                num_qubits += self.compute_graph.nodes[subcircuit_idx]["effective"]

            smart_order = sorted(
                subcircuit_entry_lengths.keys(),
                key=lambda subcircuit_idx: subcircuit_entry_lengths[
                    subcircuit_idx],
            )

            edges = self.compute_graph.get_edges(from_node=None, to_node=None)

            reconstructed_prob = None
            for edge_bases in itertools.product(["I", "X", "Y", "Z"],
                                                repeat=len(edges)):
                self.compute_graph.assign_bases_to_edges(edge_bases=edge_bases,
                                                    edges=edges)
                summation_term = []
                for subcircuit_idx in smart_order:
                    subcircuit_entry_init_meas = self.compute_graph.get_init_meas(
                        subcircuit_idx=subcircuit_idx
                    )
                    subcircuit_entry_prob = \
                    self.subcircuit_entry_probs[subcircuit_idx][
                        subcircuit_entry_init_meas
                    ]
                    summation_term.append(subcircuit_entry_prob)
                if reconstructed_prob is None:
                    reconstructed_prob = reduce(np.kron, summation_term)
                else:
                    reconstructed_prob += reduce(np.kron, summation_term)
                self.compute_graph.remove_bases_from_edges(
                    edges=self.compute_graph.edges)

            reconstructed_prob *= 1 / 2 ** self.num_cuts

            self.dd_bins[recursion_layer] = dd_schedule
            self.dd_bins[recursion_layer]["smart_order"] = smart_order
            self.dd_bins[recursion_layer]["bins"] = reconstructed_prob
            self.dd_bins[recursion_layer]["expanded_bins"] = []

            """ Sort and truncate the largest bins """
            has_merged_states = False
            for subcircuit_idx in dd_schedule["subcircuit_state"]:
                if "merged" in dd_schedule["subcircuit_state"][subcircuit_idx]:
                    has_merged_states = True
                    break
            if recursion_layer < self.recursion_depth - 1 and has_merged_states:
                bin_indices = np.argpartition(
                    reconstructed_prob, -self.recursion_depth
                )[-self.recursion_depth :]
                for bin_id in bin_indices:
                    if reconstructed_prob[bin_id] > 1 / 2**num_qubits / 10:
                        largest_bins.append(
                            {
                                "recursion_layer": recursion_layer,
                                "bin_id": bin_id,
                                "prob": reconstructed_prob[bin_id],
                            }
                        )
                largest_bins = sorted(
                    largest_bins, key=lambda bin: bin["prob"], reverse=True
                )[: self.recursion_depth]
            recursion_layer += 1

    def initialize_dynamic_definition_schedule(self):
        schedule = {}
        schedule["subcircuit_state"] = {}
        schedule["upper_bin"] = None

        subcircuit_capacities = {
            subcircuit_idx: self.compute_graph.nodes[subcircuit_idx]["effective"]
            for subcircuit_idx in self.compute_graph.nodes
        }
        subcircuit_active_qubits = self.distribute_load(
            capacities=subcircuit_capacities
        )
        for subcircuit_idx in subcircuit_active_qubits:
            num_zoomed = 0
            num_active = subcircuit_active_qubits[subcircuit_idx]
            num_merged = (
                self.compute_graph.nodes[subcircuit_idx]["effective"]
                - num_zoomed
                - num_active
            )
            schedule["subcircuit_state"][subcircuit_idx] = [
                "active" for _ in range(num_active)
            ] + ["merged" for _ in range(num_merged)]
        return schedule

    def next_dynamic_definition_schedule(self, recursion_layer, bin_id):
        num_active = 0
        for subcircuit_idx in self.dd_bins[recursion_layer]["subcircuit_state"]:
            num_active += self.dd_bins[recursion_layer]["subcircuit_state"][
                subcircuit_idx
            ].count("active")
        binary_bin_idx = bin(bin_id)[2:].zfill(num_active)
        smart_order = self.dd_bins[recursion_layer]["smart_order"]
        next_dd_schedule = {
            "subcircuit_state": copy.deepcopy(
                self.dd_bins[recursion_layer]["subcircuit_state"]
            )
        }
        binary_state_idx_ptr = 0
        for subcircuit_idx in smart_order:
            for qubit_ctr, qubit_state in enumerate(
                next_dd_schedule["subcircuit_state"][subcircuit_idx]
            ):
                if qubit_state == "active":
                    next_dd_schedule["subcircuit_state"][subcircuit_idx][qubit_ctr] = (
                        int(binary_bin_idx[binary_state_idx_ptr])
                    )
                    binary_state_idx_ptr += 1
        next_dd_schedule["upper_bin"] = (recursion_layer, bin_id)

        subcircuit_capacities = {
            subcircuit_idx: next_dd_schedule["subcircuit_state"][subcircuit_idx].count(
                "merged"
            )
            for subcircuit_idx in next_dd_schedule["subcircuit_state"]
        }
        subcircuit_active_qubits = self.distribute_load(
            capacities=subcircuit_capacities
        )
        # print('subcircuit_active_qubits:',subcircuit_active_qubits)
        for subcircuit_idx in next_dd_schedule["subcircuit_state"]:
            num_active = subcircuit_active_qubits[subcircuit_idx]
            for qubit_ctr, qubit_state in enumerate(
                next_dd_schedule["subcircuit_state"][subcircuit_idx]
            ):
                if qubit_state == "merged" and num_active > 0:
                    next_dd_schedule["subcircuit_state"][subcircuit_idx][
                        qubit_ctr
                    ] = "active"
                    num_active -= 1
            assert num_active == 0
        return next_dd_schedule

    def distribute_load(self, capacities):
        if self.mem_limit is None:
            total_load = sum(capacities.values())
        else:
            total_load = min(sum(capacities.values()), self.mem_limit)
        total_capacity = sum(capacities.values())
        loads = {subcircuit_idx: 0 for subcircuit_idx in capacities}

        for slot_idx in loads:
            loads[slot_idx] = int(capacities[slot_idx] / total_capacity * total_load)
        total_load -= sum(loads.values())

        for slot_idx in loads:
            while total_load > 0 and loads[slot_idx] < capacities[slot_idx]:
                loads[slot_idx] += 1
                total_load -= 1
        assert total_load == 0
        return loads

    def merge_states_into_bins(self, dd_schedule):
        """
        The first merge of subcircuit probs using the target number of bins
        Saves the overhead of writing many states in the first SM recursion
        """
        merged_subcircuit_entry_probs = {}
        for subcircuit_idx in self.compute_graph.nodes:
            merged_subcircuit_entry_probs[subcircuit_idx] = {}
            for subcircuit_entry_init_meas in self.subcircuit_entry_probs[
                subcircuit_idx
            ]:
                unmerged_prob_vector = self.subcircuit_entry_probs[subcircuit_idx][
                    subcircuit_entry_init_meas
                ]
                merged_subcircuit_entry_probs[subcircuit_idx][
                    subcircuit_entry_init_meas
                ] = merge_prob_vector(
                    unmerged_prob_vector=unmerged_prob_vector,
                    qubit_states=dd_schedule["subcircuit_state"][subcircuit_idx],
                )
        return merged_subcircuit_entry_probs


def merge_prob_vector(unmerged_prob_vector, qubit_states):
    num_active = qubit_states.count("active")
    if num_active == len(qubit_states):
        # short-circuit if no merging is needed
        return np.copy(unmerged_prob_vector)

    num_merged = qubit_states.count("merged")
    merged_prob_vector = np.zeros(2**num_active, dtype="float32")

    for active_qubit_states in itertools.product(["0", "1"], repeat=num_active):
        if len(active_qubit_states) > 0:
            merged_bin_id = int("".join(active_qubit_states), 2)
        else:
            merged_bin_id = 0
        for merged_qubit_states in itertools.product(["0", "1"], repeat=num_merged):
            active_ptr = 0
            merged_ptr = 0
            binary_state_id = ""
            for qubit_state in qubit_states:
                if qubit_state == "active":
                    binary_state_id += active_qubit_states[active_ptr]
                    active_ptr += 1
                elif qubit_state == "merged":
                    binary_state_id += merged_qubit_states[merged_ptr]
                    merged_ptr += 1
                else:
                    binary_state_id += "%s" % qubit_state
            state_id = int(binary_state_id, 2)
            merged_prob_vector[merged_bin_id] += unmerged_prob_vector[state_id]
    return merged_prob_vector
