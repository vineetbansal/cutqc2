import itertools
import copy
import numpy as np


class ComputeGraph(object):
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, subcircuit_idx, attributes):
        self.nodes[subcircuit_idx] = attributes

    def remove_node(self, subcircuit_idx):
        """
        Remove a node from the compute graph
        """
        del self.nodes[subcircuit_idx]

    def add_edge(self, u_for_edge, v_for_edge, attributes):
        self.edges.append((u_for_edge, v_for_edge, attributes))

    def get_edges(self, from_node, to_node):
        """
        Get edges in the graph based on some given conditions:
        1. If from_node is given. Only retain edges from the node.
        2. If to_node is given. Only retain edges to the node.
        """
        edges = []
        for edge in self.edges:
            u_for_edge, v_for_edge, _ = edge
            match_from_node = from_node is None or u_for_edge == from_node
            match_to_node = to_node is None or v_for_edge == to_node
            if match_from_node and match_to_node:
                edges.append(edge)
        return edges

    def assign_bases_to_edges(self, edge_bases, edges):
        """Assign the edge_bases to edges"""
        for edge_basis, edge in zip(edge_bases, edges):
            assert edge in self.edges
            u_for_edge, v_for_edge, attributes = edge
            attributes["basis"] = edge_basis

    def remove_bases_from_edges(self, edges):
        """Remove the edge_bases from edges"""
        for edge in edges:
            u_for_edge, v_for_edge, attributes = edge
            if "basis" in attributes:
                del attributes["basis"]

    def remove_all_bases(self):
        for edge in self.edges:
            u_for_edge, v_for_edge, attributes = edge
            if "basis" in attributes:
                del attributes["basis"]

    def get_init_meas(self, subcircuit_idx):
        """Get the entry_init, entry_meas for a given node"""
        node_attributes = self.nodes[subcircuit_idx]
        bare_subcircuit = node_attributes["subcircuit"]
        entry_init = ["zero"] * bare_subcircuit.num_qubits
        edges_to_node = self.get_edges(from_node=None, to_node=subcircuit_idx)
        for edge in edges_to_node:
            _, v_for_edge, edge_attributes = edge
            assert v_for_edge == subcircuit_idx
            entry_init[bare_subcircuit.qubits.index(edge_attributes["rho_qubit"])] = (
                edge_attributes["basis"]
            )

        entry_meas = ["comp"] * bare_subcircuit.num_qubits
        edges_from_node = self.get_edges(from_node=subcircuit_idx, to_node=None)
        for edge in edges_from_node:
            u_for_edge, _, edge_attributes = edge
            assert u_for_edge == subcircuit_idx
            entry_meas[bare_subcircuit.qubits.index(edge_attributes["O_qubit"])] = (
                edge_attributes["basis"]
            )
        return (tuple(entry_init), tuple(entry_meas))

    def get_contraction_edges(
        self, leading_subcircuit_idx, contracted_subcircuits_indices
    ):
        """
        Edges connecting the leading subcircuit and any one of the contracted subcircuits
        """
        contraction_edges = []
        for edge in self.edges:
            u_for_edge, v_for_edge, _ = edge
            if (
                u_for_edge in contracted_subcircuits_indices
                and v_for_edge in contracted_subcircuits_indices
            ):
                continue
            elif (
                u_for_edge == leading_subcircuit_idx
                and v_for_edge in contracted_subcircuits_indices
            ):
                contraction_edges.append(edge)
            elif (
                v_for_edge == leading_subcircuit_idx
                and u_for_edge in contracted_subcircuits_indices
            ):
                contraction_edges.append(edge)
        return contraction_edges

    def get_leading_edges(self, leading_subcircuit_idx, contracted_subcircuits_indices):
        """
        Edges only connecting the leading subcircuit
        """
        leading_edges = []
        for edge in self.edges:
            u_for_edge, v_for_edge, _ = edge
            if (
                u_for_edge in contracted_subcircuits_indices
                and v_for_edge in contracted_subcircuits_indices
            ):
                continue
            elif (
                u_for_edge == leading_subcircuit_idx
                and v_for_edge not in contracted_subcircuits_indices
            ):
                leading_edges.append(edge)
            elif (
                v_for_edge == leading_subcircuit_idx
                and u_for_edge not in contracted_subcircuits_indices
            ):
                leading_edges.append(edge)
        return leading_edges

    def get_trailing_edges(
        self, leading_subcircuit_idx, contracted_subcircuits_indices
    ):
        """
        Edges only connecting the contracted subcircuits
        """
        trailing_edges = []
        for edge in self.edges:
            u_for_edge, v_for_edge, _ = edge
            if (
                u_for_edge in contracted_subcircuits_indices
                and v_for_edge in contracted_subcircuits_indices
            ):
                continue
            elif (
                u_for_edge in contracted_subcircuits_indices
                and v_for_edge != leading_subcircuit_idx
            ):
                trailing_edges.append(edge)
            elif (
                v_for_edge in contracted_subcircuits_indices
                and u_for_edge != leading_subcircuit_idx
            ):
                trailing_edges.append(edge)
        return trailing_edges

    def get_contracted_edges(self, contracted_subcircuits_indices):
        """
        Edges in between the contracted subcircuits
        """
        contracted_edges = []
        for edge in self.edges:
            u_for_edge, v_for_edge, _ = edge
            if (
                u_for_edge in contracted_subcircuits_indices
                and v_for_edge in contracted_subcircuits_indices
            ):
                contracted_edges.append(edge)
        return contracted_edges


def get_cut_qubit_pairs(complete_path_map):
    """
    Get O-Rho cut qubit pairs
    """
    O_rho_pairs = []
    for input_qubit in complete_path_map:
        path = complete_path_map[input_qubit]
        if len(path) > 1:
            for path_ctr, item in enumerate(path[:-1]):
                O_qubit_tuple = item
                rho_qubit_tuple = path[path_ctr + 1]
                O_rho_pairs.append((O_qubit_tuple, rho_qubit_tuple))
    return O_rho_pairs


def get_instance_init_meas(init_label, meas_label):
    """
    Convert subcircuit entry init,meas into subcircuit instance init,meas
    """
    init_combinations = []
    for x in init_label:
        if x == "zero":
            init_combinations.append(["zero"])
        elif x == "I":
            init_combinations.append(["+zero", "+one"])
        elif x == "X":
            init_combinations.append(["2plus", "-zero", "-one"])
        elif x == "Y":
            init_combinations.append(["2plusI", "-zero", "-one"])
        elif x == "Z":
            init_combinations.append(["+zero", "-one"])
        else:
            raise Exception("Illegal initilization symbol :", x)
    init_combinations = list(itertools.product(*init_combinations))

    subcircuit_init_meas = []
    for init in init_combinations:
        subcircuit_init_meas.append((tuple(init), tuple(meas_label)))
    return subcircuit_init_meas


def convert_to_physical_init(init):
    coefficient = 1
    for idx, x in enumerate(init):
        if x == "zero":
            continue
        elif x == "+zero":
            init[idx] = "zero"
        elif x == "+one":
            init[idx] = "one"
        elif x == "2plus":
            init[idx] = "plus"
            coefficient *= 2
        elif x == "-zero":
            init[idx] = "zero"
            coefficient *= -1
        elif x == "-one":
            init[idx] = "one"
            coefficient *= -1
        elif x == "2plusI":
            init[idx] = "plusI"
            coefficient *= 2
        else:
            raise Exception("Illegal initilization symbol :", x)
    return coefficient, tuple(init)


def generate_compute_graph(counter, subcircuits, complete_path_map):
    """
    Generate the connection graph among subcircuits
    """
    compute_graph = ComputeGraph()
    for subcircuit_idx in counter:
        subcircuit_attributes = copy.deepcopy(counter[subcircuit_idx])
        subcircuit_attributes["subcircuit"] = subcircuits[subcircuit_idx]
        compute_graph.add_node(
            subcircuit_idx=subcircuit_idx, attributes=subcircuit_attributes
        )
    for circuit_qubit in complete_path_map:
        path = complete_path_map[circuit_qubit]
        for counter in range(len(path) - 1):
            upstream_subcircuit_idx = path[counter]["subcircuit_idx"]
            downstream_subcircuit_idx = path[counter + 1]["subcircuit_idx"]
            compute_graph.add_edge(
                u_for_edge=upstream_subcircuit_idx,
                v_for_edge=downstream_subcircuit_idx,
                attributes={
                    "O_qubit": path[counter]["subcircuit_qubit"],
                    "rho_qubit": path[counter + 1]["subcircuit_qubit"],
                },
            )
    return compute_graph


def generate_subcircuit_entries(compute_graph):
    """
    subcircuit_entries[subcircuit_idx][entry_init, entry_meas] = subcircuit_entry_term
    subcircuit_entry_term (list): (coefficient, instance_init, instance_meas)
    subcircuit_entry = Sum(coefficient*subcircuit_instance)

    subcircuit_instances[subcircuit_idx] = [(instance_init,instance_meas)]
    """
    subcircuit_entries = {}
    subcircuit_instances = {}
    for subcircuit_idx in compute_graph.nodes:
        # print('subcircuit_%d'%subcircuit_idx)
        bare_subcircuit = compute_graph.nodes[subcircuit_idx]["subcircuit"]
        subcircuit_entries[subcircuit_idx] = {}
        subcircuit_instances[subcircuit_idx] = []
        from_edges = compute_graph.get_edges(from_node=subcircuit_idx, to_node=None)
        to_edges = compute_graph.get_edges(from_node=None, to_node=subcircuit_idx)
        subcircuit_edges = from_edges + to_edges
        for subcircuit_edge_bases in itertools.product(
            ["I", "X", "Y", "Z"], repeat=len(subcircuit_edges)
        ):
            # print('subcircuit_edge_bases =',subcircuit_edge_bases)
            subcircuit_entry_init = ["zero"] * bare_subcircuit.num_qubits
            subcircuit_entry_meas = ["comp"] * bare_subcircuit.num_qubits
            for edge_basis, edge in zip(subcircuit_edge_bases, subcircuit_edges):
                (
                    upstream_subcircuit_idx,
                    downstream_subcircuit_idx,
                    edge_attributes,
                ) = edge
                if subcircuit_idx == upstream_subcircuit_idx:
                    O_qubit = edge_attributes["O_qubit"]
                    subcircuit_entry_meas[bare_subcircuit.qubits.index(O_qubit)] = (
                        edge_basis
                    )
                elif subcircuit_idx == downstream_subcircuit_idx:
                    rho_qubit = edge_attributes["rho_qubit"]
                    subcircuit_entry_init[bare_subcircuit.qubits.index(rho_qubit)] = (
                        edge_basis
                    )
                else:
                    raise IndexError(
                        "Generating entries for a subcircuit. subcircuit_idx should be either upstream or downstream"
                    )
            # print('subcircuit_entry_init =',subcircuit_entry_init)
            # print('subcircuit_entry_meas =',subcircuit_entry_meas)
            subcircuit_instance_init_meas = get_instance_init_meas(
                init_label=subcircuit_entry_init, meas_label=subcircuit_entry_meas
            )
            subcircuit_entry_term = []
            for init_meas in subcircuit_instance_init_meas:
                instance_init, instance_meas = init_meas
                coefficient, instance_init = convert_to_physical_init(
                    init=list(instance_init)
                )
                if (instance_init, instance_meas) not in subcircuit_instances[
                    subcircuit_idx
                ]:
                    subcircuit_instances[subcircuit_idx].append(
                        (instance_init, instance_meas)
                    )
                subcircuit_entry_term.append(
                    (coefficient, (instance_init, instance_meas))
                )
                # print('%d *'%coefficient, instance_init, instance_meas)
            subcircuit_entries[subcircuit_idx][
                (tuple(subcircuit_entry_init), tuple(subcircuit_entry_meas))
            ] = subcircuit_entry_term
    return subcircuit_entries, subcircuit_instances


def get_reconstruction_qubit_order(full_circuit, complete_path_map, subcircuits):
    """
    Get the output qubit in the full circuit for each subcircuit
    Qiskit orders the full circuit output in descending order of qubits
    """
    subcircuit_out_qubits = {
        subcircuit_idx: [] for subcircuit_idx in range(len(subcircuits))
    }
    for input_qubit in complete_path_map:
        path = complete_path_map[input_qubit]
        output_qubit = path[-1]
        subcircuit_out_qubits[output_qubit["subcircuit_idx"]].append(
            (output_qubit["subcircuit_qubit"], full_circuit.qubits.index(input_qubit))
        )
    for subcircuit_idx in subcircuit_out_qubits:
        subcircuit_out_qubits[subcircuit_idx] = sorted(
            subcircuit_out_qubits[subcircuit_idx],
            key=lambda x: subcircuits[subcircuit_idx].qubits.index(x[0]),
            reverse=True,
        )
        subcircuit_out_qubits[subcircuit_idx] = [
            x[1] for x in subcircuit_out_qubits[subcircuit_idx]
        ]
    return subcircuit_out_qubits


def read_dd_bins(subcircuit_out_qubits, dd_bins):
    num_qubits = sum(
        [
            len(subcircuit_out_qubits[subcircuit_idx])
            for subcircuit_idx in subcircuit_out_qubits
        ]
    )
    reconstructed_prob = np.zeros(2**num_qubits, dtype=np.float32)
    # print(subcircuit_out_qubits)
    for recursion_layer in dd_bins:
        # print('-'*20,'Verify Recursion Layer %d'%recursion_layer,'-'*20)
        # [print(field,dd_bins[recursion_layer][field]) for field in dd_bins[recursion_layer]]
        num_active = sum(
            [
                dd_bins[recursion_layer]["subcircuit_state"][subcircuit_idx].count(
                    "active"
                )
                for subcircuit_idx in dd_bins[recursion_layer]["subcircuit_state"]
            ]
        )
        for bin_id, bin_prob in enumerate(dd_bins[recursion_layer]["bins"]):
            if bin_prob > 0 and bin_id not in dd_bins[recursion_layer]["expanded_bins"]:
                binary_bin_id = bin(bin_id)[2:].zfill(num_active)
                # print('dd bin %s'%binary_bin_id)
                binary_full_state = ["" for _ in range(num_qubits)]
                for subcircuit_idx in dd_bins[recursion_layer]["smart_order"]:
                    subcircuit_state = dd_bins[recursion_layer]["subcircuit_state"][
                        subcircuit_idx
                    ]
                    for subcircuit_qubit_idx, qubit_state in enumerate(
                        subcircuit_state
                    ):
                        qubit_idx = subcircuit_out_qubits[subcircuit_idx][
                            subcircuit_qubit_idx
                        ]
                        if qubit_state == "active":
                            binary_full_state[qubit_idx] = binary_bin_id[0]
                            binary_bin_id = binary_bin_id[1:]
                        else:
                            binary_full_state[qubit_idx] = "%s" % qubit_state
                # print('reordered qubit state = {}'.format(binary_full_state))
                merged_qubit_indices = []
                for qubit, qubit_state in enumerate(binary_full_state):
                    if qubit_state == "merged":
                        merged_qubit_indices.append(qubit)
                num_merged = len(merged_qubit_indices)
                average_state_prob = bin_prob / 2**num_merged
                for binary_merged_state in itertools.product(
                    ["0", "1"], repeat=num_merged
                ):
                    for merged_qubit_ctr in range(num_merged):
                        binary_full_state[merged_qubit_indices[merged_qubit_ctr]] = (
                            binary_merged_state[merged_qubit_ctr]
                        )
                    full_state = "".join(binary_full_state)[::-1]
                    full_state_idx = int(full_state, 2)
                    reconstructed_prob[full_state_idx] = average_state_prob
                #     print('--> full state {} {:d}. p = {:.3e}'.format(full_state,full_state_idx,average_state_prob))
                # print()
    return reconstructed_prob
