import itertools, copy, math
from cutqc2.cutqc.cutqc.compute_graph import ComputeGraph


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
