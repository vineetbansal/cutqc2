import itertools
from copy import deepcopy
import numpy as np
from cutqc2.cutqc.cutqc.compute_graph import ComputeGraph
from cutqc2.cutqc.helper_functions.non_ibmq_functions import evaluate_circ
from cutqc2.cutqc.helper_functions.conversions import quasi_to_real
from cutqc2.cutqc.helper_functions.metrics import MSE
from cutqc2.cutqc.cutqc.dynamic_definition import read_dd_bins


class CutSolution:
    def __init__(self, *, circuit, subcircuits, complete_path_map, num_cuts):
        self.circuit = circuit
        self.subcircuits = subcircuits
        self.complete_path_map = complete_path_map
        self.num_cuts = num_cuts
        self.annotated_subcircuits = {}  # populated by `populate_annotated_subcircuits()`

        self.populate_annotated_subcircuits()
        self.populate_compute_graph()
        self.populate_subcircuit_entries()

    def __len__(self):
        return len(self.subcircuits)

    def __iter__(self):
        return iter(self.subcircuits)

    def __getitem__(self, item):
        return self.subcircuits[item]

    def populate_compute_graph(self):
        """
        Generate the connection graph among subcircuits
        """
        annotated_subcircuits = self.annotated_subcircuits
        subcircuits = self.subcircuits
        complete_path_map = self.complete_path_map

        compute_graph = ComputeGraph()
        for subcircuit_idx in annotated_subcircuits:
            subcircuit_attributes = deepcopy(annotated_subcircuits[subcircuit_idx])
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
        self.compute_graph = compute_graph

    def populate_subcircuit_entries(self):

        compute_graph = self.compute_graph

        subcircuit_entries = {}
        subcircuit_instances = {}

        for subcircuit_idx in compute_graph.nodes:
            bare_subcircuit = compute_graph.nodes[subcircuit_idx]["subcircuit"]
            subcircuit_entries[subcircuit_idx] = {}
            subcircuit_instances[subcircuit_idx] = []
            from_edges = compute_graph.get_edges(from_node=subcircuit_idx,
                                                 to_node=None)
            to_edges = compute_graph.get_edges(from_node=None,
                                               to_node=subcircuit_idx)
            subcircuit_edges = from_edges + to_edges
            for subcircuit_edge_bases in itertools.product(
                    ["I", "X", "Y", "Z"], repeat=len(subcircuit_edges)
            ):
                subcircuit_entry_init = ["zero"] * bare_subcircuit.num_qubits
                subcircuit_entry_meas = ["comp"] * bare_subcircuit.num_qubits
                for edge_basis, edge in zip(subcircuit_edge_bases,
                                            subcircuit_edges):
                    (
                        upstream_subcircuit_idx,
                        downstream_subcircuit_idx,
                        edge_attributes,
                    ) = edge
                    if subcircuit_idx == upstream_subcircuit_idx:
                        O_qubit = edge_attributes["O_qubit"]
                        subcircuit_entry_meas[
                            bare_subcircuit.qubits.index(O_qubit)] = (
                            edge_basis
                        )
                    elif subcircuit_idx == downstream_subcircuit_idx:
                        rho_qubit = edge_attributes["rho_qubit"]
                        subcircuit_entry_init[
                            bare_subcircuit.qubits.index(rho_qubit)] = (
                            edge_basis
                        )
                    else:
                        raise IndexError(
                            "Generating entries for a subcircuit. subcircuit_idx should be either upstream or downstream"
                        )

                subcircuit_instance_init_meas = self.get_instance_init_meas(
                    initializations=subcircuit_entry_init,
                    measurements=subcircuit_entry_meas
                )
                subcircuit_entry_term = []
                for init_meas in subcircuit_instance_init_meas:
                    instance_init, instance_meas = init_meas
                    coefficient, instance_init = self.convert_to_physical_init(
                        init=list(instance_init)
                    )
                    if (instance_init, instance_meas) not in \
                            subcircuit_instances[
                                subcircuit_idx
                            ]:
                        subcircuit_instances[subcircuit_idx].append(
                            (instance_init, instance_meas)
                        )
                    subcircuit_entry_term.append(
                        (coefficient, (instance_init, instance_meas))
                    )
                subcircuit_entries[subcircuit_idx][
                    (
                    tuple(subcircuit_entry_init), tuple(subcircuit_entry_meas))
                ] = subcircuit_entry_term

        self.subcircuit_entries, self.subcircuit_instances = subcircuit_entries, subcircuit_instances

    @staticmethod
    def get_instance_init_meas(initializations, measurements):
        init_combinations = []
        for initialization in initializations:
            match initialization:
                case "zero":
                    init_combinations.append(["zero"])
                case "I":
                    init_combinations.append(["+zero", "+one"])
                case "X":
                    init_combinations.append(["2plus", "-zero", "-one"])
                case "Y":
                    init_combinations.append(["2plusI", "-zero", "-one"])
                case "Z":
                    init_combinations.append(["+zero", "-one"])
                case _:
                    raise Exception("Illegal initialization symbol :", initialization)

        init_combinations = list(itertools.product(*init_combinations))

        subcircuit_init_meas = []
        for init in init_combinations:
            subcircuit_init_meas.append((tuple(init), tuple(measurements)))
        return subcircuit_init_meas

    def convert_to_physical_init(self, init):
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

    def populate_annotated_subcircuits(self):
        for subcircuit_idx, subcircuit in enumerate(self.subcircuits):
            self.annotated_subcircuits[subcircuit_idx] = {
                "effective": subcircuit.num_qubits,
            }

        for input_qubit, path in self.complete_path_map.items():
            if len(path) > 1:
                for j, O_qubit in enumerate(path[:-1]):
                    rho_qubit = path[j + 1]
                    from_subcircuit_index = O_qubit["subcircuit_idx"]
                    to_subcircuit_index = rho_qubit["subcircuit_idx"]

                    self.annotated_subcircuits[from_subcircuit_index]["effective"] -= 1

    def get_reconstruction_qubit_order(self):
        """
        Get the output qubit in the full circuit for each subcircuit
        Qiskit orders the full circuit output in descending order of qubits
        """
        subcircuit_out_qubits = {
            subcircuit_idx: [] for subcircuit_idx in range(len(self))
        }
        for input_qubit, path in self.complete_path_map.items():
            output_qubit = path[-1]
            subcircuit_out_qubits[output_qubit["subcircuit_idx"]].append(
                (output_qubit["subcircuit_qubit"],
                 self.circuit.qubits.index(input_qubit))
            )
        for subcircuit_idx in subcircuit_out_qubits:
            subcircuit_out_qubits[subcircuit_idx] = sorted(
                subcircuit_out_qubits[subcircuit_idx],
                key=lambda x: self[subcircuit_idx].qubits.index(x[0]),
                reverse=True,
            )
            subcircuit_out_qubits[subcircuit_idx] = [
                x[1] for x in subcircuit_out_qubits[subcircuit_idx]
            ]
        return subcircuit_out_qubits

    def full_verify(self, dd_bins):
        ground_truth = evaluate_circ(circuit=self.circuit, backend="statevector_simulator")
        subcircuit_out_qubits = self.get_reconstruction_qubit_order()
        reconstructed_prob = read_dd_bins(
            subcircuit_out_qubits=subcircuit_out_qubits, dd_bins=dd_bins
        )
        real_probability = quasi_to_real(
            quasiprobability=reconstructed_prob, mode="nearest"
        )
        approximation_error = (
            MSE(target=ground_truth, obs=real_probability)
            * 2**self.circuit.num_qubits
            / np.linalg.norm(ground_truth) ** 2
        )
        return reconstructed_prob, approximation_error