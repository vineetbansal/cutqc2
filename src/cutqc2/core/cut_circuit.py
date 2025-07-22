from pathlib import Path
from copy import deepcopy
import itertools
import numpy as np
from typing import Self

from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.circuit import Qubit, QuantumRegister, CircuitInstruction
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode, DAGCircuit

from cutqc2.cutqc.cutqc.evaluator import run_subcircuit_instances, attribute_shots
from cutqc2.cutqc.cutqc.dynamic_definition import read_dd_bins, DynamicDefinition
from cutqc2.cutqc.cutqc.compute_graph import ComputeGraph
from cutqc2.cutqc.helper_functions.non_ibmq_functions import evaluate_circ
from cutqc2.cutqc.helper_functions.conversions import quasi_to_real
from cutqc2.cutqc.helper_functions.metrics import MSE
from cutqc2.core.dag import DagNode, DAGEdge


class WireCutGate(UnitaryGate):
    """
    Custom gate to represent a wire cut in a quantum circuit.
    """

    def __init__(self):
        super().__init__(data=[[1, 0], [0, 1]], num_qubits=1, label="//")
        # The super constructor initializes name as "unitary" - use our own
        self.name = "cut"


class CutCircuit:
    def __init__(
        self,
        circuit: QuantumCircuit,
        add_labels: bool = True,
    ):
        self.check_valid(circuit)

        self.raw_circuit = circuit.copy()
        self.unlabeled_circuit = circuit.copy()
        if add_labels:
            self.circuit = self.get_labeled_circuit(circuit.copy())
        else:
            self.circuit = circuit.copy()

        self.inter_wire_dag = self.get_inter_wire_dag(self.circuit)
        self.inter_wire_dag_metadata = self.get_dag_metadata(self.inter_wire_dag)

        self.subcircuits = []
        # location of cuts, as (<wire_index>, <gate_index>) tuples
        self.cuts: list[tuple[int, int]] = []

        self._reconstruction_qubit_order = None

    def __str__(self):
        return str(self.unlabeled_circuit.draw(output="text", fold=-1))

    def __len__(self):
        return len(self.subcircuits)

    def __iter__(self):
        return iter(self.subcircuits)

    def __getitem__(self, item):
        return self.subcircuits[item]

    @staticmethod
    def check_valid(circuit: QuantumCircuit):
        unitary_factors = circuit.num_unitary_factors()
        assert unitary_factors == 1, (
            f"Input circuit is not fully connected thus does not need cutting. Number of unitary factors = {unitary_factors}"
        )

        assert circuit.num_clbits == 0, (
            "Please remove classical bits from the circuit before cutting"
        )
        dag = circuit_to_dag(circuit)
        for op_node in dag.topological_op_nodes():
            assert len(op_node.qargs) <= 2, (
                "CutQC currently does not support >2-qubit gates"
            )
            assert op_node.op.name != "barrier", (
                "Please remove barriers from the circuit before cutting"
            )

    @staticmethod
    def from_file(filepath: str | Path, *args, **kwargs) -> Self:
        if isinstance(filepath, str):
            filepath = Path(filepath)

        # Keep imports local to this function
        from cutqc2.io.h5 import h5_to_cut_circuit

        supported_formats = {".h5": h5_to_cut_circuit}
        assert filepath.suffix in supported_formats, "Unsupported format"
        return supported_formats[filepath.suffix](filepath, *args, **kwargs)

    @staticmethod
    def get_labeled_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Get a labeled version of the circuit where each instruction is labeled
        """
        labeled_instructions = []
        for i, instr in enumerate(list(circuit.data)):
            label = f"{i:04d}"
            new_op = instr.operation.copy().to_mutable()
            new_op.label = label
            instr = CircuitInstruction(
                operation=new_op, qubits=instr.qubits, clbits=instr.clbits
            )
            labeled_instructions.append(instr)

        labeled_circuit = QuantumCircuit.from_instructions(
            labeled_instructions, qubits=circuit.qubits, clbits=circuit.clbits
        )
        labeled_circuit.qregs = circuit.qregs

        return labeled_circuit

    @staticmethod
    def get_inter_wire_dag(circuit: QuantumCircuit) -> DAGCircuit:
        """
        Get the dag for the stripped version of a circuit where we only
        preserve gates that span two wires.
        """
        dag = DAGCircuit()
        for qreg in circuit.qregs:
            dag.add_qreg(qreg)

        for vertex in circuit_to_dag(circuit).topological_op_nodes():
            if len(vertex.qargs) == 2 and vertex.op.name != "barrier":
                dag.apply_operation_back(op=vertex.op, qargs=vertex.qargs)
        return dag

    @staticmethod
    def get_dag_metadata(dag: DAGCircuit) -> dict:
        edges = []
        node_name_ids = {}
        id_node_names = {}
        id_to_dag_edge = {}
        vertex_ids = {}
        curr_node_id = 0
        qubit_gate_counter = {}

        for qubit in dag.qubits:
            qubit_gate_counter[qubit] = 0

        for vertex in dag.topological_op_nodes():
            if len(vertex.qargs) != 2:
                raise Exception("vertex does not have 2 qargs!")

            arg0, arg1 = vertex.qargs

            dag_edge = DAGEdge(
                DagNode(
                    name=vertex.label,
                    wire_index=arg0._index,
                    gate_index=qubit_gate_counter[arg0],
                ),
                DagNode(
                    name=vertex.label,
                    wire_index=arg1._index,
                    gate_index=qubit_gate_counter[arg1],
                ),
            )
            vertex_name = str(dag_edge)

            qubit_gate_counter[arg0] += 1
            qubit_gate_counter[arg1] += 1
            if vertex_name not in node_name_ids and hash(vertex) not in vertex_ids:
                node_name_ids[vertex_name] = curr_node_id
                id_node_names[curr_node_id] = vertex_name
                id_to_dag_edge[curr_node_id] = dag_edge
                vertex_ids[hash(vertex)] = curr_node_id

                curr_node_id += 1

        for u, v, _ in dag.edges():
            if isinstance(u, DAGOpNode) and isinstance(v, DAGOpNode):
                u_id = vertex_ids[hash(u)]
                v_id = vertex_ids[hash(v)]
                edges.append((u_id, v_id))

        n_vertices = dag.size()

        return {
            "n_vertices": n_vertices,
            "edges": edges,
            "vertex_ids": node_name_ids,
            "id_vertices": id_node_names,
            "id_to_dag_edge": id_to_dag_edge,
        }

    @staticmethod
    def get_initializations(paulis: list[str]) -> list[tuple[int, tuple[str]]]:
        """
        Get coefficients and kets used for each term in the expansion of the
        trace operators for each of the Pauli bases (eq. 2 in paper).

                 0   1   +   i
            0    1
            I    1   1
            X   -1  -1   2
            Y   -1  -1       2
            Z    1  -1

        """
        terms = {
            "zero": {"zero": 1},
            "I": {"zero": 1, "one": 1},
            "X": {"zero": -1, "one": -1, "plus": 2},
            "Y": {"zero": -1, "one": -1, "plusI": 2},
            "Z": {"zero": 1, "one": -1},
        }

        substitution_lists = [terms[pauli].items() for pauli in paulis]

        result = []
        for combo in itertools.product(*substitution_lists):
            coeff = 1
            labels = []
            for label, c in combo:
                coeff *= c
                labels.append(label)
            result.append((coeff, tuple(labels)))
        return result

    def find_cuts(
        self,
        max_subcircuit_width: int,
        max_cuts: int,
        num_subcircuits: list[int],
        max_subcircuit_cuts: int,
        subcircuit_size_imbalance: int,
    ):
        from cutqc2.cutqc.cutqc.cutter import MIP_Model

        n_vertices, edges, vertex_ids, id_vertices, id_to_dag_edge = (
            self.inter_wire_dag_metadata["n_vertices"],
            self.inter_wire_dag_metadata["edges"],
            self.inter_wire_dag_metadata["vertex_ids"],
            self.inter_wire_dag_metadata["id_vertices"],
            self.inter_wire_dag_metadata["id_to_dag_edge"],
        )

        num_qubits = self.circuit.num_qubits
        for num_subcircuit in num_subcircuits:
            if (
                num_subcircuit * max_subcircuit_width - (num_subcircuit - 1)
                < num_qubits
                or num_subcircuit > num_qubits
                or max_cuts + 1 < num_subcircuit
            ):
                continue

            mip_model = MIP_Model(
                n_vertices=n_vertices,
                edges=edges,
                vertex_ids=vertex_ids,
                id_vertices=id_vertices,
                id_to_dag_edge=id_to_dag_edge,
                num_subcircuit=num_subcircuit,
                max_subcircuit_width=max_subcircuit_width,
                max_subcircuit_cuts=max_subcircuit_cuts,
                subcircuit_size_imbalance=subcircuit_size_imbalance,
                num_qubits=num_qubits,
                max_cuts=max_cuts,
            )

            if mip_model.solve():
                return mip_model.cut_edges_pairs
            else:
                raise RuntimeError("No viable cuts found")

    def add_cut_at_position(self, wire_index: int, gate_index: int):
        cut_qubit = self.circuit.qubits[wire_index]

        cut_wire_position = 0
        found = False
        for i, instr in enumerate(self.circuit.data):
            if cut_qubit in instr.qubits:  # we're on the right wire
                if cut_wire_position == gate_index:
                    cut_instr = CircuitInstruction(WireCutGate(), qubits=(cut_qubit,))
                    self.unlabeled_circuit.data.insert(i + 1, cut_instr)
                    found = True
                    break
                cut_wire_position += 1

        if found:
            self.cuts.append((wire_index, gate_index))
        else:
            raise ValueError(
                f"Gate index '{gate_index}' or wire {wire_index} not found in the circuit. Cannot add cut."
            )

    def add_cut(self, label: str, wire_index: int):
        """
        Add a cut to the circuit at the position of the instruction with the specified label.
        Args:
            label: The label of the instruction after which the cut should be made.
            wire_index: The index of the wire where the cut should be made.
        """
        cut_qubit = self.circuit.qubits[wire_index]
        gate_counter = {qubit: 0 for qubit in self.circuit.qubits}

        found = False
        for i, instr in enumerate(self.circuit.data):
            if instr.operation.label == label:
                cut_qubit = self.circuit.qubits[wire_index]
                cut_instr = CircuitInstruction(WireCutGate(), qubits=(cut_qubit,))
                # insert the cut instruction right after the current instruction
                self.circuit.data.insert(i + 1, cut_instr)
                # Also add to unlabeled circuit (for visualization purposes)
                self.unlabeled_circuit.data.insert(i + 1, cut_instr)
                found = True
                break
            else:
                for qubit in instr.qubits:
                    gate_counter[qubit] += 1

        if found:
            gate_index = gate_counter[cut_qubit]
            self.cuts.append((wire_index, gate_index))
        else:
            raise ValueError(
                f"Label '{label}' or wire {wire_index} not found in the circuit. Cannot add cut."
            )

    def add_cuts(
        self,
        cut_edges: list[tuple[DAGEdge, DAGEdge]],
        generate_subcircuits: bool = True,
    ):
        # validate cut_edges
        for cut_edge in cut_edges:
            p, q = cut_edge[0] | cut_edge[1]
            if q - p != 1:
                raise ValueError(
                    "Invalid cut - the cut edge does not connect two adjacent gates."
                )

        for cut_edge in cut_edges:
            edge0, edge1 = cut_edge
            self.add_cut(p.name, p.wire_index)

        if generate_subcircuits:
            self.generate_subcircuits()

    @property
    def num_cuts(self) -> int:
        return len(self.cuts)

    @property
    def reconstruction_qubit_order(self) -> dict[int, list[int]]:
        subcircuit_out_qubits = {
            subcircuit_idx: [] for subcircuit_idx in range(len(self))
        }

        for qubit, mappings in self.qubit_mapping.items():
            output_subcircuit_index, output_qubit = list(mappings.items())[-1]
            subcircuit_out_qubits[output_subcircuit_index].append(
                (
                    output_qubit,
                    self.circuit.qubits.index(qubit),
                )
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

    @reconstruction_qubit_order.setter
    def reconstruction_qubit_order(self, value: dict[int, list[int]]):
        self._reconstruction_qubit_order = deepcopy(value)

    def generate_subcircuits(self):
        # TODO: Cache results intelligently based on cut positions:
        # We may not have to regenerate everything in all cases
        def remap_qubits(
            subcircuit_ops: list[DAGOpNode],
            subcircuit_qubits: set[Qubit],
            subcircuit_i: int,
            qubit_mapping: dict[Qubit, dict[int, Qubit]],
        ) -> QuantumCircuit:
            subcircuit_qubit_i = 0
            subcircuit_size = len(subcircuit_qubits)
            for qubit in subcircuit_qubits:
                if subcircuit_i not in qubit_mapping[qubit]:
                    new_qubit = Qubit(
                        QuantumRegister(subcircuit_size, "q"), subcircuit_qubit_i
                    )
                    qubit_mapping[qubit][subcircuit_i] = new_qubit
                    subcircuit_qubit_i += 1

            for subcircuit_op in subcircuit_ops:
                subcircuit_op.qargs = tuple(
                    qubit_mapping[qarg][subcircuit_i] for qarg in subcircuit_op.qargs
                )

            instructions = []
            for subcircuit_op in subcircuit_ops:
                instructions.append(
                    CircuitInstruction(
                        operation=subcircuit_op.op, qubits=subcircuit_op.qargs
                    )
                )
            return QuantumCircuit.from_instructions(instructions)

        # qubit => {<subcircuit_i>: <qubit in subcircuit>}
        self.qubit_mapping = {qubit: {} for qubit in self.circuit.qubits}

        self.subcircuits = []
        subcircuit_i = 0
        subcircuit_qubits = set()
        subcircuit_ops = []

        for op_node in circuit_to_dag(self.unlabeled_circuit).topological_op_nodes():
            op_node = deepcopy(op_node)
            subcircuit_qubits |= set(op_node.qargs)
            if op_node.name == "cut":
                subcircuit = remap_qubits(
                    subcircuit_ops, subcircuit_qubits, subcircuit_i, self.qubit_mapping
                )
                self.subcircuits.append(subcircuit)

                subcircuit_i += 1
                subcircuit_qubits = set()
                subcircuit_ops = []
            else:
                subcircuit_ops.append(op_node)

        # create last subcircuit
        if subcircuit_ops:
            subcircuit = remap_qubits(
                subcircuit_ops, subcircuit_qubits, subcircuit_i, self.qubit_mapping
            )
            self.subcircuits.append(subcircuit)

        # book-keeping tasks
        self.populate_annotated_subcircuits()
        self.populate_compute_graph()
        self.populate_subcircuit_entries()

    def cut(
        self,
        max_subcircuit_width: int,
        max_cuts: int,
        num_subcircuits: list[int],
        max_subcircuit_cuts: int,
        subcircuit_size_imbalance: int,
        generate_subcircuits: bool = True,
    ):
        cut_edges_pairs = self.find_cuts(
            max_subcircuit_width=max_subcircuit_width,
            max_cuts=max_cuts,
            num_subcircuits=num_subcircuits,
            max_subcircuit_cuts=max_subcircuit_cuts,
            subcircuit_size_imbalance=subcircuit_size_imbalance,
        )

        self.add_cuts(
            cut_edges=cut_edges_pairs, generate_subcircuits=generate_subcircuits
        )

    def run_subcircuits(
        self,
        subcircuits: list[int] | None = None,
        backend: str = "statevector_simulator",
    ):
        subcircuits = subcircuits or range(len(self))
        for subcircuit in subcircuits:
            subcircuit_measured_probs = run_subcircuit_instances(
                subcircuit=self[subcircuit],
                subcircuit_instance_init_meas=self.subcircuit_instances[subcircuit],
                backend=backend,
            )
            self.subcircuit_entry_probs[subcircuit] = attribute_shots(
                subcircuit_measured_probs=subcircuit_measured_probs,
                subcircuit_entries=self.subcircuit_entries[subcircuit],
            )

    def compute_probabilities(self, mem_limit=None, recursion_depth=1):
        dd = DynamicDefinition(
            compute_graph=self.compute_graph,
            num_cuts=self.num_cuts,
            subcircuit_entry_probs=self.subcircuit_entry_probs,
            mem_limit=mem_limit,
            recursion_depth=recursion_depth,
        )

        self.approximation_bins = dd.dd_bins

    def postprocess(self) -> np.array:
        if not hasattr(self, "approximation_bins"):
            self.compute_probabilities()

        reconstructed_prob = read_dd_bins(
            subcircuit_out_qubits=self.reconstruction_qubit_order,
            dd_bins=self.approximation_bins,
        )
        reconstructed_prob = quasi_to_real(
            quasiprobability=reconstructed_prob, mode="nearest"
        )
        return reconstructed_prob

    def get_ground_truth(self, backend: str) -> np.ndarray:
        return evaluate_circ(circuit=self.raw_circuit, backend=backend)

    def verify(
        self,
        reconstructed_probabilities: np.ndarray | None = None,
        backend: str = "statevector_simulator",
        atol: float = 1e-10,
    ):
        reconstructed_probabilities = reconstructed_probabilities or self.postprocess()
        ground_truth = self.get_ground_truth(backend)

        approximation_error = (
            MSE(target=ground_truth, obs=reconstructed_probabilities)
            * 2**self.circuit.num_qubits
            / np.linalg.norm(ground_truth) ** 2
        )

        assert approximation_error < atol, (
            "Difference in cut circuit and uncut circuit is outside of floating point error tolerance"
        )

    def populate_compute_graph(self):
        """
        Generate the connection graph among subcircuits
        """
        annotated_subcircuits = self.annotated_subcircuits
        subcircuits = self.subcircuits

        self.compute_graph = ComputeGraph()
        for subcircuit_idx in annotated_subcircuits:
            subcircuit_attributes = deepcopy(annotated_subcircuits[subcircuit_idx])
            subcircuit_attributes["subcircuit"] = subcircuits[subcircuit_idx]
            self.compute_graph.add_node(
                subcircuit_idx=subcircuit_idx, attributes=subcircuit_attributes
            )

        for qubit, mappings in self.qubit_mapping.items():
            mappings_keys = list(mappings.keys())
            for upstream_subcircuit_index, downstream_subcircuit_index in zip(
                mappings_keys, mappings_keys[1:]
            ):
                self.compute_graph.add_edge(
                    u_for_edge=upstream_subcircuit_index,
                    v_for_edge=downstream_subcircuit_index,
                    attributes={
                        "O_qubit": mappings[upstream_subcircuit_index],
                        "rho_qubit": mappings[downstream_subcircuit_index],
                    },
                )

    def populate_subcircuit_entries(self):
        compute_graph = self.compute_graph

        subcircuit_entries = {}
        subcircuit_instances = {}

        for subcircuit_idx in compute_graph.nodes:
            bare_subcircuit = compute_graph.nodes[subcircuit_idx]["subcircuit"]
            subcircuit_entries[subcircuit_idx] = {}
            subcircuit_instances[subcircuit_idx] = []
            from_edges = compute_graph.get_edges(from_node=subcircuit_idx, to_node=None)
            to_edges = compute_graph.get_edges(from_node=None, to_node=subcircuit_idx)
            subcircuit_edges = from_edges + to_edges
            for subcircuit_edge_bases in itertools.product(
                ["I", "X", "Y", "Z"], repeat=len(subcircuit_edges)
            ):
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
                        subcircuit_entry_init[
                            bare_subcircuit.qubits.index(rho_qubit)
                        ] = edge_basis
                    else:
                        raise IndexError(
                            "Generating entries for a subcircuit. subcircuit_idx should be either upstream or downstream"
                        )

                subcircuit_entry_term = []
                for coeff, paulis in self.get_initializations(subcircuit_entry_init):
                    initializations_and_measurements = (
                        paulis,
                        tuple(subcircuit_entry_meas),
                    )
                    if (
                        initializations_and_measurements
                        not in subcircuit_instances[subcircuit_idx]
                    ):
                        subcircuit_instances[subcircuit_idx].append(
                            initializations_and_measurements
                        )
                    subcircuit_entry_term.append(
                        (coeff, initializations_and_measurements)
                    )

                subcircuit_entries[subcircuit_idx][
                    (tuple(subcircuit_entry_init), tuple(subcircuit_entry_meas))
                ] = subcircuit_entry_term

        self.subcircuit_entries, self.subcircuit_instances = (
            subcircuit_entries,
            subcircuit_instances,
        )
        self.subcircuit_entry_probs = {}

    def populate_annotated_subcircuits(self):
        self.annotated_subcircuits = {}
        for subcircuit_idx, subcircuit in enumerate(self.subcircuits):
            self.annotated_subcircuits[subcircuit_idx] = {
                "effective": subcircuit.num_qubits,
            }

        for mappings in self.qubit_mapping.values():
            subcircuit_indices = list(mappings.keys())
            # all but the last subcircuit index in subcircuit_indices
            # has their effective qubits reduce by 1
            for subcircuit_index in subcircuit_indices[:-1]:
                self.annotated_subcircuits[subcircuit_index]["effective"] -= 1

    def to_file(self, filepath: str | Path, *args, **kwargs) -> None:
        if isinstance(filepath, str):
            filepath = Path(filepath)

        # Keep imports local to this function
        from cutqc2.io.h5 import cut_circuit_to_h5

        supported_formats = {".h5": cut_circuit_to_h5}
        assert filepath.suffix in supported_formats, "Unsupported format"
        return supported_formats[filepath.suffix](self, filepath, *args, **kwargs)
