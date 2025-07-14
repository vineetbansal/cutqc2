from copy import deepcopy
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.circuit import Qubit, QuantumRegister, CircuitInstruction
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit.dagcircuit import DAGOpNode, DAGCircuit
from cutqc2.core.dag import DagNode, DAGEdge
from cutqc2.cutqc.cutqc.cut_solution import CutSolution


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
        cut_qubits_and_positions: list[tuple[Qubit, int]] | None = None,
        add_labels: bool = True,
    ):
        self.raw_circuit = circuit
        if add_labels:
            self.circuit = self.get_labeled_circuit(circuit.copy())
        else:
            self.circuit = circuit.copy()

        self.subcircuits = []
        for cut_qubit_and_position in cut_qubits_and_positions or []:
            self.add_cut(cut_qubit_and_position)

        self.inter_wire_dag = self.get_inter_wire_dag(self.circuit)
        self.inter_wire_dag_metadata = self.get_dag_metadata(self.inter_wire_dag)

    def __str__(self):
        return str(self.circuit)

    def __len__(self):
        return len(self.subcircuits)

    def __iter__(self):
        return iter(self.subcircuits)

    def __getitem__(self, item):
        return self.subcircuits[item]

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
        node_label_ids = {}
        dag_edge_to_id = {}
        id_node_names = {}
        id_node_labels = {}
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
                    register_name=arg0._register.name,
                    wire_index=arg0._index,
                    gate_index=qubit_gate_counter[arg0],
                ),
                DagNode(
                    register_name=arg1._register.name,
                    wire_index=arg1._index,
                    gate_index=qubit_gate_counter[arg1],
                ),
                name=vertex.label,
            )
            vertex_name = str(dag_edge)

            qubit_gate_counter[arg0] += 1
            qubit_gate_counter[arg1] += 1
            if vertex_name not in node_name_ids and id(vertex) not in vertex_ids:
                node_name_ids[vertex_name] = curr_node_id
                node_label_ids[vertex.label] = curr_node_id
                dag_edge_to_id[dag_edge] = curr_node_id
                id_node_names[curr_node_id] = vertex_name
                id_node_labels[curr_node_id] = vertex.label
                id_to_dag_edge[curr_node_id] = dag_edge
                vertex_ids[id(vertex)] = curr_node_id

                curr_node_id += 1

        for u, v, _ in dag.edges():
            if isinstance(u, DAGOpNode) and isinstance(v, DAGOpNode):
                u_id = vertex_ids[id(u)]
                v_id = vertex_ids[id(v)]
                edges.append((u_id, v_id))

        n_vertices = dag.size()

        return {
            "n_vertices": n_vertices,
            "edges": edges,
            "vertex_ids": node_name_ids,
            "id_vertices": id_node_names,
            "dag_edge_to_id": dag_edge_to_id,
            "id_to_dag_edge": id_to_dag_edge,
            "vertex_label_ids": node_label_ids,
            "id_vertex_labels": id_node_labels,
        }

    def add_cut(self, cut_qubit_and_position: tuple[Qubit, int]):
        """
        Add a cut to the circuit at the specified position.
        Args:
            cut_qubit_and_position: A tuple containing the Qubit to cut and the position
                                    in the wire where the cut should be made.
                                    The position is a 0-indexed integer indicating the gate position
                                    on the wire 'after' which the cut should be made.
                                    This tuple format is what legacy CutQC code mostly uses.
        """
        cut_qubit, cut_position = cut_qubit_and_position
        cut_instr = CircuitInstruction(WireCutGate(), qubits=(cut_qubit,))

        cut_wire_position = 0
        for i, instr in enumerate(self.circuit.data):
            if cut_qubit in instr.qubits:  # we're on the right wire
                if cut_wire_position > cut_position:
                    self.circuit.data.insert(i, cut_instr)
                    break
                cut_wire_position += 1

    def add_cut_at_label(self, label: str):
        """
        Add a cut to the circuit at the position of the instruction with the specified label.
        Args:
            label: The label of the instruction after which the cut should be made.
        """
        for i, instr in enumerate(self.circuit.data):
            if instr.operation.label == label:
                cut_qubit = instr.qubits[0]
                cut_instr = CircuitInstruction(WireCutGate(), qubits=(cut_qubit,))
                # insert the cut instruction right after the current instruction
                self.circuit.data.insert(i + 1, cut_instr)
                break

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
            self.add_cut_at_label(edge0.name)

        if generate_subcircuits:
            self.generate_subcircuits()

        # TODO: The legacy format for `complete_path_map` is a bit convoluted.
        # We should simply stick with `self.qubit_mapping` which is simpler.
        complete_path_map = {
            k: [{"subcircuit_idx": _k, "subcircuit_qubit": _v} for _k, _v in v.items()]
            for k, v in self.qubit_mapping.items()
        }

        self.cut_solution = CutSolution(
            circuit=self.raw_circuit,
            subcircuits=self.subcircuits,
            complete_path_map=complete_path_map,
            num_cuts=len(cut_edges),
        )

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

        for op_node in circuit_to_dag(self.circuit).topological_op_nodes():
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
