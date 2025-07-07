import math
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.dagcircuit import DAGOpNode
from qiskit.converters import circuit_to_dag, dag_to_circuit
import gurobipy as gp
from qiskit import QuantumCircuit, QuantumRegister
from cutqc2.cutqc.cutqc.cut_solution import CutSolution


class MIP_Model(object):
    def __init__(
        self,
        n_vertices,
        edges,
        vertex_ids,
        id_vertices,
        num_subcircuit,
        max_subcircuit_width,
        max_subcircuit_cuts,
        subcircuit_size_imbalance,
        num_qubits,
        max_cuts,
    ):
        self.check_graph(n_vertices, edges)
        self.n_vertices = n_vertices
        self.edges = edges
        self.n_edges = len(edges)
        self.vertex_ids = vertex_ids
        self.id_vertices = id_vertices
        self.num_subcircuit = num_subcircuit
        self.max_subcircuit_width = max_subcircuit_width
        self.max_subcircuit_cuts = max_subcircuit_cuts
        self.subcircuit_size_imbalance = math.sqrt(subcircuit_size_imbalance)
        self.num_qubits = num_qubits
        self.max_cuts = max_cuts

        """
        Count the number of input qubits directly connected to each node
        """
        self.vertex_weight = {}
        for node in self.vertex_ids:
            qargs = node.split(" ")
            num_in_qubits = 0
            for qarg in qargs:
                if int(qarg.split("]")[1]) == 0:
                    num_in_qubits += 1
            self.vertex_weight[node] = num_in_qubits

        self.model = gp.Model(name="cut_searching")
        self.model.params.OutputFlag = 0
        self._add_variables()
        self._add_constraints()

    def _add_variables(self):
        """
        Indicate if a vertex is in some subcircuit
        """
        self.vertex_var = []
        for i in range(self.num_subcircuit):
            subcircuit_y = []
            for j in range(self.n_vertices):
                j_in_i = self.model.addVar(lb=0.0, ub=1.0, vtype=gp.GRB.BINARY)
                subcircuit_y.append(j_in_i)
            self.vertex_var.append(subcircuit_y)

        """
        Indicate if an edge has one and only one vertex in some subcircuit
        """
        self.edge_var = []
        for i in range(self.num_subcircuit):
            subcircuit_x = []
            for j in range(self.n_edges):
                v = self.model.addVar(lb=0.0, ub=1.0, vtype=gp.GRB.BINARY)
                subcircuit_x.append(v)
            self.edge_var.append(subcircuit_x)

        """
        Total number of cuts
        add 0.1 for numerical stability
        """
        self.num_cuts = self.model.addVar(
            lb=0, ub=self.max_cuts + 0.1, vtype=gp.GRB.INTEGER, name="num_cuts"
        )

        self.subcircuit_counter = {}
        for subcircuit in range(self.num_subcircuit):
            self.subcircuit_counter[subcircuit] = {}

            self.subcircuit_counter[subcircuit]["original_input"] = self.model.addVar(
                lb=0.1,
                ub=self.max_subcircuit_width,
                vtype=gp.GRB.INTEGER,
                name="original_input_%d" % subcircuit,
            )
            self.subcircuit_counter[subcircuit]["rho"] = self.model.addVar(
                lb=0,
                ub=self.max_subcircuit_width,
                vtype=gp.GRB.INTEGER,
                name="rho_%d" % subcircuit,
            )
            self.subcircuit_counter[subcircuit]["O"] = self.model.addVar(
                lb=0,
                ub=self.max_subcircuit_width,
                vtype=gp.GRB.INTEGER,
                name="O_%d" % subcircuit,
            )
            self.subcircuit_counter[subcircuit]["d"] = self.model.addVar(
                lb=0.1,
                ub=self.max_subcircuit_width,
                vtype=gp.GRB.INTEGER,
                name="d_%d" % subcircuit,
            )
            self.subcircuit_counter[subcircuit]["size"] = self.model.addVar(
                lb=self.n_vertices
                / self.num_subcircuit
                / self.subcircuit_size_imbalance,
                ub=self.n_vertices
                / self.num_subcircuit
                * self.subcircuit_size_imbalance,
                vtype=gp.GRB.INTEGER,
                name="size_%d" % subcircuit,
            )
            if self.max_subcircuit_cuts is not None:
                self.subcircuit_counter[subcircuit]["num_cuts"] = self.model.addVar(
                    lb=0.1,
                    ub=self.max_subcircuit_cuts,
                    vtype=gp.GRB.INTEGER,
                    name="num_cuts_%d" % subcircuit,
                )
        self.model.update()

    def _add_constraints(self):
        """
        each vertex in exactly one subcircuit
        """
        for v in range(self.n_vertices):
            self.model.addConstr(
                gp.quicksum(
                    [self.vertex_var[i][v] for i in range(self.num_subcircuit)]
                ),
                gp.GRB.EQUAL,
                1,
            )

        """
        edge_var=1 indicates one and only one vertex of an edge is in subcircuit
        edge_var[subcircuit][edge] = vertex_var[subcircuit][u] XOR vertex_var[subcircuit][v]
        """
        for i in range(self.num_subcircuit):
            for e in range(self.n_edges):
                u, v = self.edges[e]
                u_vertex_var = self.vertex_var[i][u]
                v_vertex_var = self.vertex_var[i][v]
                self.model.addConstr(self.edge_var[i][e] <= u_vertex_var + v_vertex_var)
                self.model.addConstr(self.edge_var[i][e] >= u_vertex_var - v_vertex_var)
                self.model.addConstr(self.edge_var[i][e] >= v_vertex_var - u_vertex_var)
                self.model.addConstr(
                    self.edge_var[i][e] <= 2 - u_vertex_var - v_vertex_var
                )

        """
        Symmetry-breaking constraints
        Force small-numbered vertices into small-numbered subcircuits:
            v0: in subcircuit 0
            v1: in subcircuit_0 or subcircuit_1
            v2: in subcircuit_0 or subcircuit_1 or subcircuit_2
            ...
        """
        for vertex in range(self.num_subcircuit):
            self.model.addConstr(
                gp.quicksum(
                    [
                        self.vertex_var[subcircuit][vertex]
                        for subcircuit in range(vertex + 1)
                    ]
                )
                == 1
            )

        """
        Compute number of cuts
        """
        self.model.addConstr(
            self.num_cuts
            == gp.quicksum(
                [
                    self.edge_var[subcircuit][i]
                    for i in range(self.n_edges)
                    for subcircuit in range(self.num_subcircuit)
                ]
            )
            / 2
        )

        for subcircuit in range(self.num_subcircuit):
            """
            Compute number of different types of qubit in a subcircuit
            """
            self.model.addConstr(
                self.subcircuit_counter[subcircuit]["original_input"]
                == gp.quicksum(
                    [
                        self.vertex_weight[self.id_vertices[i]]
                        * self.vertex_var[subcircuit][i]
                        for i in range(self.n_vertices)
                    ]
                )
            )

            self.model.addConstr(
                self.subcircuit_counter[subcircuit]["rho"]
                == gp.quicksum(
                    [
                        self.edge_var[subcircuit][i]
                        * self.vertex_var[subcircuit][self.edges[i][1]]
                        for i in range(self.n_edges)
                    ]
                )
            )

            self.model.addConstr(
                self.subcircuit_counter[subcircuit]["O"]
                == gp.quicksum(
                    [
                        self.edge_var[subcircuit][i]
                        * self.vertex_var[subcircuit][self.edges[i][0]]
                        for i in range(self.n_edges)
                    ]
                )
            )

            self.model.addConstr(
                self.subcircuit_counter[subcircuit]["d"]
                == self.subcircuit_counter[subcircuit]["original_input"]
                + self.subcircuit_counter[subcircuit]["rho"]
            )

            self.model.addConstr(
                self.subcircuit_counter[subcircuit]["size"]
                == gp.quicksum(
                    [self.vertex_var[subcircuit][v] for v in range(self.n_vertices)]
                )
            )

            if self.max_subcircuit_cuts is not None:
                self.model.addConstr(
                    self.subcircuit_counter[subcircuit]["num_cuts"]
                    == self.subcircuit_counter[subcircuit]["rho"]
                    + self.subcircuit_counter[subcircuit]["O"]
                )

        self.model.setObjective(self.num_cuts, gp.GRB.MINIMIZE)
        self.model.update()

    def check_graph(self, n_vertices, edges):
        # 1. edges must include all vertices
        # 2. all u,v must be ordered and smaller than n_vertices
        vertices = set([i for (i, _) in edges])
        vertices |= set([i for (_, i) in edges])
        assert vertices == set(range(n_vertices))
        for u, v in edges:
            assert u < v
            assert u < n_vertices

    def solve(self):
        self.model.params.threads = 48
        self.model.Params.TimeLimit = 30
        self.model.optimize()

        if self.model.solcount > 0:
            self.objective = None
            self.subcircuits = []
            self.optimal = self.model.Status == gp.GRB.OPTIMAL
            self.runtime = self.model.Runtime
            self.node_count = self.model.nodecount
            self.mip_gap = self.model.mipgap
            self.objective = self.model.ObjVal

            for i in range(self.num_subcircuit):
                subcircuit = []
                for j in range(self.n_vertices):
                    if abs(self.vertex_var[i][j].x) > 1e-4:
                        subcircuit.append(self.id_vertices[j])
                self.subcircuits.append(subcircuit)
            assert (
                sum([len(subcircuit) for subcircuit in self.subcircuits])
                == self.n_vertices
            )

            cut_edges_idx = []
            cut_edges = []
            for i in range(self.num_subcircuit):
                for j in range(self.n_edges):
                    if abs(self.edge_var[i][j].x) > 1e-4 and j not in cut_edges_idx:
                        cut_edges_idx.append(j)
                        u, v = self.edges[j]
                        cut_edges.append((self.id_vertices[u], self.id_vertices[v]))
            self.cut_edges = cut_edges
            return True
        else:
            return False


def read_circ(circuit):
    dag = circuit_to_dag(circuit)
    edges = []
    node_name_ids = {}
    id_node_names = {}
    vertex_ids = {}
    curr_node_id = 0
    qubit_gate_counter = {}
    for qubit in dag.qubits:
        qubit_gate_counter[qubit] = 0
    for vertex in dag.topological_op_nodes():
        if len(vertex.qargs) != 2:
            raise Exception("vertex does not have 2 qargs!")

        arg0, arg1 = vertex.qargs

        vertex_name = "%s[%d]%d %s[%d]%d" % (
            arg0._register.name,
            arg0._index,
            qubit_gate_counter[arg0],
            arg1._register.name,
            arg1._index,
            qubit_gate_counter[arg1],
        )
        qubit_gate_counter[arg0] += 1
        qubit_gate_counter[arg1] += 1
        if vertex_name not in node_name_ids and id(vertex) not in vertex_ids:
            node_name_ids[vertex_name] = curr_node_id
            id_node_names[curr_node_id] = vertex_name
            vertex_ids[id(vertex)] = curr_node_id
            curr_node_id += 1

    for u, v, _ in dag.edges():
        if isinstance(u, DAGOpNode) and isinstance(v, DAGOpNode):
            u_id = vertex_ids[id(u)]
            v_id = vertex_ids[id(v)]
            edges.append((u_id, v_id))

    n_vertices = dag.size()
    return n_vertices, edges, node_name_ids, id_node_names


def cuts_parser(cuts, circ):
    dag = circuit_to_dag(circ)
    positions = []
    for position in cuts:
        source, dest = position
        source_qargs = [
            (x.split("]")[0] + "]", int(x.split("]")[1])) for x in source.split(" ")
        ]
        dest_qargs = [
            (x.split("]")[0] + "]", int(x.split("]")[1])) for x in dest.split(" ")
        ]
        qubit_cut = []
        for source_qarg in source_qargs:
            source_qubit, source_multi_Q_gate_idx = source_qarg
            for dest_qarg in dest_qargs:
                dest_qubit, dest_multi_Q_gate_idx = dest_qarg
                if (
                    source_qubit == dest_qubit
                    and dest_multi_Q_gate_idx == source_multi_Q_gate_idx + 1
                ):
                    qubit_cut.append(source_qubit)
        # if len(qubit_cut)>1:
        #     raise Exception('one cut is cutting on multiple qubits')
        for x in source.split(" "):
            if x.split("]")[0] + "]" == qubit_cut[0]:
                source_idx = int(x.split("]")[1])
        for x in dest.split(" "):
            if x.split("]")[0] + "]" == qubit_cut[0]:
                dest_idx = int(x.split("]")[1])
        multi_Q_gate_idx = max(source_idx, dest_idx)

        wire = None
        for qubit in circ.qubits:
            if qubit._register.name == qubit_cut[0].split("[")[
                0
            ] and qubit._index == int(qubit_cut[0].split("[")[1].split("]")[0]):
                wire = qubit
        tmp = 0
        all_Q_gate_idx = None
        for gate_idx, gate in enumerate(
            list(dag.nodes_on_wire(wire=wire, only_ops=True))
        ):
            if len(gate.qargs) > 1:
                tmp += 1
                if tmp == multi_Q_gate_idx:
                    all_Q_gate_idx = gate_idx
        positions.append((wire, all_Q_gate_idx))
    positions = sorted(positions, reverse=True, key=lambda cut: cut[1])
    return positions


def subcircuits_parser(subcircuit_gates, circuit):
    """
    Assign the single qubit gates to the closest two-qubit gates

    Returns:
    complete_path_map[input circuit qubit] = [{subcircuit_idx,subcircuit_qubit}]
    """

    def calculate_distance_between_gate(gate_A, gate_B):
        if len(gate_A.split(" ")) >= len(gate_B.split(" ")):
            tmp_gate = gate_A
            gate_A = gate_B
            gate_B = tmp_gate
        distance = float("inf")
        for qarg_A in gate_A.split(" "):
            qubit_A = qarg_A.split("]")[0] + "]"
            qgate_A = int(qarg_A.split("]")[-1])
            for qarg_B in gate_B.split(" "):
                qubit_B = qarg_B.split("]")[0] + "]"
                qgate_B = int(qarg_B.split("]")[-1])
                if qubit_A == qubit_B:
                    distance = min(distance, abs(qgate_B - qgate_A))
        return distance

    dag = circuit_to_dag(circuit)
    qubit_allGate_depths = {x: 0 for x in circuit.qubits}
    qubit_2qGate_depths = {x: 0 for x in circuit.qubits}
    gate_depth_encodings = {}
    for op_node in dag.topological_op_nodes():
        gate_depth_encoding = ""
        for qarg in op_node.qargs:
            gate_depth_encoding += "%s[%d]%d " % (
                qarg._register.name,
                qarg._index,
                qubit_allGate_depths[qarg],
            )
        gate_depth_encoding = gate_depth_encoding[:-1]
        gate_depth_encodings[op_node] = gate_depth_encoding
        for qarg in op_node.qargs:
            qubit_allGate_depths[qarg] += 1
        if len(op_node.qargs) == 2:
            MIP_gate_depth_encoding = ""
            for qarg in op_node.qargs:
                MIP_gate_depth_encoding += "%s[%d]%d " % (
                    qarg._register.name,
                    qarg._index,
                    qubit_2qGate_depths[qarg],
                )
                qubit_2qGate_depths[qarg] += 1
            MIP_gate_depth_encoding = MIP_gate_depth_encoding[:-1]
            for subcircuit_idx in range(len(subcircuit_gates)):
                for gate_idx in range(len(subcircuit_gates[subcircuit_idx])):
                    if (
                        subcircuit_gates[subcircuit_idx][gate_idx]
                        == MIP_gate_depth_encoding
                    ):
                        subcircuit_gates[subcircuit_idx][gate_idx] = gate_depth_encoding
                        break

    subcircuit_op_nodes = {x: [] for x in range(len(subcircuit_gates))}
    subcircuit_sizes = [0 for x in range(len(subcircuit_gates))]
    complete_path_map = {}
    for circuit_qubit in dag.qubits:
        complete_path_map[circuit_qubit] = []
        qubit_ops = dag.nodes_on_wire(wire=circuit_qubit, only_ops=True)
        for qubit_op_idx, qubit_op in enumerate(qubit_ops):
            gate_depth_encoding = gate_depth_encodings[qubit_op]
            nearest_subcircuit_idx = -1
            min_distance = float("inf")
            for subcircuit_idx in range(len(subcircuit_gates)):
                distance = float("inf")
                for gate in subcircuit_gates[subcircuit_idx]:
                    if len(gate.split(" ")) == 1:
                        # Do not compare against single qubit gates
                        continue
                    else:
                        distance = min(
                            distance,
                            calculate_distance_between_gate(
                                gate_A=gate_depth_encoding, gate_B=gate
                            ),
                        )
                if distance < min_distance:
                    min_distance = distance
                    nearest_subcircuit_idx = subcircuit_idx
            assert nearest_subcircuit_idx != -1
            path_element = {
                "subcircuit_idx": nearest_subcircuit_idx,
                "subcircuit_qubit": subcircuit_sizes[nearest_subcircuit_idx],
            }
            if (
                len(complete_path_map[circuit_qubit]) == 0
                or nearest_subcircuit_idx
                != complete_path_map[circuit_qubit][-1]["subcircuit_idx"]
            ):
                complete_path_map[circuit_qubit].append(path_element)
                subcircuit_sizes[nearest_subcircuit_idx] += 1

            subcircuit_op_nodes[nearest_subcircuit_idx].append(qubit_op)
    for circuit_qubit in complete_path_map:
        for path_element in complete_path_map[circuit_qubit]:
            path_element_qubit = QuantumRegister(
                size=subcircuit_sizes[path_element["subcircuit_idx"]], name="q"
            )[path_element["subcircuit_qubit"]]
            path_element["subcircuit_qubit"] = path_element_qubit
    subcircuits = generate_subcircuits(
        subcircuit_op_nodes=subcircuit_op_nodes,
        complete_path_map=complete_path_map,
        subcircuit_sizes=subcircuit_sizes,
        dag=dag,
    )
    return subcircuits, complete_path_map


def generate_subcircuits(subcircuit_op_nodes, complete_path_map, subcircuit_sizes, dag):
    qubit_pointers = {x: 0 for x in complete_path_map}
    subcircuits = [QuantumCircuit(x, name="q") for x in subcircuit_sizes]
    for op_node in dag.topological_op_nodes():
        subcircuit_idx = list(
            filter(
                lambda x: op_node in subcircuit_op_nodes[x], subcircuit_op_nodes.keys()
            )
        )
        assert len(subcircuit_idx) == 1
        subcircuit_idx = subcircuit_idx[0]
        subcircuit_qargs = []
        for op_node_qarg in op_node.qargs:
            if (
                complete_path_map[op_node_qarg][qubit_pointers[op_node_qarg]][
                    "subcircuit_idx"
                ]
                != subcircuit_idx
            ):
                qubit_pointers[op_node_qarg] += 1
            path_element = complete_path_map[op_node_qarg][qubit_pointers[op_node_qarg]]
            assert path_element["subcircuit_idx"] == subcircuit_idx
            subcircuit_qargs.append(path_element["subcircuit_qubit"])
        subcircuits[subcircuit_idx].append(
            instruction=op_node.op, qargs=subcircuit_qargs, cargs=None
        )
    return subcircuits


def circuit_stripping(circuit):
    # Remove all single qubit gates and barriers in the circuit
    dag = circuit_to_dag(circuit)
    stripped_dag = DAGCircuit()
    [stripped_dag.add_qreg(x) for x in circuit.qregs]
    for vertex in dag.topological_op_nodes():
        if len(vertex.qargs) == 2 and vertex.op.name != "barrier":
            stripped_dag.apply_operation_back(op=vertex.op, qargs=vertex.qargs)
    return dag_to_circuit(stripped_dag)


def find_cuts(
    circuit,
    max_subcircuit_width,
    max_cuts,
    num_subcircuits,
    max_subcircuit_cuts,
    subcircuit_size_imbalance,
    verbose,
    raise_error: bool = False,
) -> CutSolution | None:
    stripped_circ = circuit_stripping(circuit=circuit)
    n_vertices, edges, vertex_ids, id_vertices = read_circ(circuit=stripped_circ)
    num_qubits = circuit.num_qubits
    cut_solution = {}

    for num_subcircuit in num_subcircuits:
        if (
            num_subcircuit * max_subcircuit_width - (num_subcircuit - 1) < num_qubits
            or num_subcircuit > num_qubits
            or max_cuts + 1 < num_subcircuit
        ):
            continue
        kwargs = dict(
            n_vertices=n_vertices,
            edges=edges,
            vertex_ids=vertex_ids,
            id_vertices=id_vertices,
            num_subcircuit=num_subcircuit,
            max_subcircuit_width=max_subcircuit_width,
            max_subcircuit_cuts=max_subcircuit_cuts,
            subcircuit_size_imbalance=subcircuit_size_imbalance,
            num_qubits=num_qubits,
            max_cuts=max_cuts,
        )

        mip_model = MIP_Model(**kwargs)
        feasible = mip_model.solve()
        if feasible:
            positions = cuts_parser(mip_model.cut_edges, circuit)
            subcircuits, complete_path_map = subcircuits_parser(
                subcircuit_gates=mip_model.subcircuits, circuit=circuit
            )

            cut_solution = CutSolution(
                circuit=circuit,
                subcircuits=subcircuits,
                complete_path_map=complete_path_map,
                num_cuts=len(positions)
            )
            return cut_solution

    if raise_error:
        raise RuntimeError("No viable cuts found")
    return None
