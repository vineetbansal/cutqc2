class ComputeGraph:
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
