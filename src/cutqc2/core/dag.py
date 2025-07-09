from itertools import product


class DagNode:
    """
    Represents a node in a quantum circuit DAG (Directed Acyclic Graph),
    corresponding to a specific gate on a specific wire (qubit/register).

    Attributes:
        wire_index (int): The index of the wire (qubit/register).
        gate_index (int): The index of the gate on the wire.
        register_name (str): The name of the register (default 'q').
    """

    def __init__(self, wire_index: int, gate_index: int, register_name: str = "q"):
        """
        Initialize a DagNode.

        Args:
            wire_index (int): The index of the wire (qubit/register).
            gate_index (int): The index of the gate on the wire.
            register_name (str, optional): The name of the register. Defaults to 'q'.
        """
        self.wire_index = wire_index
        self.gate_index = gate_index
        self.register_name = register_name

    def __str__(self):
        """
        Return a string representation of the DagNode.

        Returns:
            str: The string in the format 'register_name[wire_index]gate_index'.
        """
        return "%s[%d]%d" % (self.register_name, self.wire_index, self.gate_index)

    def __lt__(self, other: "DagNode"):
        """
        Compare two DagNodes for ordering, first by wire_index, then by gate_index.

        Args:
            other (DagNode): The other DagNode to compare to.

        Returns:
            bool: True if this node is less than the other node.
        """
        if self.wire_index < other.wire_index:
            return True
        elif self.wire_index == other.wire_index:
            return self.gate_index < other.gate_index
        return False

    def __sub__(self, other):
        """
        Subtract two DagNodes on the same wire to get the difference in gate indices.

        Args:
            other (DagNode): The other DagNode to subtract.

        Returns:
            int: The difference in gate indices.

        Raises:
            ValueError: If the nodes are on different wires.
        """
        if self.wire_index != other.wire_index:
            raise ValueError("Cannot subtract nodes on different wires")
        return self.gate_index - other.gate_index


class DAGEdge:
    """
    Represents an edge in a quantum circuit DAG, connecting two DagNodes.

    Attributes:
        source (DagNode): The source node of the edge.
        dest (DagNode): The destination node of the edge.
    """

    def __init__(self, first: DagNode, second: DagNode):
        """
        Initialize a DAGEdge between two DagNodes.

        Args:
            first (DagNode): The first node.
            second (DagNode): The second node.
        """
        self.source, self.dest = sorted((first, second))

    def __str__(self):
        """
        Return a string representation of the DAGEdge.

        Returns:
            str: The string in the format 'source dest'.
        """
        return "%s %s" % (self.source, self.dest)

    def __or__(self, other: "DAGEdge") -> tuple[DagNode, DagNode]:
        """
        Find the pair of DagNodes (one from each edge) that share the same wire index.

        Args:
            other (DAGEdge): The other DAGEdge to compare with.

        Returns:
            tuple[DagNode, DagNode]: The pair of nodes with the same wire index, sorted.

        Raises:
            ValueError: If there is no common wire between the two edges.
        """
        for a, b in product((self.source, self.dest), (other.source, other.dest)):
            if a.wire_index == b.wire_index:
                return tuple(sorted((a, b)))

        raise ValueError("No common wire")
