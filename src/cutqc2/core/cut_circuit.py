from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.circuit.quantumregister import Qubit
from qiskit.circuit.quantumcircuitdata import CircuitInstruction


class WireCutGate(UnitaryGate):
    """
    Custom gate to represent a wire cut in a quantum circuit.
    """

    def __init__(self):
        super().__init__(data=[[1, 0], [0, 1]], num_qubits=1, label="✂️")
        # The super constructor initializes name as "unitary" - use our own
        self.name = "cut"


class CutCircuit:
    def __init__(
        self,
        circuit: QuantumCircuit,
        cut_qubits_and_positions: list[tuple[Qubit, int]] | None = None,
    ):
        self.circuit = circuit
        for cut_qubit_and_position in cut_qubits_and_positions or []:
            self.add_cut(cut_qubit_and_position)

    def __str__(self):
        return str(self.circuit)

    def add_cut(self, cut_qubit_and_position: tuple[Qubit, int]) -> QuantumCircuit:
        """
        Add a cut to the circuit at the specified position.
        Args:
            cut_qubit_and_position: A tuple containing the Qubit to cut and the position
                                    in the wire where the cut should be made.
                                    The position is a 0-indexed integer indicating the gate position
                                    on the wire 'after' which the cut should be made.
                                    This tuple format is what legacy CutQC code mostly uses.
        Returns:
            QuantumCircuit: The modified circuit with the cut added.
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
