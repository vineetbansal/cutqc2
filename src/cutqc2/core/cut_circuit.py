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
        add_labels: bool = True,
    ):
        if add_labels:
            self.circuit = self.get_labeled_circuit(circuit.copy())
        else:
            self.circuit = circuit.copy()

        for cut_qubit_and_position in cut_qubits_and_positions or []:
            self.add_cut(cut_qubit_and_position)

    def __str__(self):
        return str(self.circuit)

    @staticmethod
    def get_labeled_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
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
