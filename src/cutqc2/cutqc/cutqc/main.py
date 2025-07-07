import subprocess, os, logging
from time import perf_counter
from qiskit.converters import circuit_to_dag
from cutqc2.cutqc.cutqc.cutter import find_cuts
from cutqc2.cutqc.cutqc.evaluator import run_subcircuit_instances, attribute_shots
from cutqc2.cutqc.cutqc.dynamic_definition import DynamicDefinition


class CutQC:
    """
    The main module for CutQC
    cut --> evaluate results --> verify (optional)
    """

    @staticmethod
    def check_valid(circuit):
        """
        If the input circuit is not fully connected, it does not need CutQC to be split into smaller circuits.
        CutQC hence only cuts a circuit if it is fully connected.
        Furthermore, CutQC only supports 2-qubit gates.
        """
        if circuit.num_unitary_factors() != 1:
            raise ValueError(
                f"Input circuit is not fully connected thus does not need cutting. Number of unitary factors = {circuit.num_unitary_factors()}"
            )
        if circuit.num_clbits > 0:
            raise ValueError(
                "Please remove classical bits from the circuit before cutting")
        dag = circuit_to_dag(circuit)
        for op_node in dag.topological_op_nodes():
            if len(op_node.qargs) > 2:
                raise ValueError(
                    "CutQC currently does not support >2-qubit gates")
            if op_node.op.name == "barrier":
                raise ValueError(
                    "Please remove barriers from the circuit before cutting")

    def __init__(self, circuit, cutter_constraints, verbose):
        """
        Args:
        circuit : the input quantum circuit
        cutter_constraints : cutting constraints to satisfy

        verbose: setting verbose to True to turn on logging information.
        Useful to visualize what happens,
        but may produce very long outputs for complicated circuits.
        """
        self.check_valid(circuit=circuit)
        self.circuit = circuit
        self.cutter_constraints = cutter_constraints
        self.verbose = verbose
        self.times = {}

    def cut(self) -> None:
        """
        Cut the given circuits
        If use the MIP solver to automatically find cuts, the following are required:
        max_subcircuit_width: max number of qubits in each subcircuit

        The following are optional:
        max_cuts: max total number of cuts allowed
        num_subcircuits: list of subcircuits to try, CutQC returns the best solution found among the trials
        max_subcircuit_cuts: max number of cuts for a subcircuit
        max_subcircuit_size: max number of gates in a subcircuit
        quantum_cost_weight: quantum_cost_weight : MIP overall cost objective is given by
        quantum_cost_weight * num_subcircuit_instances + (1-quantum_cost_weight) * classical_postprocessing_cost

        Else supply the subcircuit_vertices manually
        Note that supplying subcircuit_vertices overrides all other arguments
        """
        cutter_begin = perf_counter()

        self.cut_solution = find_cuts(
            **self.cutter_constraints, circuit=self.circuit, verbose=self.verbose, raise_error=True
        )
        self.compute_graph, self.subcircuit_entries, self.subcircuit_instances = self.cut_solution.compute_graph, self.cut_solution.subcircuit_entries, self.cut_solution.subcircuit_instances
        self.times["cutter"] = perf_counter() - cutter_begin


    def evaluate(self, num_shots_fn):
        """
        num_shots_fn: a function that gives the number of shots to take for a given circuit
        """
        evaluate_begin = perf_counter()
        self.subcircuit_entry_probs = {}
        for subcircuit_index in range(len(self.cut_solution)):
            subcircuit_measured_probs = run_subcircuit_instances(
                subcircuit=self.cut_solution[subcircuit_index],
                subcircuit_instance_init_meas=self.subcircuit_instances[
                    subcircuit_index
                ],
                num_shots_fn=num_shots_fn,
            )
            self.subcircuit_entry_probs[subcircuit_index] = attribute_shots(
                subcircuit_measured_probs=subcircuit_measured_probs,
                subcircuit_entries=self.subcircuit_entries[subcircuit_index],
            )
        eval_time = perf_counter() - evaluate_begin
        self.times["evaluate"] = eval_time

    def build(self, mem_limit, recursion_depth):
        dd = DynamicDefinition(
            compute_graph=self.compute_graph,
            num_cuts=self.cut_solution.num_cuts,
            subcircuit_entry_probs=self.subcircuit_entry_probs,
            mem_limit=mem_limit,
            recursion_depth=recursion_depth,
        )
        dd.build()

        self.approximation_bins = dd.dd_bins
        self.num_recursions = len(self.approximation_bins)

    def verify(self):
        reconstructed_prob, self.approximation_error = self.cut_solution.full_verify(
            dd_bins=self.approximation_bins,
        )

