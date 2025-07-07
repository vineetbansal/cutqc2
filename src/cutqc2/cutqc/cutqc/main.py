import subprocess, os, logging
from time import perf_counter
from qiskit.converters import circuit_to_dag
from cutqc2.cutqc.cutqc.cutter import find_cuts
from cutqc2.cutqc.cutqc.evaluator import run_subcircuit_instances, attribute_shots
from cutqc2.cutqc.cutqc.dynamic_definition import DynamicDefinition


class CutQC:
    @staticmethod
    def check_valid(circuit):
        unitary_factors = circuit.num_unitary_factors()
        assert unitary_factors == 1, f"Input circuit is not fully connected thus does not need cutting. Number of unitary factors = {unitary_factors}"

        assert circuit.num_clbits == 0, "Please remove classical bits from the circuit before cutting"
        dag = circuit_to_dag(circuit)
        for op_node in dag.topological_op_nodes():
            assert len(op_node.qargs) <= 2, "CutQC currently does not support >2-qubit gates"
            assert op_node.op.name != "barrier", "Please remove barriers from the circuit before cutting"

    def __init__(self, circuit, cutter_constraints, verbose):
        self.check_valid(circuit=circuit)
        self.circuit = circuit
        self.cutter_constraints = cutter_constraints
        self.verbose = verbose

    def cut(self) -> None:
        self.cut_solution = find_cuts(
            **self.cutter_constraints, circuit=self.circuit, verbose=self.verbose, raise_error=True
        )
        self.compute_graph, self.subcircuit_entries, self.subcircuit_instances = self.cut_solution.compute_graph, self.cut_solution.subcircuit_entries, self.cut_solution.subcircuit_instances


    def evaluate(self, num_shots_fn=None):
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

