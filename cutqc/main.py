from time import perf_counter

from cutqc.cutqc_model import CutQCModel

from cutqc.cutter import find_cuts
from cutqc.evaluator import (
    run_subcircuit_instances,
    attribute_shots,
)

from cutqc.post_process_helper import (
    generate_subcircuit_entries,
    generate_compute_graph,
)


__host_machine__ = 0


class CircuitCutter:
    """
    The main module for CutQC
    cut --> evaluate results --> verify (optional)
    """

    def __init__(
        self,
        name=None,
        circuit=None,
        cutter_constraints=None,
        verbose=False,
    ):
        """
        Args:
        name: name of the input quantum circuit
        circuit: the input quantum circuit
        cutter_constraints: cutting constraints to satisfy

        verbose: setting verbose to True to turn on logging information.
                 Useful to visualize what happens,
                 but may produce very long outputs for complicated circuits.

        """
        self.name = name
        self.circuit = circuit
        self.cutter_constraints = cutter_constraints
        self.verbose = verbose
        self.times = {}

        self.compute_graph = None
        self.tmp_data_folder = None
        self.num_cuts = None
        self.complete_path_map = None
        self.subcircuits = None

    def cut(self):
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
        if self.verbose:
            print("*" * 20, "Cut %s" % self.name, "*" * 20)
            print(
                "width = %d depth = %d size = %d -->"
                % (
                    self.circuit.num_qubits,
                    self.circuit.depth(),
                    self.circuit.num_nonlocal_gates(),
                )
            )
            print(self.cutter_constraints)
        cutter_begin = perf_counter()
        cut_solution = find_cuts(
            **self.cutter_constraints, circuit=self.circuit, verbose=self.verbose
        )
        for field in cut_solution:
            self.__setattr__(field, cut_solution[field])

        if "complete_path_map" in cut_solution:
            self.has_solution = True
            self._generate_metadata()
        else:
            self.has_solution = False
        self.times["cutter"] = perf_counter() - cutter_begin

    def evaluate(self, eval_mode, num_shots_fn):
        """
        eval_mode = qasm: simulate shots
        eval_mode = sv: statevector simulation
        num_shots_fn: a function that gives the number of shots to take for a given circuit
        """
        if self.verbose:
            print("*" * 20, "evaluation mode = %s" % (eval_mode), "*" * 20)
        self.eval_mode = eval_mode
        self.num_shots_fn = num_shots_fn

        evaluate_begin = perf_counter()
        self._run_subcircuits()
        self._attribute_shots()

        ## This is the place the cutqcmodel needs to return
        self.times["evaluate"] = perf_counter() - evaluate_begin
        if self.verbose:
            print("evaluate took %e seconds" % self.times["evaluate"])

        return CutQCModel(
            self.compute_graph,
            self.complete_path_map,
            self.attributed_shots,
            self.entry_init_meas_ids,
            self.num_cuts,
            self.subcircuits,
            self.circuit,
        )

    def _generate_metadata(self):
        self.compute_graph = generate_compute_graph(
            counter=self.counter,
            subcircuits=self.subcircuits,
            complete_path_map=self.complete_path_map,
        )

        (
            self.subcircuit_entries,
            self.subcircuit_instances,
        ) = generate_subcircuit_entries(compute_graph=self.compute_graph)

        if self.verbose:
            print("--> %s subcircuit_entries:" % self.name)
            for subcircuit_idx in self.subcircuit_entries:
                print(
                    "Subcircuit_%d has %d entries"
                    % (subcircuit_idx, len(self.subcircuit_entries[subcircuit_idx]))
                )

    def _run_subcircuits(self):
        """
        Run all the subcircuit instances
        subcircuit_instance_probs[subcircuit_idx][(init,meas)] = measured prob
        """
        if self.verbose:
            print("--> Running Subcircuits %s" % self.name)

        self.instance_init_meas_ids = {}

        self.prob_dict, self.instance_init_meas_ids = run_subcircuit_instances(
            subcircuits=self.subcircuits,
            subcircuit_instances=self.subcircuit_instances,
            eval_mode=self.eval_mode,
            num_shots_fn=self.num_shots_fn,
            data_folder=self.tmp_data_folder,
        )

    def _attribute_shots(self):
        """
        Attribute the subcircuit_instance shots into respective subcircuit entries
        subcircuit_entry_probs[subcircuit_idx][entry_init, entry_meas] = entry_prob
        """
        if self.verbose:
            print("--> Attribute shots %s" % self.name)

        self.entry_init_meas_ids = {}
        self.attributed_shots, self.entry_init_meas_ids = attribute_shots(
            subcircuit_entries=self.subcircuit_entries,
            subcircuits=self.subcircuits,
            eval_mode=self.eval_mode,
            instance_init_meas_ids=self.instance_init_meas_ids,
            prob_dict=self.prob_dict,
            num_workers=20,
        )
