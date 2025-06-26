"""
Title: single_node_full_example.py
Description: Shows the creation, cutting, evaluation, and reconstruction of a circuit 
"""

from cutqc import CircuitCutter, CircuitReconstructor
from cutqc import generate_circ


if __name__ == "__main__":
    circ_type = "adder"
    circ_size = 10
    max_width = 10

    # Generate Example Circuit and Initialize CutQC
    circuit = generate_circ(
        num_qubits=circ_size,
        depth=1,
        circuit_type=circ_type,
        reg_name="q",
        connected_only=True,
        seed=None,
    )

    cutter = CircuitCutter(
        name="%s_%d" % (circ_type, circ_size),
        circuit=circuit,
        cutter_constraints={
            "max_subcircuit_width": max_width,
            "max_subcircuit_cuts": 10,
            "subcircuit_size_imbalance": 2,
            "max_cuts": 10,
            "num_subcircuits": [2, 3, 4, 5, 6, 8],
        },
    )

    print("--- Cut --- ")
    cutter.cut()

    if not cutter.has_solution:
        raise Exception("The input circuit and constraints have no viable cuts")

    print("--- Evaluate ---")
    cutqc_model = cutter.evaluate(eval_mode="sv", num_shots_fn=None)

    # Initiate Reconstruct
    print("--- Reconstruct ---")
    reconstructor = CircuitReconstructor(cutqc_model, mem_limit=32, recursion_depth=1)
    compute_time = reconstructor.build()
    cutqc_model.verify()
    print("Completed")
