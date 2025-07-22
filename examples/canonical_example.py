# Title: 'canonical_example.py'
# Description: A prototypical adder circuit is used as example to demonstrate cutting and reconstruction in cutqc2

from math import pi
import qiskit

from cutqc2.core.cut_circuit import CutCircuit
from cutqc2.cutqc.helper_functions.benchmarks import generate_circ 

def supremacy():  
  num_qubits=6
  circuit = generate_circ(
      num_qubits=num_qubits,
      depth=1,
      circuit_type="supremacy",
      reg_name="q",
      connected_only=True,
      seed=None,
  )
  
  import math
  
  cutter_constraints = {
          "max_subcircuit_width":  math.ceil(circuit.num_qubits / 4 * 3),
          "max_subcircuit_cuts": 10,
          "subcircuit_size_imbalance": 2,
          "max_cuts": 10,
          "num_subcircuits": [3]
      } 
  return circuit, cutter_constraints

if __name__ == "__main__":
    circuit, cutter_constraints = supremacy ()    
    cut_circuit = CutCircuit(circuit)
        
    # Cut the circuit automatically with specified parameters.
    cut_circuit.cut(
        max_subcircuit_width=cutter_constraints["max_subcircuit_width"],
        max_subcircuit_cuts=cutter_constraints["max_subcircuit_cuts"],
        subcircuit_size_imbalance=cutter_constraints["subcircuit_size_imbalance"],
        max_cuts=cutter_constraints["max_cuts"],
        num_subcircuits=cutter_constraints["num_subcircuits"],
    )
    
    # Notice the location of the cut as the '//' marker gate.
    print(cut_circuit)

    print("----- Subcircuits -----")
    for subcircuit in cut_circuit:
        print(subcircuit)
        print()

    # Run all the subcircuits - by default we use the `statevector_simulator`
    # backend from qiskit.
    cut_circuit.run_subcircuits()

    # Save the cut circuit to a file
    # We could have done this at any point after creating the `CutCircuit` object.
    cut_circuit.to_file("tutorial.h5")

    # Load the cut circuit from the file
    cut_circuit = CutCircuit.from_file("tutorial.h5")

    # We perform the postprocessing step on the reloaded cut-circuit,
    # though we could have done this on the original cut-circuit as well.
    cut_circuit.postprocess()

    # Verification involves comparing the generated probabilities
    # with the expected probabilities from a simulation of the uncut circuit.
    # By default we use the `statevector_simulator` backend from qiskit.
    cut_circuit.verify()


