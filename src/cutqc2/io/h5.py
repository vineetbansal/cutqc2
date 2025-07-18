from pathlib import Path
import h5py
import numpy as np
from qiskit.qasm2 import dumps, loads
from cutqc2 import __version__
from cutqc2.core.cut_circuit import CutCircuit


def cut_circuit_to_h5(
    cut_circuit: CutCircuit, filepath: str | Path, *args, **kwargs
) -> None:
    if isinstance(filepath, str):
        filepath = Path(filepath)

    with h5py.File(filepath, "w") as f:
        str_type = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("version", data=__version__, dtype=str_type)
        f.create_dataset(
            "uncut_circuit", data=dumps(cut_circuit.raw_circuit), dtype=str_type
        )

        if cut_circuit.num_cuts > 0:
            cuts_array = np.array(
                cut_circuit.cuts,
                dtype=np.dtype([("wire_index", "i4"), ("gate_index", "i4")]),
            )
            f.create_dataset("cuts", data=cuts_array)

            group = f.create_group("subcircuit_mapping")
            for key, value in cut_circuit.reconstruction_qubit_order.items():
                group.create_dataset(str(key), data=np.array(value, dtype="int"))

            if cut_circuit.subcircuit_entry_probs:
                group = f.create_group("subcircuit_probability_vector")
                for (
                    subcircuit_idx,
                    entry_probs,
                ) in cut_circuit.subcircuit_entry_probs.items():
                    group = f.create_group(
                        "subcircuit_probability_vector/" + str(subcircuit_idx)
                    )
                    for k, v in entry_probs.items():
                        key = "_".join(["-".join(k[0]), "-".join(k[1])])
                        group.create_dataset(key, data=np.array(v, dtype="float64"))


def h5_to_cut_circuit(filepath: str | Path, *args, **kwargs) -> CutCircuit:
    with h5py.File(filepath, "r") as f:
        qasm_str = f["uncut_circuit"][()].decode("utf-8")
        cut_circuit = CutCircuit(loads(qasm_str))

        if "cuts" in f:
            cuts = f["cuts"][()]
            for cut in cuts:
                cut_circuit.add_cut_at_position(
                    wire_index=cut["wire_index"], gate_index=cut["gate_index"]
                )
            cut_circuit.generate_subcircuits()

        if "subcircuit_mapping" in f:
            mapping = {}
            for key, ds in f["subcircuit_mapping"].items():
                mapping[int(key)] = ds[()].astype(int).tolist()
            cut_circuit.reconstruction_qubit_order = mapping

        if "subcircuit_probability_vector" in f:
            probs_group = f["subcircuit_probability_vector"]
            entry_probs = {}
            for subcircuit_idx in probs_group:
                subgrp = probs_group[subcircuit_idx]
                prob_dict = {}
                for key, ds in subgrp.items():
                    str_a, str_b = key.split("_")
                    tuple_key = (tuple(str_a.split("-")), tuple(str_b.split("-")))
                    prob_dict[tuple_key] = ds[()].astype(float)
                entry_probs[int(subcircuit_idx)] = prob_dict
            cut_circuit.subcircuit_entry_probs = entry_probs

        if "probability_vector" in f:
            pass

    return cut_circuit
