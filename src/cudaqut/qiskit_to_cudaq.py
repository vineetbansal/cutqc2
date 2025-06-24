import ast
import qiskit
from cudaq.kernel.ast_bridge import compile_to_mlir
from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime


class Kernel:
    def __init__(self, kernel_name: str,
                 qiskit_circuit: qiskit.QuantumCircuit):
        self.kernel_name = kernel_name
        self.ast_module = self.get_cudaq_ast_module(self.kernel_name,
                                                    qiskit_circuit)
        self.ast_module_src = ast.unparse(self.ast_module)

        # Ideally something like this should work,
        # but looks like `self.parentFrame = inspect.stack()[2].frame`
        # in `kernel_decorator.py` in cudaq source is an unjustified assumption
        # and is messing things up.

        # ... = PyKernelDecorator(function="placeholder_string",
        #                   funcSrc=self.ast_module_src,
        #                   kernelName=self.kernel_name,
        #                   location=('0', 1),
        #                   signature={},
        #                   overrideGlobalScopedVars={"foo": "bar"})

    def compile(self) -> None:
        self.module, _, _ = compile_to_mlir(self.ast_module, capturedDataStorage=None)

    def __call__(self) -> None:
        cudaq_runtime.pyAltLaunchKernel(self.kernel_name, self.module)

    @staticmethod
    def get_cudaq_ast_module(name: str,
                             qc: qiskit.QuantumCircuit) -> ast.Module:

        single_qubit_gate_mapping = {
            "h": "h",
            "t": "t",
            "rx": "rx"
        }

        multi_qubit_gate_mapping = {
            "cz": "cz",
        }

        def single_qubit_gate(qc: qiskit.QuantumCircuit,
                              gate: str,
                              instr: qiskit.circuit.quantumcircuitdata.CircuitInstruction) -> ast.Expr:

            params = []
            for param in instr.params:
                params.append(
                    ast.Constant(param)
                )

            # TODO: It would be nice to obtain the index from the Qubit object
            # itself, but there seems to be no way other than to reach for
            # the private `._index` attribute.
            qubit_index = qc.qubits.index(instr.qubits[0])

            return ast.Expr(
                value=ast.Call(
                    func=ast.Name(
                        id=single_qubit_gate_mapping[gate],
                        ctx=ast.Load()
                    ),
                    args=[
                        *params,
                        ast.Subscript(
                            value=ast.Name(id='qubits', ctx=ast.Load()),
                            slice=ast.Constant(qubit_index),
                            ctx=ast.Load()
                        )],
                    keywords=[]
                )
            )

        def multi_qubit_gate(qc: qiskit.QuantumCircuit,
                             gate: str,
                             instr: qiskit.circuit.quantumcircuitdata.CircuitInstruction) -> ast.Expr:
            assert len(
                instr.qubits) == 2, "Only two-qubit multi-qubit gates are supported for now"

            # TODO: It would be nice to obtain the index from the Qubit object
            # itself, but there seems to be no way other than to reach for
            # the private `._index` attribute.
            from_qubit_index = qc.qubits.index(instr.qubits[0])
            to_qubit_index = qc.qubits.index(instr.qubits[1])

            return ast.Expr(
                value=ast.Call(
                    func=ast.Name(
                        id=multi_qubit_gate_mapping[gate],
                        ctx=ast.Load()
                    ),
                    args=[
                        ast.Subscript(
                            value=ast.Name(id='qubits', ctx=ast.Load()),
                            slice=ast.Constant(from_qubit_index),
                            ctx=ast.Load()
                        ),
                        ast.Subscript(
                            value=ast.Name(id='qubits', ctx=ast.Load()),
                            slice=ast.Constant(to_qubit_index),
                            ctx=ast.Load()
                        ),
                    ],
                    keywords=[]
                )
            )

        body = []
        qvec_assign = ast.Assign(
            lineno=None,
            targets=[ast.Name(id='qubits', ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(value=ast.Name(id='cudaq', ctx=ast.Load()),
                                   attr='qvector', ctx=ast.Load()),
                args=[ast.Constant(value=qc.num_qubits)],
                keywords=[]
            )
        )
        body.append(qvec_assign)

        for instr in qc.data:
            instr_name = instr.name
            if instr_name in ("reset", "measure", "measure_all", "barrier"):
                continue  # TODO: Handle these!
            elif instr_name in single_qubit_gate_mapping:
                body.append(
                    single_qubit_gate(qc, instr_name, instr)
                )
            elif instr_name in multi_qubit_gate_mapping:
                body.append(
                    multi_qubit_gate(qc, instr_name, instr)
                )
            else:
                raise NotImplementedError(f"Unsupported gate: {instr_name}")

        return ast.Module(
            body=[
                ast.FunctionDef(
                    name=name,
                    lineno=None,
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[]
                    ),
                    body=body,
                    decorator_list=[],
                    returns=None,
                    type_comment=None
                )
            ],
            type_ignores=[]
        )
