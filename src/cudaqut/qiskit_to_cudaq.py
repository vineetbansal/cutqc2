import ast
import qiskit
from cudaq.kernel.ast_bridge import compile_to_mlir
from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime


class Kernel:
    def __init__(self, kernel_name: str, qiskit_circuit: qiskit.QuantumCircuit):
        self.kernel_name = kernel_name
        self.ast_module = self.get_cudaq_ast_module(self.kernel_name, qiskit_circuit)
        self.ast_module_src = ast.unparse(self.ast_module)

        self.compile()

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
    def get_cudaq_ast_module(name: str, qiskit_circuit: qiskit.QuantumCircuit) -> ast.Module:
        # This is just a hardcoded module corresponding to Figure 4 in the
        # paper. Replace with a dynamically constructed one.
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
                    body=[
                        ast.Assign(
                            lineno=None,
                            targets=[ast.Name(id='qubits', ctx=ast.Store())],
                            value=ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id='cudaq', ctx=ast.Load()),
                                    attr='qvector',
                                    ctx=ast.Load()
                                ),
                                args=[ast.Constant(value=5)],
                                keywords=[]
                            )
                        ),
                        # H gates
                        *[ast.Expr(
                            value=ast.Call(
                                func=ast.Name(id='h', ctx=ast.Load()),
                                args=[ast.Subscript(
                                    value=ast.Name(id='qubits', ctx=ast.Load()),
                                    slice=ast.Constant(i),
                                    ctx=ast.Load()
                                )],
                                keywords=[]
                            )
                        ) for i in range(5)],
                        # CZ(0,1)
                        ast.Expr(
                            value=ast.Call(
                                func=ast.Name(id='cz', ctx=ast.Load()),
                                args=[
                                    ast.Subscript(value=ast.Name(id='qubits',
                                                                 ctx=ast.Load()),
                                                  slice=ast.Constant(0),
                                                  ctx=ast.Load()),
                                    ast.Subscript(value=ast.Name(id='qubits',
                                                                 ctx=ast.Load()),
                                                  slice=ast.Constant(1),
                                                  ctx=ast.Load())
                                ],
                                keywords=[]
                            )
                        ),
                        # T gates on 2, 3, 4
                        *[ast.Expr(
                            value=ast.Call(
                                func=ast.Name(id='t', ctx=ast.Load()),
                                args=[ast.Subscript(
                                    value=ast.Name(id='qubits', ctx=ast.Load()),
                                    slice=ast.Constant(i),
                                    ctx=ast.Load()
                                )],
                                keywords=[]
                            )
                        ) for i in [2, 3, 4]],
                        # CZ(0,2)
                        ast.Expr(
                            value=ast.Call(
                                func=ast.Name(id='cz', ctx=ast.Load()),
                                args=[
                                    ast.Subscript(value=ast.Name(id='qubits',
                                                                 ctx=ast.Load()),
                                                  slice=ast.Constant(0),
                                                  ctx=ast.Load()),
                                    ast.Subscript(value=ast.Name(id='qubits',
                                                                 ctx=ast.Load()),
                                                  slice=ast.Constant(2),
                                                  ctx=ast.Load())
                                ],
                                keywords=[]
                            )
                        ),
                        # RX(pi/2, qubit 4)
                        ast.Expr(
                            value=ast.Call(
                                func=ast.Name(id='rx', ctx=ast.Load()),
                                args=[
                                    ast.BinOp(
                                        left=ast.Attribute(
                                            value=ast.Name(id='math',
                                                           ctx=ast.Load()),
                                            attr='pi', ctx=ast.Load()),
                                        op=ast.Div(),
                                        right=ast.Constant(value=2)
                                    ),
                                    ast.Subscript(value=ast.Name(id='qubits',
                                                                 ctx=ast.Load()),
                                                  slice=ast.Constant(4),
                                                  ctx=ast.Load())
                                ],
                                keywords=[]
                            )
                        ),
                        # RX(pi/2, qubit 0), RX(pi/2, qubit 1)
                        *[ast.Expr(
                            value=ast.Call(
                                func=ast.Name(id='rx', ctx=ast.Load()),
                                args=[
                                    ast.BinOp(
                                        left=ast.Attribute(
                                            value=ast.Name(id='math',
                                                           ctx=ast.Load()),
                                            attr='pi', ctx=ast.Load()),
                                        op=ast.Div(),
                                        right=ast.Constant(value=2)
                                    ),
                                    ast.Subscript(value=ast.Name(id='qubits',
                                                                 ctx=ast.Load()),
                                                  slice=ast.Constant(i),
                                                  ctx=ast.Load())
                                ],
                                keywords=[]
                            )
                        ) for i in [0, 1]],
                        # CZ(2,4)
                        ast.Expr(
                            value=ast.Call(
                                func=ast.Name(id='cz', ctx=ast.Load()),
                                args=[
                                    ast.Subscript(value=ast.Name(id='qubits',
                                                                 ctx=ast.Load()),
                                                  slice=ast.Constant(2),
                                                  ctx=ast.Load()),
                                    ast.Subscript(value=ast.Name(id='qubits',
                                                                 ctx=ast.Load()),
                                                  slice=ast.Constant(4),
                                                  ctx=ast.Load())
                                ],
                                keywords=[]
                            )
                        ),
                        # T gates on 0 and 1
                        *[ast.Expr(
                            value=ast.Call(
                                func=ast.Name(id='t', ctx=ast.Load()),
                                args=[ast.Subscript(
                                    value=ast.Name(id='qubits', ctx=ast.Load()),
                                    slice=ast.Constant(i),
                                    ctx=ast.Load()
                                )],
                                keywords=[]
                            )
                        ) for i in [0, 1]],
                        # CZ(2,3)
                        ast.Expr(
                            value=ast.Call(
                                func=ast.Name(id='cz', ctx=ast.Load()),
                                args=[
                                    ast.Subscript(value=ast.Name(id='qubits',
                                                                 ctx=ast.Load()),
                                                  slice=ast.Constant(2),
                                                  ctx=ast.Load()),
                                    ast.Subscript(value=ast.Name(id='qubits',
                                                                 ctx=ast.Load()),
                                                  slice=ast.Constant(3),
                                                  ctx=ast.Load())
                                ],
                                keywords=[]
                            )
                        ),
                        # RX(pi/2, 4)
                        ast.Expr(
                            value=ast.Call(
                                func=ast.Name(id='rx', ctx=ast.Load()),
                                args=[
                                    ast.BinOp(
                                        left=ast.Attribute(
                                            value=ast.Name(id='math',
                                                           ctx=ast.Load()),
                                            attr='pi', ctx=ast.Load()),
                                        op=ast.Div(),
                                        right=ast.Constant(value=2)
                                    ),
                                    ast.Subscript(value=ast.Name(id='qubits',
                                                                 ctx=ast.Load()),
                                                  slice=ast.Constant(4),
                                                  ctx=ast.Load())
                                ],
                                keywords=[]
                            )
                        ),
                        # Final H gates
                        *[ast.Expr(
                            value=ast.Call(
                                func=ast.Name(id='h', ctx=ast.Load()),
                                args=[ast.Subscript(
                                    value=ast.Name(id='qubits', ctx=ast.Load()),
                                    slice=ast.Constant(i),
                                    ctx=ast.Load()
                                )],
                                keywords=[]
                            )
                        ) for i in range(5)]
                    ],
                    decorator_list=[
                    ],
                    returns=None,
                    type_comment=None
                )
            ],
            type_ignores=[]
        )
