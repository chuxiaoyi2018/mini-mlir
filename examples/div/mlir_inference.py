import pymlir
import numpy as np

op_name = 'div'
Op_name = op_name.capitalize()

g_mlir_module = pymlir.module()
g_mlir_module.load(f'{op_name}_lower.mlir')

g_mlir_module.set_tensor(f'onnx::{Op_name}_0', -np.ones((2,72)).astype(np.float32))
g_mlir_module.set_tensor(f'onnx::{Op_name}_1', -np.ones((2,72)).astype(np.float32))
g_mlir_module.invoke()
print(g_mlir_module.get_tensor(f'2_{Op_name}'))


g_mlir_module.set_tensor(f'onnx::{Op_name}_0', np.ones((2,72)).astype(np.float32))
g_mlir_module.set_tensor(f'onnx::{Op_name}_1', np.ones((2,72)).astype(np.float32))
g_mlir_module.invoke()
print(g_mlir_module.get_tensor(f'2_{Op_name}'))
