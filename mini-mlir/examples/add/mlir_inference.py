import pymlir
import numpy as np

g_mlir_module = pymlir.module()
g_mlir_module.load('add_lower.mlir')

g_mlir_module.set_tensor('onnx::Add_0', -np.ones((2,72)).astype(np.float32))
g_mlir_module.set_tensor('onnx::Add_1', -np.ones((2,72)).astype(np.float32))
g_mlir_module.invoke()
print(g_mlir_module.get_tensor('2_Add'))


g_mlir_module.set_tensor('onnx::Add_0', np.ones((2,72)).astype(np.float32))
g_mlir_module.set_tensor('onnx::Add_1', np.ones((2,72)).astype(np.float32))
g_mlir_module.invoke()
print(g_mlir_module.get_tensor('2_Add'))
