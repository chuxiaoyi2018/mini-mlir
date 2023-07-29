import pymlir
import numpy as np

g_mlir_module = pymlir.module()
g_mlir_module.load('relu_lower.mlir')

g_mlir_module.set_tensor('x', -np.ones((2,72)).astype(np.float32))
g_mlir_module.invoke()
print(g_mlir_module.get_tensor('1_Relu'))


g_mlir_module.set_tensor('x', np.ones((2,72)).astype(np.float32))
g_mlir_module.invoke()
print(g_mlir_module.get_tensor('1_Relu'))
