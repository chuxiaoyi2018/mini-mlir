import onnx
from onnx import helper
from onnx import TensorProto
import numpy as np

a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [2, 3])
b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [3, 4])

z = helper.make_tensor_value_info('z', TensorProto.FLOAT, [2, 4])

node = helper.make_node('Gemm', ['a', 'b'], ['z'],)

graph = helper.make_graph([node],'gemm', [a, b], [z],)

model = helper.make_model(graph)

onnx.checker.check_model(model)

print(model)

onnx.save(model, 'gemm.onnx')

