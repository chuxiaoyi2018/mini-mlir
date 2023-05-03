import onnx
import numpy as np
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

def dtype_to_tensortype(dtype):
    if dtype == float:
        return TensorProto.FLOAT
    elif dtype == int:
        return TensorProto.INT64
    elif dtype == bool:
        return TensorProto.BOOL
    else:
        raise RuntimeError(dtype, " type no support")

def const_tensor(name, shape, type, value=None):
    if value is not None:
        init = value
    else:
        num = 1
        for s in shape:
          num = s * num
        init = np.random.rand(num)
    tensor = helper.make_tensor(name, dtype_to_tensortype(type), shape, (init.flatten().astype(type)))
    return tensor

def tensor(name, shape=None, type=None):
    if shape is not None:
        tensor = helper.make_tensor_value_info(name, dtype_to_tensortype(type), shape)
    else :
        tensor = helper.make_empty_tensor_value_info(name)
    return tensor

def gen_model(node, input, output, init, path):
    graph_def = helper.make_graph(
        node,
        "utest-model",
        input,
        output,
        init,
    )
    model_def = helper.make_model(graph_def, producer_name='onnx-example')
    onnx.save(model_def, path)

