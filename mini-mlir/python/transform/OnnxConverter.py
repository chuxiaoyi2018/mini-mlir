# ONNX Node define:
# https://github.com/onnx/onnx/blob/main/docs/Operators.md

from .MLIRImporter import MLIRImporter
from .BaseConverter import BaseConverter
from onnx import numpy_helper, mapping
from numbers import Number
import onnxsim.onnx_simplifier as onnxsim

import onnx
import numpy as np

onnx_attr_translator = {
    "axis": lambda x: int(x),
    "axes": lambda x: [int(a) for a in x],
    "dtype": lambda x: onnx_dtype(x),
    "keepdims": lambda x: bool(x),
    "to": lambda x: onnx_dtype(x),
}


def translate_onnx(key, val):
    return onnx_attr_translator.get(key, lambda x: x)(val)


def onnx_dtype(dtype):
    if isinstance(dtype, Number):
        onnx_dtype = dtype
    elif isinstance(dtype, str):
        onnx_dtype = onnx.TensorProto.DataType.Value(dtype)
    else:
        raise RuntimeError("dtype should be number or str.")
    return mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_dtype]


def convert_onnx_attribute_proto(attr_proto):
    if attr_proto.HasField('f'):
        return attr_proto.f
    elif attr_proto.HasField('i'):
        return attr_proto.i
    elif attr_proto.HasField('s'):
        return attr_proto.s
    elif attr_proto.HasField('t'):
        return attr_proto.t  # this is a proto!
    elif attr_proto.floats:
        return list(attr_proto.floats)
    elif attr_proto.ints:
        return list(attr_proto.ints)
    elif attr_proto.strings:
        str_list = list(attr_proto.strings)
        return str_list
    elif attr_proto.name:
        name_list = list(attr_proto.name)
        return name_list
    else:
        raise ValueError("Unsupported ONNX attribute: {}".format(attr_proto))


class BaseNode():
    def __init__(self, info):
        self.name = str(info["name"])
        self.op_type = str(info["op_type"])
        self.attrs = dict(info["attrs"])
        self.inputs = list(info["inputs"])
        self.outputs = list(info["outputs"])


class OnnxNode(BaseNode):
    def __init__(self, node):
        info = dict()
        info["name"] = node.output[0]
        info["op_type"] = node.op_type
        info["attrs"] = [(attr.name, \
                          translate_onnx(attr.name, convert_onnx_attribute_proto(attr))) \
                          for attr in node.attribute]
        info["inputs"] = node.input
        info["outputs"] = node.output
        super().__init__(info)
        self.node_proto = node


class OnnxConverter(BaseConverter):
    def __init__(self, model_name: str, onnx_file: str, input_shapes: list, mlir_file: str):
        super().__init__()
        self.model_name = model_name
        self.weight_file = "{}_top_weight.npz".format(model_name)
        self.mlir_file = mlir_file
        self.input_names = list()
        self.output_names = list()
        self.model = None
        self.mlir = None
        self.load_onnx_model(onnx_file, input_shapes)
        self.init_MLIRImporter()

        self.onnxop_factory = {
            "Add": lambda node: self.convert_add_op(node),
            "Sub": lambda node: self.convert_sub_op(node),
            "Div": lambda node: self.convert_div_op(node),
            "AveragePool": lambda node: self.convert_avgpool_op(node),
            "BatchNormalization": lambda node: self.convert_batchnorm_op(node),
            "Conv": lambda node: self.convert_conv_op(node),
            "Flatten": lambda node: self.convert_flatten_op(node),
            "Gemm": lambda node: self.convert_gemm_op(node),
            "GlobalAveragePool": lambda node: self.convert_global_avgpool_op(node),
            "GlobalMaxPool": lambda node: self.convert_global_maxpool_op(node),
            "MaxPool": lambda node: self.convert_maxpool_op(node),
            "Relu": lambda node: self.convert_relu_op(node),
            "Reshape": lambda node: self.convert_reshape_op(node)
        }

    def __del__(self):
        del self.mlir

    def get_outputs(self, model: onnx.ModelProto):
        initializer_names = [x.name for x in model.graph.initializer]
        return [opt for opt in model.graph.output if opt.name not in initializer_names]

    def get_inputs(self, model: onnx.ModelProto):
        initializer_names = [x.name for x in model.graph.initializer]
        return [ipt for ipt in model.graph.input if ipt.name not in initializer_names]

    def get_input_names(self, model: onnx.ModelProto):
        input_names = [ipt.name for ipt in self.get_inputs(model)]
        return input_names

    def get_input_types(self, model: onnx.ModelProto):
        input_types = []
        for input in self.get_inputs(model):
            if input.type.tensor_type.elem_type in [onnx.TensorProto.INT64, onnx.TensorProto.INT32]:
                input_types.append('INT32')
            else:
                input_types.append('F32')
        return input_types

    def get_output_types(self, model: onnx.ModelProto):
        output_types = []
        for output in self.get_outputs(model):
            if output.type.tensor_type.elem_type in [
                    onnx.TensorProto.INT64, onnx.TensorProto.INT32
            ]:
                output_types.append('INT32')
            else:
                output_types.append('F32')
        return output_types

    def get_shape_from_value_info_proto(self, v: onnx.ValueInfoProto):
        return [dim.dim_value for dim in v.type.tensor_type.shape.dim]

    def get_input_shapes(self, model: onnx.ModelProto):
        inputs = self.get_inputs(model)
        return [self.get_shape_from_value_info_proto(i) for i in inputs]

    def input_shape_assign(self, input_shapes):
        inputs = self.get_inputs(self.model)
        outputs = self.get_outputs(self.model)
        shape_changed = False
        no_shape = True

    def load_onnx_model(self, onnx_file, input_shapes: list):
        self.model = onnx.load(onnx_file)
        self.input_names = self.get_input_names(self.model)
        self.num_input = len(self.input_names)
        self.input_shape_assign(input_shapes)
        self.input_shapes = self.get_input_shapes(self.model)
        self.input_types = self.get_input_types(self.model)
        self.output_types = self.get_output_types(self.model)
        
        model_simplified, is_ok = onnxsim.simplify(self.model)
        if is_ok:
            self.model = model_simplified
        # add all weight
        for tensor in self.model.graph.initializer:
            name = tensor.name
            # all weight convert to fp32
            # TODO: support other type
            data = numpy_helper.to_array(tensor).astype(np.float32)
            self.addTensor(name, data)
        # add all shape info
        for input in self.model.graph.input:
            if not self.isTensor(input.name):
                shape = [i.dim_value for i in input.type.tensor_type.shape.dim]
                self.addShape(input.name, shape)
        for info in self.model.graph.value_info:
            shape = [i.dim_value for i in info.type.tensor_type.shape.dim]
            self.addShape(info.name, shape)
        for output in self.model.graph.output:
            if not self.isTensor(output.name):
                self.output_names.append(output.name)
                shape = [i.dim_value for i in output.type.tensor_type.shape.dim]
                self.addShape(output.name, shape)
        onnx.save(self.model, "{}_debug.onnx".format(self.model_name))

    def model_shape_infer(self, input_shapes):
        inputs = onnxsim.get_inputs(self.model)
        no_shape = True

        def check_shape(l, r):
            if no_shape == False and l != r:
                raise KeyError("input shapes error:{}, {} vs {}".format(input_shapes, l, r))

        if len(input_shapes) > 0:
            no_shape = False
            check_shape(self.num_input, len(input_shapes))
        for idx, input in enumerate(inputs):
            _dims = input.type.tensor_type.shape.dim
            num_dims = len(_dims)
            if no_shape == False:
                check_shape(num_dims, len(input_shapes[idx]))
            _shape = []
            for _i, _dim in enumerate(_dims):
                if _dim.dim_value <= 0:
                    _dim.dim_value = 1 if no_shape else input_shapes[idx][_i]
                elif not no_shape:
                    check_shape(_dim.dim_value, input_shapes[idx][_i])
                _shape.append(_dim.dim_value)
            self.addShape(input.name, _shape)
        self.model = onnx.shape_inference.infer_shapes(self.model)

    def init_MLIRImporter(self):
        input_shapes = list()
        for _name in self.input_names:
            input_shapes.append(self.getShape(_name))
        output_shapes = list()
        for _name in self.output_names:
            output_shapes.append(self.getShape(_name))
        # init importer
        self.mlir = MLIRImporter(input_shapes, output_shapes, self.weight_file)

    def run(self):
        """convert all to mlir"""
        # add input op
        for idx, _name in enumerate(self.input_names):
            input_op = self.mlir.create_input_op(_name, idx)
            self.addOperand(_name, input_op)

        def NoneAndRaise(node):
            raise RuntimeError("{} Op not support now".format(node.op_type))

        for n in self.model.graph.node:
            node = OnnxNode(n)
            self.onnxop_factory.get(node.op_type, lambda x: NoneAndRaise(x))(node)

        # add return op
        return_op = list()
        # Set output
        for idx, _name in enumerate(self.output_names):
            op = self.getOperand(_name)
            return_op.append(op)

        self.mlir.create_return_op(return_op)
        mlir_txt = self.mlir.print_module()
        with open(self.mlir_file, "w") as f:
            f.write(mlir_txt)
        self.WeightToNpz(self.weight_file)
        print("Save mlir file: {}".format(self.mlir_file))

    def convert_add_op(self, onnx_node):
        assert (len(onnx_node.inputs) == 2)
        if self.isTensor(onnx_node.inputs[0]) or self.isTensor(onnx_node.inputs[1]):
            # TODO: support tensor
            raise RuntimeError("not support Tensor")
        op0 = self.getOperand(onnx_node.inputs[0])
        op1 = self.getOperand(onnx_node.inputs[1])
        p = {'name': "{}_{}".format(onnx_node.name, onnx_node.op_type)}
        output_shape = self.getShape(onnx_node.name)
        add_op = self.mlir.create_add_op([op0, op1], output_shape, **p)
        self.addOperand(onnx_node.name, add_op)
        return

    def convert_sub_op(self, onnx_node):
        assert (len(onnx_node.inputs) == 2)
        if self.isTensor(onnx_node.inputs[0]) or self.isTensor(onnx_node.inputs[1]):
            # TODO: support tensor
            raise RuntimeError("not support Tensor")
        op0 = self.getOperand(onnx_node.inputs[0])
        op1 = self.getOperand(onnx_node.inputs[1])
        p = {'name': "{}_{}".format(onnx_node.name, onnx_node.op_type)}
        output_shape = self.getShape(onnx_node.name)
        add_op = self.mlir.create_sub_op([op0, op1], output_shape, **p)
        self.addOperand(onnx_node.name, add_op)
        return

    def convert_div_op(self, onnx_node):
        assert (len(onnx_node.inputs) == 2)
        if self.isTensor(onnx_node.inputs[0]) or self.isTensor(onnx_node.inputs[1]):
            # TODO: support tensor
            raise RuntimeError("not support Tensor")
        op0 = self.getOperand(onnx_node.inputs[0])
        op1 = self.getOperand(onnx_node.inputs[1])
        p = {'name': "{}_{}".format(onnx_node.name, onnx_node.op_type)}
        output_shape = self.getShape(onnx_node.name)
        add_op = self.mlir.create_div_op([op0, op1], output_shape, **p)
        self.addOperand(onnx_node.name, add_op)
        return

    def convert_batchnorm_op(self, onnx_node):
        assert (onnx_node.op_type == "BatchNormalization")
        # TODO: support batchnorm
        raise RuntimeError("not support {}".format(onnx_node.op_type))

    def convert_conv_op(self, onnx_node):
        assert (onnx_node.op_type == "Conv")
        op = self.getOperand(onnx_node.inputs[0])
        kernel_shape = onnx_node.attrs['kernel_shape']
        dim = len(kernel_shape)
        dilations = onnx_node.attrs.get("dilations", dim * [1])
        group = onnx_node.attrs.get("group", 1)
        pads = onnx_node.attrs.get("pads", dim * 2 * [0])
        strides = onnx_node.attrs.get("strides", dim * [1])
        operands = list()
        operands.append(op)
        filter_op = self.getWeightOp(onnx_node.inputs[1])
        operands.append(filter_op)
        if len(onnx_node.inputs) > 2:
            bias_op = self.getWeightOp(onnx_node.inputs[2])
        else:
            bias_op = self.mlir.none_op
        operands.append(bias_op)
        p = {
            'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
            'kernel_shape': kernel_shape,
            'strides': strides,
            'dilations': dilations,
            'pads': pads,
            'group': group,
            'do_relu': False,
            'ins': [],
        }
        output_shape = self.getShape(onnx_node.name)
        new_op = self.mlir.create_conv_op(operands, output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_flatten_op(self, onnx_node):
        assert (onnx_node.op_type == "Flatten")
        op = self.getOperand(onnx_node.inputs[0])
        p = {
            'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
        }
        output_shape = self.getShape(onnx_node.name)
        new_op = self.mlir.create_reshape_op([op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_gemm_op(self, onnx_node):
        assert (onnx_node.op_type == "Gemm")
        #(M, K) * (K, N) => (M, N)
        op = self.getOperand(onnx_node.inputs[0])
        alpha = onnx_node.attrs.get('alpha', 1)
        beta = onnx_node.attrs.get('beta', 1)
        trans_a = onnx_node.attrs.get('transA', 0)
        trans_b = onnx_node.attrs.get('transB', 0)
        # TODO:support more situations
        assert (trans_a == 0)
        operands = list()
        operands.append(op)
        B = onnx_node.inputs[1]
        assert (self.isTensor(B))
        if trans_b == 1 or alpha != 1:
            _tensor = self.getTensor(B)
            if trans_b == 1:
                _tensor = np.ascontiguousarray(np.transpose(_tensor, (1, 0)))
            if alpha != 1:
                _tensor *= alpha
            B += '_fix'
            self.addTensor(B, _tensor)
        operands.append(self.getWeightOp(B))
        if len(onnx_node.inputs) > 2 and beta != 0:
            C = onnx_node.inputs[2]
            if beta != 1:
                _tensor = self.getTensor(C)
                _tensor *= beta
                C += '_fix'
                self.addTensor(C, _tensor)
            operands.append(self.getWeightOp(C))
        else:
            operands.append(self.mlir.none_op)
        p = {'name': "{}_{}".format(onnx_node.name, onnx_node.op_type), 'do_relu': False}
        output_shape = self.getShape(onnx_node.name)
        new_op = self.mlir.create_matmul_op(operands, output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_global_maxpool_op(self, onnx_node):
        assert (onnx_node.op_type == "GlobalMaxPool")
        op = self.getOperand(onnx_node.inputs[0])
        input_shape = self.getShape(onnx_node.inputs[0])
        num_dim = len(input_shape) - 2
        assert (num_dim > 0)
        p = {
            'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
            'kernel_shape': input_shape[2:],
            'strides': num_dim * [1],
            'pads': num_dim * 2 * [0],
            'count_include_pad': True,
            'do_relu': False,
        }
        output_shape = self.getShape(onnx_node.name)
        new_op = self.mlir.create_maxpool_op([op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_global_avgpool_op(self, onnx_node):
        assert (onnx_node.op_type == "GlobalAveragePool")
        op = self.getOperand(onnx_node.inputs[0])
        input_shape = self.getShape(onnx_node.inputs[0])
        num_dim = len(input_shape) - 2
        assert (num_dim > 0)
        p = {
            'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
            'kernel_shape': input_shape[2:],
            'strides': num_dim * [1],
            'pads': num_dim * 2 * [0],
            'count_include_pad': True,
            'do_relu': False,
        }
        output_shape = self.getShape(onnx_node.name)
        new_op = self.mlir.create_avgpool_op([op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_avgpool_op(self, onnx_node):
        assert (onnx_node.op_type == "AveragePool")
        op = self.getOperand(onnx_node.inputs[0])
        kernel_shape = onnx_node.attrs['kernel_shape']
        count_include_pad = onnx_node.attrs.get('count_include_pad', False)
        dim = len(kernel_shape)
        pads = onnx_node.attrs.get("pads", dim * 2 * [0])
        strides = onnx_node.attrs.get("strides", kernel_shape)
        p = {
            'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
            'kernel_shape': kernel_shape,
            'strides': strides,
            'pads': pads,
            'count_include_pad': count_include_pad,
            'do_relu': False,
        }
        output_shape = self.getShape(onnx_node.name)
        new_op = self.mlir.create_avgpool_op([op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_maxpool_op(self, onnx_node):
        assert (onnx_node.op_type == "MaxPool")
        op = self.getOperand(onnx_node.inputs[0])
        kernel_shape = onnx_node.attrs['kernel_shape']
        count_include_pad = onnx_node.attrs.get('count_include_pad', False)
        dim = len(kernel_shape)
        pads = onnx_node.attrs.get("pads", dim * 2 * [0])
        strides = onnx_node.attrs.get("strides", kernel_shape)
        p = {
            'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
            'kernel_shape': kernel_shape,
            'strides': strides,
            'pads': pads,
            'count_include_pad': count_include_pad,
            'do_relu': False,
        }
        output_shape = self.getShape(onnx_node.name)
        new_op = self.mlir.create_maxpool_op([op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_relu_op(self, onnx_node):
        assert (onnx_node.op_type == "Relu")
        op = self.getOperand(onnx_node.inputs[0])
        output_shape = self.getShape(onnx_node.name)
        p = {'name': "{}_{}".format(onnx_node.name, onnx_node.op_type)}
        new_op = self.mlir.create_relu_op([op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_reshape_op(self, onnx_node):
        assert (onnx_node.op_type == "Reshape")
        # TODO: support reshape
        raise RuntimeError("not support {}".format(onnx_node.op_type))

