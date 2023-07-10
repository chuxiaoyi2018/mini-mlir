# ONNX Node define:
# https://github.com/onnx/onnx/blob/main/docs/Operators.md

from .MLIRImporter import MLIRImporter
from .BaseConverter import BaseConverter
from onnx import numpy_helper, mapping
from numbers import Number
import onnxsim.onnx_simplifier as onnxsim
from .OnnxOpt import onnx_opt


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
    def __init__(self, model_name: str, onnx_file: str, input_shapes: list, mlir_file: str, chip: str):
        super().__init__()
        self.model_name = model_name
        self.weight_file = "{}_top_weight.npz".format(model_name)
        self.mlir_file = mlir_file
        self.input_names = list()
        self.output_names = list()
        self.model = None
        self.mlir = None
        self.chip = chip
        
        self.load_onnx_model(onnx_file, input_shapes)
        self.init_MLIRImporter()
        self.opset = self.model.opset_import[-1].version

        self.onnxop_factory = {
            "Add": lambda node: self.convert_add_op(node),
            "Sub": lambda node: self.convert_sub_op(node),
            "Div": lambda node: self.convert_div_op(node),
            "AveragePool": lambda node: self.convert_avgpool_op(node),
            "BatchNormalization": lambda node: self.convert_batchnorm_op(node),
            "Conv": lambda node: self.convert_conv_op(node),
            "Concat": lambda node: self.convert_concat_op(node),
            "Erf": lambda node: self.convert_erf_op(node),
            "Flatten": lambda node: self.convert_flatten_op(node),
            "Gather": lambda node: self.convert_gather_op(node),
            "Gemm": lambda node: self.convert_gemm_op(node),
            "GELU": lambda node: self.convert_gelu_op(node),
            "GlobalAveragePool": lambda node: self.convert_global_avgpool_op(node),
            "GlobalMaxPool": lambda node: self.convert_global_maxpool_op(node),
            "Mul":lambda node:self.convert_mul_op(node),
            "MatMul":lambda node:self.convert_gemm_op(node),
            "MaxPool": lambda node: self.convert_maxpool_op(node),
            "ReduceMean": lambda node: self.convert_reduce_mean_op(node),
            "Relu": lambda node: self.convert_relu_op(node),
            "Reshape": lambda node: self.convert_reshape_op(node),
            "Slice": lambda node: self.convert_slice_op(node),
            "Sqrt": lambda node: self.convert_sqrt_op(node),
            "Split": lambda node: self.convert_split_op(node),
            "Softmax": lambda node: self.convert_softmax_op(node),
            "Transpose": lambda node: self.convert_transpose_op(node),
            "Pow": lambda node: self.convert_pow_op(node),
            "Reshape": lambda node: self.convert_reshape_op(node),
            "LayerNormalization": lambda node: self.convert_layer_norm_op(node),
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
        inputs_list = [self.get_shape_from_value_info_proto(i) for i in inputs]

        # transpose from [1,3,224,224] to [1,224,224,3]
        # if self.chip == "cpu" and len(inputs_list) == 1 and len(inputs_list[0]) == 4:
        #     inputs_list = [[i[0], i[2], i[3], i[1]] for i in inputs_list]
        return inputs_list

    def input_shape_assign(self, input_shapes):
        inputs = self.get_inputs(self.model)
        outputs = self.get_outputs(self.model)
        shape_changed = False
        no_shape = True

    def load_onnx_model(self, onnx_file, input_shapes: list):
        # transpose from 
        # if self.chip == "cpu" and len(input_shapes) == 1 and len(input_shapes[0]) == 4:
        #     input_shapes = [[shape[0], shape[2], shape[3], shape[1]] for shape in input_shapes]

        self.model = onnx.load(onnx_file)
        self.input_names = self.get_input_names(self.model)
        self.num_input = len(self.input_names)
        self.input_shape_assign(input_shapes)
        self.input_shapes = self.get_input_shapes(self.model)
        self.input_types = self.get_input_types(self.model)
        self.output_types = self.get_output_types(self.model)
        
        print("begin onnxsim")
        model_simplified, is_ok = onnxsim.simplify(self.model)
        print("end onnxsim")

        if is_ok:
            self.model = model_simplified

        print("beginn onnxopt")
        if is_ok:
            # fuse ops such as layernorm gelu...
            self.model, self.node_name_mapping = onnx_opt(self.model, True)
            # pass
        print("end onnxopt")
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
                # transpose from [1,3,224,224] to [1,224,224,3]
                # if self.chip == "cpu" and len(shape) == 4:
                #     shape = [shape[0], shape[2], shape[3], shape[1]]
                self.addShape(input.name, shape)
        for info in self.model.graph.value_info:
            shape = [i.dim_value for i in info.type.tensor_type.shape.dim]
            self.addShape(info.name, shape)
        for output in self.model.graph.output:
            if not self.isTensor(output.name):
                self.output_names.append(output.name)
                shape = [i.dim_value for i in output.type.tensor_type.shape.dim]
                # transpose from [1,3,224,224] to [1,224,224,3]
                # if self.chip == "cpu" and len(shape) == 4:
                #     shape = [shape[0], shape[2], shape[3], shape[1]]
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
        lhs = onnx_node.inputs[0]
        rhs = onnx_node.inputs[1]
        rhs_shape = self.getShape(rhs)
        lhs_shape = self.getShape(lhs)
        p = {'name': "{}_{}".format(onnx_node.name, onnx_node.op_type)}

        if self.isWeight(lhs) and not self.isWeight(rhs):
            onnx_node.inputs[0], onnx_node.inputs[1] = onnx_node.inputs[1], onnx_node.inputs[0]
            self.convert_add_op(onnx_node)
            return


        if self.chip == "cpu":
            if len(lhs_shape) == 3 and len(rhs_shape) == 1:
                weight = self.tensors[rhs]
                self.tensors[rhs] = weight.reshape(1, 1, rhs_shape[0])
                self.shapes[rhs] = self.tensors[rhs].shape

        
        if not self.isWeight(onnx_node.inputs[0]) and not self.isWeight(onnx_node.inputs[1]):
            op0 = self.getOperand(onnx_node.inputs[0])
            op1 = self.getOperand(onnx_node.inputs[1])
            output_shape = self.getShape(onnx_node.name)
            add_op = self.mlir.create_add_op([op0, op1], output_shape, **p)
        elif not self.isWeight(onnx_node.inputs[0]) or self.isWeight(onnx_node.inputs[1]):
            op0 = self.getOperand(onnx_node.inputs[0])
            op1 = self.getWeightOp(onnx_node.inputs[1])
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
        lhs = onnx_node.inputs[0]
        rhs = onnx_node.inputs[1]
        rhs_shape = self.getShape(rhs)
        lhs_shape = self.getShape(lhs)
        if not self.isWeight(onnx_node.inputs[0]) and not self.isWeight(onnx_node.inputs[1]):
            op0 = self.getOperand(onnx_node.inputs[0])
            op1 = self.getOperand(onnx_node.inputs[1])
            p = {'name': "{}_{}".format(onnx_node.name, onnx_node.op_type)}
            output_shape = self.getShape(onnx_node.name)
            div_op = self.mlir.create_div_op([op0, op1], output_shape, **p)
        elif not self.isWeight(onnx_node.inputs[0]) and self.isWeight(onnx_node.inputs[1]):
            if len(rhs_shape) == 0:
                op0 = self.getOperand(onnx_node.inputs[0])
                op1 = self.getWeight(onnx_node.inputs[1])
                p = {
                    'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
                    'const_val': 1 / self.getWeight(rhs).flatten()[0],
                }
                output_shape = self.getShape(onnx_node.name)
                div_op = self.mlir.create_mulconst_op([op0], output_shape, **p)
            else:
                raise RuntimeError("not support Tensor")
        else:
            RuntimeError("not support Tensor")
        self.addOperand(onnx_node.name, div_op)
        return

    def convert_batchnorm_op(self, onnx_node):
        assert (onnx_node.op_type == "BatchNormalization")
        # TODO: support batchnorm
        raise RuntimeError("not support {}".format(onnx_node.op_type))

    def convert_conv_op(self, onnx_node):
        assert (onnx_node.op_type == "Conv")
        if self.chip == "cpu" and len(onnx_node.inputs) == 3:
            # transpose from (OC,IC,KH,KW) to (OC,KH,KW,IC)
            # self.tensors[onnx_node.inputs[1]] = self.tensors[onnx_node.inputs[1]].transpose(0,2,3,1)
            # self.shapes[onnx_node.inputs[1]] = self.tensors[onnx_node.inputs[1]].shape
            # output_shape = self.shapes[onnx_node.outputs[0]]
            # self.shapes[onnx_node.outputs[0]] = [output_shape[0], output_shape[2], output_shape[3], output_shape[1]]
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
        else:
            raise RuntimeError("not support now")

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
        assert (onnx_node.op_type == "Gemm" or onnx_node.op_type == 'MatMul')
        # (M, K) * (K, N) => (M, N)
        alpha = onnx_node.attrs.get('alpha', 1)
        beta = onnx_node.attrs.get('beta', 1)
        trans_a = onnx_node.attrs.get('transA', 0)
        trans_b = onnx_node.attrs.get('transB', 0)
        # TODO:support more situations
        assert (trans_a == 0)
        operands = list()
        A = onnx_node.inputs[0]
        B = onnx_node.inputs[1]
        A_shape = self.getShape(A)
        B_shape = self.getShape(B)

        in_op = self.getOperand(A)
        operands.append(in_op)

        # reorder for tosa
        if self.chip == "cpu":
            if alpha == 1 and beta == 1 and trans_a == 0 and trans_b == 0 and len(A_shape) == 3 and len(B_shape) == 2:
                assert A_shape[-1] == B_shape[0]
                weight = self.tensors[B]
                self.tensors[B] = weight.reshape(1, B_shape[0], B_shape[1])
                self.shapes[B] = self.tensors[B].shape

        if self.isWeight(B):
            if trans_b == 1 or alpha != 1:
                _tensor = self.getWeight(B)
                if trans_b == 1:
                    _tensor = np.ascontiguousarray(np.transpose(_tensor, (1, 0)))
                if alpha != 1:
                    _tensor *= alpha
                B += '_fix'
                self.addWeight(B, _tensor)
            operands.append(self.getWeightOp(B))
        else:
            operands.append(self.getOperand(B))
        if len(onnx_node.inputs) > 2 and beta != 0:
            C = onnx_node.inputs[2]
            if self.isWeight(C):
                if beta != 1:
                    _tensor = self.getWeight(C)
                    _tensor *= beta
                    C += '_fix'
                    self.addWeight(C, _tensor)
                operands.append(self.getWeightOp(C))
            else:
                operands.append(self.getOperand(C))
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
        op = self.getOperand(onnx_node.inputs[0])
        output_shape = self.getShape(onnx_node.name)
        p = {'name': "{}_{}".format(onnx_node.name, onnx_node.op_type)}
        new_op = self.mlir.create_reshape_op([op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)
    
    def convert_softmax_op(self, onnx_node):
        assert (onnx_node.op_type in ("Softmax"))
        op = self.getOperand(onnx_node.inputs[0])
        output_shape = self.getShape(onnx_node.name)
        axis_default = -1 if self.opset >= 13 else 1
        axis = onnx_node.attrs.get('axis', axis_default)
        if axis < 0:
            axis += len(output_shape)
        p = {
            'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
            'axis': axis,
        }
        new_op = self.mlir.create_softmax_op([op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_transpose_op(self, onnx_node):
        assert (onnx_node.op_type in ("Transpose"))
        op = self.getOperand(onnx_node.inputs[0])
        input_shape = self.getShape(onnx_node.inputs[0])
        output_shape = self.getShape(onnx_node.name)
        # default revert it, eg: shape (2, 3, 4)->(4, 3, 2), per=[2, 1, 0]
        perm_default = list(np.arange(len(input_shape))[::-1])
        transpose_perm = onnx_node.attrs.get('perm', perm_default)
        assert (len(input_shape) == len(transpose_perm))
        p = {
            'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
            'transpose_perm': transpose_perm,
        }
        new_op = self.mlir.create_permute_op([op], output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_concat_op(self, onnx_node):
        assert (onnx_node.op_type == "Concat")
        output_shape = self.getShape(onnx_node.name)
        num_dims = len(output_shape)
        axis = onnx_node.attrs['axis']
        if axis < 0:
            axis += num_dims
        operands = list()
        weight_data = None
        for x in onnx_node.inputs:
            x_shape = self.getShape(x)
            num_elem = np.prod(x_shape)
            if num_elem == 0:
                print("WARNING:{}'s shape is strange {}".format(x, x_shape))
                continue
            if self.isWeight(x):
                data = self.getWeight(x)
                if weight_data is not None:
                    weight_data = np.concatenate((weight_data, data), axis=axis)
                else:
                    weight_data = data
                continue
            else:
                if weight_data is not None:
                    w_name = x + "_weight"
                    self.addWeight(w_name, weight_data)
                    operands.append(self.getWeightOp(w_name))
                    weight_data = None
                operands.append(self.getOperand(x))
        if len(operands) == 0:
            # all weight
            self.addWeight(onnx_node.name, weight_data)
            return
        if weight_data is not None:
            w_name = onnx_node.name + "_weight"
            self.addWeight(w_name, weight_data)
            operands.append(self.getWeightOp(w_name))
        if len(operands) == 1:
            self.addOperand(onnx_node.name, operands[0])
            return
        p = {"name": "{}_{}".format(onnx_node.name, onnx_node.op_type), "axis": axis}
        new_op = self.mlir.create_concat_op(operands, output_shape, **p)
        self.addOperand(onnx_node.name, new_op)

    def convert_layer_norm_op(self, onnx_node):
        assert (onnx_node.op_type == "LayerNormalization")
        assert (len(onnx_node.inputs) <= 3)
        input_shape = self.getShape(onnx_node.inputs[0])
        num_dims = len(input_shape)
        axis = onnx_node.attrs.get("axis", -1)
        if axis < 0:
            axis += num_dims
        normalized_shape = input_shape[axis:]
        eps = onnx_node.attrs.get("epsilon", 1e-05)
        if type(eps) == list and len(eps) == 1:
            eps = eps[0]
        # stash_type is not important
        wb_shape = [1 if i < axis else input_shape[i] for i in range(num_dims)]
        input_opd = self.getOperand(onnx_node.inputs[0])
        scale_opd = self.mlir.none_op
        bias_opd = self.mlir.none_op
        if len(onnx_node.inputs) != 3:
            raise ValueError(f"not support layernorm when len(onnx_node.inputs) == {len(onnx_node.inputs)}")
        output_shape = self.getShape(onnx_node.name)

        p = {
            'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
            'axis': axis,
            'normalized_shape': normalized_shape,
            'eps': eps,
        }
        out_op = self.mlir.create_layer_norm_op([input_opd, scale_opd, bias_opd], output_shape, **p)
        self.addOperand(onnx_node.name, out_op)


    def convert_split_op(self, onnx_node):
        assert (onnx_node.op_type == "Split")
        input_shape = self.getShape(onnx_node.inputs[0])
        num_output = len(onnx_node.outputs)
        num_dims = len(input_shape)
        axis = onnx_node.attrs['axis']
        if axis < 0:
            axis += num_dims
        slice = input_shape[axis] // num_output
        split = None
        # to avoid the case that split attr in input
        if len(onnx_node.inputs) > 1:
            split = self.getWeight(onnx_node.inputs[1]).astype(int)
        else:
            split = onnx_node.attrs.get('split', [slice] * num_output)
        op = self.getOperand(onnx_node.inputs[0])

        offset = 0
        # replace the split with slice
        for i, name in zip(split, onnx_node.outputs):
            output_shape = list(input_shape)
            output_shape[axis] = i
            slice_offset = [0] * num_dims
            slice_offset[axis] = offset
            slice_step = [1] * num_dims
            slice_end = [input_shape[i] for i in range(num_dims)]
            offset = offset + i
            slice_end[axis] = offset
            p = {
                'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
                'offset': list(slice_offset),
                'steps': list(slice_step),
                'ends': list(slice_end),
            }
            new_op = self.mlir.create_slice_op([op]+[self.mlir.none_op]*3, output_shape, **p) 
            self.addOperand(name, new_op)


    def convert_gelu_op(self, onnx_node):
        # 0.5 * val * (1.0 + std::erf(val / std::sqrt(2.0)));
        assert (onnx_node.op_type == "GELU")
        assert (len(onnx_node.inputs) == 1)
        operand = self.getOperand(onnx_node.inputs[0])
        output_shape = self.getShape(onnx_node.name)
        p = {
            'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
        }
        new_op = self.mlir.create_gelu_op([operand], output_shape, **p) 
        self.addOperand(onnx_node.name, new_op)

    def convert_slice_op(self, onnx_node):
        assert (onnx_node.op_type == "Slice")
        if len(onnx_node.inputs) == 5:
            input_name, start_name, end_name, axes_name, step_name = onnx_node.inputs
            in0 = self.getOperand(onnx_node.inputs[0])
            in0_shape = self.getShape(onnx_node.inputs[0])
            out_shape = self.getShape(onnx_node.name)
            
            if self.isScalar(onnx_node.inputs[1]) and self.isScalar(onnx_node.inputs[2]) and self.isScalar(onnx_node.inputs[4]):
                axis = int(self.tensors[axes_name].flatten()[0])
                start = int(self.getScalar(onnx_node.inputs[1]))
                end = int(self.getScalar(onnx_node.inputs[2]))
                step = int(self.getScalar(onnx_node.inputs[4]))
                offset = list(range(start, end, step))
                slice_shape = list(np.take(np.ones(in0_shape), offset, axis=axis).shape)

                # start_list and size_list for tosa
                if self.chip == 'cpu':
                    size = len(in0_shape)
                    start_list =  [int(self.getScalar(onnx_node.inputs[1]))] * size
                    size_list = slice_shape
                    
                # add slice
                p = {
                    'name': "{}_Slice_{}".format(onnx_node.name, onnx_node.op_type),
                    'axis': axis,
                    'offset': offset,
                    'start_list': start_list,
                    'size_list': size_list
                }
                slice_op = self.mlir.create_slice_op([in0], slice_shape, **p)
                self.addOperand(onnx_node.name, slice_op)
                return
            else:
                raise RuntimeError("not support now")
        else:
            raise RuntimeError("not support now")

    def convert_gather_op(self, onnx_node):
        assert (onnx_node.op_type == "Gather")
        in0 = self.getOperand(onnx_node.inputs[0])
        in0_shape = self.getShape(onnx_node.inputs[0])
        out_shape = self.getShape(onnx_node.name)
        axis = onnx_node.attrs.get('axis', 0)

        if self.isScalar(onnx_node.inputs[1]):
            offset = int(self.getScalar(onnx_node.inputs[1]))
            if offset < 0:
                offset = in0_shape[axis] + offset
            slice_offset = [0] * len(in0_shape)
            slice_step = [1] * len(in0_shape)
            slice_end = [in0_shape[i] for i in range(len(in0_shape))]
            slice_offset[axis] = offset
            slice_end[axis] = offset + 1
            slice_shape = list(np.take(np.ones(in0_shape), np.array([offset]), axis=axis).shape)

            # add slice
            p = {
                'name': "{}_Slice_{}".format(onnx_node.name, onnx_node.op_type),
                'offset': list(slice_offset),
                'steps': list(slice_step),
                'ends': list(slice_end),
            }
            slice_op = self.mlir.create_slice_op([in0]+[self.mlir.none_op]*3, slice_shape, **p)
            # add reshape
            p = {
                'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
            }
            new_op = self.mlir.create_reshape_op([slice_op], out_shape, **p)
            self.addOperand(onnx_node.name, new_op)
            return
        else:
            raise RuntimeError("not support now")
        
    def convert_reduce_mean_op(self, onnx_node):
        assert (onnx_node.op_type =="ReduceMean")
        output_shape = self.getShape(onnx_node.name)

        operand = self.getOperand(onnx_node.inputs[0])
        input_shape = self.getShape(onnx_node.inputs[0])
        num_dims = len(input_shape)
        axes = onnx_node.attrs.get("axes", [-1])

        # axes : list  
        # axis : int
        if axes == [-1]:
            axis = axes[0]
            if axis < 0:
                axis += num_dims
            p = {
                'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
                'axis': axis
            }
            new_op = self.mlir.create_reduce_mean_op([operand], output_shape, **p)
            self.addOperand(onnx_node.name, new_op)
        else:
            raise RuntimeError("not support now")

    def convert_pow_op(self, onnx_node):
        assert (onnx_node.op_type == "Pow")
        assert (len(onnx_node.inputs) == 2)
        base = onnx_node.inputs[0]
        expn = onnx_node.inputs[1]
        if self.isScalar(expn):
            base_op = self.getOperand(base)
            expn_const = self.getScalar(expn)
            output_shape = self.getShape(onnx_node.name)
            if expn_const == 2.0:
                p = {
                    'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
                }
                mul_op = self.mlir.create_mul_op([base_op, base_op], output_shape, **p)
                self.addOperand(onnx_node.name, mul_op)
                return
            else:
                raise RuntimeError("Not implemented")
        else:
            raise RuntimeError("Not implemented")
        
    def convert_sqrt_op(self, onnx_node):
        assert (onnx_node.op_type == "Sqrt")
        operand = self.getOperand(onnx_node.inputs[0])
        output_shape = self.getShape(onnx_node.name)
        p = {
            'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
        }
        sqrt_op = self.mlir.create_sqrt_op([operand], output_shape, **p)
        self.addOperand(onnx_node.name, sqrt_op)

    def convert_mul_op(self, onnx_node):
        assert (onnx_node.op_type == "Mul")
        assert (len(onnx_node.inputs) == 2)
        lhs = onnx_node.inputs[0]
        rhs = onnx_node.inputs[1]
        lhs_shape = self.getShape(lhs)
        rhs_shape = self.getShape(rhs)
        if self.isWeight(lhs) and not self.isWeight(rhs):
            onnx_node.inputs[0], onnx_node.inputs[1] = rhs, lhs
            self.convert_mul_op(onnx_node)
            return

        op0 = self.getOperand(lhs)
        output_shape = self.getShape(onnx_node.name)

        if (not self.isWeight(lhs)) and self.isWeight(rhs):
            # weight reorder for tosa
            if len(lhs_shape) == 3 and (len(rhs_shape) == 1 or len(rhs_shape) == 0) and self.chip == "cpu":
                weight = self.getWeight(rhs)
                weight = weight.reshape(1,1,-1)
                self.tensors[rhs] = weight
                self.shapes[rhs] = weight.shape
                weight_op = self.getWeightOp(rhs)
                p = {
                    'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
                }
                mul_op = self.mlir.create_mul_op([op0, weight_op], output_shape, **p)
                self.addOperand(onnx_node.name, mul_op)
                return
            else:
                raise RuntimeError("not support now")
        else:
            if lhs_shape == rhs_shape:
                op1 = self.getOperand(rhs)
                p = {
                    'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
                }
                mul_op = self.mlir.create_mul_op([op0, op1], output_shape, **p)
                self.addOperand(onnx_node.name, mul_op)
            else:
                raise RuntimeError("not support now")

    def convert_erf_op(self, onnx_node):
        assert (onnx_node.op_type == "Erf")
        op = self.getOperand(onnx_node.inputs[0])
        output_shape = self.getShape(onnx_node.name)
        p = {
            'name': "{}_{}".format(onnx_node.name, onnx_node.op_type),
        }
        erf_op = self.mlir.create_erf_op([op], output_shape, **p)
        self.addOperand(onnx_node.name, erf_op)