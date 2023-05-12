# ONNX Node define:
# https://github.com/onnx/onnx/blob/main/docs/Operators.md



from .MLIRImporter import MLIRImporter
from .BaseConverter import BaseConverter
from numbers import Number
import numpy as np
import torch

class Platform:
    ONNX = "ONNX"
    TORCH = "TORCH"
    TFLITE = "TFLITE"
    CAFFE = "CAFFE"
    TPULANG = "TPULANG"

class BaseNode():

    def __init__(self, info):
        self.name = str(info["name"])
        self.op_type = str(info["op_type"])
        self.inputs = list(info["inputs"])
        self.outputs = list(info["outputs"])

class TorchNode(BaseNode):

    def __init__(self, node):
        info = dict()
        op_type = node.kind()
        info["op_type"] = op_type if not op_type.endswith("_") else op_type[:-1]
        info["inputs"] = [inp.debugName() for inp in node.inputs()]
        info["outputs"] = [outp.debugName() for outp in node.outputs()]
        info["name"] = info["outputs"][0]
        super().__init__(info)
        self.node_proto = node

def get_attr(model: torch.jit.RecursiveScriptModule, node: torch.Node):
    if node.kind() == 'prim::Param':
        return (model, '')
    if node.kind() == 'prim::GetAttr':
        name = node.s('name')
        obj, parent = get_attr(model, node.input().node())
        return (getattr(obj, name), parent + '.' + name if len(parent) > 0 else name)

def get_constant(node: torch.Node):
    """Retrieve a constant associated with this prim::Constant node"""
    attribute_names = node.attributeNames()
    num_attributes = len(attribute_names)
    name = node.output().debugName()
    is_tensor = False
    type = node.output().type().kind()
    value = None
    if type == "NoneType":
        return name, None, True
    elif num_attributes == 1:
        attr_name = attribute_names[0]
        if type == "IntType":
            value = node.i(attr_name)
        elif type == "BoolType":
            value = bool(node.i(attr_name))
        elif type in ["FloatType", "LongType"]:
            value = node.f(attr_name)
        elif type in ["DeviceObjType", "StringType"]:
            value = node.s(attr_name)
        elif type in ["TensorType", "CompleteTensorType"]:
            is_tensor = True
            tensor = node.t(attr_name)
            if tensor.is_cuda:
                tensor = tensor.cpu()
            value = tensor.numpy()
        else:
            raise NotImplementedError("Unsupported type: %s" % type)
    else:
        assert num_attributes == 0
        return None
    return name, value, is_tensor


class TorchConverter(BaseConverter):
    def __init__(self, model_name: str, torch_file: str, input_shapes: list, input_types: list, output_names: list, mlir_file: str):
        super().__init__()
        self.model_name = model_name
        self.weight_file = "{}_top_origin_weight.npz".format(model_name)
        self.model = None
        self.mlir = None
        self.node_name_mapping = {}  # used in torch opt
        self.load_torch_model(torch_file, input_shapes, input_types, output_names)
        self.init_MLIRImporter()
        self.unranked_type = self.mlir.get_tensor_type([])
        self.converted_nodes = list()
        self.const_val = dict()
        self.preprocess_args = dict()
        self.mlir_file = mlir_file
        self.op_factory = {
            "aten::matmul": lambda node: self.convert_matmul_op(node),
            "aten::add": lambda node: self.convert_add_op(node),
            ###### prim #####
            "prim::Constant": lambda node: self.convert_constant(node),
            "prim::GetAttr": lambda node: self.convert_get_attr(node),
        }
        # yapf: enable
        self.check_op_types()

    def __del__(self):
        if self.mlir != None:
            del self.mlir
            self.mlir = None

    def check_op_types(self):
        op_types = self.get_all_op_types()
        known_ops = list(self.op_factory.keys())

        unknown_ops = []
        for op_type in op_types:
            if op_type not in known_ops:
                if not (op_type.endswith("_") and op_type[:-1] in known_ops):
                    unknown_ops.append(op_type)
        if len(unknown_ops) != 0:
            raise RuntimeError(
                "The following operators are not implemented: {}".format(unknown_ops))

    def get_all_op_types(self):
        """Return all operator names in the input graph"""
        self.nodes = list(self.graph.nodes())
        prim_blocks = ["prim::If", "prim::Loop"]
        for prim in prim_blocks:
            prim_nodes = self.graph.findAllNodes(prim, recurse=True)
            for prim_node in prim_nodes:
                for block in prim_node.blocks():
                    self.nodes += block.nodes()
        return set(node.kind() for node in self.nodes)

    def get_input_by_name(self, input):
        return self.const_val[input] if input in self.const_val.keys() else self.getOp(input)

    def get_loc(self, names):
        if isinstance(names, str):
            return Location.fused([Location.name(names)], context=self.mlir.ctx)
        elif isinstance(names, list):
            return Location.fused([Location.name(n) for n in names], context=self.mlir.ctx)
        else:
            raise RuntimeError("Unknown names:{}".format(names))

    def load_torch_model(self, torch_file, input_shapes: list, input_types: list,
                         output_names: list):
        if isinstance(torch_file, str):
            self.model = torch.jit.load(torch_file, map_location=torch.device('cpu'))
        else:
            self.model = torch_file
        self.model.eval()
        self.graph = self.model.inlined_graph
        self.state_dict = self.model.state_dict()
        is_module = isinstance(self.model, torch.jit.ScriptModule)
        inputs = list(self.graph.inputs())
        inputs = inputs[1:] if is_module else inputs
        self.input_names = []
        if len(input_shapes) != len(inputs):
            raise RuntimeError(f"Input shape not match inputs: {input_shapes}")
        for s, inp in zip(input_shapes, inputs):
            self.input_names.append(inp.debugName())
            self.addShape(inp.debugName(), s)
        self.output_names = []
        if output_names:
            self.output_names = output_names
        else:
            for outp in self.graph.outputs():
                if outp.node().kind() == 'prim::TupleConstruct' or \
                   outp.node().kind() == 'prim::ListConstruct':
                    ins = outp.node().inputs()
                    self.output_names.extend([i.debugName() for i in ins])
                else:
                    self.output_names.append(outp.debugName())
        self.weight_names = []
        self.num_input = len(self.input_names)
        self.num_output = len(self.output_names)
        self.input_shapes = input_shapes
        self.input_types = []
        for t in input_types:
            if t.lower() not in self.TypeMap:
                raise RuntimeError(f"Unknown type {t}")
            self.input_types.append(self.TypeMap[t.lower()])
        self.output_shapes = [[]] * self.num_output


    def init_MLIRImporter(self):
        # init importer
        self.mlir = MLIRImporter(self.input_shapes, self.output_shapes, self.model_name,
                                 [], [])
        self.weight_file = self.mlir.weight_file

    def run(self):
        """convert all to mlir"""
        # add input op
        for idx, _name in enumerate(self.input_names):
            input_ = self.mlir.create_input_op(_name, idx)
            self.addOperand(_name, input_)

        def NoneAndRaise(node):
            raise RuntimeError("{} Op not support now".format(node.op_type))

        self.tensor_list = {}
        self.converted_nodes.clear()
        for node in self.graph.nodes():
            self.converted_nodes.append(TorchNode(node))
        # checkout all type is supported
        unsupported = set()
        for n in self.converted_nodes:
            if n.op_type not in self.op_factory:
                unsupported.add(n.op_type)
        if unsupported:
            raise RuntimeError("Op not support:{}".format(unsupported))

        self.generate_list_map()
        for n in self.converted_nodes:
            self.op_factory.get(n.op_type, lambda x: NoneAndRaise(x))(n)
        # add return op
        return_op = list()
        # Set output
        for idx, _name in enumerate(self.output_names):
            op = self.getOperand(_name)
            return_op.append(op)

        self.mlir.create_return_op(return_op)
        mlir_txt = self.mlir.print_module()
        mlir_file = self.mlir_file
        with open(mlir_file, "w") as f:
            f.write(mlir_txt)
        self.WeightToNpz(self.weight_file)
        print("Save mlir file: {}".format(mlir_file))

    def convert_matmul_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        op1 = self.getOp(torch_node.inputs[1])
        p   = {"name": torch_node.name, "do_relu": False}
        operands = list()
        operands.append(op0).append(op1)
        output_shape = self.getShape(torch_node.name)
        new_op = self.mlir.MatMulOp(operands, output_shape, **p)
        self.addOperand(torch_node.name, new_op)

    def convert_add_op(self, torch_node: TorchNode):
        op0 = self.getOp(torch_node.inputs[0])
        op1 = self.getOp(torch_node.inputs[1])
        p   = {"name": torch_node.name, "do_relu": False}
        operands = list()
        operands.append(op0).append(op1)
        output_shape = self.getShape(torch_node.name)
        new_op = self.mlir.AddOp(operands, output_shape, **p)
        self.addOperand(torch_node.name, new_op)

    def convert_constant(self, torch_node: TorchNode):
        name, value, is_tensor = get_constant(torch_node.node_proto)
        if is_tensor:
            self.const_val[name] = value
        else:
            self.const_val[name] = np.array(value)
        self.addTensor(name, self.const_val[name])
    
    def convert_get_attr(self, torch_node: TorchNode):
        obj, name = get_attr(self.model, torch_node.node_proto)
        if isinstance(obj, Number):
            self.const_val[name] = np.array(obj)
        else:
            self.const_val[name] = obj
        self.addTensor(name, self.const_val[name])