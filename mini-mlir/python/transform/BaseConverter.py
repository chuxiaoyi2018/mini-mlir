import numpy as np
class BaseConverter(object):
    def __init__(self):
        self.operands = dict()
        self.tensors = dict()
        self.shapes = dict()

    def run(self):
        raise NotImplementedError('run')

    def addShape(self, name, shape):
        if isinstance(shape, tuple):
            shape = list(shape)
        elif not isinstance(shape, list):
            raise KeyError("{}:{} unknown shape".format(name, shape))
        if name in self.shapes:
            if self.shapes[name] != shape:
                raise KeyError("shape {} conflict".format(name))
        self.shapes[name] = shape

    def getShape(self, name):
        if name not in self.shapes:
            raise KeyError("shape {} not found".format(name))
        return self.shapes[name]

    def addOperand(self, name, op):
        if name in self.operands:
            if self.operands[name] != op:
                raise KeyError("operand {} conflict".format(name))
            return
        self.operands[name] = op

    def getOperand(self, name):
        if name not in self.operands:
            raise KeyError("operand {} not found".format(name))
        return self.operands[name]

    def addTensor(self, name, data):
        if name in self.tensors:
            raise KeyError("tensor {} conflict".format(name))
        if not isinstance(data, np.ndarray):
            raise KeyError("tensor data must be numpy array")
        self.tensors[name] = data
        self.addShape(name, data.shape)

    def isTensor(self, name):
        if name in self.tensors:
            return True
        return False

    def getTensor(self, name):
        if name not in self.tensors:
            raise KeyError("No {} tensor in model".format(name))
        return self.tensors[name]

    def addWeight(self, name, data):
        if not isinstance(data, np.ndarray):
            raise KeyError("tensor data must be numpy array")
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        if name in self.tensors:
            if np.all(self.tensors[name] == data):
                return
            raise KeyError("tensor {} conflict".format(name))
        if len(data.shape) == 0:
            data = data.reshape([1])
        # all weight convert to f32.
        self.tensors[name] = data
        self.addShape(name, data.shape)

    def isWeight(self, name):
        if name in self.tensors:
            return True
        return False

    def getWeight(self, name):
        if name not in self.tensors:
            raise KeyError("No {} tensor in model".format(name))
        return self.tensors[name]

    def getWeightOp(self, name):
        if name not in self.tensors:
            raise KeyError("Should addTensor first:{}!!!".format(name))
        op = self.mlir.create_weight_op(name, self.getShape(name))
        self.addOperand(name, op)
        return op

    def WeightToNpz(self, weight_file):
        tensor_npz = {}
        for name in self.tensors:
            if name in self.operands:
                tensor_npz[name] = self.tensors[name]
        np.savez(weight_file, **tensor_npz)

    def isScalar(self, name):
        if not self.isWeight(name): return False
        if np.prod(self.getShape(name)) == 1: return True
        w = self.getWeight(name)
        return np.all(w == w.flatten()[0])

    def getScalar(self, name):
        if not self.isScalar(name):
            raise RuntimeError("Not Scalar")
        return self.getWeight(name).flatten()[0]

