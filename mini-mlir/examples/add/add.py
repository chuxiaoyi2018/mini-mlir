import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, x, y):
        return x + y
shape0 = (2, 72)
shape1 = (2, 72)
model = Model().eval()

torch.onnx.export(
        model,
        (torch.rand(shape0), torch.rand(shape1)),
        'add.onnx',
        ['a', 'b'],
        ['Y'],
        opset_version=11
        )
