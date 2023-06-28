import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, x):
        return F.softmax(x)
shape0 = (1, 2, 2, 72)
shape1 = (1, 2, 2, 72)
model = Model().eval()

torch.onnx.export(
        model,
        torch.rand(shape0),
        'softmax.onnx',
        ['a'],
        ['Y'],
        opset_version=11
        )
