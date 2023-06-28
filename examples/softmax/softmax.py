import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, x):
        z = torch.nn.functional.softmax(x)
        return z
shape0 = (2, 72)
model = Model().eval()

torch.onnx.export(
        model,
        (torch.rand(shape0)),
        'softmax.onnx',
        ['input'],
        ['output'],
        opset_version=13
        )
