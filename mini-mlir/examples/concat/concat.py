import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, x, y):
        z = torch.concat((x, y), 1)
        return z
shape0 = (2, 3)
shape1 = (2, 3)
x = torch.rand(shape0)
y = torch.rand(shape1)
model = Model().eval()


torch.onnx.export(
        model,
        (x, y),
        'concat.onnx',
        ['input'],
        ['output'],
        opset_version=13
        )

