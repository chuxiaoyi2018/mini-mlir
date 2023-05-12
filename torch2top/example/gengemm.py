import torch 


class Gemm(torch.nn.Module):
    def __init__(self):
        super(Gemm, self).__init__()
        self.t = torch.nn.Parameter(torch.randn(2, 1))
    def forward(self, x, y):
        return torch.matmul(x, y) + self.t

shape0 = (2, 72)
shape1 = (72, 1)
model = Gemm().eval()
jit_model = torch.jit.trace(model, (torch.randn(shape0), torch.randn(shape1)))
jit_model.save("gemm.pt")

