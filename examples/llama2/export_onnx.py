import torch
from model import ModelArgs, Transformer

checkpoint = torch.load('stories42M.pt')
gptconf = ModelArgs(**checkpoint['model_args'])
model = Transformer(gptconf)

state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict, strict=False)
model.eval()

class model_lite(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.freqs_cos = model.freqs_cos[:SEQ_LENGTH]
        self.freqs_sin = model.freqs_sin[:SEQ_LENGTH]

    def forward(self, x):
        x = self.model.tok_embeddings(x)
        x = self.model.layers[0](x, self.freqs_cos, self.freqs_sin)
        #for layer in self.model.vit.encoder.layer:
        #   x = layer(x)[0]
        x = self.model.norm(x[:,0:1])
        x = self.model.output(x)
        return x

SEQ_LENGTH = 256

def convert_model():
    # input
    input_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.long)

    torch.onnx.export(
        model, (input_ids),
        f'llama2.onnx',
        verbose=False,
        input_names=['input_ids'],
        output_names=['logits'],
        do_constant_folding=True,
        opset_version=15)
    
def convert_lite_model():
    # input
    input_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.long)
    lite_model = model_lite(model)

    torch.onnx.export(
        lite_model, (input_ids),
        f'llama2-lite.onnx',
        verbose=False,
        input_names=['input_ids'],
        output_names=['logits'],
        do_constant_folding=True,
        opset_version=15)

convert_lite_model()
