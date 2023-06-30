from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import torch

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])


class vit_lite(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        x = self.model.vit.embeddings(x)
        x = self.model.vit.encoder.layer[0](x)
        x = self.model.vit.layernorm(x[0])
        x = self.model.classifier(x[:,0])
        return x
            
lite_model = vit_lite(model)
torch.onnx.export(lite_model, torch.randn(1,3,224,224), 'vit-lite.onnx', verbose=False)
