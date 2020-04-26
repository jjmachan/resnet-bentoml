from PIL import Image

import torch
from torchvision import transforms

import bentoml
from bentoml.artifact import PytorchModelArtifact
from bentoml.handlers import ImageHandler

classes = ['ant', 'bee']
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
cpu = torch.device('cpu')

@bentoml.env(pip_dependencies=['torch', 'torchvision'])
@bentoml.artifacts([PytorchModelArtifact('model')])
class AntOrBeeClassifier(bentoml.BentoService):

    @bentoml.api(ImageHandler)
    def predict(self, img):
        img = Image.fromarray(img)
        img = transform(img)

        self.artifacts.model.eval()
        outputs = self.artifacts.model(img.unsqueeze(0))
        _, idxs = outputs.topk(1)
        idx = idxs.squeeze().item()
        return classes[idx]
