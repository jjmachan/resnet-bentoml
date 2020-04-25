from PIL import Image

from torchvision import transforms

import bentoml
from bentoml.artifact import PytorchModelArtifact
from bentoml.handlers import ImageHandler

classes = ['ant', 'bee']

@bentoml.env(pip_dependencies=['torch', 'torchvision'])
@bentoml.artifacts(PytorchModelArtifact('net'))
class AndorBeeClassifier(bentoml.BentoService):

    @bentoml.api(ImageHandler)
    def predict(self, img):
        print(type(img))
        print(img)
