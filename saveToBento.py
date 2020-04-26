import argparse

import torch.nn as nn
from torchvision import models

import utils
from classificationService import AntOrBeeClassifier

def saveToBento(checkpoint):
    model_state_dict, _, _, _, _ = utils.load_model(checkpoint)

    # Define the model
    model_ft =  models.resnet18(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)

    # Load saved model
    model_ft.load_state_dict(model_state_dict)

    # Add model to Bento ML
    bento_svc = AntOrBeeClassifier()
    bento_svc.pack('model', model_ft)

    # Save Bento Service
    saved_path = bento_svc.save()
    print('Bento Service Saved in ', saved_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help='Checkpoint to load the model')
    args = parser.parse_args()

    saveToBento(args.checkpoint)
