from torchvision.models import resnet34, ResNet34_Weights
import torch.nn as nn


def build_model(weights=ResNet34_Weights.DEFAULT, fine_tune=True, num_classes=1):
    model = resnet34(weights)

    if fine_tune:
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        for params in model.parameters():
            params.requires_grad = False

    model.fc = nn.Linear(512, num_classes)
    return model

