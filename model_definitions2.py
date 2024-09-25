# model_definitions.py

import torch
import torch.nn as nn
from torchvision.models import resnet50

class ModelA(nn.Module):
    def __init__(self, original_model=None):
        super(ModelA, self).__init__()
        if original_model is None:
            original_model = resnet50(pretrained=False)
            original_model.load_state_dict(torch.load('./resnet50-pretrained.pth'))
        self.features = nn.Sequential(
            original_model.conv1,
            original_model.bn1,
            original_model.relu,
            original_model.maxpool,
            original_model.layer1,
            original_model.layer2  # New split after layer2
        )

    def forward(self, x):
        x = self.features(x)
        return x

class ModelB(nn.Module):
    def __init__(self, original_model=None):
        super(ModelB, self).__init__()
        if original_model is None:
            original_model = resnet50(pretrained=False)
            original_model.load_state_dict(torch.load('./resnet50-pretrained.pth'))
        self.features = nn.Sequential(
            original_model.layer3,
            original_model.layer4,
            original_model.avgpool
        )
        self.fc = original_model.fc

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
