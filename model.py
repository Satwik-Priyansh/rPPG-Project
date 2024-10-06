import torch.nn as nn
from torchvision import models

class CNNrPPG(nn.Module):
    def __init__(self):
        super(CNNrPPG, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.fc = nn.Linear(512, 1)

    def forward(self, x):
        return self.base_model(x)
