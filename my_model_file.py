import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import DenseNet121_Weights

class MyDenseNet(nn.Module):
    def __init__(self, num_classes=200, pretrained=False):
        super().__init__()
        
        # Загрузка предобученной модели
        if pretrained:
            self.model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        else:
            self.model = models.densenet121(weights=None)  # Если не нужно, то загружаем модель без весов

        # Изменяем классификатор на нужное количество классов
        self.model.classifier = nn.Linear(in_features=1024, out_features=num_classes)

        # Замораживаем параметры
        for param in self.model.parameters():
            param.requires_grad = False

        # Разморозка высокоуровневых слоев
        for param in self.model.features.denseblock4.parameters():
            param.requires_grad = True

        # Разморозка классификатора
        self.model.classifier.weight.requires_grad = True
        self.model.classifier.bias.requires_grad = True

    def forward(self, x):
        return self.model(x)
