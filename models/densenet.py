import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from .models import register


__all__ = ['densenet121']

class DenseNet(nn.Module):

    def __init__(self, pretrained):
        super(DenseNet, self).__init__()

        if pretrained :
            print('init model from official.')
        model = torchvision.models.densenet121(pretrained=pretrained)
        self.out_dim = model.classifier.in_features
        self.model = model.features
        # print(model)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        features = self.model(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out

    def forward(self, x):
        return self._forward_impl(x)

@register('densenet121')
def densenet121(pretrained=False, progress=True, **kwargs):
    return DenseNet(pretrained=pretrained)
