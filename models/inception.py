import torch
import torch.nn as nn
import torchvision

from .models import register

__all__ = ['inception_v3']

class Inception(nn.Module):

    def __init__(self, pretrained):
        super(Inception, self).__init__()

        if pretrained :
            print('init model from official.')
        self.model = torchvision.models.inception_v3(pretrained=pretrained)
        self.model.aux_logits = False
        self.out_dim = self.model.fc.in_features

        self.model.fc = nn.Identity()
        # print(self.model)
        

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.model(x)
        x = torch.flatten(x, 1)
        # print(x.shape)
        return x

    def forward(self, x):
        return self._forward_impl(x)

@register('inception_v3')
def inception_v3(pretrained=False, progress=True, **kwargs):
    return Inception(pretrained=pretrained)

