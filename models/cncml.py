import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import utils
from .models import register


@register('cncml')
class CNCML(nn.Module):

    def __init__(self, encoder, encoder_args={}, method='cos',
                 temp=10., temp_learnable=True):
        print('init meta model. method {}, temp {}, temp_learnable {}'.format(method, temp, temp_learnable))
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        self.method = method

        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

    def forward(self, x_shot, x_query, nearest_point=False):
        shot_shape = x_shot.shape[:-3]
        query_shape = x_query.shape[:-3]
        img_shape = x_shot.shape[-3:]
 
        # print('step1- x_shot shape:{}, x_query shape:{}'.format(x_shot.shape, x_query.shape))
        x_shot = x_shot.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)
        # print('step2- x_shot shape:{}, x_query shape:{}'.format(x_shot.shape, x_query.shape))
        x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))
        x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
        # print('step3- x_shot shape:{}, x_query shape:{}'.format(x_shot.shape, x_query.shape))
        x_shot = x_shot.view(*shot_shape, -1)
        x_query = x_query.view(*query_shape, -1)
        # print('step4- x_shot shape:{}, x_query shape:{}'.format(x_shot.shape, x_query.shape))

        if self.method == 'cos':
            if not nearest_point:
                x_shot = x_shot.mean(dim=-2)
            # print('step5- x_shot shape:{}, x_query shape:{}'.format(x_shot.shape, x_query.shape))
            x_shot = F.normalize(x_shot, dim=-1)
            x_query = F.normalize(x_query, dim=-1)
            metric = 'dot'
        elif self.method == 'sqr':
            x_shot = x_shot.mean(dim=-2)
            metric = 'sqr'

        logits = utils.compute_logits(
                x_query, x_shot, metric=metric, temp=self.temp, nearest_point=nearest_point)
        return logits

