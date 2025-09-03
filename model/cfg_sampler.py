import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
    
class ClassifierFreeSampleModel_inpaint(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model  # model is the actual model to run

    def forward(self, x, timesteps, y=None, **kwargs):
        y_uncond = deepcopy(y)
        x_uncond = x.clone()
        x_uncond[:, -1] = y_uncond['inpainted_motion_uncond'][:, -1]
        out = self.model(x, timesteps, y=y, **kwargs)
        out_uncond = self.model(x_uncond, timesteps, y=y_uncond, **kwargs)
        return out + y['scale'] * (out - out_uncond)

