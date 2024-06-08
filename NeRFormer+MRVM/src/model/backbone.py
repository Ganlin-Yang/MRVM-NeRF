from .transformer import NerFormer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


class NetWrapper(nn.Module):
    def __init__(self, conf, d_latent=512, d_in=78, d_out=4):
        super().__init__()
        self.hidden_size = conf['nerformer']['hidden_size']
        self.rgb_fc = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size//2),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_size//2, d_out))
        
        self.nerformer = NerFormer.from_conf(conf['nerformer'], d_in=d_in, d_latent=d_latent, d_out=d_out) 
           
    
    def forward(self, rgb_feat, position, mask_index=None):
        out_feat = self.nerformer(rgb_feat, position, mask_index=mask_index) # shape: (SB, num_rays, num_points, hidden_size)
        outputs = self.rgb_fc(out_feat)   # shape: (SB, num_rays, num_points, 4)
        return outputs, out_feat

    @classmethod
    def from_conf(cls, conf, d_in, d_latent, d_out, **kwargs):
        return cls(
            conf = conf,
            d_latent = d_latent,
            d_in = d_in,
            d_out = d_out
        )