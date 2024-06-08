import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

class FeedForward(nn.Module):
    def __init__(self, dim, hid_dim, dp_rate):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, dim)
        self.dp = nn.Dropout(dp_rate)
        self.activ = nn.ReLU()

    def forward(self, x):
        x = self.dp(self.activ(self.fc1(x)))
        x = self.dp(self.fc2(x))
        return x

class Attention(nn.Module):
    def __init__(self, dim, n_heads, dp_rate):
        super(Attention, self).__init__()
        
        self.q_fc = nn.Linear(dim, dim, bias=False)
        self.k_fc = nn.Linear(dim, dim, bias=False)
        self.v_fc = nn.Linear(dim, dim, bias=False)
        self.out_fc = nn.Linear(dim, dim)
        self.dp = nn.Dropout(dp_rate)
        self.n_heads = n_heads
       

    def forward(self, x):
        q = self.q_fc(x)
        q = q.view(x.shape[0], x.shape[1], self.n_heads, -1).permute(0, 2, 1, 3)
        k = self.k_fc(x)
        k = k.view(x.shape[0], x.shape[1], self.n_heads, -1).permute(0, 2, 1, 3)
        v = self.v_fc(x)
        v = v.view(x.shape[0], x.shape[1], self.n_heads, -1).permute(0, 2, 1, 3)
        attn = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(q.shape[-1])
        attn = torch.softmax(attn, dim=-1)
        attn = self.dp(attn)

        out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous()
        out = out.view(x.shape[0], x.shape[1], -1)
        out = self.dp(self.out_fc(out))
        
        return out

class Transformer(nn.Module):
    def __init__(
        self, dim, ff_hid_dim, n_heads, ff_dp_rate, attn_dp_rate,
    ):
        super(Transformer, self).__init__()
        self.attn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ff_norm = nn.LayerNorm(dim, eps=1e-6)

        self.ff = FeedForward(dim, ff_hid_dim, ff_dp_rate)
        self.attn = Attention(dim, n_heads, attn_dp_rate)

    def forward(self, x):
        residue = x
        x = self.attn_norm(x)
        x = self.attn(x)
        x = x + residue

        residue = x
        x = self.ff_norm(x)
        x = self.ff(x)
        x = x + residue

        return x


class NerFormer(nn.Module):
    def __init__(self, layer_num=4, hidden_size=128, num_head=4, in_feature_size=512, in_position_size=78, output_size=4):
        super(NerFormer, self).__init__()
        self.rgbfeat_fc = nn.Linear(in_feature_size, hidden_size)
        self.color_dir_fc = nn.Linear(hidden_size+7, hidden_size)
        self.positionfeat_fc = nn.Sequential(
            nn.Linear(in_position_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, 1, hidden_size))
        trunc_normal_(self.mask_token, mean=0., std=.02)
        self.view_trans = nn.ModuleList([])
        self.ray_trans = nn.ModuleList([])
        for i in range(layer_num):
            # view transformer
            view_trans = Transformer(
                dim=hidden_size,
                ff_hid_dim=int(hidden_size * 4),
                n_heads=num_head,
                ff_dp_rate=0.1,
                attn_dp_rate=0.1,
            )
            self.view_trans.append(view_trans)
            # ray transformer
            ray_trans = Transformer(
                dim=hidden_size,
                ff_hid_dim=int(hidden_size * 4),
                n_heads=num_head,
                ff_dp_rate=0.1,
                attn_dp_rate=0.1,
            )
            self.ray_trans.append(ray_trans)
        self.hidden_size = hidden_size
        self.norm = nn.LayerNorm(hidden_size)
        self.view_aggregator = nn.Linear(self.hidden_size, 1)

    def forward(self, rgb_feat, position, mask_index=None):
        
        SB, num_views, num_rays, num_points, _ = rgb_feat.shape
        assert position.shape[0]==SB and position.shape[1]==num_rays and position.shape[2]==num_points
        rgb_feat_ = self.rgbfeat_fc(rgb_feat[...,7:]) # shape: (SB, num_views, num_rays, num_points, hidden_size)
        rgb_feat = self.color_dir_fc(torch.cat((rgb_feat_,rgb_feat[...,:7]),dim=-1)) # (SB, num_views, num_rays, num_points, hidden_size)
        position_feature = self.positionfeat_fc(position) # shape: (SB, num_rays, num_points, hidden_size)
        
        # Perform masking operation during pretraining stage (only operate on fine sampling stage)
        if mask_index is not None: # mask_index.shape: (SB, num_views, num_rays, num_points, 1)
            mask_token = self.mask_token.expand(rgb_feat.shape)
            rgb_feat = rgb_feat * (1-mask_index) + mask_token * mask_index
        
        # Perform positional encoding 
        x = rgb_feat + position_feature.unsqueeze(1) # shape: (SB, num_views, num_rays, num_points, hidden_size)
        x = x.permute(0,2,1,3,4).reshape(-1, num_points, self.hidden_size) # shape: (SB * num_rays * num_views, num_points, hidden_size)
        # transformer modules
        for i, (viewtrans, raytrans) in enumerate(
            zip(self.view_trans, self.ray_trans)
        ):  
            
            x = x.reshape(-1, num_views, num_points, self.hidden_size) # shape: (SB * num_rays, num_views, num_points, hidden_size)
            x = x.transpose(1,2).reshape(-1, num_views, self.hidden_size)  # shape: (SB * num_rays * num_points, num_views, hidden_size)
            # view transformer 
            x = viewtrans(x)
            # ray transformer
            x = x.reshape(-1, num_points, num_views, self.hidden_size)   # shape: (SB * num_rays, num_points, num_views, hidden_size)
            x = x.transpose(1, 2).reshape(-1, num_points, self.hidden_size)  # shape: (SB * num_rays * num_views, num_points, hidden_size)
            x = raytrans(x)
        
        # normalize 
        out_feat = self.norm(x)  # shape : (SB * num_rays * num_views, num_points, hidden_size)
        out_feat = out_feat.reshape(-1, num_rays, num_views, num_points, self.hidden_size)
        view_weighter = self.view_aggregator(out_feat).softmax(dim=2)  # shape: (SB , num_rays , num_views, num_points, 1)
        aggregated_feat = (view_weighter * out_feat).sum(dim=2) # shape: (SB, num_rays, num_points, hidden_size)
        return aggregated_feat

    @classmethod
    def from_conf(cls, conf, d_in, d_latent, d_out, **kwargs):
        # PyHocon construction
        return cls(
            layer_num = conf.get_int("layer_num", 4),
            hidden_size = conf.get_int("hidden_size", 128),
            num_head = conf.get_int("num_head", 4),
            in_feature_size= d_latent,
            in_position_size = d_in,
            output_size = d_out
        )
