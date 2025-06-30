import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
import numpy as np
import pickle
import cv2
import os
from torchvision.utils import save_image

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input

class Generate_gate(nn.Module):
    def __init__(self):
        super(Generate_gate, self).__init__()
        self.proj = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(512,256, 1),
                                  nn.ReLU(),
                                  nn.Conv2d(256,512, 1),
                                  nn.ReLU())

        self.epsilon = 1e-8
    def forward(self, x):

        alpha = self.proj(x)
        gate = (alpha**2) / (alpha**2 + self.epsilon)

        return gate

def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False


def freeze_direct(layer):
    for param in layer.parameters():
        param.requires_grad = False


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8),
        res_blocks=3,
        dropout=0,
        with_noise_level_emb=True,
        image_size=128
    ):
        super().__init__()

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=False)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                        dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

        self.mask_tail = FCN()

        self.channel_attention = SqueezeExcitationLayer(pre_channel, pre_channel, ratio=4, layer_name='channel_layer')

        ###################### init CSNorm ##################
        self.gate = Generate_gate()
        for i in range(512):
            setattr(self, 'CSN_' + str(i), nn.InstanceNorm2d(1, affine=True))
            freeze_direct(getattr(self, 'CSN_' + str(i)))
        freeze(self.gate)

    def forward(self, x, mtm, time, continous):
        x_lr = x[:, :3, :, :]
        x_mask = x[:, 3, :, :].unsqueeze(1)
        x_noisy = x[:, 4:, :, :]
        # updated_mask = self.mask_update(x_noisy, x_mask)
        # x_updated_mask = updated_mask.detach()
        x = torch.cat((x_lr, x_mask, x_noisy), dim=1)

        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None

        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)
        if continous:
            gate = self.gate(x)
            lq_copy = torch.cat([getattr(self, 'CSN_' + str(i))(x[:,i,:,:][:,None,:,:]) for i in range(512)], dim=1)
            x = gate * (lq_copy) + (1-gate) * x

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                feat = feats.pop()
                # if x.shape[2]!=feat.shape[2] or x.shape[3]!=feat.shape[3]:
                #     feat = F.interpolate(feat, x.shape[2:])
                x = layer(torch.cat((x, feat), dim=1), t)
                mtm = F.interpolate(mtm, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
                x = x + (x * mtm)
                
            else:
                x = layer(x)
                mtm = F.interpolate(mtm, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
                x = x + (x * mtm)

        #f1 = self.final_conv(x)
        #attention_x = x + (x * mtm)
        #save_image(f1, 'ret_img-1.jpg')
        #x_attention=self.channel_attention(x)

        #gate = self.gate(attention_x)
        #lq_copy = torch.cat([getattr(self, 'CSN_' + str(i))(attention_x[:,i,:,:][:,None,:,:]) for i in range(64)], dim=1)
        #gate_x = gate * (lq_copy) + (1-gate) * attention_x

        f = self.final_conv(x)

        m=self.mask_tail(x)

        #save_image(f, 'ret_img-2.jpg')

        return f, m



class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fcn = nn.Sequential(
            # nn.Conv2d(4, 32, 3, padding=1),
            # nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        # xin = torch.cat((x, mask), dim=1)
        return self.fcn(x)


class SqueezeExcitationLayer(nn.Module):
    def __init__(self, in_channels, out_dim, ratio, layer_name):
        super(SqueezeExcitationLayer, self).__init__()
        self.layer_name = layer_name
        self.out_dim = out_dim
        self.ratio = ratio
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, out_dim // ratio)
        self.fc2 = nn.Linear(out_dim // ratio, out_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_x):
        squeeze = self.global_avg_pooling(input_x)
        excitation = self.fc1(squeeze.view(squeeze.size(0), -1))
        excitation = self.relu(excitation)

        excitation = self.fc2(excitation)
        excitation = self.sigmoid(excitation)

        excitation = excitation.view(-1, self.out_dim, 1, 1)

        scale = input_x * excitation + input_x

        return scale