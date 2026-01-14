"""
ScaFu: A Lightweight Scale Fusion Module for Transformer-based Object Detection
Copyright (c) 2026 The ScaFu Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIMv2 (https://github.com/Intellindust-AI-Lab/DEIMv2)
Copyright (c) 2025 huangshihua. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DINOv3 (https://github.com/facebookresearch/dinov3)
Copyright (c) Meta Platforms, Inc. and affiliates.
This software may be used and distributed in accordance with the terms of the DINOv3 License Agreement.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import torchvision.ops as ops

from functools import partial
from ..core import register
from .vit_tiny import VisionTransformer
from .dinov3 import DinoVisionTransformer


class ScaleAttention(nn.Module):
    def __init__(self, in_dim, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_dim * num_scales, num_scales, kernel_size=1, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, scale_feats):
        concat_feat = torch.cat(scale_feats, dim=1)
        global_feat = self.avg_pool(concat_feat)
        attn_weight = self.fc(global_feat)

        new_attn_weight = attn_weight.clone()
        new_attn_weight[:, 0:1, :, :] = new_attn_weight[:, 0:1, :, :] + 0.1
        new_attn_weight = new_attn_weight / new_attn_weight.sum(dim=1, keepdim=True)

        fused_feat = 0
        for i in range(self.num_scales):
            fused_feat += scale_feats[i] * new_attn_weight[:, i:i+1, :, :]
        return fused_feat


class LocalConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3):
        super().__init__()
        self.dcn = nn.Conv2d(
            in_dim, out_dim, kernel_size=kernel_size,
            padding=1, stride=1, bias=False
        )
        self.norm = nn.SyncBatchNorm(out_dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.dcn(x)))


class DynamicScaleFusionModule(nn.Module):
    def __init__(self, in_dim=16, out_dims=[32, 64, 64], target_scales=[4,8,12,16,24,32]):
        super().__init__()
        self.in_dim = in_dim
        self.out_dims = out_dims
        self.target_scales = target_scales

        self.scale_convs = nn.ModuleDict()
        for s in target_scales:
            stride = s // 4
            if s == 4:
                self.scale_convs[f's{s}'] = nn.Sequential(
                    nn.Conv2d(in_dim, in_dim, 3, stride=stride, padding=1, groups=in_dim, bias=False),
                    nn.SyncBatchNorm(in_dim),
                    nn.GELU(),
                    LocalConvLayer(in_dim, in_dim)
                )
                self.s4_shortcut = nn.Conv2d(in_dim, in_dim, 1, stride=stride, padding=0, bias=False)
            else:
                self.scale_convs[f's{s}'] = nn.Sequential(
                    nn.Conv2d(in_dim, in_dim, 3, stride=stride, padding=1, groups=in_dim, bias=False),
                    nn.SyncBatchNorm(in_dim),
                    nn.GELU(),
                    LocalConvLayer(in_dim, in_dim)
                )

        self.cross_scale_interact = nn.ModuleDict({
            'c2': LocalConvLayer(in_dim, in_dim),
            'c3': LocalConvLayer(in_dim, in_dim),
            'c4': LocalConvLayer(in_dim, in_dim)
        })

        self.attn_c2 = ScaleAttention(in_dim, num_scales=3)
        self.attn_c3 = ScaleAttention(in_dim, num_scales=3)
        self.attn_c4 = ScaleAttention(in_dim, num_scales=3)

        self.proj_c2 = nn.Sequential(
            LocalConvLayer(in_dim, out_dims[0]),
            nn.Conv2d(out_dims[0], out_dims[0], 1, 1, bias=False)
        )
        self.proj_c3 = nn.Sequential(
            LocalConvLayer(in_dim, out_dims[1]),
            nn.Conv2d(out_dims[1], out_dims[1], 1, 1, bias=False)
        )
        self.proj_c4 = nn.Sequential(
            LocalConvLayer(in_dim, out_dims[2]),
            nn.Conv2d(out_dims[2], out_dims[2], 1, 1, bias=False)
        )

    def forward(self, c1_sta):
        scale_feats = {}
        for s in self.target_scales:
            if s == 4:
                scale_feat = self.scale_convs[f's{s}'](c1_sta)
                shortcut_feat = self.s4_shortcut(c1_sta)
                scale_feats[f's{s}'] = scale_feat + shortcut_feat
            else:
                scale_feats[f's{s}'] = self.scale_convs[f's{s}'](c1_sta)

        s4_interact = self.cross_scale_interact['c2'](scale_feats['s4'])
        s4_interact_down = F.interpolate(s4_interact, size=scale_feats['s8'].shape[2:],
                                         mode='bilinear', align_corners=False)
        scale_feats['s8'] = scale_feats['s8'] + s4_interact_down

        s12_interact = self.cross_scale_interact['c3'](scale_feats['s12'])
        s12_interact_down = F.interpolate(s12_interact, size=scale_feats['s16'].shape[2:],
                                          mode='bilinear', align_corners=False)
        scale_feats['s16'] = scale_feats['s16'] + s12_interact_down

        s24_interact = self.cross_scale_interact['c4'](scale_feats['s24'])
        s24_interact_down = F.interpolate(s24_interact, size=scale_feats['s32'].shape[2:],
                                          mode='bilinear', align_corners=False)
        scale_feats['s32'] = scale_feats['s32'] + s24_interact_down

        s4 = F.interpolate(scale_feats['s4'], scale_factor=0.5, mode='bilinear', align_corners=False)
        s8 = scale_feats['s8']
        s12 = F.interpolate(scale_feats['s12'], size=s8.shape[2:], mode='bilinear', align_corners=False)
        c2_fused = self.attn_c2([s4, s8, s12])
        c2_out = self.proj_c2(c2_fused)

        s8_c3 = F.interpolate(scale_feats['s8'], scale_factor=0.5, mode='bilinear', align_corners=False)
        s12_c3 = F.interpolate(scale_feats['s12'], size=scale_feats['s16'].shape[2:], mode='bilinear', align_corners=False)
        s16_c3 = scale_feats['s16']
        c3_fused = self.attn_c3([s8_c3, s12_c3, s16_c3])
        c3_out = self.proj_c3(c3_fused)

        s16_c4 = F.interpolate(scale_feats['s16'], scale_factor=0.5, mode='bilinear', align_corners=False)
        s24_c4 = F.interpolate(scale_feats['s24'], size=scale_feats['s32'].shape[2:], mode='bilinear', align_corners=False)
        s32_c4 = scale_feats['s32']
        c4_fused = self.attn_c4([s16_c4, s24_c4, s32_c4])
        c4_out = self.proj_c4(c4_fused)

        return c2_out, c3_out, c4_out


class SpatialPriorModulev2(nn.Module):
    def __init__(self, inplanes=16, use_dynamic_fusion=True, fusion_strength=0.5):
        super().__init__()
        self.use_dynamic_fusion = use_dynamic_fusion
        self.fusion_strength = fusion_strength
        self.inplanes = inplanes

        self.stem = nn.Sequential(
            *[
                nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(inplanes),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )
        self.conv2 = nn.Sequential(
            *[
                nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(2 * inplanes),
            ]
        )
        self.conv3 = nn.Sequential(
            *[
                nn.GELU(),
                nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(4 * inplanes),
            ]
        )
        self.conv4 = nn.Sequential(
            *[
                nn.GELU(),
                nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(4 * inplanes),
            ]
        )

        if self.use_dynamic_fusion:
            self.dynamic_fusion = DynamicScaleFusionModule(
                in_dim=inplanes,
                out_dims=[2*inplanes, 4*inplanes, 4*inplanes],
                target_scales=[4,8,12,16,24,32]
            )
            self.gate_c2 = nn.Sigmoid()
            self.gate_c3 = nn.Sigmoid()
            self.gate_c4 = nn.Sigmoid()

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)

        if self.use_dynamic_fusion:
            fused_c2, fused_c3, fused_c4 = self.dynamic_fusion(c1)

            gate_c2 = self.gate_c2(torch.mean(c2, dim=[2,3], keepdim=True)) * self.fusion_strength
            gate_c3 = self.gate_c3(torch.mean(c3, dim=[2,3], keepdim=True)) * self.fusion_strength
            gate_c4 = self.gate_c4(torch.mean(c4, dim=[2,3], keepdim=True)) * self.fusion_strength

            c2 = c2 + fused_c2 * gate_c2
            c3 = c3 + fused_c3 * gate_c3
            c4 = c4 + fused_c4 * gate_c4

        return c2, c3, c4


@register()
class DINOv3ScaFu(nn.Module):
    def __init__(
        self,
        name=None,
        weights_path=None,
        interaction_indexes=[],
        finetune=True,
        embed_dim=192,
        num_heads=3,
        patch_size=16,
        use_scafu=True,
        conv_inplane=16,
        hidden_dim=None,
        use_dynamic_fusion=True,
        fusion_strength=0.5
    ):
        super(DINOv3ScaFu, self).__init__()
        if 'dinov3' in name:
            self.dinov3 = DinoVisionTransformer(name=name)
            if weights_path is not None and os.path.exists(weights_path):
                print(f'Loading ckpt from {weights_path}...')
                self.dinov3.load_state_dict(torch.load(weights_path))
            else:
                print('Training DINOv3 from scratch...')
        else:
            self.dinov3 =  VisionTransformer(embed_dim=embed_dim, num_heads=num_heads, return_layers=interaction_indexes)
            if weights_path is not None and os.path.exists(weights_path):
                print(f'Loading ckpt from {weights_path}...')
                self.dinov3._model.load_state_dict(torch.load(weights_path))
            else:
                print('Training ViT-Tiny from scratch...')

        embed_dim = self.dinov3.embed_dim
        self.interaction_indexes = interaction_indexes
        self.patch_size = patch_size

        if not finetune:
            self.dinov3.eval()
            self.dinov3.requires_grad_(False)

        self.use_scafu = use_scafu
        if use_scafu:
            print(f"Using Enhanced Dynamic Fusion ScaFu (strength={fusion_strength})")
            self.scafu = SpatialPriorModulev2(
                inplanes=conv_inplane,
                use_dynamic_fusion=use_dynamic_fusion,
                fusion_strength=fusion_strength
            )
        else:
            conv_inplane = 0

        hidden_dim = hidden_dim if hidden_dim is not None else embed_dim
        self.convs = nn.ModuleList([
            nn.Conv2d(embed_dim + conv_inplane*2, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(embed_dim + conv_inplane*4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(embed_dim + conv_inplane*4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        ])
        self.norms = nn.ModuleList([
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim)
        ])

    def forward(self, x):
        H_c, W_c = x.shape[2] // 16, x.shape[3] // 16
        H_toks, W_toks = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        bs, C, h, w = x.shape

        if len(self.interaction_indexes) > 0 and not isinstance(self.dinov3, VisionTransformer):
            all_layers = self.dinov3.get_intermediate_layers(
                x, n=self.interaction_indexes, return_class_token=True
            )
        else:
            all_layers = self.dinov3(x)

        if len(all_layers) == 1:
            all_layers = [all_layers[0], all_layers[0], all_layers[0]]

        sem_feats = []
        num_scales = len(all_layers) - 2
        for i, sem_feat in enumerate(all_layers):
            feat, _ = sem_feat
            sem_feat = feat.transpose(1, 2).view(bs, -1, H_c, W_c).contiguous()
            resize_H, resize_W = int(H_c * 2**(num_scales-i)), int(W_c * 2**(num_scales-i))
            sem_feat = F.interpolate(sem_feat, size=[resize_H, resize_W], mode="bilinear", align_corners=False)
            sem_feats.append(sem_feat)

        fused_feats = []
        if self.use_scafu:
            c2_scafu, c3_scafu, c4_scafu = self.scafu(x)
            detail_feats = [c2_scafu, c3_scafu, c4_scafu]
            for sem_feat, detail_feat in zip(sem_feats, detail_feats):
                fused_feats.append(torch.cat([sem_feat, detail_feat], dim=1))
        else:
            fused_feats = sem_feats

        c2 = self.norms[0](self.convs[0](fused_feats[0]))
        c3 = self.norms[1](self.convs[1](fused_feats[1]))
        c4 = self.norms[2](self.convs[2](fused_feats[2]))

        return c2, c3, c4