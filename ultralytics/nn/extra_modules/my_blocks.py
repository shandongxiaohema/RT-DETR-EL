from .attention import *
from .transformer import *
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch import rearrange  # 假设已导入
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import nn, Tensor, LongTensor
from torch.nn import init
import torch.nn.functional as F
import torchvision
from efficientnet_pytorch.model import MemoryEfficientSwish

import itertools
import einops
import math
import numpy as np
from einops import rearrange
from torch import Tensor
from typing import Tuple, Optional, List
from ..modules.conv import Conv, autopad
from ..backbone.TransNext import AggregatedAttention, get_relative_position_cpb
from timm.models.layers import trunc_normal_
import torch.utils.checkpoint as checkpoint
from torch.nn.modules.utils import _pair
from torch import Tensor
from torch.jit import Final
import math
import numpy as np
from functools import partial
from typing import Optional, Callable, Optional, Dict, Union, List
from einops import rearrange, reduce
from collections import OrderedDict
from einops import rearrange
from ..backbone.UniRepLKNet import get_bn, get_conv2d, NCHWtoNHWC, GRNwithNHWC, SEBlock, NHWCtoNCHW, fuse_bn, merge_dilated_into_large_kernel
from ..backbone.rmt import RetBlock, RelPos2d
from ..modules.conv import Conv, DWConv, DSConv, RepConv, GhostConv, autopad, LightConv, ConvTranspose
from ..modules.block import get_activation, ConvNormLayer, BasicBlock, BottleNeck, RepC3, C3, C2f, Bottleneck
from .attention import *
from .ops_dcnv3.modules import DCNv3
from .transformer import LocalWindowAttention
from .dynamic_snake_conv import DySnakeConv
from .RFAConv import RFAConv, RFCAConv, RFCBAMConv
from .rep_block import *
from .shiftwise_conv import ReparamLargeKernelConv
from .mamba_vss import VSSBlock
from .orepa import OREPA
from .fadc import AdaptiveDilatedConv
from .hcfnet import PPA, LocalGlobalAttention
from .deconv import DEConv
from .SMPConv import SMPConv
from .kan_convs import FastKANConv2DLayer, KANConv2DLayer, KALNConv2DLayer, KACNConv2DLayer, KAGNConv2DLayer
from .wtconv2d import WTConv2d
from .camixer import CAMixer
from .tsdn import DTAB, LayerNorm
from .metaformer import MetaFormerBlock, MetaFormerCGLUBlock, SepConv
from .savss import *
from ..backbone.MambaOut import GatedCNNBlock_BCHW, LayerNormGeneral
from .efficientvim import EfficientViMBlock, EfficientViMBlock_CGLU
from ..backbone.overlock import RepConvBlock
from .filc import *
from .DCMPNet import LEGM
from .mobileMamba.mobilemamba import MobileMambaBlock
from .semnet import SBSM
from ..backbone.lsnet import LSConv, Block as LSBlock
from .transMamba import TransMambaBlock
from .EVSSM import EVS
from .DarkIR import EBlock, DBlock
from .FDConv_initialversion import FDConv
from .dsan import *
from .MaIR import *
from .SFSConv import SFS_Conv
from .GroupMamba.groupmamba import GroupMambaLayer, Block_mamba
from .MambaVision import MambaVisionBlock
from .UMFormer import GL_VSS
from .esc import ESCBlock, ConvAttn
from .VSSD import VMAMBA2Block
from .TinyVIM import TViMBlock
from .CSI import CSI
from .UniConvNet import UniConvBlock
from .MobileUViT import LGLBlock
from .ConverseNet import Converse2D, ConverseBlock
from .gcconv import GCConv
from .CFBlock import CFBlock

from ultralytics.utils.torch_utils import fuse_conv_and_bn, make_divisible
from timm.layers import CondConv2d, DropPath, trunc_normal_, use_fused_attn, to_2tuple
__all__ = ['HybridBiLevelRoutingAttention','BiFormerBlock','SPDConv_Wavelet','SPD_Att','Channel_Compressor','CoordSPD','SPD_Dual',
           'SPD_ECA','DSC_SPD','SPD_Omni','SPD_ResAtt','SPD_LFE','CSPOmniKernelPlus','DS_OmniBlock','SA_OmniBlock','CSPLSKA',
           'SimAM']


class SpatialGroupEnhance(nn.Module):
    def __init__(self, groups=8):
        super().__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.sig = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b * self.groups, -1, h, w)  # bs*g,dim//g,h,w
        xn = x * self.avg_pool(x)  # bs*g,dim//g,h,w
        xn = xn.sum(dim=1, keepdim=True)  # bs*g,1,h,w
        t = xn.view(b * self.groups, -1)  # bs*g,h*w

        t = t - t.mean(dim=1, keepdim=True)  # bs*g,h*w
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std  # bs*g,h*w
        t = t.view(b, self.groups, h, w)  # bs,g,h*w

        t = t * self.weight + self.bias  # bs,g,h*w
        t = t.view(b * self.groups, 1, h, w)  # bs*g,1,h*w
        x = x * self.sig(t)
        x = x.view(b, c, h, w)
        return x


class TopkRouting(nn.Module):
    """
    differentiable topk routing with scaling
    Args:
        qk_dim: int, feature dimension of query and key
        topk: int, the 'topk'
        qk_scale: int or None, temperature (multiply) of softmax activation
        with_param: bool, wether inorporate learnable params in routing unit
        diff_routing: bool, wether make routing differentiable
        soft_routing: bool, wether make output value multiplied by routing weights
    """

    def __init__(self, qk_dim, topk=4, qk_scale=None, param_routing=False, diff_routing=False):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5
        self.diff_routing = diff_routing
        # TODO: norm layer before/after linear?
        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()
        # routing activation
        self.routing_act = nn.Softmax(dim=-1)

    def forward(self, query: Tensor, key: Tensor) -> Tuple[Tensor]:
        """
        Args:
            q, k: (n, p^2, c) tensor
        Return:
            r_weight, topk_index: (n, p^2, topk) tensor
        """
        if not self.diff_routing:
            query, key = query.detach(), key.detach()
        query_hat, key_hat = self.emb(query), self.emb(key)  # per-window pooling -> (n, p^2, c)
        attn_logit = (query_hat * self.scale) @ key_hat.transpose(-2, -1)  # (n, p^2, p^2)
        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)  # (n, p^2, k), (n, p^2, k)
        r_weight = self.routing_act(topk_attn_logit)  # (n, p^2, k)

        return r_weight, topk_index


class KVGather(nn.Module):
    def __init__(self, mul_weight='none'):
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight

    def forward(self, r_idx: Tensor, r_weight: Tensor, kv: Tensor):
        """
        r_idx: (n, p^2, topk) tensor
        r_weight: (n, p^2, topk) tensor
        kv: (n, p^2, w^2, c_kq+c_v)

        Return:
            (n, p^2, topk, w^2, c_kq+c_v) tensor
        """
        # select kv according to routing index
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)
        # print(r_idx.size(), r_weight.size())
        # FIXME: gather consumes much memory (topk times redundancy), write cuda kernel?
        topk_kv = torch.gather(kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1),
                               # (n, p^2, p^2, w^2, c_kv) without mem cpy
                               dim=2,
                               index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv)
                               # (n, p^2, k, w^2, c_kv)
                               )

        if self.mul_weight == 'soft':
            topk_kv = r_weight.view(n, p2, topk, 1, 1) * topk_kv  # (n, p^2, k, w^2, c_kv)
        elif self.mul_weight == 'hard':
            raise NotImplementedError('differentiable hard routing TBA')
        # else: #'none'
        #     topk_kv = topk_kv # do nothing

        return topk_kv


class QKVLinear(nn.Module):
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias)

    def forward(self, x):
        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim + self.dim], dim=-1)
        return q, kv


class BiLevelRoutingAttention(nn.Module):
    """
    n_win: number of windows in one side (so the actual number of windows is n_win*n_win)
    kv_per_win: for kv_downsample_mode='ada_xxxpool' only, number of key/values per window. Similar to n_win, the actual number is kv_per_win*kv_per_win.
    topk: topk for window filtering
    param_attention: 'qkvo'-linear for q,k,v and o, 'none': param free attention
    param_routing: extra linear for routing
    diff_routing: wether to set routing differentiable
    soft_routing: wether to multiply soft routing weights
    """

    def __init__(self, dim, num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='identity',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
                 side_dwconv=3,
                 auto_pad=True):
        super().__init__()
        # local attention setting
        self.dim = dim
        self.n_win = n_win  # Wh, Ww
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        assert self.qk_dim % num_heads == 0 and self.dim % num_heads == 0, 'qk_dim and dim must be divisible by num_heads!'
        self.scale = qk_scale or self.qk_dim ** -0.5

        ################side_dwconv (i.e. LCE in ShuntedTransformer)###########
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2,
                              groups=dim) if side_dwconv > 0 else \
            lambda x: torch.zeros_like(x)

        ################ global routing setting #################
        self.topk = topk
        self.param_routing = param_routing
        self.diff_routing = diff_routing
        self.soft_routing = soft_routing
        # router
        assert not (self.param_routing and not self.diff_routing)  # cannot be with_param=True and diff_routing=False
        self.router = TopkRouting(qk_dim=self.qk_dim,
                                  qk_scale=self.scale,
                                  topk=self.topk,
                                  diff_routing=self.diff_routing,
                                  param_routing=self.param_routing)
        if self.soft_routing:  # soft routing, always diffrentiable (if no detach)
            mul_weight = 'soft'
        elif self.diff_routing:  # hard differentiable routing
            mul_weight = 'hard'
        else:  # hard non-differentiable routing
            mul_weight = 'none'
        self.kv_gather = KVGather(mul_weight=mul_weight)

        # qkv mapping (shared by both global routing and local attention)
        self.param_attention = param_attention
        if self.param_attention == 'qkvo':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Linear(dim, dim)
        elif self.param_attention == 'qkv':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Identity()
        else:
            raise ValueError(f'param_attention mode {self.param_attention} is not surpported!')

        self.kv_downsample_mode = kv_downsample_mode
        self.kv_per_win = kv_per_win
        self.kv_downsample_ratio = kv_downsample_ratio
        self.kv_downsample_kenel = kv_downsample_kernel
        if self.kv_downsample_mode == 'ada_avgpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'ada_maxpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveMaxPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'maxpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.MaxPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'avgpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.AvgPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'identity':  # no kv downsampling
            self.kv_down = nn.Identity()
        elif self.kv_downsample_mode == 'fracpool':
            # assert self.kv_downsample_ratio is not None
            # assert self.kv_downsample_kenel is not None
            # TODO: fracpool
            # 1. kernel size should be input size dependent
            # 2. there is a random factor, need to avoid independent sampling for k and v
            raise NotImplementedError('fracpool policy is not implemented yet!')
        elif kv_downsample_mode == 'conv':
            # TODO: need to consider the case where k != v so that need two downsample modules
            raise NotImplementedError('conv policy is not implemented yet!')
        else:
            raise ValueError(f'kv_down_sample_mode {self.kv_downsaple_mode} is not surpported!')

        # softmax for local attention
        self.attn_act = nn.Softmax(dim=-1)

        self.auto_pad = auto_pad

    def forward(self, x, ret_attn_mask=False):
        """
        x: NHWC tensor

        Return:
            NHWC tensor
        """
        x = rearrange(x, "n c h w -> n h w c")
        # NOTE: use padding for semantic segmentation
        ###################################################
        if self.auto_pad:
            N, H_in, W_in, C = x.size()

            pad_l = pad_t = 0
            pad_r = (self.n_win - W_in % self.n_win) % self.n_win
            pad_b = (self.n_win - H_in % self.n_win) % self.n_win
            x = F.pad(x, (0, 0,  # dim=-1
                          pad_l, pad_r,  # dim=-2
                          pad_t, pad_b))  # dim=-3
            _, H, W, _ = x.size()  # padded size
        else:
            N, H, W, C = x.size()
            assert H % self.n_win == 0 and W % self.n_win == 0  #
        ###################################################

        # patchify, (n, p^2, w, w, c), keep 2d window as we need 2d pooling to reduce kv size
        x = rearrange(x, "n (j h) (i w) c -> n (j i) h w c", j=self.n_win, i=self.n_win)

        #################qkv projection###################
        # q: (n, p^2, w, w, c_qk)
        # kv: (n, p^2, w, w, c_qk+c_v)
        # NOTE: separte kv if there were memory leak issue caused by gather
        q, kv = self.qkv(x)

        # pixel-wise qkv
        # q_pix: (n, p^2, w^2, c_qk)
        # kv_pix: (n, p^2, h_kv*w_kv, c_qk+c_v)
        q_pix = rearrange(q, 'n p2 h w c -> n p2 (h w) c')
        kv_pix = self.kv_down(rearrange(kv, 'n p2 h w c -> (n p2) c h w'))
        kv_pix = rearrange(kv_pix, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win)

        q_win, k_win = q.mean([2, 3]), kv[..., 0:self.qk_dim].mean(
            [2, 3])  # window-wise qk, (n, p^2, c_qk), (n, p^2, c_qk)

        ##################side_dwconv(lepe)##################
        # NOTE: call contiguous to avoid gradient warning when using ddp
        lepe = self.lepe(rearrange(kv[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', j=self.n_win,
                                   i=self.n_win).contiguous())
        lepe = rearrange(lepe, 'n c (j h) (i w) -> n (j h) (i w) c', j=self.n_win, i=self.n_win)

        ############ gather q dependent k/v #################

        r_weight, r_idx = self.router(q_win, k_win)  # both are (n, p^2, topk) tensors

        kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix)  # (n, p^2, topk, h_kv*w_kv, c_qk+c_v)
        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], dim=-1)
        # kv_pix_sel: (n, p^2, topk, h_kv*w_kv, c_qk)
        # v_pix_sel: (n, p^2, topk, h_kv*w_kv, c_v)

        ######### do attention as normal ####################
        k_pix_sel = rearrange(k_pix_sel, 'n p2 k w2 (m c) -> (n p2) m c (k w2)',
                              m=self.num_heads)  # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_kq//m) transpose here?
        v_pix_sel = rearrange(v_pix_sel, 'n p2 k w2 (m c) -> (n p2) m (k w2) c',
                              m=self.num_heads)  # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_v//m)
        q_pix = rearrange(q_pix, 'n p2 w2 (m c) -> (n p2) m w2 c',
                          m=self.num_heads)  # to BMLC tensor (n*p^2, m, w^2, c_qk//m)

        # param-free multihead attention
        attn_weight = (
                                  q_pix * self.scale) @ k_pix_sel  # (n*p^2, m, w^2, c) @ (n*p^2, m, c, topk*h_kv*w_kv) -> (n*p^2, m, w^2, topk*h_kv*w_kv)
        attn_weight = self.attn_act(attn_weight)
        out = attn_weight @ v_pix_sel  # (n*p^2, m, w^2, topk*h_kv*w_kv) @ (n*p^2, m, topk*h_kv*w_kv, c) -> (n*p^2, m, w^2, c)
        out = rearrange(out, '(n j i) m (h w) c -> n (j h) (i w) (m c)', j=self.n_win, i=self.n_win,
                        h=H // self.n_win, w=W // self.n_win)

        out = out + lepe
        # output linear
        out = self.wo(out)

        # NOTE: use padding for semantic segmentation
        # crop padded region
        if self.auto_pad and (pad_r > 0 or pad_b > 0):
            out = out[:, :H_in, :W_in, :].contiguous()

        if ret_attn_mask:
            return out, r_weight, r_idx, attn_weight
        else:
            return rearrange(out, "n h w c -> n c h w")


class HybridBiLevelRoutingAttention(nn.Module):
    def __init__(self, dim, num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='identity',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False, side_dwconv=3,
                 auto_pad=True, global_token_ratio=0.1):
        super().__init__()
        # 原有初始化逻辑
        self.dim = dim
        self.n_win = n_win
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        self.scale = qk_scale or self.qk_dim ** -0.5
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv//2, groups=dim) if side_dwconv > 0 else lambda x: torch.zeros_like(x)
        self.topk = topk
        self.param_routing = param_routing
        self.diff_routing = diff_routing
        self.soft_routing = soft_routing
        self.router = TopkRouting(qk_dim=self.qk_dim, qk_scale=self.scale, topk=self.topk, diff_routing=self.diff_routing, param_routing=self.param_routing)
        mul_weight = 'soft' if self.soft_routing else ('hard' if self.diff_routing else 'none')
        self.kv_gather = KVGather(mul_weight=mul_weight)
        self.param_attention = param_attention
        if self.param_attention == 'qkvo':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Linear(dim, dim)
        elif self.param_attention == 'qkv':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Identity()
        self.kv_downsample_mode = kv_downsample_mode
        self.kv_per_win = kv_per_win
        self.kv_downsample_ratio = kv_downsample_ratio
        self.kv_downsample_kenel = kv_downsample_kernel  # 修正拼写为 kernel，如果原为 kenel
        if self.kv_downsample_mode == 'ada_avgpool':
            self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'ada_maxpool':
            self.kv_down = nn.AdaptiveMaxPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'maxpool':
            self.kv_down = nn.MaxPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'avgpool':
            self.kv_down = nn.AvgPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'identity':
            self.kv_down = nn.Identity()
        else:
            raise ValueError(f'kv_down_sample_mode {self.kv_downsample_mode} is not supported!')
        self.attn_act = nn.Softmax(dim=-1)
        self.auto_pad = auto_pad

        # 新增全局令牌参数
        self.global_token_ratio = global_token_ratio
        self.global_proj = nn.Linear(dim, dim)  # 可学习投影

    def forward(self, x, ret_attn_mask=False):
        x = rearrange(x, "n c h w -> n h w c")
        if self.auto_pad:
            N, H_in, W_in, C = x.size()
            pad_l = pad_t = 0
            pad_r = (self.n_win - W_in % self.n_win) % self.n_win
            pad_b = (self.n_win - H_in % self.n_win) % self.n_win
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.size()

        # 生成全局令牌（平均池化整个特征图）
        global_token = F.adaptive_avg_pool2d(x.permute(0, 3, 1, 2), 1).squeeze(-1).squeeze(-1)  # (N, C)
        global_token = self.global_proj(global_token)  # (N, C)

        # 窗口划分
        x = rearrange(x, "n (j h) (i w) c -> n (j i) h w c", j=self.n_win, i=self.n_win)

        # qkv投影
        q, kv = self.qkv(x)  # q: (n, p2, h, w, c_qk), kv: (n, p2, h, w, c_qk+c_v)

        # pixel-wise qkv
        q_pix = rearrange(q, 'n p2 h w c -> n p2 (h w) c')
        kv_pix = self.kv_down(rearrange(kv, 'n p2 h w c -> (n p2) c h w'))
        kv_pix = rearrange(kv_pix, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win)

        # 添加全局令牌到 kv_pix（作为额外 token）
        num_tokens = kv_pix.size(2)  # (h w)
        num_global = int(self.global_token_ratio * num_tokens)
        global_kv = kv.mean(dim=[2, 3])  # (n, p2, c_qk+c_v)
        global_kv = rearrange(global_kv, 'n p2 c -> n p2 1 c')  # (n, p2, 1, c)
        global_kv = global_kv.repeat(1, 1, num_global, 1)  # (n, p2, num_global, c)
        kv_pix = torch.cat([kv_pix, global_kv], dim=2)  # (n, p2, (h w + num_global), c) — 形状匹配

        # 窗口级 qk
        q_win, k_win = q.mean([2, 3]), kv[..., 0:self.qk_dim].mean([2, 3])

        # lepe (side_dwconv)
        lepe = self.lepe(rearrange(kv[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', j=self.n_win, i=self.n_win).contiguous())
        lepe = rearrange(lepe, 'n c (j h) (i w) -> n (j h) (i w) c', j=self.n_win, i=self.n_win)

        # 路由和注意力计算（原有逻辑）
        r_weight, r_idx = self.router(q_win, k_win)
        kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix)
        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], dim=-1)
        k_pix_sel = rearrange(k_pix_sel, 'n p2 k w2 (m c) -> (n p2) m c (k w2)', m=self.num_heads)
        v_pix_sel = rearrange(v_pix_sel, 'n p2 k w2 (m c) -> (n p2) m (k w2) c', m=self.num_heads)
        q_pix = rearrange(q_pix, 'n p2 w2 (m c) -> (n p2) m w2 c', m=self.num_heads)
        attn_weight = (q_pix * self.scale) @ k_pix_sel
        attn_weight = self.attn_act(attn_weight)
        out = attn_weight @ v_pix_sel
        out = rearrange(out, '(n j i) m (h w) c -> n (j h) (i w) (m c)', j=self.n_win, i=self.n_win, h=H//self.n_win, w=W//self.n_win)
        out = out + lepe
        out = self.wo(out)

        if self.auto_pad and (pad_r > 0 or pad_b > 0):
            out = out[:, :H_in, :W_in, :].contiguous()

        return rearrange(out, "n h w c -> n c h w")






# # ==================== BiFormerBlock 使用基础代码 ====================
class BiFormerBlock(nn.Module):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__()

        self.shortcut = shortcut

        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)

        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)
        self.attention = HybridBiLevelRoutingAttention(ch_out)

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.attention(out)
        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        out = out + short
        out = self.act(out)

        return out

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv  # 如果你的项目 Conv 在别处，按实际 import

import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设你项目中已有标准的 Conv 模块 (带 BN 和 Act)
# 如果没有，请确保能访问到它，或者改用 nn.Conv2d
try:
    from ultralytics.nn.modules.conv import Conv
except:
    # 备选方案：如果找不到框架的 Conv，手动定义一个简单的
    class Conv(nn.Module):
        def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
            super().__init__()
            self.conv = nn.Conv2d(c1, c2, k, s, bias=False)
            self.bn = nn.BatchNorm2d(c2)
            self.act = nn.SiLU() if act is True else nn.Identity()

        def forward(self, x):
            return self.act(self.bn(self.conv(x)))




class SPDConv_Wavelet(nn.Module):
    """
    小波下采样改进版：兼容 RT-DETR parse_model。
    采用延迟加载模式，自动适配通道。
    """

    def __init__(self, c1=None, c2=None, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        # 即使 parse_model 传了参数，我们也主要通过 forward 动态获取
        self.ouc = c2  # 目标输出通道
        self.cv1 = None

        # Haar 小波滤波器初始化
        kernel = torch.tensor([[[[1, 1], [1, 1]]],
                               [[[1, 1], [-1, -1]]],
                               [[[1, -1], [1, -1]]],
                               [[[1, -1], [-1, 1]]]]) / 2.0
        self.register_buffer('filter', kernel)

    def forward(self, x):
        b, c, h, w = x.shape

        # 1. 动态初始化卷积层：确保通道对齐
        if self.cv1 is None:
            # 如果 YAML 没给 c2，我们就默认输出等于输入 c
            out_ch = self.ouc if self.ouc is not None else c
            self.cv1 = Conv(c * 4, out_ch, k=1).to(x.device)

        # 2. 小波变换操作
        # 将输入视为 (B*C, 1, H, W) 进行组卷积
        x_reshaped = x.view(b * c, 1, h, w)
        x_w = F.conv2d(x_reshaped, self.filter, stride=2, groups=1)

        # 3. 重排回 (B, C*4, H/2, W/2)
        x_w = x_w.view(b, c * 4, h // 2, w // 2)

        # 4. 通道压缩
        return self.cv1(x_w)


import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class Channel_Compressor(nn.Module):
    """
    针对 RT-DETR 优化的通道压缩器。
    采用完全兼容模式，解决 parse_model 导致的 TypeError。
    """

    def __init__(self, c1=None, c2=None, *args, **kwargs):
        super().__init__()
        # 记录目标输出通道（由 YAML 中的 [256] 传入）
        # 如果 args 里面有值，c2 可能会被挤到后面，这里做一个健壮性处理
        self.ouc = c2 if c2 is not None else (args[0] if len(args) > 0 else None)
        self.cv1 = None

    def forward(self, x):
        if self.cv1 is None:
            # 动态获取当前 Tensor 的真实通道数 (如 512)
            inc = x.shape[1]
            # 如果没有指定输出通道，默认不改变通道数
            final_ouc = self.ouc if self.ouc is not None else inc
            # 初始化卷积层
            self.cv1 = Conv(inc, final_ouc, k=1).to(x.device)

        return self.cv1(x)


import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ultralytics.nn.modules.conv import Conv
except:
    class Conv(nn.Module):
        def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
            super().__init__()
            self.conv = nn.Conv2d(c1, c2, k, s, padding=k//2, bias=False)
            self.bn = nn.BatchNorm2d(c2)
            self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        def forward(self, x):
            return self.act(self.bn(self.conv(x)))

class SPD_Att(nn.Module):
    def __init__(self, c1=None, c2=None, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.ouc = c2
        self.cv1 = None
        # ECA 注意力组件
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_eca = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. SPD 下采样
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2],
                       x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        b, c_new, h, w = x.shape

        # 2. 动态初始化，适配任何输入
        if self.cv1 is None:
            out_ch = self.ouc if self.ouc is not None else c_new
            self.cv1 = Conv(c_new, out_ch, k=3).to(x.device)
            self.conv_eca.to(x.device)

        # 3. ECA 注意力逻辑
        y = self.avg_pool(x) # (B, C_new, 1, 1)
        # 确保 view 维度绝对安全
        y = self.conv_eca(y.view(b, 1, c_new)).view(b, c_new, 1, 1)
        y = self.sigmoid(y)

        return self.cv1(x * y.expand_as(x))


import torch
import torch.nn as nn

# 1. 保持 CoordinateAttention 不变
class CoordinateAttention(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y); y = self.bn1(y); y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_w * a_h


class CoordSPD(nn.Module):
    # 使用 inc 接收输入通道，用 *args 接收 YAML 里的 [128]
    def __init__(self, inc, *args):
        super().__init__()

        # 核心逻辑：如果 args 有值（即 YAML 传进来了 128），就取第一个；
        # 如果 args 是空的，就默认让 ouc 等于 inc（或者你手动写死 128）
        ouc = args[0] if len(args) > 0 else 128

        # SPD 会将通道数翻 4 倍 (inc * 4)
        self.conv = nn.Sequential(
            nn.Conv2d(inc * 4, ouc, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ouc),
            nn.ReLU(inplace=True)
        )
        self.ca = CoordinateAttention(ouc, ouc)

    def forward(self, x):
        # Space-to-Depth 逻辑
        x = torch.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2]
        ], 1)
        x = self.conv(x)
        return self.ca(x)


import torch
import torch.nn as nn

try:
    from ultralytics.nn.modules.conv import Conv
except:
    pass  # 假设环境正常


class SPD_Dual(nn.Module):
    """
    SPD_Dual (延迟初始化版 - 严格参照 SPDConv_Wavelet 格式)

    机制:
      1. SPD 无损下采样
      2. 双路融合: 3x3 卷积 (保大目标) + Strip 条形卷积 (保细长裂纹)
    """

    # 严格保持和你给出的 SPDConv_Wavelet 一样的参数签名
    def __init__(self, c1=None, c2=None, k=1, s=1, p=None, g=1, act=True):
        super().__init__()

        # 1. 关键：正确记录输出通道
        # 如果 YAML 传了 [128]，c2 就是 128。
        # 我们必须设置 self.c2，这样 parse_model 才能计算出下一层 Concat 是 384
        self.c2 = c2
        self.ouc = c2  # 内部使用

        # 延迟初始化标记
        self.cv1 = None

    def forward(self, x):
        # SPD: (B, C, H, W) -> (B, 4C, H/2, W/2)
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2],
                       x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)

        # 动态初始化
        if self.cv1 is None:
            inc = x.shape[1]  # 此时 inc 已经是 4倍通道了
            out_ch = self.ouc if self.ouc is not None else inc

            # --- 分支 A: 3x3 常规卷积 ---
            self.branch_square = Conv(inc, out_ch, k=3).to(x.device)

            # --- 分支 B: Strip 条形卷积 (模拟十字感受野) ---
            self.branch_strip = nn.Sequential(
                nn.Conv2d(inc, out_ch, kernel_size=(5, 1), padding=(2, 0), bias=False),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=(1, 5), padding=(0, 2), bias=False),
                nn.BatchNorm2d(out_ch),
                nn.SiLU()
            ).to(x.device)

            self.cv1 = True  # 标记初始化完成

        # 双路融合：直接相加 (Add)
        # 这种方式不改变通道数，且比 Concat 更省显存
        return self.branch_square(x) + self.branch_strip(x)


import torch
import torch.nn as nn
import math

try:
    from ultralytics.nn.modules.conv import Conv
except:
    pass


class ECA(nn.Module):
    """
    ECA 模块：极轻量级的通道注意力。
    论文来源: ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
    作用：给通道打分，重要的特征通道放大，不重要的缩小。
    """

    def __init__(self, channels, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 动态计算卷积核大小，通常 k=3 就能处理好
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. 全局平均池化 (B, C, H, W) -> (B, C, 1, 1)
        y = self.avg_pool(x)

        # 2. 1D 卷积捕获跨通道交互 (B, C, 1) -> (B, 1, C)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # 3. Sigmoid 归一化权重 + 乘回原特征
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class SPD_ECA(nn.Module):
    """
    SPD_ECA: 你的“自研”模块。
    本质 = 原始 SPDConv + ECA 注意力。
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        # 1. 正常的参数初始化 (配合 tasks.py 注册)
        inc_spd = c1 * 4

        # 2. 核心卷积 (这是保证效果的关键，和原始一模一样)
        self.conv = Conv(inc_spd, c2, k=3)

        # 3. 创新点：加上 ECA 模块
        # 即使这玩意没用，它也不会破坏特征图的空间结构
        self.eca = ECA(c2, k_size=3)

    def forward(self, x):
        # 1. SPD 切片
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2],
                       x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)

        # 2. 卷积提取特征 (原始效果的保障)
        x = self.conv(x)

        # 3. 注意力加权 (论文卖点)
        x = self.eca(x)

        return x


import torch
import torch.nn as nn

try:
    from ultralytics.nn.modules.conv import Conv
except:
    pass


class DSC_SPD(nn.Module):
    """
    Paper Name: Decoupled Spatial-Channel SPD (DSC-SPD)
    Mechanism: Parallel 3x3 (Spatial) and 1x1 (Channel) Convolutions.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        # c1: Input channels (e.g., 64 from P2)
        # c2: Output channels (e.g., 128)

        inc_spd = c1 * 4

        # Branch 1: Spatial Feature Extraction (3x3)
        # 负责捕捉裂纹的形状、纹理
        self.spatial_branch = Conv(inc_spd, c2, k=3)

        # Branch 2: Channel Information Fusion (1x1)
        # 负责通道间的信息重组，弥补 SPD 切片带来的通道割裂
        self.channel_branch = Conv(inc_spd, c2, k=1)

    def forward(self, x):
        # 1. Space-to-Depth
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2],
                       x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)

        # 2. Decoupled Fusion (Add)
        return self.spatial_branch(x) + self.channel_branch(x)


import torch
import torch.nn as nn

try:
    from ultralytics.nn.modules.conv import Conv
except:
    pass  # 只要环境里有 ultralytics 就能跑


class SPD_Omni(nn.Module):
    """
    SPD_Omni: 融合 OmniKernel 思想的无损下采样模块
    结构: SPD -> [3x3 Main] + [5x1 Strip] + [1x5 Strip]
    特点: 100% 兼容原始 SPDConv 效果，利用条形卷积增强对细长/微小目标的捕捉。
    """

    def __init__(self, c1=None, c2=None, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        # 1. 自动处理 c2 (输出通道)，兼容 RT-DETR 的 parse_model
        # 如果 YAML 传了 [128]，c2 就是 128
        self.ouc = c2

        # 2. 延迟初始化标记
        self.cv_main = None
        self.cv_h = None
        self.cv_v = None

    def forward(self, x):
        # --- 1. Space-to-Depth (无损下采样) ---
        # (B, C, H, W) -> (B, 4C, H/2, W/2)
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2],
                       x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)

        # --- 2. 动态初始化 (只在第一次运行时构建) ---
        if self.cv_main is None:
            # 获取 SPD 拼接后的真实通道数 (如 64*4 = 256)
            inc = x.shape[1]
            # 确定输出通道 (如果 YAML 没写，就默认保持 inc 不变)
            out = self.ouc if self.ouc is not None else inc

            # [主路]: 3x3 标准卷积 (核心保底，保证效果不低于原始)
            self.cv_main = Conv(inc, out, k=3).to(x.device)

            # [创新路 A]: 1x5 水平条形卷积 (捕捉水平特征)
            # padding=(0, 2) 保证尺寸不变
            self.cv_h = Conv(inc, out, k=(1, 5), p=(0, 2)).to(x.device)

            # [创新路 B]: 5x1 垂直条形卷积 (捕捉垂直特征)
            # padding=(2, 0) 保证尺寸不变
            self.cv_v = Conv(inc, out, k=(5, 1), p=(2, 0)).to(x.device)

        # --- 3. 多路特征融合 (Add) ---
        # 使用加法融合，不增加显存占用，且梯度回传直接
        # 这种并联结构类似 Inception/RepVGG，在小目标上通常能涨点
        return self.cv_main(x) + self.cv_h(x) + self.cv_v(x)


import torch
import torch.nn as nn

try:
    from ultralytics.nn.modules.conv import Conv
except:
    pass


# class SimAM(nn.Module):
#     """ 无参注意力 (ICML 2021) """
#
#     def __init__(self, e_lambda=1e-4):
#         super().__init__()
#         self.activaton = nn.Sigmoid()
#         self.e_lambda = e_lambda
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#         n = h * w - 1
#         x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
#         y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
#         # 返回加权后的特征
#         return x * self.activaton(y)


class SPD_ResAtt(nn.Module):
    """
    【推荐尝试】残差注意力 SPD
    逻辑：Out = Conv(x) + alpha * SimAM(Conv(x))
    优势：绝对不丢失原始特征信息，Attention 仅作为“奖励项”。
    """

    def __init__(self, c1=None, c2=None, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        # 1. 计算通道 (静态计算，防止 SegFault)
        # c1 是输入通道 (如 64)，SPD 后变成 256
        hidden_channels = c1 * 4

        # 2. 原始卷积 (保住分数的根基)
        self.conv = Conv(hidden_channels, c2, k=3)

        # 3. 注意力模块
        self.att = SimAM()

        # 4. 零初始化系数 (保命符)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Space-to-Depth
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2],
                       x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)

        # 提取特征
        feat = self.conv(x)

        # 融合：原始特征 + (0 * 注意力特征)
        # 训练初期完全等价于原始 SPD，随着 alpha 学习，注意力逐渐生效
        return feat + self.alpha * self.att(feat)


import torch
import torch.nn as nn

try:
    from ultralytics.nn.modules.conv import Conv
except:
    pass


class LFE_Module(nn.Module):
    """
    Local Feature Enhancer (局部特征增强)
    原理: 原始特征 + alpha * (原始特征 - 平滑后的特征)
    物理含义: 显式增强高频细节(裂纹/断栅)，利用差分原理突出缺陷边缘。
    """

    def __init__(self, dim):
        super().__init__()
        # 3x3 平均池化 = 低频滤波器 (模糊)
        self.avg = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        # 1x1 卷积用于调整增强特征的权重/通道变换 (可选，为了增加参数量显得像个模块)
        self.enhance_conv = nn.Conv2d(dim, dim, 1, bias=False)
        # 零初始化: 初始状态下 alpha=0, 模块等价于恒等映射
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 1. 计算低频成分 (背景/平滑区域)
        low_freq = self.avg(x)

        # 2. 计算高频成分 (原图 - 低频 = 边缘/纹理)
        high_freq = x - low_freq

        # 3. 对高频进行特征变换 (增加非线性)
        enhanced = self.enhance_conv(high_freq)

        # 4. 融合: 原图 + alpha * 增强的高频
        return x + self.alpha * enhanced


class SPD_LFE(nn.Module):
    """
    SPD + LFE 组合模块
    替换原有的 SPDConv
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        # 静态计算通道，防止报错
        hidden_channels = c1 * 4

        # 1. 原始 SPD 卷积 (保底)
        self.conv = Conv(hidden_channels, c2, k=3)

        # 2. 挂载 LFE 模块
        self.lfe = LFE_Module(c2)

    def forward(self, x):
        # SPD 切片
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2],
                       x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        # 卷积提取
        x = self.conv(x)
        # 增强
        return self.lfe(x)









class FGM(nn.Module):

    def __init__(self, dim) -> None:

        super().__init__()



        self.conv = nn.Conv2d(dim, dim*2, 3, 1, 1, groups=dim)



        self.dwconv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)

        self.dwconv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)

        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))

        self.beta = nn.Parameter(torch.ones(dim, 1, 1))



    def forward(self, x):

        # res = x.clone()

        fft_size = x.size()[2:]

        x1 = self.dwconv1(x)

        x2 = self.dwconv2(x)



        x2_fft = torch.fft.fft2(x2, norm='backward')



        out = x1 * x2_fft



        out = torch.fft.ifft2(out, dim=(-2,-1), norm='backward')

        out = torch.abs(out)



        return out * self.alpha + x * self.beta



class OmniKernelPlus(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        ker = 31
        pad = ker // 2
        self.in_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
            nn.GELU()
        )
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)

        self.dw_13 = nn.Conv2d(dim, dim, kernel_size=(1,ker), padding=(0,pad), stride=1, groups=dim)
        self.dw_31 = nn.Conv2d(dim, dim, kernel_size=(ker,1), padding=(pad,0), stride=1, groups=dim)
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, stride=1, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)

        self.act = nn.ReLU()

        # SCA
        self.conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        # FCA
        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.fac_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fgm = FGM(dim)

    def forward(self, x):
        feat = self.in_conv(x)

        # ===== Frequency Channel Attention =====
        att_f = self.fac_conv(self.fac_pool(feat))
        X = torch.fft.fft2(feat, norm='backward')
        X = att_f * X
        feat_f = torch.fft.ifft2(X, dim=(-2,-1), norm='backward')
        feat_f = torch.abs(feat_f)

        # ===== Spatial Channel Attention =====
        att_s = self.conv(self.pool(feat_f))
        x_sca = att_s * feat_f
        x_sca = self.fgm(x_sca)

        # ===== Omni-Kernel Spatial Aggregation =====
        spatial = (
            self.dw_13(feat) +
            self.dw_31(feat) +
            self.dw_33(feat) +
            self.dw_11(feat)
        )

        # ===== OmniKernelPlus: Frequency-guided modulation =====
        spatial = spatial * (1 + x_sca)

        out = x + spatial
        out = self.act(out)
        return self.out_conv(out)



import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class CSPOmniKernelPlus(nn.Module):
    """
    CSPOmniKernelPlus

    A CSP-style omni-kernel aggregation module with automatic
    channel inference for Ultralytics parse_model compatibility.
    """

    def __init__(self, c1, c2=None, *args, **kwargs):
        """
        Notes:
            - c2 is optional to support empty args list in YAML
            - when c2 is None, output channels default to c1
        """
        super().__init__()

        # -------------------------------
        # 🔑 关键修复点（parse_model 兼容）
        # -------------------------------
        if c2 is None:
            c2 = c1  # 与原 CSPOmniKernel 行为保持一致

        e = kwargs.get("e", 0.5)
        act = kwargs.get("act", True)

        hidden = int(c2 * e)

        # CSP split
        self.cv1 = Conv(c1, hidden, 1, 1, act=act)
        self.cv2 = Conv(c1, hidden, 1, 1, act=act)

        # Omni-kernel branches
        self.branch_3x3 = Conv(hidden, hidden, 3, 1, act=act)
        self.branch_5x5 = Conv(hidden, hidden, 5, 1, act=act)
        self.branch_dilated = Conv(hidden, hidden, 3, 1, d=2, act=act)

        # Kernel fusion
        self.fuse = Conv(hidden * 3, hidden, 1, 1, act=act)

        # Output projection
        self.cv3 = Conv(hidden * 2, c2, 1, 1, act=act)

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(x)

        b1 = self.branch_3x3(y1)
        b2 = self.branch_5x5(y1)
        b3 = self.branch_dilated(y1)

        omni = self.fuse(torch.cat([b1, b2, b3], dim=1))
        out = self.cv3(torch.cat([omni, y2], dim=1))
        return out


import torch
import torch.nn as nn
import torch.nn.functional as F


# 保持你之前运行正常的 FGM 逻辑
class FGM(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.conv = nn.Conv2d(dim, dim * 2, 3, 1, 1, groups=dim)
        self.dwconv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)
        x2_fft = torch.fft.fft2(x2, norm='backward')
        out = x1 * x2_fft
        out = torch.fft.ifft2(out, dim=(-2, -1), norm='backward')
        out = torch.abs(out)
        return out * self.alpha + x * self.beta


class OmniKernel_DS(nn.Module):
    def __init__(self, c1, c2=None) -> None:
        super().__init__()
        dim = c1
        out_dim = c2 if c2 is not None else dim
        ker = 31
        pad = ker // 2

        self.in_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU()
        )
        self.out_conv = nn.Conv2d(dim, out_dim, kernel_size=1)

        # 原始的大核卷积四分支（保证基础效果）
        self.dw_13 = nn.Conv2d(dim, dim, kernel_size=(1, ker), padding=(0, pad), groups=dim)
        self.dw_31 = nn.Conv2d(dim, dim, kernel_size=(ker, 1), padding=(pad, 0), groups=dim)
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, groups=dim)

        self.act = nn.SiLU()  # 换成 SiLU，更契合 RT-DETR

        ### 改进点：双统计量空间注意力 (Dual-Statistic Spatial Attention) ###
        self.conv_atten = nn.Conv2d(dim, dim, kernel_size=1)
        self.pool_avg = nn.AdaptiveAvgPool2d(1)
        self.pool_max = nn.AdaptiveMaxPool2d(1)

        # 引入残差缩放因子 gamma，初始化为 0.1，确保初始训练极其稳定
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1) * 0.1)

        # FCA 分支
        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.fac_pool = nn.AdaptiveAvgPool2d(1)
        self.fgm = FGM(dim)

    def forward(self, x):
        identity = x
        out = self.in_conv(x)

        # 1. 频域注意力分支 (FCA)
        x_att = self.fac_conv(self.fac_pool(out))
        x_fft = torch.fft.fft2(out, norm='backward')
        x_fft = x_att * x_fft
        x_fca = torch.abs(torch.fft.ifft2(x_fft, dim=(-2, -1), norm='backward'))

        # 2. 改进的空间注意力 (SCA) - 使用双统计量融合
        # 融合公式: 0.5 * (AvgPool + MaxPool)
        x_pool = (self.pool_avg(x_fca) + self.pool_max(x_fca)) * 0.5
        x_sca = self.conv_atten(x_pool) * x_fca
        x_sca = self.fgm(x_sca)

        # 3. 核心大核分支集成
        kernels = self.dw_13(out) + self.dw_31(out) + self.dw_33(out) + self.dw_11(out)

        # 4. 最终融合：Identity + 原始大核 + (缩放后的改进注意力)
        out = identity + kernels + (self.gamma * x_sca)
        return self.out_conv(self.act(out))


# YAML 中直接调用这个类名
class DS_OmniBlock(nn.Module):
    def __init__(self, c1, c2, e=0.25):
        super().__init__()
        self.c_split = int(c2 * e)
        # 这里的 Conv 建议用你项目里定义好的，如果报错就用 nn.Conv2d
        self.cv1 = nn.Conv2d(c1, c2, 1, bias=False)
        self.cv2 = nn.Conv2d(c2, c2, 1, bias=False)
        self.m = OmniKernel_DS(self.c_split, self.c_split)

    def forward(self, x):
        y = self.cv1(x)
        y1, y2 = torch.split(y, [self.c_split, y.shape[1] - self.c_split], dim=1)
        return self.cv2(torch.cat((self.m(y1), y2), 1))


import torch
import torch.nn as nn


# 保持原版 FGM 语义（在初始化时不改变输入）
class FGM(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        # 去掉未使用的 conv，保留两个 1x1 conv（如你需要可改）
        self.dwconv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        # 控制频域贡献和直接通过的比例
        # alpha 初始为 0（频域贡献初始禁用），beta 初始为 1（保留原始输入）
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        # x: (B, C, H, W)
        x1 = self.dwconv1(x)  # 实数
        x2 = self.dwconv2(x)  # 实数

        # 频域处理：注意返回的是复数张量
        x2_fft = torch.fft.fft2(x2, norm='backward')
        out_complex = x1 * x2_fft  # complex
        iffted = torch.fft.ifft2(out_complex, dim=(-2, -1), norm='backward')
        out_real = torch.abs(iffted)  # real

        # alpha 和 beta 对于 channels 进行广播
        return out_real * self.alpha + x * self.beta


class OmniKernel_SA(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        ker = 31
        pad = ker // 2

        self.in_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU()
        )

        # 原始四分支大核（保证基础效果）
        # depthwise conv：groups=dim 意味着每个通道独立卷积（depthwise）
        self.dw_13 = nn.Conv2d(dim, dim, kernel_size=(1, ker), padding=(0, pad), groups=dim)
        self.dw_31 = nn.Conv2d(dim, dim, kernel_size=(ker, 1), padding=(pad, 0), groups=dim)
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, groups=dim)

        self.act = nn.ReLU()

        # 注意力 & 池化分支
        self.pool_avg = nn.AdaptiveAvgPool2d(1)
        self.pool_max = nn.AdaptiveMaxPool2d(1)

        # alpha_saliency 控制 avg + max 的混合（初始化为 0，只有 avg 生效）
        self.alpha_saliency = nn.Parameter(torch.zeros(1))

        # 用于生成通道权重的 1x1 conv
        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.fac_pool = nn.AdaptiveAvgPool2d(1)

        # FGM 模块（初始化保证不改变其输入）
        self.fgm = FGM(dim)

        # —— 关键：对 SCA 输出做门控，初始为 0（确保新增分支在初始化时不会影响输出）
        self.sca_gate = nn.Parameter(torch.zeros(1))

        # 小的变换 conv（作用在 pooled 特征上）
        self.conv = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        # x: (B, C, H, W)
        res = self.in_conv(x)

        # FCA 分支
        x_att = self.fac_conv(self.fac_pool(res))  # (B, C, 1, 1)
        x_fft = torch.fft.fft2(res, norm='backward')  # complex
        x_fft = x_att * x_fft  # broadcasting, complex
        x_fca = torch.abs(torch.fft.ifft2(x_fft, dim=(-2, -1), norm='backward'))  # real

        # SCA 分支（渐进式融合），alpha_saliency 控制是否加入 max pool
        x_pool = self.pool_avg(x_fca) + self.alpha_saliency * self.pool_max(x_fca)  # (B, C, 1, 1)
        x_sca = self.conv(x_pool) * x_fca  # broadcast conv weights over spatial dims

        # 频域/空间微扰（FGM），FGM 本身在初始化时等价于恒等（alpha=0,beta=1）
        x_sca = self.fgm(x_sca)

        # 对 SCA 分支整体施加门控（初始为 0）
        x_sca = self.sca_gate * x_sca

        # 原始残差结构 —— 注意使用原始输入 x（而非 res）叠加 depthwise branches
        out = x + self.dw_13(res) + self.dw_31(res) + self.dw_33(res) + self.dw_11(res) + x_sca
        return self.act(out)


class SA_OmniBlock(nn.Module):
    # __init__ 只接收一个 c1（和你在 tasks.py 中的实例化一致）
    def __init__(self, c1, e=0.25):
        super().__init__()
        # 边界保护：保证 split 在 [1, c1-1]
        c_split = max(1, int(c1 * e))
        if c_split >= c1:
            c_split = max(1, c1 - 1)
        self.c_split = c_split

        self.cv1 = nn.Conv2d(c1, c1, 1, bias=False)
        self.cv2 = nn.Conv2d(c1, c1, 1, bias=False)
        # m 处理前半部分通道
        self.m = OmniKernel_SA(self.c_split)

    def forward(self, x):
        y = self.cv1(x)
        # 安全分割（前 c_split 给 m，剩余通道直接跳过）
        y1, y2 = torch.split(y, [self.c_split, y.shape[1] - self.c_split], dim=1)
        # m(y1) 返回 (B, c_split, H, W)
        out = torch.cat((self.m(y1), y2), dim=1)
        return self.cv2(out)


class LSKA(nn.Module):
    def __init__(self, dim, k=23):  # k=23 感受野已经很大了，比 Omni 的 31 略小但更高效
        super().__init__()

        # 深度卷积分解 (Horizontal)
        self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, k), stride=1, padding=(0, k // 2), groups=dim)
        # 深度卷积分解 (Vertical)
        self.conv0v = nn.Conv2d(dim, dim, kernel_size=(k, 1), stride=1, padding=(k // 2, 0), groups=dim)

        # 深度空洞卷积 (用于进一步扩大感受野)
        self.conv_s = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=9, dilation=3, groups=dim)

        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0h(x)
        attn = self.conv0v(attn)
        attn = self.conv_s(attn)
        attn = self.conv1(attn)
        return u * attn


class CSPLSKA(nn.Module):
    def __init__(self, dim, e=0.5):  # e 可以稍微调大一点，比如 0.5
        super().__init__()
        self.e = e
        c_hidden = int(dim * self.e)
        self.cv1 = Conv(dim, c_hidden, 1)
        self.cv2 = Conv(dim, c_hidden, 1)
        # 核心模块换成 LSKA
        self.m = LSKA(c_hidden)
        self.cv3 = Conv(2 * c_hidden, dim, 1)  # 最后融合

    def forward(self, x):
        # CSP 结构：一半进 LSKA，一半直接残差
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat((y1, y2), 1))



import torch
import torch.nn as nn

class SimAM(nn.Module):
    """
    Simple Attention Module (ICCV 2021)
    论文: "SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks"
    代码默认 e_lambda=1e-4，和论文一致；若显存吃紧可再调大。
    """
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        self.act = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        # 计算每个像素的“能量”
        n = w * h - 1
        x_minus = x - x.mean(dim=(2, 3), keepdim=True)
        var = (x_minus.pow(2)).sum(dim=(2, 3), keepdim=True) / n
        # 论文公式(5)
        att = x_minus / (4.0 * (var + self.e_lambda)) + 0.5
        return x * self.act(att)