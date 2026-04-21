import torch
import torch.nn as nn
import numpy as np
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from spikingjelly.clock_driven import surrogate
import torch.nn.functional as F
from functools import partial
from transformers import BertModel
from spikingjelly.clock_driven import surrogate, neuron, functional
from lib.CPG import  CPGLinear, CPG
from lib.positional_embedding import PositionEmbedding
from typing import Callable, overload
from lib.modules.aggr.gpo import GPO
from lib.spike_coder import Dynamic_Threshold_LIFNode,ConvEncoder,RepeatEncoder,DeltaEncoder,StageEncoder,LinearEncoder,RepeatTextEncoder,StageTextEncoder,LinearLNEncoder
#from lib.multi_dynamic_lif import Multi_Threshold_Acc as Multi_Dynamic_Threshold_LIF
#from lib.multi_dynamic_lif import Dynamic_Threshold_LIFNode
#from lib.GAC import GAC
from lib.multispike import Multispike, Multispike_att

import torch.distributed as dist
import torch.distributed.nn.functional as DF
backend='cupy'
detach_reset = True
model_path='bert-base-uncased'

def printSpikeInfo(layertensor, name, isprint=True):
    if isprint:
        non_zero_elm = torch.count_nonzero(layertensor).item()
        #璁＄畻闈為浂鍊肩殑涓暟
        spike_num = layertensor.sum().item()
        #璁＄畻闈為浂鍊肩殑鍜岋紝濡傛灉淇″彿鏄?,1鐨勮瘽锛屼簩鑰呭簲璇ョ浉绛?        elem_num = layertensor.numel()
        spike_rate = spike_num * 1.0 / elem_num
        sparse_rate = 1 - non_zero_elm * 1.0 / elem_num
        print('name:%s shape:%s, elem sum: %d, elem num: %d, non zero elm: %d, fire rate: %.5f, sparse rate: %.5f, is fire:%d' % (name, layertensor.shape, spike_num, elem_num, non_zero_elm, spike_rate, sparse_rate, non_zero_elm==spike_num))
    return

def elem_count(x):
    unique_values, counts = torch.unique(x, return_counts=True)
    for val, count in zip(unique_values.tolist(), counts.tolist()):
        print(f"鍏冪礌 {val} 鍑虹幇浜?{count} 娆?,flush=True)
    return 

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

    
        self.fc1_linear = nn.Linear(in_features, hidden_features)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = Multispike()

        self.fc2_linear = nn.Linear(hidden_features, out_features)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif= Multispike()
        

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T,B,N,C = x.shape

        #x=self.begin_lif(x)

        x_ = x.flatten(0, 1)#涓婅竟鐨凪LP鎶奣鍜孡灞曞钩浜嗭紝鐪嬮鏍煎ソ鍍忔槸sequence鐨勫啓娉?
        x = self.fc1_linear(x_)
        x = self.fc1_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, self.c_hidden).contiguous()
        x = self.fc1_lif(x)

        x = self.fc2_linear(x.flatten(0,1))
        x = self.fc2_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        x = self.fc2_lif(x)
        return x



class Spiking_GFNN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        hidden_expend_dim = int(2 * hidden_dim)

        self.hidden_dim = hidden_dim
        self.dim = dim

        #self.begin_lif=Multispike()
        self.to_hidden_linear = nn.Linear(dim, hidden_expend_dim)

        self.to_hidden_v_bn = nn.BatchNorm1d(hidden_dim)
        self.to_hidden_v_lif = Multispike()

        self.to_out = nn.Linear(hidden_dim, dim)
        self.to_bn = nn.BatchNorm1d(dim)
        self.to_lif = Multispike()

    def forward(self, x):
        T, B, N, C = x.shape

        #x=self.begin_lif(x)

        x_for_input = x.flatten(0, 1)

        hidden_layer = self.to_hidden_linear(x_for_input)

        v, gate = hidden_layer.chunk(2, dim=-1)

        v = self.to_hidden_v_bn(v.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, self.hidden_dim).contiguous()
        v = self.to_hidden_v_lif(v).flatten(0, 1)

        out = torch.mul(gate, v)

        out = self.to_out(out)
        out = self.to_bn(out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, self.dim).contiguous()
        out = self.to_lif(out)

        return out



class SpikingSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125

        #self.begin_lif = Multispike()

        self.q_linear = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = Multispike()

        self.k_linear = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = Multispike()

        self.v_linear = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = Multispike()

        self.attn_lif = Multispike_att()

        self.proj_linear = nn.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = Multispike()

    def forward(self, x):
        T, B, L, D = x.shape

        #x=self.begin_lif(x)

        x_for_qkv = x.flatten(0, 1)  # TB, N, C   4B锛?048,36
        q_linear_out = self.q_linear(x_for_qkv)  # [TB, N, C] linear锛?28*128
        q_linear_out = self.q_bn(q_linear_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L, D).contiguous()
        q_linear_out = self.q_lif(q_linear_out)
        q = q_linear_out.reshape(T, B, L, self.num_heads, D // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_linear_out = self.k_linear(x_for_qkv)
        k_linear_out = self.k_bn(k_linear_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L, D).contiguous()
        k_linear_out = self.k_lif(k_linear_out)
        k = k_linear_out.reshape(T, B, L, self.num_heads, D // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_linear_out = self.v_linear(x_for_qkv)
        v_linear_out = self.v_bn(v_linear_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L, D).contiguous()
        v_linear_out = self.v_lif(v_linear_out)
        v = v_linear_out.reshape(T, B, L, self.num_heads, D // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        x = k.transpose(-2, -1) @ v
        x = (q @ x) * self.scale


        x = x.transpose(2, 3).reshape(T, B, L, D).contiguous()
        x = self.attn_lif(x)
        #杈撳嚭鍏冪礌涓暟
        #elem_count(x)

        x = x.flatten(0, 1)
        x = self.proj_lif(self.proj_bn(self.proj_linear(x).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L, D))
        #x=self.proj_bn(self.proj_linear(x).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L, D)

        return x






class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SpikingSelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Spiking_GFNN(dim=dim, hidden_dim=mlp_hidden_dim)
        #self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim)

    def forward(self, x):
        x = x + self.attn(x)#TBLD
        x = x + self.mlp(x)#TBLD
        return x

        
class Spikformer(nn.Module):
    dim :int=128,
    num_pe_neuron: int = 10,
    def __init__(self,
                 embed_dims=128, num_heads=4, mlp_ratios=4, qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=8, sr_ratios=2, T=4
                 ):
        super().__init__()
        self.T = T  # time step
        self.depths = depths
        self.embed_size=int (embed_dims)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        #self.spike_coder=ConvEncoder(self.T)
        #self.spike_coder=DeltaEncoder(self.T)
        #self.spike_coder=RepeatEncoder(self.T)
        self.spike_coder=RepeatTextEncoder(self.T,self.embed_size)
        #self.spike_coder = StageTextEncoder(self.T,self.embed_size)
        #self.spike_coder=LinearLNEncoder(self.T,self.embed_size)
        #self.spike_coder=StageEncoder(self.T,self.embed_size)

        self.block = nn.ModuleList([Block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths)])


        self.apply(self._init_weights)

        self.linear=nn.Linear(2048,embed_dims)
        
        weights = [2.3026, 2.3026]
        #weights = [2.3026, 1.9,1.6094, 1.2,0.9163, 0.5]
        #weights = [2.3026, 1.9,1.6094, 1.2,0.9163, 0.5, 0.2231,0.1,0.08,0.05]
        self.weights_tensor = nn.Parameter(torch.tensor(weights).unsqueeze(0).unsqueeze(2).unsqueeze(3))#1,2,1,1,
        self.weights_tensor2 = nn.Parameter(torch.tensor(weights).unsqueeze(0).unsqueeze(2).unsqueeze(3))
        #self.gpool = GPO(32, 32)

        #self.gac=GAC(T=self.T,out_channels=self.embed_size)
        #self.img_attn_pool=AttentionPool(embed_dims)
        


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward_features(self, x):

        for blk in self.block:
            x = blk(x)
        #return x
        return x

    def forward(self, x,lengths):
        #x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1)
        functional.reset_net(self)#閲嶇疆鑷繁

        #x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)#TBCHW
        #x=self.SPS(x)#TBND

        x = self.linear(x)
        x = self.spike_coder(x)
        x = self.forward_features(x)
        

        #x=x.mean(2)#TBD#,鏈夋病鏈夊彲鑳芥槸鐩存帴mean鐨勫師鍥狅紝杩欎篃鏄疓PO鐨勪紭鍖栫殑浣嶇疆锛屽緟瀹?
        #x=self.img_attn_pool(x)
    

        
        x = x.transpose(0, 1)  # [B, T,  L,D]
        x = torch.sum(x * self.weights_tensor.to(x.device), dim=1)#BLD
        #x = x * self.weights_tensor.to(x.device)
        #x = x.flatten(1,2)
        x = l2norm(x, dim=-1) #BLD

        return x


@register_model
def img_ssa(pretrained=False, **kwargs):
    model = Spikformer(
        # img_size_h=224, img_size_w=224,
        # patch_size=16, embed_dims=768, num_heads=12, mlp_ratios=4,
        # in_channels=3, num_classes=1000, qkv_bias=False,
        # norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=12, sr_ratios=1,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model



