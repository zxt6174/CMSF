import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from spikingjelly.clock_driven import surrogate
import torch.nn.functional as F
from functools import partial
from spikingjelly.clock_driven import surrogate, neuron, functional
from lib.CPG import  CPGLinear, CPG
from lib.positional_embedding import PositionEmbedding
from typing import Callable, overload
from lib.modules.aggr.gpo import GPO
from lib.spike_coder import Dynamic_Threshold_LIFNode,ConvEncoder,RepeatEncoder,DeltaEncoder
#from lib.GAC import GAC
from lib.multispike import Multispike,Multispike_att
import torch.distributed as dist
import torch.distributed.nn.functional as DF
backend='cupy'
detach_reset = True

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

        self.begin_lif = MultiStepLIFNode(tau=2.0, v_threshold=1.0, detach_reset=True, backend='cupy')

        self.fc1_linear = nn.Linear(in_features, hidden_features)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = MultiStepLIFNode(tau=2.0, v_threshold=1.0, detach_reset=True, backend='cupy')

        self.fc2_linear = nn.Linear(hidden_features, out_features)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T,B,N,C = x.shape

        x=self.begin_lif(x)

        x_ = x.flatten(0, 1)#上边的MLP把T和L展平了，看风格好像是sequence的写法

        x = self.fc1_linear(x_)
        x = self.fc1_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, self.c_hidden).contiguous()
        x = self.fc1_lif(x)

        x = self.fc2_linear(x.flatten(0,1))
        x = self.fc2_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        return x


class Spiking_GFNN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        hidden_expend_dim = int(2 * hidden_dim)

        self.hidden_dim = hidden_dim
        self.dim = dim

        self.begin_lif=MultiStepLIFNode(tau=2.0, v_threshold=1.0, detach_reset=True, backend='cupy')
        self.to_hidden_linear = nn.Linear(dim, hidden_expend_dim)

        self.to_hidden_v_bn = nn.BatchNorm1d(hidden_dim)
        self.to_hidden_v_lif = MultiStepLIFNode(tau=2.0, v_threshold=1.0, detach_reset=True, backend='cupy')

        self.to_out = nn.Linear(hidden_dim, dim)
        self.to_bn = nn.BatchNorm1d(dim)
        #self.to_lif = MultiStepLIFNode(tau=2.0, v_threshold=1.0, detach_reset=True, backend='cupy')

    def forward(self, x):
        T, B, N, C = x.shape

        x=self.begin_lif(x)

        x_for_input = x.flatten(0, 1)

        hidden_layer = self.to_hidden_linear(x_for_input)

        v, gate = hidden_layer.chunk(2, dim=-1)

        v = self.to_hidden_v_bn(v.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, self.hidden_dim).contiguous()
        v = self.to_hidden_v_lif(v).flatten(0, 1)

        out = torch.mul(gate, v)

        out = self.to_out(out)
        out = self.to_bn(out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, self.dim).contiguous()
        #out = self.to_lif(out)

        return out



class SpikingSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125

        self.begin_lif = MultiStepLIFNode(tau=2.0, v_threshold=1.0, detach_reset=True, backend='cupy')

        self.q_linear = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, v_threshold=1.0, detach_reset=True, backend='cupy')

        self.k_linear = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, v_threshold=1.0, detach_reset=True, backend='cupy')

        self.v_linear = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, v_threshold=1.0, detach_reset=True, backend='cupy')

        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        self.proj_linear = nn.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm1d(dim)
        #self.proj_lif = MultiStepLIFNode(tau=2.0, v_threshold=1.0, detach_reset=True, backend='cupy')

    def forward(self, x):
        T, B, L, D = x.shape

        x=self.begin_lif(x)

        x_for_qkv = x.flatten(0, 1)  # TB, N, C   4B，2048,36
        q_linear_out = self.q_linear(x_for_qkv)  # [TB, N, C] linear：128*128
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
        x = x.flatten(0, 1)
        #x = self.proj_lif(self.proj_bn(self.proj_linear(x).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L, D))
        x=self.proj_bn(self.proj_linear(x).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L, D)

        return x



class SpikingBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SpikingSelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        #self.mlp = Spiking_GFNN(dim=dim,hidden_dim=mlp_hidden_dim)
        self.mlp=MLP(in_features=dim,hidden_features=mlp_hidden_dim,out_features=dim)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)

        return x

class AttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):  # x: ( T,B, L, D)
        attn_weights = self.attn(x).squeeze(-1)  # (T, B, L)
        attn_weights = torch.softmax(attn_weights, dim=2)  # (T, B, L)
        x_pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=2)  # (T, B, D)
        return x_pooled

class Spikingformer(nn.Module):
    dim :int=128,
    num_pe_neuron: int = 10
    def __init__(self,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2], T = 4, pretrained_cfg= None
                 ):
        super().__init__()

        self.depths = depths
        self.T = T
        self.embed_size = int(embed_dims)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        self.spike_coder=RepeatEncoder(self.T)

        self.block = nn.ModuleList([SpikingBlock(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths)])

        #self.pos_emb = PositionEmbedding(input_size=self.embed_size)

        self.apply(self._init_weights)

        self.linear=nn.Linear(2048,embed_dims)
        
        weights = [2.3026, 2.3026]
        #weights = [2.3026, 1.6094, 0.9163, 0.2231]
        #weights = [2.3026, 1.9,1.6094, 1.2,0.9163, 0.5, 0.2231,0.1,0.08,0.05]
        self.weights_tensor = nn.Parameter(torch.tensor(weights).unsqueeze(0).unsqueeze(2))
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
    def forward(self, x,image_lengths):
        functional.reset_net(self)
        #print("img-embed-before-input:", x.shape, x)

        x = self.linear(x)
        x=self.spike_coder(x)
        x = self.forward_features(x)
        

        x=x.mean(2)#TBD#,有没有可能是直接mean的原因，这也是GPO的优化的位置，待定

        #x=self.img_attn_pool(x)
        '''
        #多尺度,avg+max
        x_avg = x.mean(dim=2)  # (T,B, D)
        x_max, _ = x.max(dim=2)
        x = torch.cat([x_avg, x_max], dim=2)  # → (T,B, 2D)
        '''

        
        x = x.transpose(0, 1)  # [B, T, D]
        x = torch.sum(x * self.weights_tensor.to(x.device), dim=1)#BD
        x = l2norm(x, dim=-1)
     
        return x


@register_model
def img_ssa(pretrained=False, **kwargs):
    model = Spikingformer(
        **kwargs
    )
    model.default_cfg = _cfg()
    return model
