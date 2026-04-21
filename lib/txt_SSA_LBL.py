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
from lib.spike_coder import Dynamic_Threshold_LIFNode,ConvEncoder,RepeatEncoder,DeltaEncoder,LinearEncoder,RepeatTextEncoder,LinearLNEncoder,Linear2BN1dEncoder,RepeatBN1dEncoder
#from lib.multi_dynamic_lif import Multi_Threshold_Acc as Multi_Dynamic_Threshold_LIF
#from lib.multi_dynamic_lif import Dynamic_Threshold_LIFNode
#from lib.GAC import GAC
from lib.multispike import Multispike, Multispike_att
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from lib.modules.aggr.spike_rnn import SpikeRNN

import torch.distributed as dist
import torch.distributed.nn.functional as DF
backend='cupy'
detach_reset = True
model_path='bert-base-uncased'

def printSpikeInfo(layertensor, name, isprint=True):
    if isprint:
        non_zero_elm = torch.count_nonzero(layertensor).item()
        #计算非零值的个数
        spike_num = layertensor.sum().item()
        #计算非零值的和，如果信号是0,1的话，二者应该相等
        elem_num = layertensor.numel()
        spike_rate = spike_num * 1.0 / elem_num
        sparse_rate = 1 - non_zero_elm * 1.0 / elem_num
        print('name:%s shape:%s, elem sum: %d, elem num: %d, non zero elm: %d, fire rate: %.5f, sparse rate: %.5f, is fire:%d' % (name, layertensor.shape, spike_num, elem_num, non_zero_elm, spike_rate, sparse_rate, non_zero_elm==spike_num))
    return

def elem_count(x):
    unique_values, counts = torch.unique(x, return_counts=True)
    for val, count in zip(unique_values.tolist(), counts.tolist()):
        print(f"元素 {val} 出现了 {count} 次",flush=True)
    return 

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class MLP_Graph(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1_linear = nn.Linear(in_features, hidden_features)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = MultiStepLIFNode(tau=2.0, v_threshold=1.0, detach_reset=True, backend='cupy')

        self.fc2_linear = nn.Linear(hidden_features, out_features)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif= MultiStepLIFNode(tau=2.0, v_threshold=1.0, detach_reset=True, backend='cupy')
        
        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T,B,N,C = x.shape

        #x=self.begin_lif(x)

        x = x.flatten(0, 1)#上边的MLP把T和L展平了，看风格好像是sequence的写法
        x = self.fc1_linear(x)
        x = x.reshape(T,B,N,-1).permute(1,2,3,0).flatten(0,1)
        x = self.fc1_bn(x).reshape(B,N,self.c_hidden,T).permute(3,0,1,2)
        x = self.fc1_lif(x)

        x=x.flatten(0,1)
        x = self.fc2_linear(x)
        x = x.reshape(T,B,N,-1).permute(1,2,3,0).flatten(0,1)
        x = self.fc2_bn(x).reshape(B,N,self.c_output,T).permute(3,0,1,2)
        x = self.fc2_lif(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features 
        hidden_features = hidden_features 


        self.fc1_linear = nn.Linear(in_features, hidden_features)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        #self.fc1_lif = MultiStepLIFNode(tau=2.0, v_threshold=1.0, detach_reset=True, backend='cupy')

        self.fc2_linear = nn.Linear(hidden_features, out_features)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        #self.fc2_lif= MultiStepLIFNode(tau=2.0, v_threshold=1.0, detach_reset=True, backend='cupy')
        

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        B,N,C = x.shape

        #x=self.begin_lif(x)

        x = x.flatten(0, 1)#上边的MLP把T和L展平了，看风格好像是sequence的写法

        x = self.fc1_linear(x)
        x = self.fc1_bn(x)
        x = F.relu(x)

        x = self.fc2_linear(x)
        x = self.fc2_bn(x)
        x = x.reshape(B,N,-1)
        return x


class Spiking_GFNN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        hidden_expend_dim = int(2 * hidden_dim)

        self.hidden_dim = hidden_dim
        self.dim = dim

        #self.begin_lif=MultiStepLIFNode(tau=2.0, v_threshold=1.0, detach_reset=True, backend='cupy')
        self.to_hidden_linear = nn.Linear(dim, hidden_expend_dim)

        self.to_hidden_v_bn = nn.BatchNorm1d(hidden_dim)
        self.to_hidden_v_lif = MultiStepLIFNode(tau=2.0, v_threshold=1.0, detach_reset=True, backend='cupy')

        self.to_out = nn.Linear(hidden_dim, dim)
        self.to_bn = nn.BatchNorm1d(dim)
        self.to_lif = MultiStepLIFNode(tau=2.0, v_threshold=1.0, detach_reset=True, backend='cupy')

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

        #self.begin_lif = MultiStepLIFNode(tau=2.0, v_threshold=1.0, detach_reset=True, backend='cupy')

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
        self.proj_lif = MultiStepLIFNode(tau=2.0, v_threshold=1.0, detach_reset=True, backend='cupy')

    def forward(self, x):
        T, B, L, D = x.shape

        #x=self.begin_lif(x)

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
        #输出元素个数
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
        #self.spike_coder=RepeatBN1dEncoder(self.T,self.embed_size)
        self.spike_coder=RepeatTextEncoder(self.T,self.embed_size)# RepeatLN
        #self.spike_coder=Linear2BN1dEncoder(self.T,self.embed_size)
        #self.spike_coder=RepeatEncoder(self.T)
        #self.spike_coder=LinearLNEncoder(self.T,self.embed_size)
        #self.spike_coder=LinearEncoder(self.T,self.embed_size) #LinearBN

        self.block = nn.ModuleList([Block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths)])

        
        self.apply(self._init_weights)

        self.bert = BertModel.from_pretrained(model_path)
        #self.bert = nn.GRU(1024, 1024, 1, batch_first=True, bidirectional=True)
        self.bert_linear = nn.Linear(768, self.embed_size)
        #self.pos_emb=PositionEmbedding(input_size=self.embed_size)
        self.cpg = CPGLinear(input_size=self.embed_size,output_size=self.embed_size)
        self.init_lif=Dynamic_Threshold_LIFNode(tau=2.0, detach_reset=True, backend='cupy')
        #self.srnn=SpikeRNN(input_size=1024,max_length=36,hidden_size=1024,layers=1,num_steps=2)

        vocab_size = 30522
        self.token_embedding=nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dims)
        #self.res_bn2d=nn.BatchNorm2d(self.T)
        self.res_ln = nn.LayerNorm(embed_dims)
        
        temp = [2.3026, 1.9,1.6094, 1.2,0.9163, 0.5]
        weights = temp[0:self.T]
        #weights = [2.3026, 1.9,1.6094, 1.2,0.9163, 0.5, 0.2231,0.1,0.08,0.05]
        self.weights_tensor = nn.Parameter(torch.tensor(weights).unsqueeze(0).unsqueeze(2).unsqueeze(3)) # 1,T,1,1
        self.weights_tensor2 = nn.Parameter(torch.tensor(weights).unsqueeze(0).unsqueeze(2).unsqueeze(3)) # 1,T,1,1
        
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
        functional.reset_net(self)#重置自己
        
        bert_attention_mask = (x != 0).float()
        #print("text-embed-before-bert:",x.shape,x)
        x= self.bert(x, bert_attention_mask)[0]  # B x N x D
        #print("text-embed-after-bert:", x.shape, x)

        '''
        x=self.token_embedding(x)#token转换为token编码
        self.bert.flatten_parameters()
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True)
        # Forward propagate RNN
        out, _ = self.bert(packed)
        padded = pad_packed_sequence(out, batch_first=True)
        x, cap_len = padded
        x = (x[:, :, :x.size(2) // 2] + x[:, :, x.size(2) // 2:]) / 2
        print(x)
        '''

        
        x=self.bert_linear(x)#768->embed_size
       
        res_x=x

        #res_x = re_x.unsqueeze(0).repeat(self.T,1,1,1)
        #res_x = self.res_ln(res_x)

        
        #
        x=self.spike_coder(x)#浮点转脉冲4
        #res_x=self.srnn(x)
        '''
        T,B,L,D=x.shape
        x=x.permute(1,0,2,3).flatten(1,2)#B,TL,D
        x=self.cpg(x).reshape(B,T,L,D).permute(1,0,2,3).contiguous()#T,B,L,D
        x=self.init_lif(x)
        '''
        #x=self.pos_emb(x)#加入位置编码
        #res_x =x
       
        x = self.forward_features(x)#注意力块
        T_x=x
        
        #x=x.mean(2)

        #x = x.mean(0)#可以换成权重吧？
        #x=x.mean(2)#TBD
        #x=self.txt_attn_pool(x)
     

        x = x.transpose(0, 1)  # [B, T, L, D]
        #res_x = res_x.transpose(0,1)

        #T_x=x#用于后续cross-encoder的输入

        x = torch.sum(x * self.weights_tensor.to(x.device), dim=1)#BLD
        #res_x = torch.sum(res_x * self.weights_tensor.to(res_x.device), dim=1)#BLD

        #sum pooling
        #x=x.sum(dim=1)
        #res_x= res_x.sum(dim=1)

        #max pooling
        #x=x.max(dim=1)[0]
        #res_x=res_x.max(dim=1)[0]

        #LSE pooling
        #x[x==0]=-torch.inf
        #x= torch.logsumexp(x,dim=1)
        #res_x = torch.logsumexp(res_x,dim=1)

        #x=x+res_x
        x=l2norm(x,dim=-1)#使用l2可以保证长得差不多，但是数据是否有意义呢，所以不如接一个GPO
        res_x_norm=l2norm(res_x,dim=-1)
        
        #print("txt-encoder",x.shape,flush=True)#tptal-batch,dim

        return x,res_x,res_x_norm


@register_model
def txt_ssa(pretrained=False, **kwargs):
    model = Spikformer(
        # img_size_h=224, img_size_w=224,
        # patch_size=16, embed_dims=768, num_heads=12, mlp_ratios=4,
        # in_channels=3, num_classes=1000, qkv_bias=False,
        # norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=12, sr_ratios=1,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

