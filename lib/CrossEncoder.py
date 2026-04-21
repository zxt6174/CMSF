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
from transformers import BertModel,BertTokenizer



backend='cupy'
detach_reset = True
model_path='bert-base-uncased'

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def printSpikeInfo(layertensor, name, isprint=True):
    if isprint:
        non_zero_elm = torch.count_nonzero(layertensor).item()
        #计算非零值的个数
        spike_num = layertensor.sum().item()
        #计算非零值的和，如果信号是0,1的话，二者应该相等
        elem_num = layertensor.numel()
        spike_rate = spike_num * 1.0 / elem_num
        sparse_rate = 1 - non_zero_elm * 1.0 / elem_num
        print('name:%s shape:%s, elem sum: %d, elem num: %d, non zero elm: %d, fire rate: %.5f, sparse rate: %.5f, is fire:%d' % (name, layertensor.shape, spike_num, elem_num, non_zero_elm, spike_rate, sparse_rate, non_zero_elm==spike_num),flush=True)
    return

class Dynamic_Threshold_LIFNode(MultiStepLIFNode):
    def __init__(self, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, backend='torch'):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, backend)
        init_dynamic_threshold = v_threshold
        self.dynamic_threshold = nn.Parameter(torch.as_tensor(init_dynamic_threshold))

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.dynamic_threshold)

class RepeatTextEncoder(nn.Module):
    def __init__(self, output_size: int,dim: int):
        super().__init__()
        self.out_size = output_size
        self.lif = Dynamic_Threshold_LIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.norm2=nn.LayerNorm(dim)
        #self.lif=Multispike_att()
        #self.lif=MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, inputs: torch.Tensor):
        # inputs: BLD
        functional.reset_net(self)
        inputs = inputs.repeat(
            tuple([self.out_size] + torch.ones(len(inputs.size()), dtype=int).tolist())
        )  # TBLD
        inputs=self.norm2(inputs)#TBLD
        inputs = self.lif(inputs)#tbld
        return inputs

class Spiking_GFNN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        hidden_expend_dim = int(2 * hidden_dim)

        self.hidden_dim = hidden_dim
        self.dim = dim

        self.to_hidden_linear = nn.Linear(dim, hidden_expend_dim)

        self.to_hidden_v_bn = nn.BatchNorm1d(hidden_dim)
        self.to_hidden_v_lif = MultiStepLIFNode(tau=2.0, v_threshold = 1.0, detach_reset=True, backend='cupy')

        self.to_out = nn.Linear(hidden_dim, dim)
        self.to_bn = nn.BatchNorm1d(dim)
        self.to_lif = MultiStepLIFNode(tau=2.0, v_threshold = 1.0, detach_reset=True, backend='cupy')

    def forward(self, x):
        T,B,N,C = x.shape

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
class SpikingQKAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125

        # Query, Key, Value 线性映射
        self.q_linear = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.k_linear = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        '''
        self.v_linear = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        '''
        #默认阈值是1.0，设成0.5是希望捕获更多的注意力信息，也可以替换为Dynamic
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')
        self.attn_lif2 = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        # 投影层
        self.proj_linear = nn.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, query, key, value):
        """
        Args:
            query: [T, B, L_query, D] Query 特征
            key:   [T, B, L_key, D] Key 特征
            value: [T, B, L_key, D] Value 特征
        Returns:
            output: [T, B, L_query, D] 交叉注意力输出
        """
        T, B, L_query, D = query.shape
        _, _, L_key, _ = key.shape

        # Flatten 时间维度和批次维度
        query = query.flatten(0, 1)  # [TB, L_query, D]
        key = key.flatten(0, 1)      # [TB, L_key, D]
        value = value.flatten(0, 1)  # [TB, L_key, D]
        
        # Query 处理
        q_linear_out = self.q_linear(query)  # [TB, L_query, D]
        q_linear_out = self.q_bn(q_linear_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L_query, D).contiguous()
        q_linear_out = self.q_lif(q_linear_out)
        #q = q_linear_out.reshape(T, B, L_query, self.num_heads, D // self.num_heads).permute(0, 1, 3, 4, 2).contiguous()#T B h D/h L
        q = q_linear_out.reshape(T, B, self.num_heads, L_query // self.num_heads, D).contiguous()#T B h L/h D
        #q=q_linear_out.contiguous()#TBLD

        # Key 处理
        k_linear_out = self.k_linear(key)  # [TB, L_key, D]
        k_linear_out = self.k_bn(k_linear_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L_key, D).contiguous()
        k_linear_out = self.k_lif(k_linear_out)
        #k = k_linear_out.reshape(T, B, L_key, self.num_heads, D // self.num_heads).permute(0, 1, 3, 4, 2).contiguous()#T B h D/h L
        k = k_linear_out.reshape(T, B, self.num_heads, L_key // self.num_heads, D).contiguous()#T B h L/h D
        #k=k_linear_out.contiguous()#TBLD

        '''
        # Value 处理
        v_linear_out = self.v_linear(value)  # [TB, L_key, D]
        v_linear_out = self.v_bn(v_linear_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L_key, D).contiguous()
        v_linear_out = self.v_lif(v_linear_out)
        v = v_linear_out.reshape(T, B, L_key, self.num_heads, D // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        '''
        q = torch.sum(q, dim = 3, keepdim = True)
        #q_k = torch.sum(k, dim = 3, keepdim = True)#做一个尝试
        attn =self.attn_lif(q)
        #attn = self.attn_lif(q)+self.attn_lif2(q_k)
        output=torch.mul(attn,k)

        output = output.transpose(2, 3).reshape(T, B, L_query, D).contiguous()

        #后边接的这个有必要吗，结果已经是01值了
        
        output = output.flatten(0, 1)
        output = self.proj_lif(self.proj_bn(self.proj_linear(output).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L_query, D))
        
        return output

class SpikingCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125

        # Query, Key, Value 线性映射
        self.q_linear = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, v_threshold=1.0, detach_reset=True, backend='cupy')

        self.k_linear = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, v_threshold=1.0, detach_reset=True, backend='cupy')

        self.v_linear = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, v_threshold=1.0, detach_reset=True, backend='cupy')

        #默认阈值是1.0，设成0.5是希望捕获更多的注意力信息，也可以替换为Dynamic
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        # 投影层
        self.proj_linear = nn.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0, v_threshold=1.0, detach_reset=True, backend='cupy')

        #self.qkv=SCA_QKV(dim=dim)

    def forward(self, query, key, value):
    
        T, B, L_query, D = query.shape
        _, _, L_key, _ = key.shape

        # Flatten 时间维度和批次维度
        query = query.flatten(0, 1)  # [TB, L_query, D]
        key = key.flatten(0, 1)      # [TB, L_key, D]
        value = value.flatten(0, 1)  # [TB, L_key, D]
        
        # Query 处理
        q_linear_out = self.q_linear(query)  # [TB, L_query, D]
        q_linear_out = self.q_bn(q_linear_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L_query, D).contiguous()
        q_linear_out = self.q_lif(q_linear_out)
        q = q_linear_out.reshape(T, B, L_query, self.num_heads, D // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()#T B h L D/h
        #q = q_linear_out.reshape(T//self.num_heads, self.num_heads,B, L_query, D).permute(0, 2, 1, 3, 4).contiguous()#T/h B h L D

        # Key 处理
        k_linear_out = self.k_linear(key)  # [TB, L_key, D]
        k_linear_out = self.k_bn(k_linear_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L_key, D).contiguous()
        k_linear_out = self.k_lif(k_linear_out)
        k = k_linear_out.reshape(T, B, L_key, self.num_heads, D // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        #k = k_linear_out.reshape(T//self.num_heads, self.num_heads,B, L_key, D).permute(0, 2, 1, 3, 4).contiguous()#T/h B h L D

        # Value 处理
        v_linear_out = self.v_linear(value)  # [TB, L_key, D]
        v_linear_out = self.v_bn(v_linear_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L_key, D).contiguous()
        v_linear_out = self.v_lif(v_linear_out)
        v = v_linear_out.reshape(T, B, L_key, self.num_heads, D // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()#T,B h L D/h
        #v = v_linear_out.reshape(T//self.num_heads, self.num_heads,B, L_key, D).permute(0, 2, 1, 3, 4).contiguous()#T/h B h L D
        
        #用来测试
        # _,_,_=self.qkv(q,k,v)

        attn = k.transpose(-2, -1) @ v #N^2D
        output = (q @ attn) * self.scale #N^2D
        #attn=q@k.transpose(-2,-1)   
        #output=(attn@v)*self.scale

        output = output.transpose(2, 3).reshape(T, B, L_query, D).contiguous()
        #output = output.transpose(1, 2).reshape(T, B, L_query, D).contiguous()#这是我想在T维度做多头注意力
        output = self.attn_lif(output)

        output = output.flatten(0, 1)
        output = self.proj_lif(self.proj_bn(self.proj_linear(output).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L_query, D))

        return output


class SCA_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SpikingCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Spiking_GFNN(dim=dim,hidden_dim=mlp_hidden_dim)
        #self.mlp=MLP(in_features=dim,hidden_features=mlp_hidden_dim,out_features=dim)

    def forward(self, x, y):

        x= x+self.attn(x, y, y)
        x = x + self.mlp(x)

        return x

class QK_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SpikingQKAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Spiking_GFNN(dim=dim,hidden_dim=mlp_hidden_dim)
        #self.mlp=MLP(in_features=dim,hidden_features=mlp_hidden_dim,out_features=dim)

    def forward(self, x, y):

        x= x+self.attn(x, y,y)
        x = x + self.mlp(x)

        return x


class SpikingCrossAttentionFormer(nn.Module):
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
        self.spike_coder1=RepeatTextEncoder(self.T,self.embed_size)
        self.spike_coder2=RepeatTextEncoder(self.T,self.embed_size)

        self.apply(self._init_weights)

        temp = [2.3026, 1.9,1.6094, 1.2,0.9163, 0.5]
        weights = temp[0:self.T]
        self.weights_tensor_img = nn.Parameter(torch.tensor(weights).unsqueeze(0).unsqueeze(2).unsqueeze(3))
        self.weights_tensor_txt = nn.Parameter(torch.tensor(weights).unsqueeze(0).unsqueeze(2).unsqueeze(3))

        '''
        self.block = nn.ModuleList([SCA_Block(
            dim=embed_dims, num_heads=8, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths)])
        
        self.block2 = nn.ModuleList([SCA_Block(
            dim=embed_dims, num_heads=8, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths)])
        '''
        
        self.block_i=nn.ModuleList([QK_Block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths)])

        self.block_t=nn.ModuleList([QK_Block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths)])
        
        '''
        self.block_i2=nn.ModuleList([QK_Block(
            dim=embed_dims, num_heads=6, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths)])

        self.block_t2=nn.ModuleList([QK_Block(
            dim=embed_dims, num_heads=6, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths)])
        '''
        
        
        '''
        #以下是ANN的transformer
        self.cross_attn1 = nn.MultiheadAttention(embed_dims, num_heads=8, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dims)
        self.ffn1 = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 4),
            nn.ReLU(),
            nn.Linear(embed_dims * 4, embed_dims),
        )
        self.ln11 = nn.LayerNorm(embed_dims)
        #以下是ANN的cross encoder
        self.cross_attn2 = nn.MultiheadAttention(embed_dims, num_heads=8, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dims)
        self.ffn2 = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 4),
            nn.ReLU(),
            nn.Linear(embed_dims * 4, embed_dims),
        )
        self.ln22 = nn.LayerNorm(embed_dims)
        '''
        


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self,x,y):
      
        
        for blk in self.block:
            out = blk(x,y)
        return out#TBLD
    
    def forward_features2(self,x,y):
        
        
        for blk in self.block2:
            out = blk(x,y)
        return out#TBLD

    def forward_features_i(self,x,y):
        
        
        for blk in self.block_i:
            out = blk(x,y)
        return out#TBLD
    
    def forward_features_t(self,x,y):
        
        
        for blk in self.block_t:
            out = blk(x,y)
        return out#TBLD
    
    def forward_features_i2(self,x,y):
        
        
        for blk in self.block_i2:
            out = blk(x,y)
        return out#TBLD
    
    def forward_features_t2(self,x,y):
        
        
        for blk in self.block_t2:
            out = blk(x,y)
        return out#TBLD
 
   

    def forward(self, img_emb,txt_emb,F_img_emb,F_txt_emb):
        #

        #输入维度是BTLD
        #sims应该可以用上一步的返回值，bs*bs即可，因为我们没使用momentum，相似度矩阵大小固定
        #sim-t2i和sim-i2t在形状为bs*bs的情况下应该是转置矩阵的关系，
        #idx应该让上一步返回，或者dataloader返回值，用于后续计算
        '''
        _,_,L_i,_=img_emb.shape
        _,_,L_t,_=txt_emb.shape
        x=torch.cat((img_emb,txt_emb),dim=2)#TB(Li+Lt)D
        x=self.forward_features(x,x) #TB(Li+Lt)D
        img_cro_emb,txt_cro_emb=torch.split(x,[L_i,L_t],dim=2)# T B L_i D ,T B L_t D
        '''
        
        #SNN QK-Cross Attention
        functional.reset_net(self)
        
        #img_emb =img_emb.unsqueeze(0)# 1 B L D
        #txt_emb = txt_emb.unsqueeze(0)
        #img_emb=img_emb.repeat(self.T,1,1,1)
        #txt_emb=txt_emb.repeat(self.T,1,1,1)
        img_emb = self.spike_coder1(img_emb)
        txt_emb = self.spike_coder2(txt_emb)
        #F_img_emb = self.spike_coder(F_img_emb)
        #F_cap_emb =  self.spike_coder(F_cap_emb)

        
        img_cro_emb = self.forward_features_i(img_emb,txt_emb)

        txt_cro_emb = self.forward_features_t(txt_emb,img_emb) 

        #img_cro_emb = self.forward_features_i(img_cro_emb,img_cro_emb)
        #txt_cro_emb = self.forward_features_t(txt_cro_emb,txt_cro_emb)
       
        #img_cro_emb = img_cro_emb.flatten(0, 1)  # [B, T, L, D]
        img_cro_emb = img_cro_emb.transpose(0, 1)
        img_cro_emb = torch.sum(img_cro_emb * self.weights_tensor_img.to(img_emb.device), dim=1)#BLD

        #F_img_emb = F_img_emb.transpose(0,1)
        #F_img_emb = torch.sum(F_img_emb * self.weights_tensor_img.to(img_emb.device), dim=1)#BLD

        img_cro_emb = img_cro_emb + F_img_emb
        img_cro_emb=l2norm(img_cro_emb,dim=-1)

        #txt_cro_emb = txt_cro_emb.flatten(0, 1)  # [B, T, L, D]
        txt_cro_emb = txt_cro_emb.transpose(0, 1)
        txt_cro_emb = torch.sum(txt_cro_emb * self.weights_tensor_txt.to(txt_emb.device), dim=1)#BLD

        #F_cap_emb=F_cap_emb.transpose(0,1)
        #F_cap_emb = torch.sum(F_cap_emb * self.weights_tensor_txt.to(txt_emb.device), dim=1)#BLD

        txt_cro_emb = txt_cro_emb + F_txt_emb
        txt_cro_emb=l2norm(txt_cro_emb,dim=-1)

        return img_cro_emb,txt_cro_emb
        
        
        #ANN Co-Cross Attention
        '''
        img_emb = img_emb + F_img_emb
        img_emb = l2norm(img_emb,dim=-1)
        txt_emb = txt_emb + F_cap_emb
        txt_emb = l2norm(txt_emb,dim=-1)
        '''
        '''
        img_cro_emb, _ = self.cross_attn1(query=img_emb, key=txt_emb, value=txt_emb,
                                   key_padding_mask=None)  # key_padding_mask: (B, L_txt)
        img_cro_emb = img_emb + self.ln1(img_cro_emb )
        img_cro_emb = img_cro_emb + self.ln11(img_cro_emb)
        img_cro_emb=l2norm(img_cro_emb,dim=-1)

        txt_cro_emb, _ = self.cross_attn2(query=txt_emb, key=img_emb, value=img_emb,
                                   key_padding_mask=None)  # key_padding_mask: (B, L_txt)
        txt_cro_emb = txt_emb + self.ln2(txt_cro_emb )
        txt_cro_emb = txt_cro_emb + self.ln22(txt_cro_emb)
        txt_cro_emb=l2norm(txt_cro_emb,dim=-1)

        return img_cro_emb,txt_cro_emb
        '''
       
        
        '''
        output_pos=self.forward_features(txt_emb.transpose(0,1),img_emb.transpose(0,1)) #这里不需要拼接，直接使用txt-emb作为Query，img-emb作为Key-Value即可
        #TBLD
        idx=torch.Tensor(idx).cuda()
        idx = idx.view(-1,1)
        with torch.no_grad():
            sim_t2i=sim_i2t.T#表示转置，
            bs = img_emb.size(0)    #TBLD   
            weights_i2t = F.softmax(sim_i2t+1e-4,dim=1)
            weights_t2i = F.softmax(sim_t2i+1e-4,dim=1)

            mask = torch.eq(idx, idx.T)
            weights_i2t.masked_fill_(mask, 0)
            weights_t2i.masked_fill_(mask, 0) 

        # select a negative image for each text
        image_embeds_neg = []    
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(img_emb[neg_idx])# BTLD
        image_embeds_neg = torch.stack(image_embeds_neg,dim=0)#BTLD   

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(txt_emb[neg_idx])#BTLD
            #text_atts_neg.append(text.attention_mask[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg,dim=0)#BTLD  
        #text_atts_neg = torch.stack(text_atts_neg,dim=0)      

        text_embeds_all = torch.cat([txt_emb, text_embeds_neg],dim=0) #BTLD    #正-负
        #text_atts_all = torch.cat([text.attention_mask, text_atts_neg],dim=0)     

        image_embeds_all = torch.cat([image_embeds_neg,img_emb],dim=0) #BTLD  #负-正
        #image_atts_all = torch.cat([image_atts,image_atts],dim=0)

        ###
        #此时，output-pos和output-neg的维度仍然是TBLD，是否需要去掉时间维度T，然后在用于下一步的itm输入
        output_neg=self.forward_features(text_embeds_all.transpose(0,1),image_embeds_all.transpose(0,1)) #负-负 二倍bs，如果这个bs太大，也可以拆分成两个bs

        output_pos=output_pos.transpose(0,1).mean(dim=1)#TBLD->BLD
        output_neg=output_neg.transpose(0,1).mean(dim=1)#TBLD->BLD

        #只取BLD中L维度的第一个，得到BD
        vl_embeddings = torch.cat([output_pos[:,0,:], output_neg[:,0,:]],dim=0) #正-负-负
        vl_output = self.itm_head(vl_embeddings)  #取BLD的第一个token作为代表->BD->B2
        #vl_output = self.itm_head(vl_embeddings.mean(dim=1))  #取BLD平均池化作为代表->BD->B2   
        itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],  #三倍bs，bs，bs
                               dim=0).to(img_emb.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)

        ##第二遍
        vl_embeddings2 = torch.cat([output_neg[:,0,:], output_pos[:,0,:]],dim=0) #负-负-正
        vl_output2 = self.itm_head(vl_embeddings2)  #取BLD的第一个token作为代表->BD->B2
        #vl_output = self.itm_head(vl_embeddings.mean(dim=1))  #取BLD平均池化作为代表->BD->B2   
        itm_labels2 = torch.cat([torch.zeros(2*bs,dtype=torch.long),torch.ones(bs,dtype=torch.long)],  #三倍bs，bs，bs
                               dim=0).to(img_emb.device)
        loss_itm2 = F.cross_entropy(vl_output2, itm_labels2)
        #return vl_output, itm_labels
        return loss_itm + loss_itm2,vl_output
        '''

@register_model
def cross_sca(pretrained=False, **kwargs):
    model = SpikingCrossAttentionFormer(
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

