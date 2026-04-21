from turtle import forward
import torch
import torch.nn as nn
import numpy as np

from transformers import BertModel

from lib.modules.aggr.gpo import GPO
from lib.modules.mlp import MLP
import logging

import torch.nn.functional as F
bert_model_path='bert-base-uncased'

logger = logging.getLogger(__name__)

#from lib.txt_SSA import txt_ssa
#from lib.img_SSA import img_ssa
from  lib.img_SSA_LBL import img_ssa
from lib.txt_SSA_LBL import txt_ssa
#from lib.img_SSA_OST import img_ssa
#from lib.txt_SSA_OST import txt_ssa
from lib.CrossEncoder import cross_sca

from functools import partial

from lib.coding import get_coding, get_pooling,T2ICrossAttentionPool

from spikingjelly.clock_driven import surrogate, neuron, functional

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def maxk_pool1d_var(x, dim, k, lengths):
    results = list()
    lengths = list(lengths.cpu().numpy())
    lengths = [int(x) for x in lengths]
    for idx, length in enumerate(lengths):
        k = min(k, length)
        max_k_i = maxk(x[idx, :length, :], dim - 1, k).mean(dim - 1)
        results.append(max_k_i)
    results = torch.stack(results, dim=0)
    return results


def maxk_pool1d(x, dim, k):
    max_k = maxk(x, dim, k)
    return max_k.mean(dim)


def maxk(x, dim, k):
    index = x.topk(k, dim=dim)[1]
    return x.gather(dim, index)


def get_text_encoder(embed_size, no_txtnorm=False):
    
    txt_enc = txt_ssa(
        embed_dims=1024, num_heads=8, mlp_ratios=4, qkv_bias=False, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=1, sr_ratios=1, T=2
    )
    #return EncoderText(embed_size, no_txtnorm=no_txtnorm)
    return  txt_enc



def get_image_encoder(data_name, img_dim, embed_size, precomp_enc_type='basic',
                      backbone_source=None, backbone_path=None, no_imgnorm=False, opt = None):
    img_enc = img_ssa(
        embed_dims=1024, num_heads=8, mlp_ratios=4, qkv_bias=False, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=1, sr_ratios=1, T=2
    )
    return img_enc

def get_cross_encoder(embed_size, no_txtnorm=False):
    
    cross_enc = cross_sca(
        embed_dims=1024, num_heads=6, mlp_ratios=4, qkv_bias=False, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=1, sr_ratios=1, T=2
    )
    
    #return EncoderText(embed_size, no_txtnorm=no_txtnorm)
    return  cross_enc



class SimsEncoder(nn.Module):
    def __init__(self, coding_type, pooling_type, **args):
        super(SimsEncoder, self).__init__()
        self.opt = args["opt"]
        #self.coding =T2ICrossAttentionPool()
        #self.coding1 = get_coding("OptTransCoding",opt=self.opt)
        #self.coding = get_coding("HUNGARIANCoding",opt=self.opt)
        #self.coding = get_coding("VHACoding", opt=self.opt)
        #self.coding = get_coding("THACoding", opt=self.opt)
        self.coding = get_coding("VTadd_HACoding", opt=self.opt)

        self.pooling = get_pooling(pooling_type, opt=self.opt)

    def forward(self, img_emb, cap_emb, img_lens, cap_lens):
        sims = self.coding(img_emb, cap_emb, img_lens, cap_lens)
        #sims1 = self.coding1(img_emb, cap_emb, img_lens, cap_lens)
        sims = self.pooling(sims)
        #sims1 = self.pooling(sims1)
        #sims_mean= self.pooling(sims_mean)
        #sims2 = self.pooling(sims2)
        #sims =sims1*sims2
        #sims = self.coding(img_emb, cap_emb, img_lens, cap_lens)
        #sims = self.pooling(sims)
      
        return sims
