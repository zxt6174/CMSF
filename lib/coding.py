# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.module_new import get_mask, get_fgsims, get_fgmask, l2norm, cosine_similarity, SCAN_attention

from scipy.optimize import linear_sum_assignment
import ot
import itertools

EPS = 1e-8 # epsilon 
MASK = -1 # padding value

class OptTransCoding(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, imgs, caps, img_lens, cap_lens):
        max_r, max_w = int(img_lens.max()), int(cap_lens.max())

        # 获取相似度矩阵和有效位置掩码
        sims = get_fgsims(imgs, caps)[:, :, :max_r, :max_w]  # (B, B, R, W)
        valid_mask = get_fgmask(img_lens, cap_lens)[:, :, :max_r, :max_w]  # (B, B, R, W)

        # 将 padding 部分填为极小值，确保不会被选为最大值
        sims = sims.masked_fill(valid_mask == 0, MASK)  # MASK = -1 or a large negative value

        B, _, R, W = sims.shape

        # 获取行/列最大值位置
        sims_row_val, sims_row_idx = sims.max(dim=-1)  # (B, B, R): 每行最大值对应列索引 w
        sims_col_val, sims_col_idx = sims.max(dim=-2)  # (B, B, W): 每列最大值对应行索引 r

        # 广播比较：(r, w) 是否互为彼此最大
        row_idx_exp = sims_row_idx.unsqueeze(-1)  # (B, B, R, 1)
        col_idx_exp = sims_col_idx.unsqueeze(-2)  # (B, B, 1, W)

        r_idx = torch.arange(R, device=sims.device).view(1, 1, R, 1)  # (1,1,R,1)
        w_idx = torch.arange(W, device=sims.device).view(1, 1, 1, W)  # (1,1,1,W)

        row_match = (row_idx_exp == w_idx)  # (B, B, R, W)
        col_match = (col_idx_exp == r_idx)  # (B, B, R, W)

        # 找出既是行最大又是列最大的点
        mask_score = row_match & col_match  # (B, B, R, W)

        # 结合有效位置掩码，过滤掉 padding
        #mask_score = mask_score & valid_mask  # (B, B, R, W)

        # 保留匹配点的相似度，其他位置设为 MASK
        sims = sims.masked_fill(~mask_score, MASK)

        # 聚合所有匹配点的相似度作为最终得分
        # score = sims.sum(dim=(-1, -2))  # (B, B)
        
        return sims



class VTadd_HACoding(nn.Module):
    def __init__(self):
        super().__init__()
        self.topk = 20
        self.img_topk = 32
        self.cap_topk = 18
        self.alpha=0.5


    def forward(self, imgs, caps, img_lens, cap_lens):
        max_r, max_w = int(img_lens.max()), int(cap_lens.max())
        sims = get_fgsims(imgs, caps)[:, :, :max_r, :max_w]  # Bi x Bt x R x W
        mask = get_fgmask(img_lens, cap_lens)                # Bi x Bt x R x W
        sims = sims.masked_fill(mask == 0, MASK)#MASK=-1,

        #sims_mean1=torch.matmul(imgs.flatten(-2,-1),caps.flatten(-2,-1).t())

        # 取两个方向的最大值
        
        sims_row = sims.max(dim=-1)[0]  # Max over width W -> Bi x Bt x R
        sims_col = sims.max(dim=-2)[0]  # Max over region R -> Bi x Bt x W
        ##添加
        #tau = 7
        #sims_row = torch.logsumexp(sims_row / tau, dim=-1)   # over R
        #sims_col = torch.logsumexp(sims_col /tau, dim=-1)  # over W

        #topK
        #sims_row, _ = sims_row.topk(self.topk, dim=-1)
        #sims_col, _ = sims_col.topk(self.topk, dim=-1)

        sims_row = sims_row.unsqueeze(-1)  # (Bi, Bt, R, 1)
        sims_col = sims_col.unsqueeze(-2)  # (Bi, Bt, 1, W)
        
        
        #sims_combined = sims_row + sims_col  # → (Bi, Bt, R, W)应该可以广播吧
        sims_combined = sims_row * sims_col  # → (Bi, Bt, R, W)
        
        #sims_combined = sims_col * sims_row  # B, B,R
        #sims_combined, _ = sims_combined.topk(self.topk, dim=-1)
        
        #print("sim_row,sims_col",sims_row.shape,sims_col.shape,flush=True) #B B 36
        #sims_row, _ = sims_row.topk(self.topk, dim=-1)
        #sims_col, _ = sims_col.topk(self.topk, dim=-1)
       
        
        #sims_combined = sims_combined.masked_fill(mask == 0, MASK)

        #sims_combined = torch.sqrt(torch.clamp(sims_row * sims_col, min=1e-8))

        #sims_combined = (sims_row ** self.alpha) * (sims_col ** (1 - self.alpha)) #二者越接近，开方后相乘就越接近

        #sims_combined, _ = sims_combined.topk(self.topk, dim=-1)

        #sims_2dmax,_ =sims.flatten(2,3).topk(self.topk, dim=(-1))

        #sims[sims==MASK] = 0
        #sims_mean = sims.sum(dim=(-2,-1))/(sims.shape[2]*sims.shape[3])


        #mask_count = mask.sum(dim=(-2,-1))
        #sims_mean = sims.sum(dim=(-2,-1))/mask_count
        #sims_mean=sims.mean(dim=(-2,-1))
        

        return sims_combined


# Visual Hard Assignment Coding
class VHACoding(nn.Module):
    def __init__(self):
        super().__init__()
        self.topk=20

    def forward(self, imgs, caps, img_lens, cap_lens):
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        sims = get_fgsims(imgs, caps)[:,:,:max_r,:max_w]
        mask = get_fgmask(img_lens,cap_lens)
        sims = sims.masked_fill(mask == 0, MASK)
        sims = sims.max(dim=-2)[0]
        #sims, _ = sims.topk(self.topk, dim=-1)  # B,B,topk
        return sims

# Texual Hard Assignment Coding
class THACoding(nn.Module):
    def __init__(self):
        super().__init__()
        self.topk=20

    def forward(self, imgs, caps, img_lens, cap_lens):
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        sims = get_fgsims(imgs, caps)[:,:,:max_r,:max_w]
        mask = get_fgmask(img_lens,cap_lens)
        sims = sims.masked_fill(mask == 0, MASK)
        sims = sims.max(dim=-1)[0]
        #sims, _ = sims.topk(self.topk, dim=-1)  # B,B,topk
        return sims

class VSACoding(nn.Module):
    def __init__(self,temperature = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, imgs, caps, img_lens, cap_lens, return_attn=False):
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs = imgs[:,:max_r,:]
        caps = caps[:,:max_w,:]
        sims = get_fgsims(imgs, caps)[:,:,:max_r,:max_w]
        mask = get_fgmask(img_lens,cap_lens)

        # calculate attention
        sims = sims / self.temperature

        sims = torch.softmax(sims.masked_fill(mask==0, -torch.inf),dim=-1) # Bi x Bt x K x L
        sims = sims.masked_fill(mask == 0, 0)
        sims = torch.matmul(sims,caps) # Bi x Bt x K x D
        sims = torch.mul(sims.permute(1,0,2,3),imgs).permute(1,0,2,3).sum(dim=-1) \
                    /(torch.norm(sims,p=2,dim=-1,keepdim=False)+EPS) # Bi x Bt x K

        mask = get_mask(img_lens).permute(0,2,1).repeat(1,cap_lens.size(0),1)
        sims = sims.masked_fill(mask==0, -1)
        return sims

class T2ICrossAttentionPool(nn.Module):
    def __init__(self,smooth=9):
        super().__init__()
        self.labmda = smooth

    def forward(self, imgs, caps, img_lens, cap_lens):
        return self.xattn_score_t2i(imgs,caps,cap_lens)

    def xattn_score_t2i(self, images, captions, cap_lens, return_attn=False):
        """
        Images: (n_image, n_regions, d) matrix of images
        Captions: (n_caption, max_n_word, d) matrix of captions
        CapLens: (n_caption) array of caption lengths
        """
        #print("images has nan:", torch.isnan(images).any().item())
        #print("captions has nan:", torch.isnan(captions).any().item())

        similarities = []
        attentions = []
        n_image = images.size(0)
        n_caption = captions.size(0)
        for i in range(n_caption):
            # Get the i-th text description
            n_word = int(cap_lens[i].item())
            cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
            # --> (n_image, n_word, d)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            """
                word(query): (n_image, n_word, d)
                image(context): (n_image, n_regions, d)
                weiContext: (n_image, n_word, d)
                attn: (n_image, n_region, n_word)
            """
            if return_attn:
                weiContext,attn = SCAN_attention(cap_i_expand, images,self.labmda)
                attentions.append(attn)
            else:
                weiContext,_ = SCAN_attention(cap_i_expand, images,self.labmda)
            cap_i_expand = cap_i_expand.contiguous()
            weiContext = weiContext.contiguous()
            # (n_image, n_word)
            col_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)

            #这里也可以有多种pooling方法，不止mean
            col_sim = col_sim.mean(dim=1, keepdim=True)
            #col_sim = torch.logsumexp(col_sim , dim=1, keepdim=True) 


            similarities.append(col_sim)

        # (n_image, n_caption)
        similarities = torch.cat(similarities, 1)
        if return_attn:return torch.cat(attentions, 0)
        else:return similarities

    
# max pooling
class MaxPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sims):
        assert len(sims.shape)==3
        sims = sims.max(dim=-1)[0]
        return sims

# mean pooling
class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sims):
        assert len(sims.shape)==3
        lens = (sims!=MASK).sum(dim=-1)
        sims[sims==MASK] = 0
        sims = sims.sum(dim=-1)/lens
        return sims

# sum pooling
class SumPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sims):
        assert len(sims.shape)==3
        sims[sims==MASK] = 0
        sims = sims.sum(dim=-1)
        return sims

# log-sum-exp pooling
class LSEPooling(nn.Module):
    def __init__(self,temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, sims):
        if len(sims.shape)==3 :
            sims[sims==MASK] = -torch.inf
            sims = torch.logsumexp(sims/self.temperature,dim=-1)
            return sims
        else :
            sims[sims==MASK] = -torch.inf
            sims = torch.logsumexp(sims/self.temperature,dim=(-2,-1))
            return sims

# softmax pooling
class SoftmaxPooling(nn.Module):
    def __init__(self,temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, sims):
        assert len(sims.shape)==3
        sims[sims==MASK] = -torch.inf
        weight = torch.softmax(sims/self.temperature,dim=-1)
        sims = (weight*sims).sum(dim=-1)
        return sims

def get_coding(coding_type, **args):
    alpha = args["opt"].alpha
    if coding_type=="VHACoding":
        return VHACoding()
    elif coding_type=="THACoding":
        return THACoding()
    elif coding_type=="VSACoding":
        return VSACoding(alpha)
        
    elif coding_type=="VTadd_HACoding":
        return VTadd_HACoding()
        
    elif coding_type=="HUNGARIANCoding":
        return HUNGARIANCoding()
    elif coding_type=="OptTransCoding":
        return OptTransCoding()
    else:
        raise ValueError("Unknown coding type: {}".format(coding_type))

def get_pooling(pooling_type, **args):
    belta = args["opt"].belta
    if pooling_type=="MaxPooling":
        return MaxPooling()
    elif pooling_type=="MeanPooling":
        return MeanPooling()
    elif pooling_type=="SumPooling":
        return SumPooling()
    elif pooling_type=="SoftmaxPooling":
        return SoftmaxPooling(belta)
    elif pooling_type=="LSEPooling":
        return LSEPooling(belta)
    else:
        raise ValueError("Unknown pooling type: {}".format(pooling_type))



class OP():
    def __init__(self, max_iter, M, N, n_cls, b):
        super(OP, self).__init__()
        self.max_iter= max_iter
        self.M=M
        self.N=N
        self.n_cls=n_cls
        self.b=b
        self.eps=0.1
        self.EPS=1e-6

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(self.max_iter):
            r0 = r
            r = u / (torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)+self.EPS)
            c = v / (torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)+self.EPS)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break

        if torch.isnan(r).any():
            print("r-nan")

        if torch.isnan(c).any():
            print("c-nan")

        T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K

        return T
    def get_OP_distence(self,sim):
        #sim = torch.einsum('mbd,ncd->mnbc', image_features, text_features).contiguous()
        #self.M=image.shape[0]
        self.b=sim.shape[0]
        self.n_cls=sim.shape[1]
        self.M=sim.shape[2]
        self.N=sim.shape[3]

        sim =sim .permute(2,3,0,1)
        sim = sim.view(self.M, self.N, self.b * self.n_cls) #B,B,LN
        sim = sim.permute(2, 0, 1)#LN,B,B
        wdist = 1.0 - sim
        xx = torch.zeros(self.b * self.n_cls, self.M, dtype=sim.dtype, device=sim.device).fill_(1. / self.M)
        yy = torch.zeros(self.b * self.n_cls, self.N, dtype=sim.dtype, device=sim.device).fill_(1. / self.N)

        with torch.no_grad():
            KK = torch.exp(-wdist / self.eps)
            KK = torch.clamp(KK, min=1e-6)
            T = self.Sinkhorn(KK, xx, yy)
        if torch.isnan(wdist).any():
            print("wdist-nan")
        if torch.isnan(T).any():
            return None

        sim_op = torch.sum(T * sim, dim=(1, 2))
        sim_op = sim_op.contiguous().view(self.b, self.n_cls)
        return sim_op


def mask_max_similarity(sim_matrix, text_indices, img_indices):
    """
    为相似度矩阵生成一个二值掩码，只保留最大相似对位置。

    Args:
        sim_matrix (Tensor): 相似度矩阵，形状为 (B*R, B*W)
        text_indices (Tensor): 文本索引排列，形状为 (N_perm, K)
        img_indices (Tensor): 图像索引排列，形状为 (N_perm, K)

    Returns:
        mask (Tensor): 二值掩码，形状为 (B*R, B*W)，在最大匹配对处为1
    """
    # (N_perm, K): 取出每种排列对应的相似度值
    selected_similarities = sim_matrix[text_indices, img_indices]  # shape: (N_perm, K)

    # (N_perm,): 找到每组排列中相似度最大的那一组的下标
    max_indices = selected_similarities.argmax(dim=1)

    # 只保留最大值那一组的匹配位置
    mask = torch.zeros_like(sim_matrix)
    mask[text_indices[max_indices], img_indices[max_indices]] = 1
    return mask

def create_index_permutations(num_embeddings, row_size, col_size):
    """
    为匹配任务生成所有可能的 index permutation。

    Args:
        num_embeddings (int): 每组图像或文本的数量 (R 或 W)
        row_size (int): 图像总数 = B * R
        col_size (int): 文本总数 = B * W

    Returns:
        text_indices, image_indices: 均为 shape (N_perm, num_embeddings)
    """
    perms = list(itertools.permutations(range(num_embeddings)))
    permutations = torch.tensor(perms, dtype=torch.long)  # shape: (N_perm, num_embeddings)

    # 重复排列以覆盖完整 batch
    row_indices = permutations.repeat(row_size // num_embeddings, 1)
    col_indices = permutations.repeat(col_size // num_embeddings, 1)

    return row_indices, col_indices

def maximal_pair_assignment_similarity(dist):
    """
    执行最大匹配相似度（Maximal Pair Assignment Similarity）计算。

    Args:
        img_embs (Tensor): 图像嵌入，shape: (B*R, D)
        txt_embs (Tensor): 文本嵌入，shape: (B*W, D)
        image_batch_size (int): 图像 batch 大小 (B)
        img_set_size (int): 每组图像数量 (R)
        text_batch_size (int): 文本 batch 大小 (B)
        txt_set_size (int): 每组文本数量 (W)

    Returns:
        max_similarity (Tensor): 最终得分，shape: (1,)
    """
    image_batch_size=dist.shape[0]
    img_set_size=dist.shape[2]
    text_batch_size=dist.shape[1]
    txt_set_size=dist.shape[3]
    # 1. 计算归一化的余弦相似度矩阵，(B*R, B*W)
    #dist = cosine_sim(l2norm(img_embs), l2norm(txt_embs))

    # 2. 获取图像和文本总数量
    row_size = image_batch_size * img_set_size
    col_size = text_batch_size * txt_set_size

    # 3. 创建所有 index permutation（假设 R=W）
    text_index_all, image_index_all = create_index_permutations(
        img_set_size, row_size, col_size
    )

    # 4. 获取最大匹配 mask（只保留最大匹配方案）
    mask = mask_max_similarity(dist.detach(), text_index_all, image_index_all)

    # 5. 取出最大匹配对的相似度值
    max_similarity = mask * dist  # shape: (B*R, B*W)
    

    # 6. 平均池化 + 指数变换（类似 softmax 归一化）
    max_similarity = avg_pool(torch.exp(max_similarity.unsqueeze(0)) - 1) * img_set_size
    return max_similarity

def compute_optimal_transport(self, M):
    """
    Computes the optimal transport matrix and Slinkhorn distance using the
    Sinkhorn-Knopp algorithm
    Inputs:
        - M : cost matrix (n x m)
        - r : vector of marginals (n, )
        - c : vector of marginals (m, )
        - lam : strength of the entropic regularization
        - epsilon : convergence parameter
    Outputs:
        - P : optimal transport matrix (n x m)
        - dist : Sinkhorn distance
    Reference:
        https://github.com/rythei/PyTorchOT/blob/master/ot_pytorch.py#L11-L59
    """
    lam = self.lam
    maxIter = self.maxIter
    epsilon = self.epsilon
    
    n, m = M.shape
    r = torch.ones((n,)) / n  # (n,)
    c = torch.ones((m,)) / m  # (m,)
    if self.iscuda:
        r, c = r.cuda(), c.cuda()

    P = torch.exp(- lam * M)    # (n, m)
    P /= P.sum()
    u = torch.zeros(n)
    if self.iscuda:
        u = u.cuda()
    
    num_iter = 0

    while torch.max(torch.abs(u - P.sum(1))) > epsilon and num_iter < maxIter:
        u = P.sum(1)
        P *= (r / u).reshape((-1, 1))
        P *= (c / P.sum(0)).reshape((-1, 1))

        num_iter += 1

    return P / P.sum(-1), torch.sum(P*M)