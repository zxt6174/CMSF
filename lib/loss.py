import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_sim(images, captions):
    similarities = images.mm(captions.t())
    return similarities

def forward_orig(out_1, out_2):
    """

    p - positive pair
    n - negative pair
    sim - cosine similarity
    e - Euler's number

    ix - value x of input feature vector i
    tx - value x of input feature vector t

                    Similarities matrix: exp(sim(i, y))
                         +--+--+--+--+--+--+--+
                         |  |i1|i2|i3|t1|t2|t3|
     Modality            +--+--+--+--+--+--+--+
     Features            |i1|e |n |n |p |n |n |
    +--+  +--+           +--+--+--+--+--+--+--+
    |i1|  |t1|           |i2|n |e |n |n |p |n |
    +--+  +--+           +--+--+--+--+--+--+--+
    |i2|  |t2|  ------>  |i3|n |n |e |n |n |p |
    +--+  +--+           +--+--+--+--+--+--+--+
    |i3|  |t3|           |t1|p |n |n |e |n |n |
    +--+  +--+           +--+--+--+--+--+--+--+
                         |t2|n |p |n |n |e |n |
                         +--+--+--+--+--+--+--+
                         |t3|n |n |p |n |n |e |
                         +--+--+--+--+--+--+--+

    :param out_1: input feature vector i
    :param out_2: input feature vector t
    :return: NTXent loss
    """
    #out_1 = F.normalize(out_1)
    #out_2 = F.normalize(out_2)
    temperature = 1
    eps = 1e-6

    out = torch.cat([out_1, out_2], dim=0)

    # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
    # neg: [2 * batch_size]
    cov = torch.mm(out, out.t().contiguous())
    sim = torch.exp(cov / temperature)
    neg = sim.sum(dim=-1)

    # from each row, subtract e^1 to remove similarity measure for x1.x1
    row_sub = torch.Tensor(neg.shape).fill_(math.e).to(neg.device)
    neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

    # Positive similarity, pos becomes [2 * batch_size]
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    loss = -torch.log(pos / (neg + eps)).sum()
    return loss



class UTO(nn.Module):
    def __init__(self, opt):
        super(UTO, self).__init__()
        self.opt = opt
        self.l_alpha = opt.mu
        self.l_ep = opt.gama
        self.temperature = 0.1
        self.eps = 1e-5

    def forward(self, scores ):
        
        #bsize = im.size()[0]
        #scores = get_sim(im, s)
        bsize = scores.size(0)

    
        tmp  = torch.eye(bsize).cuda()   
        s_diag = tmp * scores        
        scores_ = scores - s_diag      
        S_ = torch.exp(self.l_alpha * (scores_ - self.l_ep))
    
        loss_diag_1 = - torch.log(1 + F.relu(s_diag.sum(0)))

        loss = torch.sum( torch.log(1 + S_.sum(0)) / self.l_alpha + torch.log(1 + S_.sum(1)) / self.l_alpha + loss_diag_1) / bsize
        return loss

    def info_nce_forward(self,query,key):
        # Compute similarity matrix
        logits = torch.matmul(query, key.T)  # [N, N]
        logits /= self.temperature
        # Labels: positive keys are the diagonal elements
        labels = torch.arange(logits.shape[0]).to(query.device)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss
        '''
        ####
        
        bsize = im.size(0)

        # 同模态相似度矩阵
        scores_im_im = get_sim(im, im)  # [batch_size, batch_size]
        scores_s_s = get_sim(s, s)  # [batch_size, batch_size]

        # 跨模态相似度矩阵
        scores_im_s = get_sim(im, s)  # [batch_size, batch_size]
        scores_s_im = get_sim(s, im)  # [batch_size, batch_size]

        # 构造对角掩码，用于移除自相似性
        tmp = torch.eye(bsize).cuda()
        s_diag_im_s = tmp * scores_im_s
        s_diag_s_im = tmp * scores_s_im

        # 去掉对角元素
        scores_im_s_ = scores_im_im - s_diag_im_s
        scores_s_im_ = scores_s_s - s_diag_s_im

        # 计算跨模态的相似性损失
        #目标是较少非配对的相似性，增大配对的相似性
        S_im_s = torch.exp(self.l_alpha * (scores_im_s_ - self.l_ep))
        S_s_im = torch.exp(self.l_alpha * (scores_s_im_ - self.l_ep))

        loss_diag_im_s = -torch.log(1 + F.relu(s_diag_im_s.sum(0)))
        loss_diag_s_im = -torch.log(1 + F.relu(s_diag_s_im.sum(0)))

        loss_im_s = torch.sum(torch.log(1 + S_im_s.sum(0)) / self.l_alpha +
                               torch.log(1 + S_im_s.sum(1)) / self.l_alpha +
                               loss_diag_im_s) / bsize

        loss_s_im = torch.sum(torch.log(1 + S_s_im.sum(0)) / self.l_alpha +
                             torch.log(1 + S_s_im.sum(1)) / self.l_alpha +
                             loss_diag_s_im) / bsize

        # 计算同模态的相似性损失
        #目标是减少同模态不同样本之间的相似性
        s_diag_im_im = tmp * scores_im_im
        s_diag_s_s = tmp * scores_s_s

        # 去掉对角元素
        scores_im_im_ = scores_im_im - s_diag_im_im
        scores_s_s_ = scores_s_s - s_diag_s_s

        S_im_im = torch.exp(self.l_alpha * (scores_im_im_ - self.l_ep))
        S_s_s = torch.exp(self.l_alpha * (scores_s_s_ - self.l_ep))


        loss_im_im = torch.sum(torch.log(1 + S_im_im.sum(0)) / self.l_alpha +
                              torch.log(1 + S_im_im.sum(1)) / self.l_alpha) / bsize

        loss_s_s = torch.sum(torch.log(1 + S_s_s.sum(0)) / self.l_alpha +
                              torch.log(1 + S_s_s.sum(1)) / self.l_alpha) / bsize

        # 综合同模态和跨模态的损失
        total_loss = (loss_im_im + loss_s_s + loss_im_s + loss_s_im) / 4

        return total_loss
        '''
    
    def intra_set_divergence_loss(self,set_emb, margin=0.5, scale=1.0):
        """
        Intra-Set Divergence Loss
        Encourages embeddings within the same set to be dissimilar (diverse).
        
        Args:
            set_emb: Tensor of shape [bs, K, D], batch of sets of embeddings
            margin: float, margin controlling the penalty threshold
            scale: float, scaling factor for the exponential penalty

        Returns:
            loss: scalar, normalized divergence loss over the batch
        """

        bs, K, D = set_emb.shape  # Batch size, number of embeddings per set, embedding dim

        # Compute pairwise similarity matrix within each set: [bs, K, K]
        #sim_matrix = torch.bmm(set_emb, set_emb.transpose(1, 2))  # dot product
        sim_matrix = torch.matmul(set_emb.flatten(-2,-1), set_emb.flatten(-2, -1).t())  # B B

        # Create a mask to extract upper-triangular (excluding diagonal) entries
        #mask = torch.triu(torch.ones(K, K, device=set_emb.device), diagonal=1).bool()  # [K, K]
        #mask = mask.unsqueeze(0)  # [1, K, K] -> for batch broadcasting
        mask = torch.triu(torch.ones(bs, bs, device=set_emb.device), diagonal=1).bool()  # [B, B]

        # Apply the mask to extract unique similarity pairs (shape: [bs * (K*(K-1)/2)])
        sim_values = sim_matrix.masked_select(mask)

        # Compute smooth penalty for similarity values above the margin
        loss = torch.exp((sim_values - margin) * scale)

        # Normalize by total number of valid pairs in the batch
        num_pairs = K * (K - 1) / 2
        loss = loss.sum() / (bs * num_pairs)

        return loss


    def moco_forward(self, v_q, t_k, t_q, v_k, v_queue, t_queue):
        # v positive logits: Nx1
        v_pos = torch.einsum("nc,nc->n", [v_q, t_k]).unsqueeze(-1)
        # v negative logits: NxK
        t_queue = t_queue.clone().detach()    
        v_neg = torch.einsum("nc,ck->nk", [v_q, t_queue])

        # # t positive logits: Nx1
        t_pos = torch.einsum("nc,nc->n", [t_q, v_k]).unsqueeze(-1)
        # t negative logits: NxK
        v_queue = v_queue.clone().detach()
        t_neg = torch.einsum("nc,ck->nk", [t_q, v_queue])

        v_pos_diag = torch.diag_embed(v_pos.squeeze(-1))
        v_bsize = v_pos_diag.size()[0]
        v_loss_diag = - torch.log(1 + F.relu(v_pos_diag.sum(0)))
        v_S_ = torch.exp((v_neg - self.l_ep) * self.l_alpha)
        v_S_T = v_S_.T
        v_loss = torch.sum(torch.log(1 + v_S_T.sum(0)) / self.l_alpha + v_loss_diag) / v_bsize

        t_pos_diag = torch.diag_embed(t_pos.squeeze(-1))
        t_bsize = t_pos_diag.size()[0]
        t_loss_diag = - torch.log(1 + F.relu(t_pos_diag.sum(0)))
        t_S_ = torch.exp((t_neg - self.l_ep) * self.l_alpha)
        t_S_T = t_S_.T
        t_loss = torch.sum(torch.log(1 + t_S_T.sum(0)) / self.l_alpha + t_loss_diag) / t_bsize

        return v_loss + t_loss

