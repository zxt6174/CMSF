import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss (max-margin based)
    """

    def __init__(self, opt=None, margin=0.2, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        if opt is not None:
            self.opt = opt
            self.margin = opt.margin
            self.max_violation = opt.max_violation
        else:
            self.margin = margin
            self.max_violation = max_violation

    def max_violation_on(self):
        self.max_violation = True
        print('Use VSE++ objective.')

    def max_violation_off(self):
        self.max_violation = False
        print('Use VSE0 objective.')

    def forward(self, sims):
        # compute image-sentence score matrix
        # sims = get_sim(im, s)
        diagonal = sims.diag().view(sims.size(0), 1)
        d1 = diagonal.expand_as(sims)
        d2 = diagonal.t().expand_as(sims)

        # compare every diagonal score to sims in its column
        # caption retrieval
        cost_s = (self.margin + sims - d1).clamp(min=0)
        # compare every diagonal score to sims in its row
        # image retrieval
        cost_im = (self.margin + sims - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(sims.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()

class InfoNCELoss(nn.Module):
    """
    Compute InfoNCELoss loss
    """
    def __init__(self, temperature=0.01, margin=0):
        super(InfoNCELoss, self).__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, sims):
        ## cost of image retrieval
        img_ret = sims-sims.diag().expand_as(sims).t()+self.margin
        img_ret[torch.eye(sims.size(0))>.5] = 0
        cost_im = torch.log(torch.sum(torch.exp(img_ret/self.temperature),dim=1))

        ## cost of text retrieval
        txt_ret = sims-sims.diag().expand_as(sims)+self.margin
        txt_ret[torch.eye(sims.size(0))>.5] = 0
        cost_s = torch.log(torch.sum(torch.exp(txt_ret/self.temperature),dim=0))

        return cost_s.mean() + cost_im.mean()

    def max_violation_on(self):
        return 

    def max_violation_off(self):
        return

def get_criterion(criterion,opt,**args):
    if criterion=="ContrastiveLoss":
        return ContrastiveLoss(margin=opt.margin)
    elif criterion=="InfoNCELoss":
        return InfoNCELoss(temperature=opt.temperature,
                            margin=opt.margin)
    elif criterion =="TripletLoss":
        return TripletLoss(opt=opt)
    else:
        raise ValueError("Unknown criterion type: {}".format(criterion))

def pos_neg_mask(labels):

    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) 
    neg_mask = labels.unsqueeze(0) != labels.unsqueeze(1)

    return pos_mask, neg_mask


def pos_neg_mask_xy(labels_col, labels_row):

    pos_mask = (labels_row.unsqueeze(0) == labels_col.unsqueeze(1)) 
    neg_mask = (labels_row.unsqueeze(0) != labels_col.unsqueeze(1))

    return pos_mask, neg_mask

class TripletLoss(nn.Module):

    def __init__(self, opt=None, margin=0.5, ):
        super().__init__()

        self.opt = opt
        self.margin = margin
        
        self.cut_off = 0.1
        self.d = 1024

        #if self.opt.data_name == 'coco':
        #self.nonzero_loss_cutoff = 1.9         
        #else:
        self.nonzero_loss_cutoff = 1.9
        
    def forward(self, sims, img_ids):

        sim_mat = sims
        img_ids = img_ids.cuda()
        
        '''
        if im.size(0) == s.size(0):
            pos_mask, neg_mask = pos_neg_mask(img_ids)
        else:
            pos_mask, neg_mask = pos_neg_mask_xy(torch.unique(img_ids), img_ids)
        '''
        pos_mask, neg_mask = pos_neg_mask(img_ids)

        loss_im = self.loss_forward(sim_mat, pos_mask, neg_mask)
        loss_s = self.loss_forward(sim_mat.t(), pos_mask.t(), neg_mask.t())

        loss = loss_im + loss_s

        return loss        

    def loss_forward(self, sim_mat, pos_mask, neg_mask): 

        pos_pair_idx = pos_mask.nonzero(as_tuple=False)
        anchor_idx = pos_pair_idx[:, 0]
        pos_idx = pos_pair_idx[:, 1]

        dist = (2 - 2 * sim_mat).sqrt()
        dist = dist.clamp(min=self.cut_off)

        log_weight = (2.0 - self.d) * dist.log() - ((self.d - 3.0) / 2.0) * (1.0 - 0.25 * (dist * dist)).log()
        inf_or_nan = torch.isinf(log_weight) | torch.isnan(log_weight)

        log_weight = log_weight * neg_mask  
        log_weight[inf_or_nan] = 0.      

        weight = (log_weight - log_weight.max(dim=1, keepdim=True)[0]).exp()
      
        weight = weight * (neg_mask * (dist < self.nonzero_loss_cutoff)).float() 
    
        weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-20)
       
        weight = weight[anchor_idx]
       

        # maybe not exist
        try:
            neg_idx = torch.multinomial(weight, 1).squeeze(1)   
        except Exception:
            print("是这里问题",flush=True)
            return torch.zeros([], requires_grad=True, device=sim_mat.device) 


        s_ap = sim_mat[anchor_idx, pos_idx]
        s_an = sim_mat[anchor_idx, neg_idx]  

        loss = F.relu(self.margin + s_an - s_ap) 
        loss = loss.sum()


        return loss