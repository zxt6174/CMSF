"""VSE model"""
from turtle import forward
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init

from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist

from lib.encoders import get_image_encoder, get_text_encoder, SimsEncoder,get_cross_encoder

from lib.loss import UTO
from lib.loss_new import get_criterion

import logging
import copy
import torch.nn.functional as F
logger = logging.getLogger(__name__)

from lib.CrossEncoder import cross_sca

from lib.spike_coder import RepeatTextEncoder
class CMSF(nn.Module):
    """
        The standard VSE model
    """
    def __init__(self, opt):
        super().__init__()
        # Build Models
        self.grad_clip = opt.grad_clip

       

        self.img_enc = get_image_encoder(opt.data_name, opt.img_dim, opt.embed_size,
                                         precomp_enc_type=opt.precomp_enc_type,
                                         backbone_source=opt.backbone_source,
                                         backbone_path=opt.backbone_path,
                                         no_imgnorm=opt.no_imgnorm,
                                         opt = opt
                                        )

        self.txt_enc = get_text_encoder(opt.embed_size, no_txtnorm=opt.no_txtnorm)

        self.sim_enc = SimsEncoder("VHACoding", "LSEPooling", opt=opt)

        self.criterion1 = get_criterion(criterion="InfoNCELoss",opt=opt)
        self.criterion2 = get_criterion(criterion="ContrastiveLoss",opt=opt)

        
        #self.cross_encoder = get_cross_encoder(opt.embed_size, no_txtnorm=opt.no_txtnorm)
        


        


        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.sim_enc.cuda()
            
            #self.cross_encoder.cuda()


        self.hal_loss = UTO(opt=opt)
        
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())
        params += list(self.sim_enc.parameters())
        params += list(self.criterion1.parameters())
        params += list(self.criterion2.parameters())
       
        #params += list(self.cross_encoder.parameters())

        self.params = params

        self.opt = opt

        # Set up the lr for different parts of the VSE model
        decay_factor = 1e-4
        if opt.precomp_enc_type == 'basic':
            if self.opt.optim == 'adam':
                all_text_params = list(self.txt_enc.parameters())
                bert_params = list(self.txt_enc.bert.parameters())
                bert_params_ptr = [p.data_ptr() for p in bert_params]
                text_params_no_bert = list()
                for p in all_text_params:
                    if p.data_ptr() not in bert_params_ptr:
                        text_params_no_bert.append(p)
                
                self.optimizer = torch.optim.AdamW([#AdamW
                    {'params': text_params_no_bert, 'lr': opt.learning_rate},
                    #是否冻结bert
                    {'params': bert_params, 'lr': opt.learning_rate*0.1 },#是否冻结BERT
                    {'params': self.img_enc.parameters(), 'lr': opt.learning_rate},
                    {'params': self.sim_enc.parameters(), 'lr': opt.learning_rate}
                    #{'params': self.cross_encoder.parameters(), 'lr': opt.learning_rate*10}
                    ],
                    lr=opt.learning_rate, weight_decay=decay_factor)
                
                '''
                self.optimizer = torch.optim.Adagrad([
                    {'params': cross_bert_params, 'lr': opt.learning_rate},
                    {'params': cross_params_no_bert, 'lr': opt.learning_rate}
                    ],
                    lr=opt.learning_rate, weight_decay=decay_factor)
                '''
                '''
                self.optimizer = torch.optim.SGD([
                    {'params': text_params_no_bert, 'lr': opt.learning_rate},
                    #是否冻结bert
                    {'params': bert_params, 'lr': opt.learning_rate * 0.0},#是否冻结BERT
                    {'params': self.img_enc.parameters(), 'lr': opt.learning_rate}
                    ],
                    lr=opt.learning_rate, weight_decay=decay_factor, momentum=0.9)
                '''


            elif self.opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(self.params, lr=opt.learning_rate, momentum=0.9)
            else:
                raise ValueError('Invalid optim option {}'.format(self.opt.optim))

        logger.info('Use {} as the optimizer, with init lr {}'.format(self.opt.optim, opt.learning_rate))

        self.Eiters = 0
        self.data_parallel = False

    def set_max_violation(self, max_violation):
        if max_violation:
            self.criterion.max_violation_on()
        else:
            self.criterion.max_violation_off()

    def state_dict(self):

        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(),self.sim_enc.state_dict(),self.criterion1.state_dict(),self.criterion2.state_dict()]

        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0], strict=False)
        self.txt_enc.load_state_dict(state_dict[1], strict=False)
        self.sim_enc.load_state_dict(state_dict[2],strict=False)
        self.criterion1.load_state_dict(state_dict[3], strict=False)
        self.criterion2.load_state_dict(state_dict[4], strict=False)
        #self.cross_encoder.load_state_dict(state_dict[5],strict=False)

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()
        self.sim_enc.train()
        #self.cross_encoder.train()


    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()
        self.sim_enc.eval()
        #self.cross_encoder.eval()


    def freeze_backbone(self):
        if 'backbone' in self.opt.precomp_enc_type:
            if isinstance(self.img_enc, nn.DataParallel):
                self.img_enc.module.freeze_backbone()
            else:
                self.img_enc.freeze_backbone()

    def unfreeze_backbone(self, fixed_blocks):
        if 'backbone' in self.opt.precomp_enc_type:
            if isinstance(self.img_enc, nn.DataParallel):
                self.img_enc.module.unfreeze_backbone(fixed_blocks)
            else:
                self.img_enc.unfreeze_backbone(fixed_blocks)

    def forward_emb(self, images, captions, lengths, image_lengths=None, is_train = False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        if self.opt.precomp_enc_type == 'basic':
            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()
                image_lengths = image_lengths.cuda()
            img_emb,T_img_emb,F_img_emb = self.img_enc(images, image_lengths) #BLD,BTLD
            
        else:
            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()
                #新加的
                lengths = torch.Tensor(lengths).cuda()
                image_lengths=lengths
                image_lengths = image_lengths.cuda()
            img_emb,T_img_emb,F_img_emb = self.img_enc(images,image_lengths)

        lengths = torch.Tensor(lengths).cuda()
        cap_emb,T_cap_emb ,F_cap_emb= self.txt_enc(captions, lengths) #BLD,BTLD

        #如果我在这里加上cross-encoder呢
        #img_cro_emb,cap_cro_emb=self.cross_encoder(img_emb,cap_emb,F_img_emb,F_cap_emb)

       
        

        return img_emb, cap_emb,T_img_emb,T_cap_emb,F_img_emb,F_cap_emb#, img_cro_emb, cap_cro_emb

    def forward_sim(self, img_emb, cap_emb, img_len, cap_len):
        if torch.cuda.is_available():
            if isinstance(img_len,list):
                img_len = torch.Tensor(img_len).cuda()
            else:
                img_len = img_len.cuda()
            cap_len = torch.Tensor(cap_len).cuda()

        sims = self.sim_enc(img_emb, cap_emb, img_len, cap_len)
        return sims

    def forward_loss(self, img_emb, cap_emb):
        """Compute the loss given pairs of image and caption embeddings

        """
        #loss = self.hal_loss.info_nce_forward(img_pool, cap_pool)*self.opt.loss_lamda
        loss = self.hal_loss(img_emb, cap_emb)*self.opt.loss_lamda

        self.logger.update('Le', loss.data.item(), img_emb.size(0))

        return loss
    
    def KLContrastiveSimLoss(self, logits, softlabel, tau, softlabel_tau):
        """
        KL divergence loss
        make logits and softlabel have the same distribution
        logits to softlabel
        """
        # softmax for softlabel
        sim_targets = F.softmax(softlabel / softlabel_tau, dim=1)

        # log softmax
        logit_inputs = F.log_softmax(logits / tau, dim=1)

        
        # KL divergence
        loss = F.kl_div(logit_inputs, sim_targets, reduction='batchmean')

        return loss

    def train_emb(self, images, captions,ids, lengths, image_lengths=None, #这里增加了一个参数idx
                  warmup_alpha=None ):
        """One training step given images and captions.
        """
        
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        #lengths = torch.Tensor(lengths).cuda()
        ids = torch.Tensor(ids).cuda()
        mse_loss = torch.nn.MSELoss(reduction='mean')
        
        
        # # compute the embeddings
        if self.opt.use_moco:
            img_emb, cap_emb, loss_moco = self.forward_emb(images, captions, lengths, image_lengths=image_lengths,
                                                is_train=True)
            self.logger.update('Le_moco', loss_moco.data.item(), img_emb.size(0))
        
            loss_itc = self.forward_loss(img_emb, cap_emb)
            sims=self.forward_sim(img_emb, cap_emb, image_lengths, lengths)
            loss_cross=self.criterion1(sims)
            #loss_info =self.hal_loss.info_nce_forward(img_emb,cap_emb)

            loss = loss_itc + loss_moco + loss_cross
            
            self.logger.update('Loss', loss.data.item(), img_emb.size(0))
            if self.Eiters % 100==0:
                if not dist.is_initialized() or dist.get_rank() == 0:#只让主进程print信息就行了
                    print("Loss-total:",loss.data.item(),"Loss-ITC:",loss_itc.data.item(),"Loss-moco:",loss_moco.data.item(),"Loss-cross:",loss_cross.data.item(),flush=True)
            
        else:
            img_emb, cap_emb,T_img_emb,T_cap_emb,F_img_emb,F_cap_emb = self.forward_emb(images, captions, lengths, image_lengths=image_lengths, is_train=True)
            #loss_itc = self.forward_loss(img_emb, cap_emb)
            sims=self.forward_sim(img_emb, cap_emb, image_lengths, lengths)
            #loss_contra=self.criterion2(sims)
            loss_info=self.criterion1(sims)
            ##先把你搬到forward-emb里边
            #img_cro_emb,cap_cro_emb=self.cross_encoder(img_emb,cap_emb,F_img_emb,F_cap_emb)

            sims_f=self.forward_sim(F_img_emb,F_cap_emb,image_lengths,lengths)
            loss_f=self.criterion1(sims_f)

            '''
            sims_c=self.forward_sim(img_cro_emb, cap_cro_emb, image_lengths, lengths)
            loss_c=self.criterion1(sims_c)
            sims_i=self.forward_sim(img_emb, cap_cro_emb, image_lengths, lengths)
            loss_i=self.criterion1(sims_i)
            sims_t=self.forward_sim(img_cro_emb, cap_emb,image_lengths, lengths)
            loss_t=self.criterion1(sims_t)
            
            
            sims_ii=self.forward_sim(img_emb, img_cro_emb, image_lengths, image_lengths)
            loss_ii=self.criterion1(sims_ii)
            sims_tt=self.forward_sim(cap_emb, cap_cro_emb,lengths, lengths)
            loss_tt=self.criterion1(sims_tt)
            '''
        
            

            #loss_itc=self.forward_loss(img_emb,cap_emb)
            loss= loss_info + loss_f#+loss_c+loss_i+loss_t+loss_ii+loss_tt
            if self.Eiters % 100==0:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    #print("Loss-total:", loss.data.item(),"Loss-info:", loss_info.data.item(),"Loss-ITC:", loss_itc.data.item(),flush=True)
                    print("Loss-total:", loss.data.item(),
                    "Loss-info:", loss_info.data.item(),
                    #"Loss-c:", loss_c.data.item(),
                    "Loss-f:", loss_f.data.item(),flush=True)
                    #"Loss-ii:", loss_ii.data.item(),"Loss-tt:", loss_tt.data.item(),
                    #"Loss-i:", loss_i.data.item(),"Loss-t:", loss_t.data.item(),flush=True)

        # measure accuracy and record loss
        self.optimizer.zero_grad()

        if warmup_alpha is not None:
            loss = loss * warmup_alpha

        #在这里需要同步所有的loss？
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss /= dist.get_world_size() 

        # compute gradient and update
        loss.backward()
        #with torch.autograd.detect_anomaly():
        #    loss.backward()

        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)

        self.optimizer.step()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.img_enc.parameters(), self.v_encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        for param_q, param_k in zip(self.txt_enc.parameters(), self.t_encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

        
    @torch.no_grad()
    def _dequeue_and_enqueue(self, v_key, t_key):

        v_keys = concat_all_gather(v_key)
        t_keys = concat_all_gather(t_key)

        batch_size = v_keys.shape[0]

        ptr = int(self.queue_ptr)
        #print(batch_size,self.K,flush=True)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.v_queue[:, ptr : ptr + batch_size] = v_keys.T
        self.t_queue[:, ptr : ptr + batch_size] = t_keys.T

        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output     

def compute_sim(images, captions):
    similarities = np.matmul(images, np.matrix.transpose(captions))
    return similarities


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X
