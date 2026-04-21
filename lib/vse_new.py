"""VSE model"""
from turtle import forward
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init

from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist

from lib.encoders import get_image_encoder, get_text_encoder

from lib.loss import UTO

import logging
import copy
import torch.nn.functional as F
logger = logging.getLogger(__name__)


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
                                         opt = opt)

        self.txt_enc = get_text_encoder(opt.embed_size, no_txtnorm=opt.no_txtnorm)

        self.cro_enc=get_cross_encoder()

        self.itm_head=nn.Linear(opt.embed_size,2)

        if opt.use_moco:
            self.K = opt.moco_M
            self.m = opt.moco_r
            self.v_encoder_k = copy.deepcopy(self.img_enc)
            self.t_encoder_k = copy.deepcopy(self.txt_enc)
            for param in self.v_encoder_k.parameters():
                param.requires_grad = False
            for param in self.t_encoder_k.parameters():
                param.requires_grad = False
            self.register_buffer("t_queue", torch.rand(opt.embed_size, self.K))
            self.t_queue = F.normalize(self.t_queue, dim=0)
            self.register_buffer("v_queue", torch.rand(opt.embed_size, self.K))
            self.v_queue = F.normalize(self.v_queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.cro_enc.cuda()
            self.itm_head.cuda()
            if opt.use_moco:
                self.v_encoder_k.cuda()
                self.t_encoder_k.cuda()
                self.t_queue = self.t_queue.cuda()
                self.v_queue = self.v_queue.cuda()
                self.queue_ptr = self.queue_ptr.cuda()
            

        self.hal_loss = UTO(opt=opt)
        
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())
        params += list(self.cro_enc.parameters())
        params += list(self.itm_head.parameters())

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

                self.optimizer = torch.optim.AdamW([
                    {'params': text_params_no_bert, 'lr': opt.learning_rate},
                    #是否冻结bert
                    {'params': bert_params, 'lr': opt.learning_rate * 0.0},#是否冻结BERT
                    {'params': self.img_enc.parameters(), 'lr': opt.learning_rate},
                    {'params': self.cro_enc.parameters(), 'lr': opt.learning_rate},
                    {'params': self.itm_head.parameters(), 'lr': opt.learning_rate}
                    ],
                    lr=opt.learning_rate, weight_decay=decay_factor)

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

        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(),self.cro_enc.state_dict(),self.itm_head.state_dict()]

        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0], strict=False)
        self.txt_enc.load_state_dict(state_dict[1], strict=False)
        self.cro_enc.load_state_dict(state_dict[2], strict=False)
        self.itm_head.load_state_dict(state_dict[3], strict=False)

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()
        self.cro_enc.train()
        self.itm_head.train()


    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()
        self.cro_enc.eval()
        self.itm_head.eval()


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

    def forward_cross_sim(self, images, captions, lengths, image_lengths=None,idx):
        ####这一段是新添的
        '''
        with torch.no_grad():
            self._momentum_update_key_encoder()
            image_embeds_m = self.v_encoder_k(images,image_lengths)
            #image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:,0,:]),dim=-1)  
            image_embeds_all = torch.cat([image_embeds_m.t(),self.v_queue.clone().detach()],dim=1)

            text_embeds_m = self.text_encoder_m(captions,lengths)    
            #text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1) 
            text_embeds_all = torch.cat([text_embeds_m.t(),self.t_queue.clone().detach()],dim=1)
        ####

        ######这一段是新加的
        sim_i2t = image_emb @ text_embeds_all 
        sim_t2i = cap_emb @ image_embeds_all 
        '''
        sim_i2t = image_emb @ cap_emb
        sim_t2i = cap_emb @ img_emb
        
        output_pos=self.cro_enc(img_emb,cap_emb)

        with torch.no_grad():
            bs = image.size(0)      
            weights_i2t = F.softmax(sim_i2t[:,:bs]+1e-4,dim=1)
            weights_t2i = F.softmax(sim_t2i[:,:bs]+1e-4,dim=1)

            mask = torch.eq(idx, idx.T)
            weights_i2t.masked_fill_(mask, 0)
            weights_t2i.masked_fill_(mask, 0) 

        # select a negative image for each text
        image_embeds_neg = []    
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(img_emb[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg,dim=0)   

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(cap_emb[neg_idx])
            #text_atts_neg.append(text.attention_mask[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg,dim=0)   
        #text_atts_neg = torch.stack(text_atts_neg,dim=0)      

        text_embeds_all_neg = torch.cat([cap_emb, text_embeds_neg],dim=0)     
        #text_atts_all = torch.cat([text.attention_mask, text_atts_neg],dim=0)     

        image_embeds_all_neg = torch.cat([image_embeds_neg,img_emb],dim=0)
        #image_atts_all = torch.cat([image_atts,image_atts],dim=0)

        output_neg=self.cro_enc(text_embeds_all_neg,image_embeds_all_neg)

        vl_embeddings = torch.cat([output_pos, output_neg],dim=0)

        vl_output = self.itm_head(vl_embeddings)          #（bs+2*bs, 2）二分类  

        itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                               dim=0).to(image.device)

        loss_itm = F.cross_entropy(vl_output, itm_labels) 
        ######

    def forward_emb(self, images, captions, lengths, image_lengths=None, is_train = False):
        """Compute the image and caption embeddings
        """

        # Set mini-batch dataset
        if self.opt.precomp_enc_type == 'basic':
            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()
                image_lengths = image_lengths.cuda()
            img_emb = self.img_enc(images, image_lengths)
            
        else:
            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()
            img_emb = self.img_enc(images)

        lengths = torch.Tensor(lengths).cuda()
        cap_emb = self.txt_enc(captions, lengths)



        if is_train and self.opt.use_moco:
            N = images.shape[0]
            '''
            with torch.no_grad():
                self._momentum_update_key_encoder()
                v_embed_k = self.v_encoder_k(images, image_lengths)
                t_embed_k = self.t_encoder_k(captions, lengths)
            '''
            loss_moco = self.hal_loss.moco_forward(img_emb, text_embeds_m, cap_emb, image_embeds_m, self.v_queue, self.t_queue)

            self._dequeue_and_enqueue(image_embeds_m, text_embeds_m)

            return image_embeds_all, text_embeds_all, loss_moco, loss_itm
        

        return image_embeds_all, text_embeds_all, loss_itm

    def forward_loss(self, img_emb, cap_emb):
        """Compute the loss given pairs of image and caption embeddings
        """

        loss = self.hal_loss(img_emb, cap_emb)*self.opt.loss_lamda

        self.logger.update('Le', loss.data.item(), img_emb.size(0))

        return loss

    def train_emb(self, images, captions, lengths, image_lengths=None, 
                  warmup_alpha=None ):
        """One training step given images and captions.
        """
        
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])
    
        # # compute the embeddings
        if self.opt.use_moco:

            img_emb, cap_emb, loss_moco, loss_itm = self.forward_emb(images, captions, lengths, image_lengths=image_lengths,
                                                is_train=True)
            self.logger.update('Le_moco', loss_moco.data.item(), img_emb.size(0))
        
            loss_itc = self.forward_loss(img_emb, cap_emb)

            loss = loss_itc + loss_moco + loss_itm
            
            self.logger.update('Loss', loss.data.item(), img_emb.size(0))
            if self.Eiters %100==0:
                if not dist.is_initialized() or dist.get_rank() == 0:#只让主进程print信息就行了
                    print("Loss-Total:",loss.data.item(),"Loss-ITC:",loss_itc.data.item(),"Loss-MoCo:",loss_moco.data.item(),"Loss-ITM:",loss_itm.data.item(),flush=True)
            
        else:
            img_emb, cap_emb, loss_itm = self.forward_emb(images, captions, lengths, image_lengths=image_lengths, is_train=True)
            loss_itc = self.forward_loss(img_emb, cap_emb)
            loss=loss_itc + loss_itm
            if self.Eiters % 100==0:

                if not dist.is_initialized() or dist.get_rank() == 0:
                    print("Loss-Total:", loss.data.item(), "Loss-ITC:", loss_itc.data.item(),"Loss-ITM:", loss_itm.data.item(),flush=True)

        #这里应该添加
        #image-encoder和txt-encoder应该输出四个值，用于输入到cross 的BLD，和用于计算粗排相似度的BD
        #loss=self.forward_cross_sims(image_emb,txt_emb,image_lemngth,txt_lemngth)




        # measure accuracy and record loss
        self.optimizer.zero_grad()

        if warmup_alpha is not None:
            loss = loss * warmup_alpha

        #在这里需要同步所有的loss？
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss /= dist.get_world_size() 

        # compute gradient and update
        loss.backward()

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
