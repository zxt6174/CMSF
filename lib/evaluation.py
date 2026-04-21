"""Evaluation"""
from __future__ import print_function
import logging
import time
import torch
import numpy as np
import sys
from collections import OrderedDict
from transformers import BertTokenizer

from lib.datasets import image_caption
from lib.encoders import l2norm
from lib.vse import CMSF
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

'''
from spikingjelly.clock_driven import surrogate, neuron, functional
from syops import get_model_complexity_info
'''
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=logger.info, backbone=False):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None

    max_n_word = 0
    for i, (images, image_lengths, captions, lengths, ids) in enumerate(data_loader):
        max_n_word = max(max_n_word, max(lengths))

    for i, data_i in enumerate(data_loader):

        # make sure val logger is used
        if not backbone:
            images, image_lengths, captions, lengths, ids = data_i
        else:
            images, captions, lengths, ids = data_i
        model.logger = val_logger

        # compute the embeddings
        if not backbone:
            img_emb, cap_emb,_,_,_,_ = model.forward_emb(images, captions, lengths, image_lengths=image_lengths)
        else:
            img_emb, cap_emb,_,_,_,_ = model.forward_emb(images, captions, lengths)
        #print("img-encode-data",img_emb.shape,flush=True)#[total-batch,dim]
        #print("txt-encode-data",cap_emb.shape,flush=True)#[total-batch,dim]
        

        
        if img_embs is None:
            if img_emb.dim() == 3:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))
            else:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))

            #cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), max_n_word, cap_emb.size(2))) #初始化为3D

            cap_lens = [0] * len(data_loader.dataset)
            img_lens = [0] * len(data_loader.dataset)
        # cache embeddings
        '''
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids, :] = cap_emb.data.cpu().numpy().copy()
        '''
        #换成cross fine的
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids,:max(lengths),:] = cap_emb.data.cpu().numpy().copy()


        for j, nid in enumerate(ids):
            cap_lens[nid] = lengths[j]
            img_lens[nid] = image_lengths[j]

        # measure accuracy and record loss
        # model.forward_loss(img_emb, cap_emb)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                .format(
                i, len(data_loader.dataset) // data_loader.batch_size + 1, batch_time=batch_time,
                e_log=str(model.logger)))
        del images, captions
    #return img_embs, cap_embs
    return img_embs, cap_embs,  img_lens, cap_lens



def eval_ensemble(results_paths, fold5=False):
    all_sims = []
    all_npts = []
    for sim_path in results_paths:
        results = np.load(sim_path, allow_pickle=True).tolist()
        npts = results['npts']
        sims = results['sims']
        all_npts.append(npts)
        all_sims.append(sims)
    all_npts = np.array(all_npts)
    all_sims = np.array(all_sims)
    assert np.all(all_npts == all_npts[0])
    npts = int(all_npts[0])
    sims = all_sims.mean(axis=0)

    if not fold5:
        r, rt = i2t(npts, sims, return_ranks=True)
        ri, rti = t2i(npts, sims, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        logger.info("rsum: %.1f" % rsum)
        logger.info("Average i2t Recall: %.1f" % ar)
        logger.info("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        logger.info("Average t2i Recall: %.1f" % ari)
        logger.info("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    else:
        npts = npts // 5
        results = []
        all_sims = sims.copy()
        for i in range(5):
            sims = all_sims[i * npts: (i + 1) * npts, i * npts * 5: (i + 1) * npts * 5]
            r, rt0 = i2t(npts, sims, return_ranks=True)
            logger.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(npts, sims, return_ranks=True)
            logger.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)

            if i == 0:
                rt, rti = rt0, rti0
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            logger.info("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]
        logger.info("-----------------------------------")
        logger.info("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        logger.info("rsum: %.1f" % (mean_metrics[12]))
        logger.info("Average i2t Recall: %.1f" % mean_metrics[10])
        logger.info("Image to text: %.1f %.1f %.1f %.1f %.1f" %
                    mean_metrics[:5])
        logger.info("Average t2i Recall: %.1f" % mean_metrics[11])
        logger.info("Text to image: %.1f %.1f %.1f %.1f %.1f" %
                    mean_metrics[5:10])


def evalrank(model_path, data_path=None, split='dev', fold5=False, save_path=None, cxc=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    opt.workers = 5
    # opt.use_moco = 0
    # opt.mu=70
    # opt.gama=0.6
    logger.info('Testing the best model')
    logger.info(opt)
    logger.info(checkpoint['epoch'])
    if not hasattr(opt, 'caption_loss'):
        opt.caption_loss = False

    # load vocabulary used by the model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab = tokenizer.vocab
    opt.vocab_size = len(vocab)

    opt.backbone_path = 'original_updown_backbone.pth'
    if data_path is not None:
        opt.data_path = data_path

    # construct model
    model = CMSF(opt)
    device_ids=[0]
    #model=nn.DataParallel(model,device_ids=device_ids)
    model = model.cuda()

    # model.make_data_parallel()
    # load model state
    model.load_state_dict(checkpoint['model'])
    model.val_start()
    
    logger.info('Loading dataset')
    data_loader = image_caption.get_test_loader(split, opt.data_name, tokenizer,
                                               opt.batch_size, opt.workers, opt)




    with torch.cuda.device(0):
        '''
        net=model.t_encoder_k
        ops, params = get_model_complexity_info(net, (3, 256, 256), data_loader, as_strings=True,
                                            print_per_layer_stat=True, verbose=True)
        #print('{:<30}  {:<8}'.format('Computational complexity ACs:', acs),flush=True)
        #print('{:<30}  {:<8}'.format('Computational complexity MACs:', macs),flush=True)
        print("ops",ops)
        print('{:<30}  {:<8}'.format('Number of parameters: ', params),flush=True)
        '''


    
    logger.info('Computing results...')
    with torch.no_grad():
        if opt.precomp_enc_type == 'basic':
            start = time.perf_counter()
            img_embs, cap_embs,img_lens,cap_lens = encode_data(model, data_loader)
            end = time.perf_counter()
            print("Encoding calculate time: {}".format(end - start))
            # np.save('wo_sge_img.npy', img_embs)
            # np.save('wo_sge_cap.npy', cap_embs)
            # print(type(cap_embs), type(img_embs))
        else:
            img_embs, cap_embs,img_lens,cap_lens = encode_data(model, data_loader, backbone=True)
    logger.info('Images: %d, Captions: %d' %
                (img_embs.shape[0] / 5, cap_embs.shape[0]))

    #plot_feature_distribution(img_embs,cap_embs,save_path="feature_distribution.png")


    if cxc:
        eval_cxc(img_embs, cap_embs, data_path)
    else:
        if not fold5:
            # no cross-validation, full evaluation
            img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])
            start = time.perf_counter()

            sims = compute_sim(img_embs, cap_embs,img_lens,cap_lens,model)

            npts = img_embs.shape[0]

            #1从sims里挑选top-k个样本
            #2将对应的id对应的特征，输入到cross-attention
            #3cross-attention的输出，输入到itm-head，得到对应的分类相似度
            #4经过两遍，得到图文和文图sims矩阵
            #5这两个矩阵被用来计算最后的recall值


            # if save_path is not None:
            #     np.save(save_path, {'npts': npts, 'sims': sims})
            #     logger.info('Save the similarity into {}'.format(save_path))

            end = time.perf_counter()
            logger.info("calculate similarity time: {}".format(end - start))

            r, rt = i2t(npts, sims, return_ranks=True)
            ri, rti = t2i(npts, sims, return_ranks=True)
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            logger.info("rsum: %.1f" % rsum)
            logger.info("Average i2t Recall: %.1f" % ar)
            logger.info("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
            logger.info("Average t2i Recall: %.1f" % ari)
            logger.info("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
        else:
            # 5fold cross-validation, only for MSCOCO
            results = []
            for i in range(5):
                img_embs_shard = img_embs[i * 5000:(i + 1) * 5000:5]
                cap_embs_shard = cap_embs[i * 5000:(i + 1) * 5000]
                start = time.time()

                sims = compute_sim(img_embs_shard, cap_embs_shard, img_lens,cap_lens,model)

                end = time.time()
                logger.info("calculate similarity time: {}".format(end - start))

                npts = img_embs_shard.shape[0]
                r, rt0 = i2t(npts, sims, return_ranks=True)
                logger.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
                ri, rti0 = t2i(npts, sims, return_ranks=True)
                logger.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)

                if i == 0:
                    rt, rti = rt0, rti0
                ar = (r[0] + r[1] + r[2]) / 3
                ari = (ri[0] + ri[1] + ri[2]) / 3
                rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
                logger.info("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
                results += [list(r) + list(ri) + [ar, ari, rsum]]

            logger.info("-----------------------------------")
            logger.info("Mean metrics: ")
            mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
            logger.info("rsum: %.1f" % (mean_metrics[12]))
            logger.info("Average i2t Recall: %.1f" % mean_metrics[10])
            logger.info("Image to text: %.1f %.1f %.1f %.1f %.1f" %
                        mean_metrics[:5])
            logger.info("Average t2i Recall: %.1f" % mean_metrics[11])
            logger.info("Text to image: %.1f %.1f %.1f %.1f %.1f" %
                        mean_metrics[5:10])
        
        # torch.save({'rt': rt, 'rti': rti}, 'ranks_f30k_top5_wo_sge.pth.tar')
'''
#对BD维度的数据计算相似度
def compute_sim(images, captions):
    similarities = np.matmul(images, np.matrix.transpose(captions))
    return similarities
'''
#对BLD维度的数据计算相似度，调用model.forward_sim
def compute_sim(img_embs, cap_embs, img_lens, cap_lens, model, shard_size=128):
    """
    Computer pairwise i2t image-caption distance with locality sharding
    """
    n_im_shard = (len(img_embs)-1)//shard_size + 1
    n_cap_shard = (len(cap_embs)-1)//shard_size + 1
    
    d = np.zeros((len(img_embs), len(cap_embs)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), len(img_embs))
        for j in range(n_cap_shard):
            #sys.stdout.write('\r>> shard_xattn_i2t batch (%d,%d)' % (i,j))
            cap_start, cap_end = shard_size*j, min(shard_size*(j+1), len(cap_embs))
            im = torch.from_numpy(img_embs[im_start:im_end]).float().cuda()
            s = torch.from_numpy(cap_embs[cap_start:cap_end]).float().cuda()
            il = img_lens[im_start:im_end]
            sl = cap_lens[cap_start:cap_end]
            sim = model.forward_sim(im, s, il, sl)
            d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
    #sys.stdout.write('\n')
    return d

def i2t(npts, sims, return_ranks=False, mode='coco'):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        if mode == 'coco':
            rank = 1e20
            for i in range(5 * index, 5 * index + 5, 1):
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank
            top1[index] = inds[0]
        else:
            rank = np.where(inds == index)[0][0]
            ranks[index] = rank
            top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(npts, sims, return_ranks=False, mode='coco'):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    # npts = images.shape[0]

    if mode == 'coco':
        ranks = np.zeros(5 * npts)
        top1 = np.zeros(5 * npts)
    else:
        ranks = np.zeros(npts)
        top1 = np.zeros(npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        if mode == 'coco':
            for i in range(5):
                inds = np.argsort(sims[5 * index + i])[::-1]
                ranks[5 * index + i] = np.where(inds == index)[0][0]
                top1[5 * index + i] = inds[0]
        else:
            inds = np.argsort(sims[index])[::-1]
            ranks[index] = np.where(inds == index)[0][0]
            top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


"""
    CxC related evaluation.
"""

def eval_cxc(images, captions, data_path):
    import os
    import json
    cxc_annot_base = os.path.join(data_path, 'cxc_annots')
    img_id_path = os.path.join(cxc_annot_base, 'testall_ids.txt')
    cap_id_path = os.path.join(cxc_annot_base, 'testall_capids.txt')

    images = images[::5, :]

    with open(img_id_path) as f:
        img_ids = f.readlines()
    with open(cap_id_path) as f:
        cap_ids = f.readlines()

    img_ids = [img_id.strip() for i, img_id in enumerate(img_ids) if i % 5 == 0]
    cap_ids = [cap_id.strip() for cap_id in cap_ids]

    with open(os.path.join(cxc_annot_base, 'cxc_it.json')) as f_it:
        cxc_it = json.load(f_it)
    with open(os.path.join(cxc_annot_base, 'cxc_i2i.json')) as f_i2i:
        cxc_i2i = json.load(f_i2i)
    with open(os.path.join(cxc_annot_base, 'cxc_t2t.json')) as f_t2t:
        cxc_t2t = json.load(f_t2t)

    sims = compute_sim(images, captions)
    t2i_recalls = cxc_inter(sims.T, img_ids, cap_ids, cxc_it['t2i'])
    i2t_recalls = cxc_inter(sims, cap_ids, img_ids, cxc_it['i2t'])
    logger.info('T2I R@1: {}, R@5: {}, R@10: {}'.format(*t2i_recalls))
    logger.info('I2T R@1: {}, R@5: {}, R@10: {}'.format(*i2t_recalls))

    i2i_recalls = cxc_intra(images, img_ids, cxc_i2i)
    t2t_recalls = cxc_intra(captions, cap_ids, cxc_t2t, text=True)
    logger.info('I2I R@1: {}, R@5: {}, R@10: {}'.format(*i2i_recalls))
    logger.info('T2T R@1: {}, R@5: {}, R@10: {}'.format(*t2t_recalls))


def cxc_inter(sims, data_ids, query_ids, annot):
    ranks = list()
    for idx, query_id in enumerate(query_ids):
        if query_id not in annot:
            raise ValueError('unexpected query id {}'.format(query_id))
        pos_data_ids = annot[query_id]
        pos_data_ids = [pos_data_id for pos_data_id in pos_data_ids if str(pos_data_id[0]) in data_ids]
        pos_data_indices = [data_ids.index(str(pos_data_id[0])) for pos_data_id in pos_data_ids]
        rank = 1e20
        inds = np.argsort(sims[idx])[::-1]
        for pos_data_idx in pos_data_indices:
            tmp = np.where(inds == pos_data_idx)[0][0]
            if tmp < rank:
                rank = tmp
        ranks.append(rank)
    ranks = np.array(ranks)
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    return (r1, r5, r10)


def cxc_intra(embs, data_ids, annot, text=False):
    pos_thresh = 3.0 if text else 2.5 # threshold for positive pairs according to the CxC paper

    sims = compute_sim(embs, embs)
    np.fill_diagonal(sims, 0)

    ranks = list()
    for idx, data_id in enumerate(data_ids):
        sim_items = annot[data_id]
        pos_items = [item for item in sim_items if item[1] >= pos_thresh]
        rank = 1e20
        inds = np.argsort(sims[idx])[::-1]
        if text:
            coco_pos = list(range(idx // 5 * 5, (idx // 5 + 1) * 5))
            coco_pos.remove(idx)
            pos_indices = coco_pos
            pos_indices.extend([data_ids.index(str(pos_item[0])) for pos_item in pos_items])
        else:
            pos_indices = [data_ids.index(str(pos_item[0])) for pos_item in pos_items]
            if len(pos_indices) == 0:  # skip it since there is positive example in the annotation
                continue
        for pos_idx in pos_indices:
            tmp = np.where(inds == pos_idx)[0][0]
            if tmp < rank:
                rank = tmp
        ranks.append(rank)

    ranks = np.array(ranks)
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    return (r1, r5, r10)

def plot_feature_distribution(image_features, text_features, save_path="feature_distribution.png"):
    """
    通过降维生成图像和文本特征的分布图，支持多种降维方法，并保存图片。

    Parameters:
    - image_features: 图像特征，形状为 (B, D)
    - text_features: 文本特征，形状为 (B, D)
    - save_path: 保存图片的路径，默认为 "feature_distribution.png"
    """
    # 合并图像和文本特征
    all_features = np.vstack((image_features, text_features))
    labels = np.array([0] * image_features.shape[0] + [1] * text_features.shape[0])  # 图像为 0，文本为 1
    
    # 降维方法列表
    methods = {
        "PCA": PCA(n_components=2),
        "t-SNE": TSNE(n_components=2, perplexity=20, random_state=42),
        "UMAP": umap.UMAP(n_components=2, random_state=42)
    }

    # 遍历每种降维方法
    for method_name, method in methods.items():
        # 进行降维
        reduced_features = method.fit_transform(all_features)

        # 创建图形进行显示
        plt.figure(figsize=(10, 10)) 

        # 可视化，图像特征用红色圆点，文本特征用绿色叉号
        plt.scatter(
            reduced_features[labels == 0, 0], reduced_features[labels == 0, 1],
            c='red', marker='o', label='Image', alpha=0.7, edgecolors='k', s=50
        )
        plt.scatter(
            reduced_features[labels == 1, 0], reduced_features[labels == 1, 1],
            c='green', marker='x', label='Text', alpha=0.7, edgecolors='k', s=50
        )

        # 设置图形标题和标签
        plt.title(f"Feature Distribution using {method_name}",fontsize=20)
        plt.xlabel("Dimension 1",fontsize=18)
        plt.ylabel("Dimension 2",fontsize=18)
        plt.legend(loc="upper right",fontsize=20)
        plt.grid(True)

        # 保存每个降维方法的图像
        method_save_path = f"{method_name}_feature_distribution_USER-big.png"
        plt.savefig(method_save_path, dpi=300)
        plt.close()  # 关闭当前图形

    # 可以选择输出一个汇总的图像，或者仅保存单独的图像
    print("Images saved for each dimensionality reduction method.")
