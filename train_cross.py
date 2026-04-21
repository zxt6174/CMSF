"""Training script"""
import os
import time
import numpy as np
import torch
#print(torch.cuda.device_count(),flush=True)
#print(os.environ,flush=True)
import torch.nn as nn
from transformers import BertTokenizer

from lib.datasets import image_caption
from lib.encoders import l2norm
from lib.vse_cross import CMSF
from lib.evaluation_cross import i2t, t2i, AverageMeter, LogCollector, encode_data, compute_sim

from torch.nn.parallel import DistributedDataParallel as DDP
import subprocess
import torch.distributed as dist

import logging
import tensorboard_logger as tb_logger
from datetime import datetime
import arguments
import random
import torch.backends.cudnn as cudnn

def setup_distributed(backend="nccl", port=None):
  
    num_gpus = torch.cuda.device_count()
    '''
    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    '''
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )

def logging_func(log_file, message):
    with open(log_file,'a') as f:
        f.write(message)
    f.close()


def init_seeds(seed=0, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def opt_process(opt):

    postfix = datetime.now().strftime("%m%d_%H%M%S")
    # ??????????????????
    '''
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    cudnn.benchmark = True
    cudnn.deterministic = False
    '''
    if opt.precomp_enc_type == 'basic':
        opt.logger_name = opt.logger_name + opt.data_name + "/butd_region_bert_"  + postfix + "/log"
        opt.model_name = opt.model_name + opt.data_name + "/butd_region_bert_" + postfix + "/checkpoints"
    elif opt.precomp_enc_type == 'backbone':
        opt.logger_name = opt.logger_name + opt.data_name + "/butd_grid_bert_"  + postfix + "/log"
        opt.model_name = opt.model_name + opt.data_name + "/butd_grid_bert_" + postfix + "/checkpoints"
    # ????????????????????????
        if not os.path.exists(opt.model_name):
            os.makedirs(opt.model_name)
        if not os.path.exists(opt.logger_name):
            os.makedirs(opt.logger_name)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():

    setup_distributed()
    # ????????????????????????
    parser = arguments.get_argument_parser()
    opt = parser.parse_args()
    opt_process(opt)

    rank = int(os.environ["RANK"])
    # ????????????????????????

    # ??????????????????
    rank = int(os.environ["RANK"])
    logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO if rank in [0] else logging.WARN)
    #logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    logger = logging.getLogger(__name__)
    logger.info(opt)

    # Load Tokenizer and Vocabulary
    bert_model_path = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)

    vocab = tokenizer.vocab
    opt.vocab_size = len(vocab)

    train_loader, val_loader = image_caption.get_loaders(
        opt.data_path, opt.data_name, tokenizer, opt.batch_size, opt.workers, opt)

    

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)

    
    model = CMSF(opt)
    model.to(device)
    # Convert BatchNorm to SyncBatchNorm
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    # ????????????????????????
    
    

    lr_schedules = [15,30]

    # optionally resume from a checkpoint
    start_epoch = 0
    if opt.resume:
        if os.path.isfile(opt.resume):
            logger.info("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another training
            model.Eiters = checkpoint['Eiters']
            logger.info("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                        .format(opt.resume, start_epoch, best_rsum))
            # validate(opt, val_loader, model)
            if opt.reset_start_epoch:
                start_epoch = 0
        else:
            logger.info("=> no checkpoint found at '{}'".format(opt.resume))

    # Train the Model
    best_rsum = 0
    for epoch in range(start_epoch, opt.num_epochs):
        logger.info(opt.logger_name)
        logger.info(opt.model_name)

        adjust_learning_rate(opt, model.optimizer, epoch, lr_schedules)

        # train for one epoch
        train(opt, train_loader, model, epoch, val_loader)
        

        print(f"讻时模型的参数? {count_parameters(model):,}")

        # evaluate on validation set
        rsum = validate(opt, val_loader, model, epoch)
        

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        # ??????????????????
        if not dist.is_initialized() or dist.get_rank() == 0:
            if not os.path.exists(opt.model_name):
                os.mkdir(opt.model_name)
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_rsum': best_rsum,
                'opt': opt,
                'Eiters': model.Eiters,
            }, is_best, filename='checkpoint_{}.pth'.format(epoch), prefix=opt.model_name + '/')


def train(opt, train_loader, model, epoch, val_loader):
    # average meters to record the training statistics
    logger = logging.getLogger(__name__)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()
        
    end = time.time()

    # ??????????????????
    train_loader.sampler.set_epoch(epoch)

    for i, train_data in enumerate(train_loader):
        # switch to train mode
        model.train_start()
        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model

        #images, img_lengths, captions, lengths, _, = train_data
        if opt.precomp_enc_type == 'basic':
            images, img_lengths, captions, lengths, ids = train_data # ????????????????????????
            #model.train_emb(images, captions, lengths, image_lengths=img_lengths)
        else:
            images, captions, lengths, ids = train_data
            img_lengths=lengths
            #model.train_emb(images, captions, lengths)
        model.train_emb(images, captions, ids,lengths, image_lengths=img_lengths)


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # logger.info log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(
                    epoch, i, len(train_loader.dataset) // train_loader.batch_size + 1, batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)


def validate(opt, val_loader, model, epoch):
    logger = logging.getLogger(__name__)
    
    model.val_start()
    with torch.no_grad():
        # compute the encoding for all the validation images and captions
        img_embs, cap_embs, img_lens, cap_lens = encode_data(model, val_loader, opt.log_step, logging.info)

    img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])

    start = time.time()
    # sims = compute_sim(img_embs, cap_embs)
    with torch.no_grad():
        sims = compute_sim(img_embs, cap_embs, img_lens, cap_lens, model)
    end = time.time()
    logger.info("calculate similarity time:".format(end - start))

    # caption retrieval
    npts = img_embs.shape[0]
    # (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, cap_lens, sims)
    (r1, r5, r10, medr, meanr) = i2t(npts, sims)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))
    # image retrieval
    # (r1i, r5i, r10i, medri, meanr) = t2i(img_embs, cap_embs, cap_lens, sims)
    (r1i, r5i, r10i, medri, meanr) = t2i(npts, sims)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanr))
    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i
    logger.info('Current rsum is {}'.format(currscore))

    # record metrics in tensorboard
    tb_logger.log_value('r1', r1, step=model.Eiters)
    tb_logger.log_value('r5', r5, step=model.Eiters)
    tb_logger.log_value('r10', r10, step=model.Eiters)
    tb_logger.log_value('medr', medr, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('rsum', currscore, step=model.Eiters)
    
    message = "Epoch: %d: Image to text: (%.1f, %.1f, %.1f) " % (epoch, r1, r5, r10)
    message += "Text to image: (%.1f, %.1f, %.1f) " % (r1i, r5i, r10i)
    message += "rsum: %.1f\n" % currscore

    log_file = os.path.join(opt.logger_name, "performance.log")
    logging_func(log_file, message)

    return currscore


def save_checkpoint(state, is_best, filename='checkpoint.pth', prefix=''):
    logger = logging.getLogger(__name__)
    tries = 50

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            #torch.save(state, prefix + filename)
            if is_best:
                torch.save(state, prefix + 'model_best.pth')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        logger.info('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error

def adjust_learning_rate(opt, optimizer, epoch, lr_schedules):
    logger = logging.getLogger(__name__)
    """Sets the learning rate to the initial LR
       decayed by 10 every opt.lr_update epochs"""
    if epoch in lr_schedules:
        logger.info('Current epoch num is {}, decrease all lr by 10'.format(epoch, ))
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            #
            new_lr = old_lr * 0.1
            param_group['lr'] = new_lr
            logger.info('new lr {}'.format(new_lr))
            print("new lr",new_lr)

if __name__ == '__main__':
    main()
