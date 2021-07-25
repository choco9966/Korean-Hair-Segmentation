import os
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast
from torch.optim import Adam
from torch import nn, Tensor
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from typing import Optional
from torch.utils.data import Dataset, DataLoader
from warmup_scheduler import GradualWarmupScheduler
from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler

from modules.utils import load_yaml, init_logger
from modules.scheduler import *
from modules.models import Encoder
from modules.dataset import *
from modules.transform import get_transforms_train, get_transforms_preprocessing

CONFIG_PATH = './config/config.yaml'
config = load_yaml(CONFIG_PATH)

# LOG
VERSION = config['LOG']['version']

# DATALOADER
NUM_WORKERS = config['DATALOADER']['num_workers']

# PREPROCESSING
PRE_GRADIENT_ACCUMULATION_STEPS = config['PREPROCESSING']['gradient_accumulation_steps']
PRE_MAX_GRAD_NORM = config['PREPROCESSING']['max_grad_norm']
PRE_APEX = config['PREPROCESSING']['apex']
PRE_PRINT_FREQ = config['PREPROCESSING']['print_freq']
PRE_SCHEDULER = config['PREPROCESSING']['scheduler']
PRE_PATIENCE = config['PREPROCESSING']['patience']
PRE_EPS = config['PREPROCESSING']['eps']
PRE_T_MAX  = config['PREPROCESSING']['T_max']
PRE_MIN_LR = config['PREPROCESSING']['min_lr']
PRE_T_0 = config['PREPROCESSING']['T_0']
PRE_COSINE_EPO = config['PREPROCESSING']['cosine_epo']
PRE_WARMUP_EPO = config['PREPROCESSING']['warmup_epo']
PRE_FREEZE_EPO = config['PREPROCESSING']['freeze_epo']
PRE_WARMUP_FACTOR = config['PREPROCESSING']['warmup_factor']
PRE_ENCODER_TYPE = config['PREPROCESSING']['encoder_type']
PRE_DECODER_TYPE = config['PREPROCESSING']['decoder_type']
PRE_WEIGHT_DECAY = config['PREPROCESSING']['weight_decay']
PRE_ENCODER_LR = config['PREPROCESSING']['encoder_lr']
PRE_BATCH_SIZE = config['PREPROCESSING']['batch_size']
PRE_EPOCHS = PRE_COSINE_EPO + PRE_WARMUP_EPO + PRE_FREEZE_EPO 

# TRAIN
GRADIENT_ACCUMULATION_STEPS = config['TRAIN']['gradient_accumulation_steps']
MAX_GRAD_NORM = config['TRAIN']['max_grad_norm']
APEX = config['TRAIN']['apex']
PRINT_FREQ = config['TRAIN']['print_freq']
SCHEDULER = config['TRAIN']['scheduler']
PATIENCE = config['TRAIN']['patience']
EPS = config['TRAIN']['eps']
T_MAX  = config['TRAIN']['T_max']
MIN_LR = config['TRAIN']['min_lr']
T_0 = config['TRAIN']['T_0']
COSINE_EPO = config['TRAIN']['cosine_epo']
WARMUP_EPO = config['TRAIN']['warmup_epo']
FREEZE_EPO = config['TRAIN']['freeze_epo']
WARMUP_FACTOR = config['TRAIN']['warmup_factor']
ENCODER_TYPE = config['TRAIN']['encoder_type']
DECODER_TYPE = config['TRAIN']['decoder_type']
WEIGHT_DECAY = config['TRAIN']['weight_decay']
ENCODER_LR = config['TRAIN']['encoder_lr']
BATCH_SIZE = config['TRAIN']['batch_size']
SELF_CUTMIX = config['TRAIN']['self_cutmix']
CUTMIX_THRESHOLD = config['TRAIN']['cutmix_threshold']
COLORING = config['TRAIN']['coloring']
COLORING_THRESHOLD = config['TRAIN']['coloring_threshold']
LOSS_SMOOTH_FACTOR = config['TRAIN']['loss_smooth_factor']
PROSPECTIVE_FILTERING = config['TRAIN']['prospective_filtering']
PRETRAINED = config['TRAIN']['pretrained']
BREAK_EPOCH = config['TRAIN']['break_epoch']

EPOCHS = COSINE_EPO + WARMUP_EPO + FREEZE_EPO 



os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = 'model/preprocessing'
OUTPUT_TRAIN_DIR = 'output/runs'


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

def rand_bbox(size, rat):
    W = size[2]
    H = size[3]
    min_rat, max_rat = rat
    cut_w = np.random.randint(W*min_rat, W*max_rat)
    cut_h = np.random.randint(H*min_rat, H*max_rat)

    bbox_list = []
    for i in range(2):
        cx = np.random.randint(0.1*W,0.9*W)
        cy = np.random.randint(0.1*H,0.9*H)
        bbx1 = cx - cut_w // 2
        bby1 = cy - cut_h // 2
        bbx2 = cx + cut_w // 2
        bby2 = cy + cut_h // 2
        bbox_list.append([bbx1, bby1, bbx2, bby2])

        if bbx1 < 0 or bbx2 >= W:
            return []
        if bby1 < 0 or bby2 >= H:
            return []

    return bbox_list

def get_iou(preds, targets):
    preds_mask = preds>0
    targets_mask = targets>0
    
    intersection = torch.sum(preds_mask*targets_mask, axis=(1,2,3))
    union = torch.sum(preds_mask, axis=(1,2,3)) + torch.sum(targets_mask, axis=(1,2,3)) - intersection
    return intersection/union




__all__ = ["SoftBCEWithLogitsLoss"]


class SoftBCEWithLogitsLoss(nn.Module):

    __constants__ = ["weight", "pos_weight", "reduction", "ignore_index", "smooth_factor"]

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        ignore_index: Optional[int] = -100,
        reduction: str = "mean",
        smooth_factor: Optional[float] = None,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_factor = smooth_factor
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        if self.smooth_factor is not None:
            soft_targets = (1 - y_true) * self.smooth_factor + y_true * (1 - self.smooth_factor)
        else:
            soft_targets = y_true

        loss = F.binary_cross_entropy_with_logits(
            y_pred, soft_targets, self.weight, pos_weight=self.pos_weight, reduction="none"
        )

        if self.ignore_index is not None:
            not_ignored_mask = y_true != self.ignore_index
            loss *= not_ignored_mask.type_as(loss)

        if self.reduction == "mean":
            loss = loss.mean()

        if self.reduction == "sum":
            loss = loss.sum()

        return loss

def train_pre_fn(train_loader, encoder1, encoder2, criterion1, criterion2, 
             optimizer1, optimizer2, epoch,
             scheduler1, scheduler2, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    dice_coeffs = AverageMeter()
    # switch to train mode
    encoder1.train()
    encoder2.train()

    scaler = torch.cuda.amp.GradScaler()

    start = end = time.time()
    global_step = 0
    for step, (images, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        targets = targets.float().to(device)
        batch_size = images.size(0)

        # =========================
        # zero_grad()
        # =========================
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        if PRE_APEX:
            with autocast():
                y_preds = encoder(images)
                loss = criterion(y_preds, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            y_preds1 = encoder1(images)
            y_preds2 = encoder2(images)

            loss1 = criterion1(y_preds1, targets)
            loss2 = criterion1(y_preds2, targets)

            ind_1_sorted = np.argsort(np.sum(loss1.data.reshape(len(y_preds1), -1).detach().cpu().numpy(), axis=1))
            loss_1_sorted = loss1[ind_1_sorted] 

            ind_2_sorted = np.argsort(np.sum(loss2.data.reshape(len(y_preds1), -1).detach().cpu().numpy(), axis=1))
            loss_2_sorted = loss2[ind_2_sorted]  

            forget_rate = 0.1
            remember_rate = 1 - forget_rate
            num_remember = int(remember_rate * len(loss_1_sorted))

            ind_1_update=ind_1_sorted[:num_remember]
            ind_2_update=ind_2_sorted[:num_remember]

            # exchange
            loss_1_update = criterion2(y_preds1[ind_2_update], targets[ind_2_update])
            loss_2_update = criterion2(y_preds2[ind_1_update], targets[ind_1_update])

            loss_1_update.backward()
            loss_2_update.backward()

        # record loss
        losses.update((loss_1_update.item() + loss_2_update.item())/2, batch_size)
        if PRE_GRADIENT_ACCUMULATION_STEPS > 1:
            loss = loss / PRE_GRADIENT_ACCUMULATION_STEPS

        #loss.backward()
        encoder_grad_norm1 = torch.nn.utils.clip_grad_norm_(encoder1.parameters(), PRE_MAX_GRAD_NORM)
        encoder_grad_norm2 = torch.nn.utils.clip_grad_norm_(encoder2.parameters(), PRE_MAX_GRAD_NORM)
        if (step + 1) % PRE_GRADIENT_ACCUMULATION_STEPS == 0:
            if PRE_APEX:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer1.step()
                optimizer2.step()
            global_step += 1

        # record dice_coeff
        dice_coeff = get_dice_coeff((y_preds1+y_preds2)/2, targets)
        dice_coeffs.update(dice_coeff, batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % PRE_PRINT_FREQ == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Dice_coeff: {dice_coeff.val:.4f}({dice_coeff.avg:.4f}) '
                  'Encoder Grad1: {encoder_grad_norm1:.4f}  '
                  'Encoder Grad2: {encoder_grad_norm2:.4f}  '
                  'Encoder LR: {encoder_lr:.6f}  '
                  .format(
                   epoch+1, step, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, dice_coeff=dice_coeffs,
                   remain=timeSince(start, float(step+1)/len(train_loader)),
                   encoder_grad_norm1=encoder_grad_norm1,
                   encoder_grad_norm2=encoder_grad_norm2,
                   encoder_lr=scheduler1.get_lr()[0],
                   ))
    return losses.avg, dice_coeffs.avg


def train_pre_loop(folds, fold, LOGGER):
    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================    
    trn_idx = folds[folds['fold'] != fold].index
    train_folds = folds.loc[trn_idx].reset_index(drop=True)

    # ====================================================
    # scheduler 
    # ====================================================
    def get_scheduler(optimizer):
        if PRE_SCHEDULER=='ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=PRE_FACTOR, patience=PRE_PATIENCE, verbose=True, eps=PRE_EPS)
        elif PRE_SCHEDULER=='CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, T_max=PRE_T_MAX, eta_min=PRE_MIN_LR, last_epoch=-1)
        elif PRE_SCHEDULER=='CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=PRE_T_0, T_mult=1, eta_min=PRE_MIN_LR, last_epoch=-1)
        elif PRE_SCHEDULER=='GradualWarmupSchedulerV2':
            scheduler_cosine=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, PRE_COSINE_EPO)
            scheduler_warmup=GradualWarmupSchedulerV2(optimizer, multiplier=PRE_WARMUP_FACTOR, total_epoch=PRE_WARMUP_EPO, after_scheduler=scheduler_cosine)
            scheduler=scheduler_warmup        
        return scheduler

    # ====================================================
    # model & optimizer
    # ====================================================
    encoder1 = Encoder(PRE_ENCODER_TYPE, PRE_DECODER_TYPE, pretrained=False)
    encoder1.to(device)

    encoder2 = Encoder(PRE_ENCODER_TYPE, PRE_DECODER_TYPE, pretrained=False)
    encoder2.to(device)

    if len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) > 1:
        encoder1 = nn.DataParallel(encoder1)
        encoder2 = nn.DataParallel(encoder2)


    optimizer1 = Adam(encoder1.parameters(), lr=PRE_ENCODER_LR, weight_decay=PRE_WEIGHT_DECAY, amsgrad=False)
    scheduler1 = get_scheduler(optimizer1)

    optimizer2 = Adam(encoder2.parameters(), lr=PRE_ENCODER_LR, weight_decay=PRE_WEIGHT_DECAY, amsgrad=False)
    scheduler2 = get_scheduler(optimizer2)
    # Log the network weight histograms (optional)

    # ====================================================
    # loop
    # ====================================================
    criterion1 = SoftBCEWithLogitsLoss(smooth_factor=0.05, reduction="none")
    criterion2 = SoftBCEWithLogitsLoss(smooth_factor=0.05)

    best_score = 0
    best_loss = np.inf

    for epoch in range(PRE_EPOCHS):
        if epoch >= 1: 
            break 
        start_time = time.time()
        train_folds_sample = train_folds.sample(frac=0.2, random_state=epoch).reset_index(drop=True)
        train_dataset = TrainDataset(train_folds_sample, transform=get_transforms_preprocessing(data='train'))

        train_loader = DataLoader(train_dataset, 
                                  batch_size=PRE_BATCH_SIZE, 
                                  shuffle=True, 
                                  num_workers=NUM_WORKERS, 
                                  pin_memory=True,
                                  drop_last=True)

        # train
        avg_loss, avg_tr_dice_coeff = train_pre_fn(train_loader, encoder1, encoder2, criterion1, criterion2, optimizer1, optimizer2, epoch, scheduler1, scheduler2, device)

        # scoring
        #score = get_score(valid_labels, text_preds)
        score = avg_tr_dice_coeff

        if isinstance(scheduler1, ReduceLROnPlateau):
            scheduler1.step(score)
            scheduler2.step(score)
        elif isinstance(scheduler1, CosineAnnealingLR):
            scheduler1.step()
            scheduler2.step()
        elif isinstance(scheduler1, CosineAnnealingWarmRestarts):
            scheduler1.step()
            scheduler2.step()
        elif isinstance(scheduler1, GradualWarmupSchedulerV2):
            scheduler1.step(epoch)
            scheduler2.step(epoch)

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Score: {avg_tr_dice_coeff:.4f}')

        model_to_save1 = encoder1.module if hasattr(encoder1, 'module') else encoder1
        model_to_save2 = encoder2.module if hasattr(encoder2, 'module') else encoder2

        if score > best_score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'encoder': model_to_save1.state_dict(), 
                        'optimizer': optimizer1.state_dict(), 
                        'scheduler': scheduler1.state_dict(), 
                       },
                        OUTPUT_DIR+f'/{PRE_ENCODER_TYPE}_{PRE_DECODER_TYPE}_fold{fold}_{VERSION}_1.pth')
                        #f'./{PRE_ENCODER_TYPE}_{PRE_DECODER_TYPE}_fold{fold}_{VERSION}_1.pth')
            torch.save({'encoder': model_to_save2.state_dict(), 
                        'optimizer': optimizer2.state_dict(), 
                        'scheduler': scheduler2.state_dict(), 
                       },
                        OUTPUT_DIR+f'/{PRE_ENCODER_TYPE}_{PRE_DECODER_TYPE}_fold{fold}_{VERSION}_2.pth')


def train_fn(train_loader, encoder, criterion, 
             optimizer, epoch,
             scheduler, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    dice_coeffs = AverageMeter()
    real_batch = AverageMeter()
    cutmix_counter = AverageMeter()
    coloring_counter = AverageMeter()
    # switch to train mode
    encoder.train()
    
    scaler = torch.cuda.amp.GradScaler()
    
    start = end = time.time()
    global_step = 0
    for step, (images, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        targets = targets.float().to(device)
        batch_size = images.size(0)
        
        if COLORING and np.random.random() < COLORING_THRESHOLD:
            coloring_counter.update(1)
            rgb_gain = np.random.random((batch_size,3))
            rgb_gain[:,0] = 0
            masks = targets

            images = images * torch.tensor([ 0.229, 0.224, 0.225 ]).reshape(3,1,1).to(device)
            images = images + torch.tensor([ 0.485, 0.456, 0.406 ]).reshape(3,1,1).to(device)
            images = images + images * (torch.tensor(rgb_gain).reshape(batch_size,3,1,1).to(device) * masks.repeat(1,3,1,1))
            images = images - torch.tensor([ 0.485, 0.456, 0.406 ]).reshape(3,1,1).to(device)
            images = images / torch.tensor([ 0.229, 0.224, 0.225 ]).reshape(3,1,1).to(device)
            images = images.float()
        else:
            coloring_counter.update(0)

        if SELF_CUTMIX and np.random.random() < CUTMIX_THRESHOLD:
            bbox_list = rand_bbox(images.size(), (0.1, 0.4))
            if len(bbox_list) == 2:
                bbx1_1, bby1_1, bbx2_1, bby2_1 = bbox_list[0]
                bbx1_2, bby1_2, bbx2_2, bby2_2 = bbox_list[1]
                images[:, :, bbx1_1:bbx2_1, bby1_1:bby2_1] = images[:, :, bbx1_2:bbx2_2, bby1_2:bby2_2]
                targets[:, :, bbx1_1:bbx2_1, bby1_1:bby2_1] = targets[:, :, bbx1_2:bbx2_2, bby1_2:bby2_2]
                cutmix_counter.update(1)
            else:
                cutmix_counter.update(0)
        else:
            cutmix_counter.update(0)

        # =========================
        # zero_grad()
        # =========================
        optimizer.zero_grad()
        if APEX:
            with autocast():
                y_preds = encoder(images)
                loss = criterion(y_preds, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            y_preds = encoder(images)
            if PROSPECTIVE_FILTERING and epoch >= 1:
                threshold = 0.5
                iou = get_iou(y_preds, targets)
                train_filter = iou>threshold
                batch_size = torch.sum(train_filter)
                y_preds = y_preds[train_filter]
                targets = targets[train_filter]
            loss = criterion(y_preds, targets)
            loss.backward()
        real_batch.update(batch_size)
        # record loss
        losses.update(loss.item(), batch_size)
        if GRADIENT_ACCUMULATION_STEPS > 1:
            loss = loss / GRADIENT_ACCUMULATION_STEPS
        encoder_grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), MAX_GRAD_NORM)
        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            if APEX:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            global_step += 1
            
        # record dice_coeff
        dice_coeff = get_dice_coeff(y_preds, 
                                    targets)
        dice_coeffs.update(dice_coeff, batch_size)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % PRINT_FREQ == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Dice_coeff: {dice_coeff.val:.4f}({dice_coeff.avg:.4f}) '
                  'Encoder Grad: {encoder_grad_norm:.4f}  '
                  'Encoder LR: {encoder_lr:.6f}  '
                  .format(
                   epoch+1, step, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, dice_coeff=dice_coeffs,
                   remain=timeSince(start, float(step+1)/len(train_loader)),
                   encoder_grad_norm=encoder_grad_norm,
                   encoder_lr=scheduler.get_lr()[0],
                   ))
    return losses.avg, dice_coeffs.avg


def train_loop(folds, fold, LOGGER):
    LOGGER.info(f"========== All dataset training ==========")

    # ====================================================
    # loader
    # ====================================================    
    trn_idx = folds[folds['fold'] != fold].index
    train_folds = folds.loc[trn_idx].reset_index(drop=True)

    train_dataset = TrainDataset(train_folds, transform=get_transforms_train(data='train'))
    train_loader = DataLoader(train_dataset, 
                              batch_size=BATCH_SIZE, 
                              shuffle=True, 
                              num_workers=NUM_WORKERS, 
                              pin_memory=True,
                              drop_last=True)

    
    # ====================================================
    # scheduler 
    # ====================================================
    def get_scheduler(optimizer):
        if SCHEDULER=='ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=FACTOR, patience=PATIENCE, verbose=True, eps=EPS)
        elif SCHEDULER=='CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=MIN_LR, last_epoch=-1)
        elif SCHEDULER=='CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=1, eta_min=MIN_LR, last_epoch=-1)
        elif SCHEDULER=='GradualWarmupSchedulerV2':
            scheduler_cosine=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, COSINE_EPO)
            scheduler_warmup=GradualWarmupSchedulerV2(optimizer, multiplier=WARMUP_FACTOR, total_epoch=WARMUP_EPO, after_scheduler=scheduler_cosine)
            scheduler=scheduler_warmup        
        return scheduler

    # ====================================================
    # model & optimizer
    # ====================================================
    encoder = Encoder(ENCODER_TYPE, DECODER_TYPE, pretrained=PRETRAINED)
    encoder.to(device)
    
    if len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) > 1:
        encoder = nn.DataParallel(encoder)

    optimizer = Adam(encoder.parameters(), lr=ENCODER_LR, weight_decay=WEIGHT_DECAY, amsgrad=False)
    scheduler = get_scheduler(optimizer)

    # ====================================================
    # loop
    # ====================================================
    criterion = DiceBCELoss()

    best_score = 0
    best_loss = np.inf
    
    for epoch in range(EPOCHS):
        if epoch >= BREAK_EPOCH: 
            break 
        start_time = time.time()
        
        # train
        avg_loss, avg_tr_dice_coeff = train_fn(train_loader, encoder, criterion, optimizer, epoch, scheduler, device)
        
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(score)
        elif isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()
        elif isinstance(scheduler, GradualWarmupSchedulerV2):
            scheduler.step(epoch)

        elapsed = time.time() - start_time
        
        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Score: {avg_tr_dice_coeff:.4f}')
 
        model_to_save = encoder.module if hasattr(encoder, 'module') else encoder
        LOGGER.info(f'Epoch {epoch+1} - Save Model')
        torch.save({'encoder': model_to_save.state_dict(), 
                    'optimizer': optimizer.state_dict(), 
                    'scheduler': scheduler.state_dict(), 
                   },
                    OUTPUT_TRAIN_DIR+f'/{ENCODER_TYPE}_{DECODER_TYPE}_{VERSION}_epoch{epoch}.pth')

