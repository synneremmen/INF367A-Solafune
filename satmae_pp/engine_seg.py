# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch
#import wandb
import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        images, targets = samples['image'], samples['mask']
        images = images.to(device, dtype=torch.float32, non_blocking=True)
        targets = targets.to(device, dtype=torch.long,  non_blocking=True)

        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)


        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss = loss / accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        if device.type == 'cuda' and torch.cuda.is_initialized():
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

            '''
            if args.local_rank == 0 and args.wandb is not None:
                try:
                    wandb.log({'train_loss_step': loss_value_reduce,
                               'train_lr_step': max_lr, 'epoch_1000x': epoch_1000x})
                except ValueError:
                    pass
            '''

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




@torch.no_grad()
def evaluate(data_loader, model, device, num_classes):
    """
    Run the model on the validation set and compute pixel Cross-Entropy loss
    plus mean IoU over all classes.
    """
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Val:'

    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch['image'].to(device, non_blocking=True)  # (B, C, H, W)
        masks  = batch['mask'].to(device,  non_blocking=True)  # (B, H, W)

        with torch.cuda.amp.autocast():
            logits = model(images)                # (B, K, H, W)
            loss   = criterion(logits, masks)

        # Predictions and per-class IoU
        preds = logits.argmax(dim=1)             # (B, H, W)
        iou_list = []
        for cls in range(num_classes):
            pred_inds   = (preds == cls)
            target_inds = (masks == cls)
            inter = (pred_inds & target_inds).sum().float()
            union = (pred_inds | target_inds).sum().float()
            iou_list.append(inter / union if union > 0 else torch.tensor(1.0, device=device))
        mIoU = torch.mean(torch.stack(iou_list)).item()

        metric_logger.update(loss=loss.item(), mIoU=mIoU)

    metric_logger.synchronize_between_processes()
    print(f'* Mean IoU {metric_logger.mIoU.global_avg:.3f}  Loss {metric_logger.loss.global_avg:.3f}')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
