# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional
import time 
import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
import pdb
import warnings
warnings.filterwarnings('ignore')

def get_tau(start_tau, end_tau, ite, total):
    tau = start_tau + (end_tau - start_tau) * ite / total 
    return tau 

ite_step = 0
def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, mask=None, args=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    # pdb.set_trace()
    total_iteration = len(data_loader) * (args.epochs)
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        global ite_step
        optimizer.zero_grad()
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if args.token_selection:
            tau = get_tau(10, 0.1, ite_step, total_iteration)
        else:
            tau = -1 

        with torch.cuda.amp.autocast():
            if args.pruning_type == 'structure':
                outputs, atten_pruning_indicator = model(samples, tau=tau, number=args.token_number)
            else:
                outputs = model(samples, tau=tau, number=args.token_number)
                atten_pruning_indicator = None

            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
            
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), 
                                    create_graph=is_second_order)
        if mask is not None: 
            mask.step(pruning_type=args.pruning_type)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # update sparse topology
        
        ite_step = mask.steps
        if ite_step % args.update_frequency == 0 and ite_step < args.t_end * total_iteration:
            mask.at_end_of_epoch(pruning_type=args.pruning_type, 
                                indicator_list=atten_pruning_indicator)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_training_time(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, mask=None, args=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    # pdb.set_trace()
    total_time = 0
    total_iteration = len(data_loader) * (args.epochs)
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():

            start = time.time()
            if args.pruning_type == 'structure':
                outputs, atten_pruning_indicator = model(samples)

            elif args.token_selection:
                outputs = model(samples, tau=10, number=args.token_number)
                atten_pruning_indicator = None

            else:
                outputs = model(samples)
                atten_pruning_indicator = None

            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
            
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), 
                                    create_graph=is_second_order)

        end = time.time()
        total_time += end-start 
        global ite_step        
        ite_step += 1
        if ite_step % 100 == 0:
            print(total_time)
            total_time = 0
        

        # if mask is not None: 
        #     mask.step(pruning_type=args.pruning_type)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # update sparse topology


        # if ite_step % args.update_frequency == 0 and ite_step < args.t_end * total_iteration:
        #     mask.at_end_of_epoch(pruning_type=args.pruning_type, 
        #                         indicator_list=atten_pruning_indicator)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    if args.token_selection:
        tau = 1
    else:
        tau = -1

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            if args.pruning_type == 'structure':
                output, atten_pruning_indicator = model(images, tau=tau, number=args.token_number)
            else:
                output = model(images, tau=tau, number=args.token_number)
                atten_pruning_indicator = None                

            # output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
