import math
import sys
from typing import Iterable, Optional
import torch
from timm.utils import ModelEma
import utils
from einops import rearrange

def train_class_batch(model, samples, target, criterion, input_chans):
    # Pass input_chans as a keyword argument to ensure it maps correctly to the model's forward pass
    outputs = model(samples, input_chans=input_chans)
    loss = criterion(outputs, target)
    return loss, outputs

def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, ch_names=None, is_binary=True):
    
    # 1. Convert string channel names to the integer indices required by the spatial map
    input_chans = None
    if ch_names is not None:
        input_chans = utils.get_input_chans(ch_names)
        
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step 
        
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        # 2. Rescale and Reshape samples for LaBraM's patch-based architecture
        # Dividing by 100 is the standard normalization for this foundation model
        samples = samples.float().to(device, non_blocking=True) / 100
        # Rearrange (Batch, Channels, Time) -> (Batch, Channels, Patches, Timepoints_per_patch)
        samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)
        
        targets = targets.to(device, non_blocking=True)
        if is_binary:
            targets = targets.float().unsqueeze(-1)

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(
                model, samples, targets, criterion, input_chans)
        else:
            with torch.amp.autocast('cuda'):
                loss, output = train_class_batch(
                    model, samples, targets, criterion, input_chans)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= update_freq
        if loss_scaler is None:
            model.backward(loss)
            model.step()
            if (data_iter_step + 1) % update_freq == 0:
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if is_binary:
            # detach and move to cpu for metrics calculation
            probs = torch.sigmoid(output).detach().cpu().numpy()
            targs = targets.detach().cpu().numpy()
            class_acc = utils.get_metrics(probs, targs, ["accuracy"], is_binary)["accuracy"]
        else:
            class_acc = (output.max(-1)[-1] == targets.squeeze()).float().mean()
            
        metric_logger.update(loss=loss_value, class_acc=class_acc, loss_scale=loss_scale_value)
        
        lrs = [group["lr"] for group in optimizer.param_groups]
        metric_logger.update(lr=max(lrs), min_lr=min(lrs))
        
        if log_writer is not None:
            log_writer.update(loss=loss_value, class_acc=class_acc, head="loss")
            log_writer.set_step()

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, header='Test:', ch_names=None, metrics=['acc'], is_binary=True):
    input_chans = None
    if ch_names is not None:
        input_chans = utils.get_input_chans(ch_names)
        
    criterion = torch.nn.BCEWithLogitsLoss() if is_binary else torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")

    model.eval()
    pred, true = [], []
    
    for step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        EEG, target = batch[0], batch[-1]
        EEG = EEG.float().to(device, non_blocking=True) / 100
        EEG = rearrange(EEG, 'B N (A T) -> B N A T', T=200)
        target = target.to(device, non_blocking=True)
        if is_binary:
            target = target.float().unsqueeze(-1)
        
        with torch.amp.autocast('cuda'):
            output = model(EEG, input_chans=input_chans)
            loss = criterion(output, target)
        
        output = torch.sigmoid(output).cpu() if is_binary else output.cpu()
        target = target.cpu()

        results = utils.get_metrics(output.numpy(), target.numpy(), metrics, is_binary)
        pred.append(output)
        true.append(target)

        metric_logger.update(loss=loss.item())
        for key, value in results.items():
            metric_logger.meters[key].update(value, n=EEG.shape[0])
            
    metric_logger.synchronize_between_processes()
    pred = torch.cat(pred, dim=0).numpy()
    true = torch.cat(true, dim=0).numpy()

    ret = utils.get_metrics(pred, true, metrics, is_binary, 0.5)
    ret['loss'] = metric_logger.loss.global_avg
    return ret