# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, DINO, and BIOT code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# https://github.com/ycq091044/BIOT
# ---------------------------------------------------------

import io
import os
import math
import time
import json
import glob
from collections import defaultdict, deque
import datetime
import numpy as np
from timm.utils import get_state_dict

from pathlib import Path
import argparse

import torch
import torch.distributed as dist
from torch import inf
import h5py

from tensorboardX import SummaryWriter
from data_processor.dataset import ShockDataset
import pickle
from scipy.signal import resample
from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


standard_1020 = [
    'FP1', 'FPZ', 'FP2', 
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', \
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', \
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', \
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', \
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', \
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10', \
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', \
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2', \
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2', \
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8', \
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8', \
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h', \
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]

# standard_1020 =['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF',
#  'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF',
#  'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF',
#  'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF',
#  'EEG PZ-REF', 'EEG ROC-REF', 'EEG LOC-REF', 'EEG EKG1-REF', 'EMG-REF',
#  'EEG 26-REF', 'EEG 27-REF', 'EEG 28-REF', 'EEG 29-REF', 'EEG 30-REF',
#  'EEG T1-REF', 'EEG T2-REF']

# standard_1020 = ['CH0', 'CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'CH8', 
#                  'CH9', 'CH10', 'CH11', 'CH12', 'CH13', 'CH14', 'CH15', 'CH16', 
#                  'CH17', 'CH18', 'CH19', 'CH20', 'CH21', 'CH22']

# standard_1020 = ['E1', 'E2', 'E3']

# standard_1020 = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
#                   'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ']

# standard_1020 = ['FEAT_0', 'FEAT_1', 'FEAT_2', 'FEAT_3', 'FEAT_4', 'FEAT_5', 'FEAT_6', 'FEAT_7', 'FEAT_8', 'FEAT_9', 
#                   'FEAT_10', 'FEAT_11', 'FEAT_12', 'FEAT_13', 'FEAT_14', 'FEAT_15', 'FEAT_16', 'FEAT_17', 'FEAT_18', 
#                   'FEAT_19', 'FEAT_20', 'FEAT_21', 'FEAT_22', 'FEAT_23', 'FEAT_24', 'FEAT_25', 'FEAT_26', 'FEAT_27']

# Generates the exact same list automatically
# standard_1020 = [f'FEAT_{i}' for i in range(129)]

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")

def get_model(model):
    if isinstance(model, torch.nn.DataParallel) \
      or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model
            
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)
    
    def update_image(self, head='images', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            self.writer.add_image(head + "/" + k, v, self.step if step is None else step)
            
    def flush(self):
        self.writer.flush()


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False):
    world_size = get_world_size()

    if world_size == 1:
        return tensor
    dist.all_reduce(tensor, op=op, async_op=async_op)

    return tensor

def all_gather_batch(tensors):
    """
    Performs all_gather operation on the provided tensors.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(
            tensor_all,
            tensor,
            async_op=False  # performance opt
        )

        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor

class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def all_gather_batch_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []

    for tensor in tensors:
        tensor_all = GatherLayer.apply(tensor)
        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor

def _get_rank_env():
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    else:
        return int(os.environ['OMPI_COMM_WORLD_RANK'])


def _get_local_rank_env():
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    else:
        return int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])


def _get_world_size_env():
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    else:
        return int(os.environ['OMPI_COMM_WORLD_SIZE'])


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = _get_rank_env()
        args.world_size = _get_world_size_env()  # int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = _get_local_rank_env()
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True, layer_names=None):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters, layer_names=layer_names)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict): 
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0, layer_names=None) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    
    parameters = [p for p in parameters if p.grad is not None]
        
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        # total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
        layer_norm = torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters])
        total_norm = torch.norm(layer_norm, norm_type)
        # print(layer_norm.max(dim=0))
        
        if layer_names is not None:
            if torch.isnan(total_norm) or torch.isinf(total_norm) or total_norm > 1.0:
                value_top, name_top = torch.topk(layer_norm, k=5)
                print(f"Top norm value: {value_top}")
                print(f"Top norm name: {[layer_names[i][7:] for i in name_top.tolist()]}")
        
    return total_norm


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


# def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None, optimizer_disc=None, save_ckpt_freq=1):
#     output_dir = Path(args.output_dir)
#     epoch_name = str(epoch)

#     if not getattr(args, 'enable_deepspeed', False):
#         checkpoint_paths = [output_dir / 'checkpoint.pth']
#         if epoch == 'best':
#             checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name),]
#         elif (epoch + 1) % save_ckpt_freq == 0:
#             checkpoint_paths.append(output_dir / ('checkpoint-%s.pth' % epoch_name))

#     for checkpoint_path in checkpoint_paths:
#         to_save = {
#             'model': model_without_ddp.state_dict(),
#             'optimizer': optimizer.state_dict(),
#             'epoch': epoch,
#             'args': args,
#         }
#         if loss_scaler is not None:
#             to_save['scaler'] = loss_scaler.state_dict()

#         # --- 1. Handle Model EMA (Update dict only) ---
#         if model_ema is not None:
#             # Try .module (Newer timm)
#             if hasattr(model_ema, 'module'):
#                 to_save['model_ema'] = model_ema.module.state_dict()
#             # Try .ema (Older timm / LaBraM) - Fixes your crash
#             elif hasattr(model_ema, 'ema'):
#                 to_save['model_ema'] = model_ema.ema.state_dict()
#             # Fallback
#             elif hasattr(model_ema, 'state_dict'):
#                 to_save['model_ema'] = model_ema.state_dict()
#             else:
#                 print("Warning: Could not save model_ema (attributes not found).")

#         # --- 2. Handle Discriminator (Update dict only) ---
#         if optimizer_disc is not None:
#             to_save['optimizer_disc'] = optimizer_disc.state_dict()

#         # --- 3. SAVE TO DISK (CRITICAL: Must be outside the 'if' blocks) ---
#         save_on_master(to_save, checkpoint_path)
#     else:
#         client_state = {'epoch': epoch}
#         if model_ema is not None:
#             client_state['model_ema'] = get_state_dict(model_ema)
#         model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)           

import os

def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None, optimizer_disc=None, **kwargs):

    # --- 1. DETERMINE IF WE SHOULD SAVE (Rank 0 only) ---
    # If distributed is initialized, check rank. Otherwise, assume we are the master.
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() != 0:
            return # Workers exit here. Only Master continues.
    
    # --- 2. SETUP PATHS ---
    output_dir = str(args.output_dir)
    epoch_name = str(epoch)
    checkpoint_name = f"checkpoint-{epoch_name}.pth"
    checkpoint_path = os.path.join(output_dir, checkpoint_name)

    # --- 3. HANDLE DEEPSPEED (If applicable) ---
    if hasattr(args, 'enable_deepspeed') and args.enable_deepspeed:
        client_state = {'epoch': epoch}
        if model_ema is not None:
            # Robust EMA check
            if hasattr(model_ema, 'module'):
                client_state['model_ema'] = model_ema.module.state_dict()
            elif hasattr(model_ema, 'ema'):
                client_state['model_ema'] = model_ema.ema.state_dict()
            else:
                client_state['model_ema'] = model_ema.state_dict()
        
        # DeepSpeed handles its own saving
        model.save_checkpoint(save_dir=output_dir, tag=f"checkpoint-{epoch_name}", client_state=client_state)
        return

    # --- 4. PREPARE DICTIONARY (Standard PyTorch) ---
    # print(f"[Save Debug] Preparing dictionary for {checkpoint_path}...")
    to_save = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'args': args,
    }

    if loss_scaler is not None:
        to_save['scaler'] = loss_scaler.state_dict()

    # EMA Handling (The Critical Part)
    if model_ema is not None:
        try:
            if hasattr(model_ema, 'module'):
                to_save['model_ema'] = model_ema.module.state_dict()
            elif hasattr(model_ema, 'ema'):
                to_save['model_ema'] = model_ema.ema.state_dict()
            else:
                to_save['model_ema'] = model_ema.state_dict()
        except Exception as e:
            print(f"[Save Warning] Failed to grab EMA weights: {e}")

    if optimizer_disc is not None:
        to_save['optimizer_disc'] = optimizer_disc.state_dict()

    # --- 5. WRITE TO DISK ---
    try:
        # Create directory if it doesn't exist (Fixes FileNotFoundError)
        os.makedirs(output_dir, exist_ok=True)
        
        # Use torch.save directly (Bypasses save_on_master issues)
        torch.save(to_save, checkpoint_path)
        print(f"[Save Success] Saved checkpoint to {checkpoint_path}")
        
    except Exception as e:
        print(f"\n[CRITICAL SAVE ERROR] Could not write file: {e}")
        # We print but don't raise, so training doesn't crash just because save failed
        print(f"\n[CRITICAL SAVE ERROR] Could not write file: {e}")
        # We print but don't raise, so training doesn't crash just because save failed


def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None, optimizer_disc=None):
    output_dir = Path(args.output_dir)
    
    if not getattr(args, 'enable_deepspeed', False):
        # torch.amp
        if args.auto_resume and len(args.resume) == 0:
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint.pth'))
            if len(all_checkpoints) > 0:
                args.resume = os.path.join(output_dir, 'checkpoint.pth')
            else:
                all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
                latest_ckpt = -1
                for ckpt in all_checkpoints:
                    t = ckpt.split('-')[-1].split('.')[0]
                    if t.isdigit():
                        latest_ckpt = max(int(t), latest_ckpt)
                if latest_ckpt >= 0:
                    args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
            print("Auto resume checkpoint: %s" % args.resume)

        if args.resume:
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            model_without_ddp.load_state_dict(checkpoint['model']) # strict: bool=True, , strict=False
            print("Resume checkpoint %s" % args.resume)
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                print(f"Resume checkpoint at epoch {checkpoint['epoch']}")
                args.start_epoch = 1#checkpoint['epoch'] + 1
                if hasattr(args, 'model_ema') and args.model_ema:
                    _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                print("With optim & sched!")
            if 'optimizer_disc' in checkpoint:
                optimizer_disc.load_state_dict(checkpoint['optimizer_disc'])
    else:
        # deepspeed, only support '--auto_resume'.
        if args.auto_resume:
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, 'checkpoint-%d' % latest_ckpt)
                print("Auto resume checkpoint: %d" % latest_ckpt)
                _, client_states = model.load_checkpoint(args.output_dir, tag='checkpoint-%d' % latest_ckpt)
                args.start_epoch = client_states['epoch'] + 1
                if model_ema is not None:
                    if args.model_ema:
                        _load_checkpoint_for_ema(model_ema, client_states['model_ema'])

def create_ds_config(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.output_dir, "latest"), mode="w") as f:
        pass

    args.deepspeed_config = os.path.join(args.output_dir, "deepspeed_config.json")
    with open(args.deepspeed_config, mode="w") as writer:
        ds_config = {
            "train_batch_size": args.batch_size * args.update_freq * get_world_size(),
            "train_micro_batch_size_per_gpu": args.batch_size,
            "steps_per_print": 1000,
            "optimizer": {
                "type": "Adam",
                "adam_w_mode": True,
                "params": {
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "bias_correction": True,
                    "betas": [
                        0.9,
                        0.999
                    ],
                    "eps": 1e-8
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 7,
                "loss_scale_window": 128
            }
        }

        writer.write(json.dumps(ds_config, indent=2))


def build_pretraining_dataset(datasets: list, time_window: list, stride_size=200, start_percentage=0, end_percentage=1):
    shock_dataset_list = []
    ch_names_list = []
    for dataset_list, window_size in zip(datasets, time_window):
        dataset = ShockDataset([Path(file_path) for file_path in dataset_list], window_size * 200, stride_size, start_percentage, end_percentage)
        shock_dataset_list.append(dataset)
        ch_names_list.append(dataset.get_ch_names())
    return shock_dataset_list, ch_names_list


def get_input_chans(ch_names):
    input_chans = [0] # for cls token
    for ch_name in ch_names:
        input_chans.append(standard_1020.index(ch_name) + 1)
    return input_chans


class TUABLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y
    

# class TUEVLoader(torch.utils.data.Dataset):
#     def __init__(self, root, files, sampling_rate=200):
#         self.root = root
#         self.files = files
#         self.default_rate = 200
#         self.sampling_rate = sampling_rate

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, index):
#         sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
#         X = sample["signal"]
#         X = X.astype(np.float32)
#         if self.sampling_rate != self.default_rate:
#             X = resample(X, 5 * self.sampling_rate, axis=-1)
#         Y = int(sample["label"][0] - 1)
#         X = torch.FloatTensor(X)
#         return X, Y


# class TUEVLoader(torch.utils.data.Dataset):
#     def __init__(self, root, files, sampling_rate=200):
#         self.root = root
#         self.files = files
#         self.default_rate = 200
#         self.sampling_rate = sampling_rate

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, index):
#         with open(os.path.join(self.root, self.files[index]), "rb") as f:
#             sample = pickle.load(f)
        
#         X = sample["signal"]
        
#         # 1. 确保数据类型为 float32
#         # 2. 核心修复：检查并转换字节序
#         if X.dtype.byteorder not in ('=', '|'):
#             X = X.byteswap().newbyteorder()
        
#         X = X.astype(np.float32)

#         if self.sampling_rate != self.default_rate:
#             # 假设这里的 resample 是 scipy.signal.resample 或类似的
#             X = resample(X, 5 * self.sampling_rate, axis=-1)
            
#         Y = int(sample["label"][0] - 1)
        
#         # 此时 X 已经是原生字节序，转换不会报错
#         X = torch.from_numpy(X) 
#         return X, Y

class TUEVLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with open(os.path.join(self.root, self.files[index]), "rb") as f:
            sample = pickle.load(f)
        
        X = sample["signal"]
        
        # --- 彻底解决字节序问题的三步走 ---
        # 1. 如果 X 是从 Big-endian 机器/格式保存的，先交换字节
        if X.dtype.byteorder == '>' or (X.dtype.byteorder == '|' and X.dtype.str.endswith('f4') == False):
            # 如果明确标记为大端序，进行交换
            X = X.byteswap().newbyteorder('=')
        
        # 2. 使用 np.array 强制创建一个全新的、原生字节序的副本
        # copy=True 和 dtype=np.float32 会强制内存重新分配
        X = np.array(X, dtype=np.float32, copy=True)
        
        # 3. 确保内存是连续的（Dataloader collate 的硬性要求）
        X = np.ascontiguousarray(X)
        
        if self.sampling_rate != self.default_rate:
            # resample 通常会返回一个 native order 的数组，但我们保持谨慎
            X = resample(X, 5 * self.sampling_rate, axis=-1)
            X = np.ascontiguousarray(X, dtype=np.float32)
            
        Y = int(sample["label"][0] - 1)
        
        # 使用 torch.as_tensor 比 torch.from_numpy 在这种情况下更稳健
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        
        return X_tensor, Y
    

def sliding_window_eeg(data, index, window_size, step_size):
    # n_samples, n_channels, n_timepoints = data.shape
    # n_windows = (n_timepoints - window_size) // step_size + 1
    # segmented = np.zeros((n_samples, n_windows, n_channels, window_size))

    
    start = (index - 1) * step_size
    end = start + window_size
    

    return data[:, :, start:end]  # shape: (n_samples, n_channels, window_size)

def align_labels(labels, n_windows):
    # print("the shape of n_windows: "+str(n_windows))
    # print(n_windows)
    return np.repeat(labels, n_windows)  # shape: (n_samples * n_windows,)


class BCICIV2b_EEGDataset(torch.utils.data.Dataset):
    def __init__(self, subject_ids, window_index=8, window_size = 500, step_size=100, transform=None, sr=1000):
        self.subject_ids = subject_ids
        self.window_index = window_index
        self.window_size = window_size
        self.step_size = step_size
        self.transform = transform
        self.sampling_rate = sr
        self.data_subjs = []
        self.labels_subjs = []

        # Load data (example)
        self.X, self.y = self.load_subject_data(self.subject_ids)

    def load_subject_data(self, subject_ids):
        # Load from HDF5 or another source
        # Return (X, y) as numpy arrays
        Subj_id = {'S01': 1, 'S02': 2, 'S03': 3, 'S04': 4, 'S05': 5, 'S06': 6, 'S07': 7, 'S08': 8, 'S09': 9}

        for subject_id in subject_ids:
            subject_id = Subj_id[subject_id]

            x1 = np.load("/homes/xw2336/data/BCICIV_2b/Subj{0}_1_X.npy".format(subject_id))
            x2 = np.load("/homes/xw2336/data/BCICIV_2b/Subj{0}_2_X.npy".format(subject_id))

            y1 = np.load("/homes/xw2336/data/BCICIV_2b/Subj{0}_1_y.npy".format(subject_id))
            y2 = np.load("/homes/xw2336/data/BCICIV_2b/Subj{0}_2_y.npy".format(subject_id))

            x_combined_data = np.concatenate([x1, x2], axis=0)
            y_combined_labels = np.concatenate([y1, y2], axis=0)
            self.data_subjs.append(x_combined_data)
            self.labels_subjs.append(y_combined_labels)
        

        x_combined_data = np.concatenate([self.data_subjs[i] for i in range(len(self.data_subjs))], axis=0)
        y_combined_labels = np.concatenate([self.labels_subjs[i] for i in range(len(self.labels_subjs))], axis=0)
            

        x_seg_data = sliding_window_eeg(x_combined_data, self.window_index, self.window_size, self.step_size)
        # n_windows = x_seg_data.shape[1]

        # x_seg_data = x_seg_data.reshape(-1, x_seg_data.shape[2], x_seg_data.shape[3])

        # labels = align_labels(y_combined_labels, n_windows)

        return x_seg_data, y_combined_labels
    

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        # Handle the "0 length" issue safely
        if x is None or x.shape[-1] == 0:
            # If the specific window index failed, we create a dummy 
            # 500-sample window so the training loop doesn't crash.
            x = np.zeros((3, 500), dtype=np.float32)
            print(f"Warning: Empty data at idx {idx}. Using zero-padding.")

        # Resample from 500 (250Hz) to 400 (200Hz) for LaBraM
        from scipy.signal import resample
        x_resampled = resample(x, 400, axis=-1)

        return torch.tensor(x_resampled, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    
def prepare_TUEV_dataset(root):
    # set random seed
    seed = 4523
    np.random.seed(seed)

    train_files = os.listdir(os.path.join(root, "processed_train"))
    val_files = os.listdir(os.path.join(root, "processed_eval"))
    test_files = os.listdir(os.path.join(root, "processed_test"))

    # prepare training and test data loader
    train_dataset = TUEVLoader(
        os.path.join(
            root, "processed_train"), train_files
    )
    test_dataset = TUEVLoader(
        os.path.join(
            root, "processed_test"), test_files
    )
    val_dataset = TUEVLoader(
        os.path.join(
            root, "processed_eval"), val_files
    )
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def prepare_TUAB_dataset(root):
    # set random seed
    seed = 12345
    np.random.seed(seed)

    train_files = os.listdir(os.path.join(root, "train"))
    np.random.shuffle(train_files)
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = TUABLoader(os.path.join(root, "train"), train_files)
    test_dataset = TUABLoader(os.path.join(root, "test"), test_files)
    val_dataset = TUABLoader(os.path.join(root, "val"), val_files)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset

def prepare_BCIIV2b_dataset(subjs, window_size):
    # set random seed
    seed = 42
    np.random.seed(seed)

    tr_subjs, val_subjs, te_subjs = subjs

    # CRITICAL FIX: Explicitly assign window_size and window_index
    # We use window_index=8 as you requested
    train_dataset = BCICIV2b_EEGDataset(tr_subjs, window_index=8, window_size=500)
    test_dataset = BCICIV2b_EEGDataset(te_subjs, window_index=8, window_size=500)
    val_dataset = BCICIV2b_EEGDataset(val_subjs, window_index=8, window_size=500)

    return train_dataset, val_dataset, test_dataset


def get_metrics(output, target, metrics, is_binary, threshold=0.5):
    if is_binary:
        if 'roc_auc' not in metrics or sum(target) * (len(target) - sum(target)) != 0:  # to prevent all 0 or all 1 and raise the AUROC error
            results = binary_metrics_fn(
                target,
                output,
                metrics=metrics,
                threshold=threshold,
            )
        else:
            results = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }
    else:
        results = multiclass_metrics_fn(
            target, output, metrics=metrics
        )
    return results
