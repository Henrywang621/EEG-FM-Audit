"""
by Wei-Bang Jiang
https://github.com/935963004/NeuroLM
Adapted for TUEV Dataset
"""

import os
import time
import argparse
from contextlib import nullcontext

import numpy as np
import torch
import torch._dynamo.config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model.model_vq import VQ_Align
from model.model_neural_transformer import NTConfig
from dataset import PickleLoader
from pathlib import Path
from utils import cosine_scheduler
import math
import torch._dynamo 

# Standard fix for DDP with torch.compile
torch._dynamo.config.optimize_ddp = False

master_process = None; device = None; dtype = None
ctx = None; ddp_rank = None; device_type = None
ddp = None; ddp_world_size = None; ddp_local_rank = None

def init(args):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank
    backend = 'nccl'
    device = 'cuda'
    # Mixed precision setup
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1

    torch.manual_seed(args.seed + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

def main(args):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank

    init(args)

    # TUEV specific checkpoint directory
    checkpoint_out_dir = os.path.join(args.out_dir, 'checkpoints/VQ_tuev')
    if master_process:
        os.makedirs(checkpoint_out_dir, exist_ok=True)

    # Text data loader for Domain Alignment
    data_dir = os.path.join(args.out_dir, 'text')
    def get_batch(split):
        if split == 'train':
            data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - args.block_size, (args.text_batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+args.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+args.block_size]).astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    # DATA LOADING: Updated for TUEV folder structure
    print(f'Loading TUEV training data from {args.dataset_dir}...')
    files = Path(args.dataset_dir, 'processed_train').rglob('*.pkl')
    files = [file for file in files]
    if not files:
        raise FileNotFoundError(f"No .pkl files found in {os.path.join(args.dataset_dir, 'processed_train')}")
    
    dataset_train = PickleLoader(files)
    print(f'Finished! Found {len(files)} files.')

    if ddp:
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True)
        data_loader_train = torch.utils.data.DataLoader(dataset_train, sampler=sampler_train, batch_size=args.batch_size, num_workers=10, pin_memory=True, drop_last=True)
    else:
        data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, num_workers=10, pin_memory=True, drop_last=True, shuffle=True)

    iter_num = 0
    # Model Configuration (Standard NeuroLM VQ)
    encoder_args = dict(n_layer=12, n_head=12, n_embd=768, block_size=1024, bias=False, dropout=0., num_classes=0, in_chans=1, out_chans=16)
    decoder_args = dict(n_layer=4, n_head=12, n_embd=768, block_size=1024, bias=False, dropout=0., num_classes=0, in_chans=128)

    init_from = 'resume' if os.path.exists(os.path.join(checkpoint_out_dir, 'ckpt.pt')) else 'scratch'

    if init_from == 'scratch':
        print("Initializing a new model from scratch")
        encoder_conf = NTConfig(**encoder_args)
        decoder_conf = NTConfig(**decoder_args)
        model = VQ_Align(encoder_conf, decoder_conf)
        start_epoch = 0
    elif init_from == 'resume':
        print(f"Resuming training from {checkpoint_out_dir}")
        ckpt_path = os.path.join(checkpoint_out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        # Load args and state_dict...
        encoder_conf = NTConfig(**checkpoint['encoder_args'])
        decoder_conf = NTConfig(**checkpoint['decoder_args'])
        model = VQ_Align(encoder_conf, decoder_conf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix): state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        start_epoch = checkpoint['epoch'] + 1

    model.to(device)
    scaler = torch.amp.GradScaler(device_type, enabled=(dtype == 'float16'))
    optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type)
    
    if init_from == 'resume': optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None 

    if args.compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    num_training_steps_per_epoch = len(dataset_train) // args.batch_size // ddp_world_size
    lr_schedule_values = cosine_scheduler(args.learning_rate, args.min_lr, args.epochs, num_training_steps_per_epoch, warmup_epochs=args.warmup_epochs)

    # TRAINING LOOP
    X_text, Y_text = get_batch('train') 
    t0 = time.time()
    raw_model = model.module if ddp else model 
    
    for epoch in range(start_epoch, args.epochs):
        for step, (batch) in enumerate(data_loader_train):
            lr = lr_schedule_values[iter_num] if args.decay_lr else args.learning_rate
            for param_group in optimizer.param_groups: param_group['lr'] = lr

            if ddp: model.require_backward_grad_sync = (step + 1) % args.gradient_accumulation_steps == 0
            
            # Unpack TUEV data
            X, Y_freq, Y_raw, input_chans, input_time, input_mask = batch
            X = X.float().to(device, non_blocking=True)
            Y_raw = Y_raw.float().to(device, non_blocking=True)
            Y_freq = Y_freq.float().to(device, non_blocking=True)

            # NORMALIZATION: Instance-based normalization is critical for TUEV voltage scales
            x_mean = X.mean(dim=-1, keepdim=True)
            x_std = X.std(dim=-1, keepdim=True)
            X = (X - x_mean) / (x_std + 1e-6)
            Y_raw = (Y_raw - x_mean) / (x_std + 1e-6)
            Y_freq = Y_freq / (x_std + 1e-6)

            input_chans, input_time, input_mask = [t.to(device, non_blocking=True) for t in [input_chans, input_time, input_mask]]

            with ctx:
                # Calculate alignment alpha
                alpha = 2 / (1 + math.exp(-10 * iter_num / (args.epochs * num_training_steps_per_epoch))) - 1
                loss, domain_loss, log = model(X, Y_freq, Y_raw, input_chans, input_time, input_mask, alpha)
                domain_loss2 = model(X_text) # Multimodal text alignment
                total_loss = (loss + domain_loss + domain_loss2) / args.gradient_accumulation_steps
            
            scaler.scale(total_loss).backward()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if (iter_num + 1) % args.log_interval == 0 and master_process:
                print(f"epoch {epoch} step [{step + 1}/{num_training_steps_per_epoch}]: loss {log['train/total_loss']:.4f}")

            X_text, Y_text = get_batch('train')
            iter_num += 1
        
        # Epoch Checkpointing
        if master_process:
            checkpoint = {'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(), 'encoder_args': encoder_args, 'decoder_args': decoder_args, 'iter_num': iter_num, 'epoch': epoch}
            torch.save(checkpoint, os.path.join(checkpoint_out_dir, 'ckpt.pt'))
            if (epoch + 1) % args.save_ckpt_freq == 0:
                torch.save(checkpoint, os.path.join(checkpoint_out_dir, f'ckpt-{epoch}.pt'))

    if ddp: destroy_process_group()

def get_args():
    parser = argparse.ArgumentParser('VQ TUEV training script')
    parser.add_argument('--out_dir', default='./')
    parser.add_argument('--dataset_dir', default='./')
    parser.add_argument('--log_interval', default=10, type=int)
    parser.add_argument('--wandb_log', default=False, action='store_true')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--text_batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--save_ckpt_freq', default=10, type=int)
    parser.add_argument('--block_size', default=1024, type=int)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--decay_lr', default=True, action='store_false')
    parser.add_argument('--seed', default=1337, type=int)
    parser.add_argument('--compile', default=False, action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args)
    print('TUEV VQ Training Done!!!!')