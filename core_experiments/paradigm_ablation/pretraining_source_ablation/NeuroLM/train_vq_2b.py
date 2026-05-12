"""
by Wei-Bang Jiang
https://github.com/935963004/NeuroLM
Adapted for BCI Competition IV 2b Dataset
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
from torch.utils.data import Dataset

from model.model_vq import VQ_Align
from model.model_neural_transformer import NTConfig
from pathlib import Path
from utils import cosine_scheduler

# Ensure these are imported from your project utilities
from downstream_dataset import get_chans, extract_single_window
from scipy.signal import resample
from einops import rearrange
import tiktoken
import math
import torch._dynamo 

# Standard fix for DDP with torch.compile
torch._dynamo.config.optimize_ddp = False

master_process = None; device = None; dtype = None
ctx = None; ddp_rank = None; device_type = None
ddp = None; ddp_world_size = None; ddp_local_rank = None

# ==========================================================
# BCICIV 2B DATALOADER
# ==========================================================
class BCICIV2bLoader(Dataset):
    def __init__(self, subject_ids, root_path, is_instruct=False, is_val=False, 
                 eeg_max_len=1024, text_max_len=128):
        
        self.root_path = root_path
        self.subject_ids = subject_ids
        
        # --- Windowing Parameters ---
        self.window_index = 8
        self.window_size = 500  # 2.0 seconds @ 250Hz
        self.step_size = 100
        
        self.original_rate = 250
        self.target_rate = 200  # NeuroLM Requirement
        
        self.is_instruct = is_instruct
        self.is_val = is_val
        self.eeg_max_len = eeg_max_len
        self.text_max_len = text_max_len
        self.ch_names = ['C3', 'CZ', 'C4']

        # 1. Load Data
        print(f"Loading data for subjects: {subject_ids}...")
        self.X, self.y = self.load_subject_data(self.subject_ids)
        print(f"Loaded. X: {self.X.shape}, Y: {self.y.shape}")

        # 2. Dynamic Label Mapping
        unique_labels = np.unique(self.y)
        unique_labels.sort()
        
        if len(unique_labels) >= 2:
            self.label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
        else:
            self.label_map = {unique_labels[0]: 0}

        # 3. Setup Tokenizer
        if is_instruct:
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            q_str = 'Question: Is the subject imagining a right hand movement? Answer:'
            
            self.text = {
                0: torch.IntTensor([50257] + encode(q_str + ' No <|endoftext|>')),
                1: torch.IntTensor([50257] + encode(q_str + ' Yes <|endoftext|>'))
            }
            self.prompt = torch.IntTensor([50257] + encode(q_str))

    def load_subject_data(self, subject_ids):
        data_list = []
        labels_list = []
        
        Subj_id_map = {'S01': 1, 'S02': 2, 'S03': 3, 'S04': 4, 'S05': 5, 
                       'S06': 6, 'S07': 7, 'S08': 8, 'S09': 9}

        for subj_str in subject_ids:
            sid = Subj_id_map.get(subj_str, 1)
            
            paths = [
                (os.path.join(self.root_path, f"Subj{sid}_1_X.npy"), os.path.join(self.root_path, f"Subj{sid}_1_y.npy")),
                (os.path.join(self.root_path, f"Subj{sid}_2_X.npy"), os.path.join(self.root_path, f"Subj{sid}_2_y.npy"))
            ]

            for x_path, y_path in paths:
                if not os.path.exists(x_path): continue
                
                x_data = np.load(x_path)
                y_data = np.load(y_path)
                
                x_data = np.nan_to_num(x_data, nan=0.0)

                if x_data.shape[1] > x_data.shape[2] and x_data.shape[2] == 3:
                     x_data = x_data.transpose(0, 2, 1)

                x_sliced = extract_single_window(x_data, self.window_index, self.window_size, self.step_size)
                
                data_list.append(x_sliced)
                labels_list.append(y_data)

        if not data_list:
            return np.array([]), np.array([])

        X_all = np.concatenate(data_list, axis=0)
        y_all = np.concatenate(labels_list, axis=0)
        return X_all, y_all

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        X_raw = self.X[index]
        Y_raw = self.y[index]
        Y_mapped = self.label_map[Y_raw]
        
        new_samples = int(X_raw.shape[1] * self.target_rate / self.original_rate)
        X_resampled = resample(X_raw, new_samples, axis=1)
        data = torch.FloatTensor(X_resampled / 100.0)

        num_seconds = data.size(1) // 200
        input_time = [i for i in range(num_seconds) for _ in range(data.size(0))]
        data = rearrange(data, 'C (S T) -> (S C) T', T=200)
        
        ch_names_list = self.ch_names * num_seconds
        input_chans = torch.IntTensor(get_chans(ch_names_list))
        input_time = torch.IntTensor(input_time)
        
        gpt_mask = torch.tril(torch.ones(data.size(0), data.size(0))).view(1, data.size(0), data.size(0))
        num_chans = len(self.ch_names)
        for i in range(num_seconds):
            s, e = i*num_chans, (i+1)*num_chans
            gpt_mask[:, s:e, s:e] = 1

        if not self.is_instruct:
            return data, Y_mapped, input_chans, input_time, gpt_mask.bool()

        # Instruct logic omitted for brevity as it is unused in VQ training
        pass

# ==========================================================
# TRAINING INITIALIZATION
# ==========================================================
def init(args):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank
    backend = 'nccl'
    device = 'cuda'
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

# ==========================================================
# MAIN TRAINING LOOP
# ==========================================================
def main(args):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank

    init(args)

    # Changed output dir to BCICIV2b
    checkpoint_out_dir = os.path.join(args.out_dir, 'checkpoints/VQ_bciciv2b')
    if master_process:
        os.makedirs(checkpoint_out_dir, exist_ok=True)

    data_dir = os.path.join('/homes/xw2336/data_portal/LLM_eva_fast/NeuroLM', 'text')
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

    # DATA LOADING: Initialized with BCICIV2bLoader
    print(f'Initializing BCICIV 2b Training Loader...')
    train_subjects = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09']
    
    dataset_train = BCICIV2bLoader(
        subject_ids=train_subjects, 
        root_path=args.dataset_dir, 
        is_instruct=False # VQ training doesn't use the text-instruction formatting
    )

    if ddp:
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True)
        data_loader_train = torch.utils.data.DataLoader(dataset_train, sampler=sampler_train, batch_size=args.batch_size, num_workers=10, pin_memory=True, drop_last=True)
    else:
        data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, num_workers=10, pin_memory=True, drop_last=True, shuffle=True)

    iter_num = 0
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

    X_text, Y_text = get_batch('train') 
    t0 = time.time()
    raw_model = model.module if ddp else model 
    
    for epoch in range(start_epoch, args.epochs):
        for step, (batch) in enumerate(data_loader_train):
            lr = lr_schedule_values[iter_num] if args.decay_lr else args.learning_rate
            for param_group in optimizer.param_groups: param_group['lr'] = lr

            if ddp: model.require_backward_grad_sync = (step + 1) % args.gradient_accumulation_steps == 0
            
            # --- DYNAMIC TARGET GENERATION ---
            # Unpack BCICIV2b Loader Output
            data, Y_mapped, input_chans, input_time, gpt_mask = batch
            
            # Setup X and compute Y_raw & Y_freq targets dynamically
            X = data.float().to(device, non_blocking=True)
            Y_raw = X.clone()
            Y_freq = torch.abs(torch.fft.rfft(X, dim=-1))[..., :-1]
            
            # Create a 1D validity mask for the sequence (all ones since no padding in this mode)
            input_mask = torch.ones(X.shape[:2], dtype=torch.bool, device=device)

            # NORMALIZATION
            x_mean = X.mean(dim=-1, keepdim=True)
            x_std = X.std(dim=-1, keepdim=True)
            X = (X - x_mean) / (x_std + 1e-6)
            Y_raw = (Y_raw - x_mean) / (x_std + 1e-6)
            Y_freq = Y_freq / (x_std + 1e-6)

            input_chans = input_chans.to(device, non_blocking=True)
            input_time = input_time.to(device, non_blocking=True)

            with ctx:
                alpha = 2 / (1 + math.exp(-10 * iter_num / (args.epochs * num_training_steps_per_epoch))) - 1
                loss, domain_loss, log = model(X, Y_freq, Y_raw, input_chans, input_time, input_mask, alpha)
                domain_loss2 = model(X_text)
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
        
        if master_process:
            checkpoint = {'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(), 'encoder_args': encoder_args, 'decoder_args': decoder_args, 'iter_num': iter_num, 'epoch': epoch}
            torch.save(checkpoint, os.path.join(checkpoint_out_dir, 'ckpt.pt'))
            if (epoch + 1) % args.save_ckpt_freq == 0:
                torch.save(checkpoint, os.path.join(checkpoint_out_dir, f'ckpt-{epoch}.pt'))

    if ddp: destroy_process_group()

def get_args():
    parser = argparse.ArgumentParser('VQ BCICIV 2b training script')
    parser.add_argument('--out_dir', default='./')
    parser.add_argument('--dataset_dir', default='/homes/xw2336/data/BCICIV_2b')
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
    print('BCICIV 2b VQ Training Done!!!!')