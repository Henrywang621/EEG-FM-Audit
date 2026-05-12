import os
import time
import math
import argparse
from contextlib import nullcontext
import warnings

# Suppress harmless warnings to keep logs clean
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset

from model.model_neurolm import NeuroLM
from model.model_vq import VQ
from model.model_neural_transformer import NTConfig
from model.model import GPTConfig
from pathlib import Path
from utils import cosine_scheduler
from collections import OrderedDict

# BCI specific imports
from downstream_dataset import get_chans, extract_single_window
from scipy.signal import resample
from einops import rearrange

master_process = None; device = None; dtype = None
ctx = None; ddp_rank = None; device_type = None
ddp = None; ddp_world_size = None; ddp_local_rank = None


# ==========================================================
# BCICIV 2B DATALOADER (Adapted for GPT Pretraining)
# ==========================================================
class BCICIV2bLoader(Dataset):
    def __init__(self, subject_ids, root_path):
        self.root_path = root_path
        self.subject_ids = subject_ids
        
        # --- Windowing Parameters ---
        self.window_index = 8
        self.window_size = 500  # 2.0 seconds @ 250Hz
        self.step_size = 100
        
        self.original_rate = 250
        self.target_rate = 200  # NeuroLM Requirement
        self.ch_names = ['C3', 'CZ', 'C4']

        # 1. Load Data
        print(f"Loading data for subjects: {subject_ids}...")
        self.X, self.y = self.load_subject_data(self.subject_ids)
        print(f"Loaded. X: {self.X.shape}, Y: {self.y.shape}")

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
        return np.concatenate(data_list, axis=0), np.concatenate(labels_list, axis=0)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        X_raw = self.X[index]
        
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

        # Pretraining requires sequence lengths and validity masks
        input_mask = torch.ones(data.size(0), dtype=torch.bool)
        num_tokens = data.size(0)

        # Output tuple required by NeuroLM Pretraining loop
        return data, input_chans, input_time, input_mask, gpt_mask.bool(), num_chans, num_tokens


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


def main(args):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank

    init(args)

    if args.compile:
        if master_process:
            print("WARNING: Disabling torch.compile for stability.")
        args.compile = False

    checkpoint_out_dir = os.path.join(args.out_dir, 'checkpoints/NeuroLM-B_bciciv2b')
    if master_process:
        os.makedirs(checkpoint_out_dir, exist_ok=True)

    # --- 1. Text Data Loader ---
    data_dir = os.path.join(args.out_dir, 'text')
    def get_batch(split):
        if split == 'train':
            data = np.memmap(os.path.join("/homes/xw2336/data_portal/LLM_eva_fast/NeuroLM/text", 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join("/homes/xw2336/data_portal/LLM_eva_fast/NeuroLM/text", 'val.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - args.block_size, (args.text_batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i + args.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + args.block_size]).astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    # --- 2. EEG Data Loader (BCICIV 2b) ---
    train_subjects = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08']
    val_subjects = ['S09']

    dataset_train = BCICIV2bLoader(subject_ids=train_subjects, root_path=args.dataset_dir)
    dataset_val = BCICIV2bLoader(subject_ids=val_subjects, root_path=args.dataset_dir)

    if ddp:
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True)
        data_loader_train = torch.utils.data.DataLoader(dataset_train, sampler=sampler_train, batch_size=args.eeg_batch_size, num_workers=4, pin_memory=True, drop_last=True)
        sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False)
        data_loader_val = torch.utils.data.DataLoader(dataset_val, sampler=sampler_val, batch_size=int(1.5 * args.eeg_batch_size), num_workers=4, pin_memory=True, drop_last=False)
    else:
        data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.eeg_batch_size, num_workers=4, pin_memory=True, drop_last=True, shuffle=True)
        data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=int(1.5 * args.eeg_batch_size), num_workers=4, pin_memory=True, drop_last=False, shuffle=False)

    # --- 3. Load VQ Tokenizer ---
    encoder_args = dict(n_layer=12, n_head=12, n_embd=768, block_size=1024, bias=False, dropout=0., num_classes=0, in_chans=1, out_chans=16)
    decoder_args = dict(n_layer=4, n_head=12, n_embd=768, block_size=1024, bias=False, dropout=0., num_classes=0, in_chans=128)
    tokenizer_ckpt_path = os.path.join(args.out_dir, args.tokenizer_path)
    tokenizer_checkpoint = torch.load(tokenizer_ckpt_path, map_location="cpu", weights_only=False)
    
    tokenizer_checkpoint_model_args = tokenizer_checkpoint['encoder_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias']: encoder_args[k] = tokenizer_checkpoint_model_args[k]
    tokenizer_checkpoint_model_args = tokenizer_checkpoint['decoder_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias']: decoder_args[k] = tokenizer_checkpoint_model_args[k]
        
    encoder_conf = NTConfig(**encoder_args)
    decoder_conf = NTConfig(**decoder_args)
    tokenizer = VQ(encoder_conf, decoder_conf)
    tokenizer_state_dict = tokenizer_checkpoint['model']
    
    unwanted_prefix = '_orig_mod.'
    for k,v in list(tokenizer_state_dict.items()):
        if k.startswith(unwanted_prefix): tokenizer_state_dict[k[len(unwanted_prefix):]] = tokenizer_state_dict.pop(k)
            
    all_keys = list(tokenizer_state_dict.keys())
    new_dict = OrderedDict()
    for key in all_keys:
        if key.startswith('VQ.'): new_dict[key[3:]] = tokenizer_state_dict[key]
    tokenizer.load_state_dict(new_dict)
    tokenizer.eval()
    tokenizer.to(device)
    tokenizer_checkpoint = None

    # --- 4. Initialize / Resume Model ---
    init_from = 'resume' if os.path.exists(os.path.join(checkpoint_out_dir, 'ckpt.pt')) else 'gpt2'
    iter_num, start_epoch, best_val_loss = 0, 0, float('inf')
    model_args = dict(n_layer=12, n_head=12, n_embd=768, block_size=args.block_size, bias=False, vocab_size=50257, dropout=0.0) 
    
    if init_from == 'resume':
        ckpt_path = os.path.join(checkpoint_out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model_args.update(checkpoint['model_args'])
        gptconf = GPTConfig(**model_args)
        model = NeuroLM(gptconf, init_from='scratch')
        state_dict = checkpoint['model']
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix): state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    else:
        gptconf = GPTConfig(**model_args)
        model = NeuroLM(gptconf, tokenizer_ckpt_path, init_from=init_from)

    model.to(device)
    scaler = torch.amp.GradScaler(device_type, enabled=(dtype == 'float16'))
    optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type)
    if init_from == 'resume': optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None 

    if ddp: model = DDP(model, device_ids=[ddp_local_rank])

    if args.wandb_log and master_process:
        import wandb
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        wandb.init(project=args.wandb_project, name=args.wandb_runname, dir=os.path.join(args.out_dir, 'wandb'), resume=True)

    num_training_steps_per_epoch = len(data_loader_train)
    lr_schedule_values = cosine_scheduler(args.learning_rate, args.min_lr, args.epochs, num_training_steps_per_epoch, warmup_epochs=args.warmup_epochs)

    # --- 7. Training Loop ---
    X_text, Y_text = get_batch('train') 
    t0 = time.time()
    raw_model = model.module if ddp else model 
    tokenizer.to(torch.float32)

    for epoch in range(start_epoch, args.epochs):
        for step, (batch) in enumerate(data_loader_train):
            lr = lr_schedule_values[iter_num] if args.decay_lr else args.learning_rate
            for param_group in optimizer.param_groups: param_group['lr'] = lr

            # Unpacked 7 parameters matching modified BCICIV2bLoader output
            X_eeg, input_chans, input_time, input_mask, gpt_mask, num_chans, num_tokens = batch
            X_eeg = X_eeg.float().to(device, non_blocking=True)
            X_eeg = (X_eeg - X_eeg.mean(dim=-1, keepdim=True)) / (X_eeg.std(dim=-1, keepdim=True) + 1e-6)        
            
            input_chans, input_time, input_mask, gpt_mask = [t.to(device, non_blocking=True) for t in [input_chans, input_time, input_mask, gpt_mask]]

            with torch.no_grad():
                with ctx:
                    Y_eeg = torch.full((X_eeg.size(0), X_eeg.size(1)), fill_value=-1-raw_model.GPT2.config.vocab_size).to(device, non_blocking=True)
                    with torch.amp.autocast('cuda', enabled=False): 
                        codebook_indices = tokenizer.get_codebook_indices(X_eeg.float(), input_chans, input_time, input_mask)
                    for i, (num_chan, num_token) in enumerate(zip(num_chans, num_tokens)):
                        Y_eeg[i, :num_token - num_chan] = codebook_indices[i, num_chan:num_token]

            if ddp: model.require_backward_grad_sync = (step + 1) % args.gradient_accumulation_steps == 0
            with ctx:
                loss1, log1, _ = model(X_eeg, Y_eeg, None, None, input_chans, input_time, input_mask, eeg_mask=gpt_mask)
                loss2, log2, _ = model(None, None, X_text, Y_text)
                loss = (loss1 + loss2) / args.gradient_accumulation_steps 
            
            scaler.scale(loss).backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            X_text, Y_text = get_batch('train')
            if (iter_num + 1) % args.log_interval == 0 and master_process:
                print(f"epoch {epoch} step [{step + 1}/{num_training_steps_per_epoch}]: train loss {log1['train/loss'] + log2['train/loss']:.4f}", flush=True)
            iter_num += 1

        # --- Validation & Smart Checkpointing ---
        print(f"End of epoch {epoch}. Running validation...", flush=True)
        val_loss, val_accuracy = evaluate(model, raw_model, tokenizer, data_loader_val)
        
        if master_process:
            ppl = math.exp(val_loss) if not math.isnan(val_loss) else float('nan')
            print(f"Evaluate : loss {val_loss:.4f}, accuracy {val_accuracy:.4f}, perplexity {ppl:.4f}")
            
            checkpoint = {'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(), 'model_args': model_args, 'iter_num': iter_num, 'epoch': epoch, 'best_val_loss': best_val_loss}
            
            # Always save latest for resuming
            torch.save(checkpoint, os.path.join(checkpoint_out_dir, 'ckpt.pt'))

            # Save best model only if improved and not NaN
            if not math.isnan(val_loss) and val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint['best_val_loss'] = best_val_loss
                print(f"NEW BEST! Saving to ckpt_best.pt", flush=True)
                torch.save(checkpoint, os.path.join(checkpoint_out_dir, 'ckpt_best.pt'))

    if ddp: destroy_process_group()

@torch.no_grad()
def evaluate(model, raw_model, tokenizer, dataloader):
    model.eval()
    loss, acc = [], []
    for _, (batch) in enumerate(dataloader):
        X_eeg, input_chans, input_time, input_mask, gpt_mask, num_chans, num_tokens = batch
        X_eeg = X_eeg.to(device, non_blocking=True)
        X_eeg = (X_eeg - X_eeg.mean(dim=-1, keepdim=True)) / (X_eeg.std(dim=-1, keepdim=True) + 1e-6)
        input_chans, input_time, input_mask, gpt_mask = [t.to(device, non_blocking=True) for t in [input_chans, input_time, input_mask, gpt_mask]]
        
        Y_eeg = torch.full((X_eeg.size(0), X_eeg.size(1)), fill_value=-1-raw_model.GPT2.config.vocab_size).to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=False):
            codebook_indices = tokenizer.get_codebook_indices(X_eeg.float(), input_chans, input_time, input_mask)
        for i, (num_chan, num_token) in enumerate(zip(num_chans, num_tokens)):
            Y_eeg[i, :num_token - num_chan] = codebook_indices[i, num_chan:num_token]
        
        with ctx:
            _, log, _ = model(X_eeg, Y_eeg, None, None, input_chans, input_time, input_mask, eeg_mask=gpt_mask)
        loss.append(log['val/loss']); acc.append(log['val/accuracy'])
    model.train()
    return np.mean(loss) if loss else float('nan'), np.mean(acc) if acc else float('nan')

def get_args():
    parser = argparse.ArgumentParser('Pretraining script', add_help=False)
    parser.add_argument('--out_dir', default='./')
    parser.add_argument('--dataset_dir', default='/homes/xw2336/data/BCICIV_2b')
    parser.add_argument('--tokenizer_path', default='checkpoints/VQ_bciciv2b/ckpt.pt')
    parser.add_argument('--log_interval', default=10, type=int)
    parser.add_argument('--wandb_log', default=False, action='store_true')
    parser.add_argument('--wandb_project', default='NeuroLM')
    parser.add_argument('--wandb_runname', default='pretrain')
    parser.add_argument('--wandb_api_key', type=str)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--eeg_batch_size', default=16, type=int)
    parser.add_argument('--text_batch_size', default=4, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--warmup_epochs', default=2, type=int)
    parser.add_argument('--block_size', default=1024, type=int)
    parser.add_argument('--learning_rate', type=float, default=1e-6)
    parser.add_argument('--min_lr', type=float, default=6e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-1)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--decay_lr', default=True, action='store_false')
    parser.add_argument('--seed', default=1337, type=int)
    parser.add_argument('--compile', default=False, action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args)