import os
import time
import argparse
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader, DistributedSampler

# NeuroLM Specific Imports
from model.model_neurolm import NeuroLM
from model.model import GPTConfig
from pathlib import Path
import tiktoken
from datetime import timedelta
from utils import prepare_TUAB_dataset, cosine_scheduler, get_metrics

# Global DDP Variables
master_process = None; device = None; dtype = None
ctx = None; ddp_rank = None; ddp_local_rank = None; ddp_world_size = None

def init_ddp():
    global ctx, master_process, ddp_rank, ddp_local_rank, ddp_world_size, device, dtype
    init_process_group(backend='nccl', timeout=timedelta(minutes=30))
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0)
    
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)

def get_loaders(args, seed):
    # EXTRACTED FROM TUAB SCRIPT: Path and parameter configuration
    dataset_train, dataset_test, dataset_val = prepare_TUAB_dataset(
        Path(args.dataset_dir, 'data_portal/TUAB/processed'), 
        is_instruct=True, eeg_max_len=276, text_max_len=80
    )
    
    # TUAB Metadata for Binary Classification
    info = {
        'name': 'TUAB', 
        'is_binary': True, 
        'num_classes': 2, 
        'result_idx': 7,
        'metrics': ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy", "f1", "cohen_kappa"],
        'label_dic': {'Yes': 1, 'No': 0},
        'dataset_val': dataset_val, 
        'dataset_test': dataset_test
    }

    sampler = DistributedSampler(dataset_train, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True, seed=seed)
    train_loader = DataLoader(dataset_train, sampler=sampler, batch_size=args.eeg_batch_size, num_workers=4, pin_memory=True)
    
    val_loader = DataLoader(dataset_val, batch_size=args.eeg_batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(dataset_test, batch_size=args.eeg_batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader, info

@torch.no_grad()
def evaluate(model, info, loader, decode):
    model.eval()
    preds, targets = [], []
    for batch in loader:
        X_eeg, X_text, label, chans, t_steps, eeg_mask, gpt_mask = [b.to(device, non_blocking=True) if isinstance(b, torch.Tensor) else b for b in batch]
        with ctx:
            generated = model.generate(X_eeg.float(), X_text, chans, t_steps, eeg_mask, eeg_text_mask=gpt_mask, max_new_tokens=5)
            for t in generated[:, 1:]:
                pred_str = decode(t.tolist())
                try:
                    p = pred_str.split(' ')[info['result_idx']]
                    # Basic cleaning to catch labels like "Yes" or "No"
                    p = p.strip().capitalize()
                    if p.startswith('('): p = p[:3] 
                    
                    val = info['label_dic'][p]
                    # TUAB Binary Handling: get_metrics expects scalars for binary
                    preds.append(val)
                except:
                    # Fallback for parsing errors
                    preds.append(0)
            targets.append(label.cpu())
    
    model.train()
    return get_metrics(np.array(preds), torch.cat(targets).numpy(), info['metrics'], info['is_binary'])

def main(args):
    init_ddp()
    seeds = [int(s) for s in args.seeds.split(',')]
    all_seed_results = []
    enc = tiktoken.get_encoding("gpt2")
    decode = lambda l: enc.decode(l)

    for seed in seeds:
        if master_process: print(f"\n>>> Starting Seed: {seed}")
        torch.manual_seed(seed + ddp_rank)
        
        # 1. Initialize Model Configuration
        model_args = dict(n_layer=12, n_head=12, n_embd=768, block_size=args.block_size, 
                          bias=True, vocab_size=50257, dropout=0.1)
        conf = GPTConfig(**model_args)
        model = NeuroLM(conf, init_from='scratch')
        
        # 2. Weight Loading
        ckpt = torch.load(os.path.join(args.out_dir, args.NeuroLM_path), map_location=device, weights_only=False)
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()}
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        
        if args.compile: model = torch.compile(model)
        model = DDP(model, device_ids=[ddp_local_rank])
        raw_model = model.module

        # 3. Setup Data & Optim
        train_loader, val_loader, test_loader, info = get_loaders(args, seed)
        optimizer = raw_model.configure_optimizers(args.weight_decay, args.learning_rate, (0.9, 0.95), 'cuda')
        scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))
        lr_schedule = cosine_scheduler(args.learning_rate, args.min_lr, args.epochs, len(train_loader))

        best_val_bacc = -1.0
        best_test_metrics = {}

        # 4. Training Loop
        for epoch in range(args.epochs):
            train_loader.sampler.set_epoch(epoch)
            for step, batch in enumerate(train_loader):
                it = epoch * len(train_loader) + step
                for param_group in optimizer.param_groups: param_group['lr'] = lr_schedule[it]

                X_eeg, X_text, Y_text, chans, t_steps, eeg_mask, gpt_mask = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
                Y_eeg = torch.full((X_eeg.size(0), X_eeg.size(1)), -1-50257).to(device)

                with ctx:
                    loss, _, _ = model(X_eeg.float(), Y_eeg, X_text, Y_text, chans, t_steps, eeg_mask, eeg_text_mask=gpt_mask)
                    loss = loss / args.gradient_accumulation_steps
                
                scaler.scale(loss).backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            # 5. Evaluation
            val_res = evaluate(raw_model, info, val_loader, decode)
            test_res = evaluate(raw_model, info, test_loader, decode)
            
            if val_res['balanced_accuracy'] > best_val_bacc:
                best_val_bacc = val_res['balanced_accuracy']
                best_test_metrics = test_res
                if master_process:
                    print(f"Seed {seed} | Ep {epoch} | Best Val BACC: {best_val_bacc:.4f} | Test BACC: {test_res['balanced_accuracy']:.4f}")
                    # REPRODUCTION-FRIENDLY SAVE: Includes model_args and config
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'model_args': model_args,
                        'seed': seed,
                        'epoch': epoch,
                        'val_bacc': best_val_bacc
                    }
                    torch.save(checkpoint, os.path.join(args.out_dir, f'ckpt_tuab_best_seed_{seed}.pt'))
            dist.barrier()

        all_seed_results.append(best_test_metrics)
        del model, optimizer, train_loader; gc.collect(); torch.cuda.empty_cache()
        dist.barrier()

    # 6. Final Aggregate Reporting
    if master_process:
        print("\n" + "="*50 + "\nFINAL TUAB AGGREGATED TEST RESULTS (Mean ± Std)\n" + "="*50)
        for m in ['balanced_accuracy', 'f1_weighted', 'cohen_kappa']:
            if m in all_seed_results[0]:
                vals = [r[m] for r in all_seed_results]
                print(f"{m:<20}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--NeuroLM_path', type=str, required=True)
    parser.add_argument('--seeds', type=str, default="16")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--eeg_batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--min_lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-1)
    parser.add_argument('--block_size', type=int, default=1024)
    parser.add_argument('--compile', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    main(get_args())